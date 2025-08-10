# https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html

# train_frcnn_pennfudan.py
import os
os.getcwd()
os.chdir(r'D:\DataBase\Data_Image')

import torch 
import torchvision
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.ops as ops

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1) Dataset: Penn-Fudan
# 폴더 구조 예시:
#   PennFudanPed/
#     PNGImages/
#     PedMasks/
# 다운로드: https://www.cis.upenn.edu/~jshi/ped_html/


#########################################################################################################################
class PennFudanDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # 이미지/마스크 로드
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask_np = np.array(mask)
        obj_ids = np.unique(mask_np)
        obj_ids = obj_ids[1:]  # 0은 배경

        masks = mask_np == obj_ids[:, None, None]

        boxes = []
        for m in masks:
            pos = np.where(m)
            if len(pos[0]) == 0 or len(pos[1]) == 0:
                # 빈 마스크 방지
                boxes.append([0,0,1,1])
                continue
            ymin, xmin = np.min(pos[0]), np.min(pos[1])
            ymax, xmax = np.max(pos[0]), np.max(pos[1])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)  # 사람=1
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)  # crowd 없음

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


#########################################################################################################################
def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(train=True):
    t = [T.ToTensor()]
    if train:
        # 간단한 수평 뒤집기
        t.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(t)


#########################################################################################################################
# 2) Model: Faster R-CNN (ResNet50-FPN v2) 미세튜닝
def get_model(num_classes=2):
    # pretrained weights로 백본/헤드 초기화
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    # 분류기 헤드 교체 (배경 포함 클래스 수)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model


model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
model
model.transform
model.backbone
model.backbone.body
model.rpn
model.roi_heads

model.roi_heads.box_predictor

# for name, module in model.backbone.body.named_children():
#     print(name, ":", module.__class__.__name__)


model.roi_heads.box_predictor.cls_score


#########################################################################################################################
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    losses_avg = 0.0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses_avg += loss.item()
    return losses_avg / len(data_loader)

@torch.no_grad()
def evaluate_and_visualize(model, dataset, device, save_dir="pred_vis"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    for i in range(min(5, len(dataset))):
        img, _ = dataset[i]
        img_gpu = img.to(device)
        outputs = model([img_gpu])[0]
        # 시각화 저장(간단히 PIL에 그리기)
        to_pil = T.ToPILImage()
        im = to_pil(img)
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(im)
        boxes = outputs["boxes"].cpu()
        scores = outputs["scores"].cpu()
        labels = outputs["labels"].cpu()
        for b, s, l in zip(boxes, scores, labels):
            if s < 0.5: 
                continue
            x1, y1, x2, y2 = b.tolist()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), f"person {s:.2f}", fill="red")
        im.save(os.path.join(save_dir, f"pred_{i}.png"))


def draw_boxes(bboxes, color='r', linewidth=2):
    """
    현재 figure의 이미지 위에 bounding box를 그려주는 함수
    bboxes: Tensor 또는 list, [[x_min, y_min, x_max, y_max], ...]
    """
    ax = plt.gca()  # 현재 axis 가져오기
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=linewidth, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)



#########################################################################################################################
# import torchvision
# from torchvision.models.detection import FasterRCNN
# from torchvision.models.detection.rpn import AnchorGenerator

# # 1. ResNet18 불러오기
# backbone = torchvision.models.resnet18(weights="IMAGENET1K_V1")  # pretrained=True 와 동일
# # 마지막 FC 제거
# backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))
# backbone.out_channels = 512  # ResNet18 마지막 레이어 출력 채널 수

# # 2. RPN Anchor 설정
# anchor_generator = AnchorGenerator(
#     sizes=((32, 64, 128, 256, 512),),
#     aspect_ratios=((0.5, 1.0, 2.0),)
# )

# # 3. ROI Pooler 설정
# roi_pooler = torchvision.ops.MultiScaleRoIAlign(
#     featmap_names=['0'], output_size=7, sampling_ratio=2
# )

# # 4. Faster R-CNN 모델 생성
# model = FasterRCNN(
#     backbone,
#     num_classes=2,  # background + 1 class
#     rpn_anchor_generator=anchor_generator,
#     box_roi_pool=roi_pooler
# )


#########################################################################################################################
root = "PennFudanPed"   # 압축 푼 폴더 경로로 변경
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train/Val split (간단히 앞뒤로 분리)
dataset = PennFudanDataset(root, transforms=get_transform(train=False))
dataset_test = PennFudanDataset(root, transforms=get_transform(train=False))

# indices = torch.randperm(len(dataset)).tolist()     # permute indices
indices = torch.arange(len(dataset))

split = int(0.8 * len(indices))
dataset = torch.utils.data.Subset(dataset, indices[:split])
dataset_test = torch.utils.data.Subset(dataset_test, indices[split:])

train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)
test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

# Explore Dataset -----------------------------------------------------------------------------
# Dataset Sample
dataset[0][1].keys()       # ['boxes', 'labels', 'masks', 'image_id', 'area', 'iscrowd'])
dataset[0][1]['boxes']
dataset[0][1]['labels']
dataset[0][1]['masks']      # segmentation mask
dataset[0][1]['image_id']
dataset[0][1]['area']
dataset[0][1]['iscrowd']        # 객체 군집 여부 (Tensor of shape [N], 값 0 또는 1)

# image + box
plt.imshow(dataset[0][0].permute(1,2,0))
draw_boxes(dataset[0][1]['boxes'])
plt.show()

# mask
plt.imshow(dataset[0][1]['masks'][0] + dataset[0][1]['masks'][1])
plt.show()
#----------------------------------------------------------------------------------------------
#########################################################################################################################


from torchvision.ops import nms, roi_align


# -----------------------------
# A. 간단한 ImageList 컨테이너(배치 텐서 + 각 이미지 크기)
# -----------------------------
class ImageList:
    def __init__(self, tensors: torch.Tensor, sizes):
        self.tensors = tensors   # B x C x Hmax x Wmax (패딩됨)
        self.sizes = sizes       # [(h1, w1), (h2, w2), ...] 리사이즈 이후 크기


# -----------------------------
# B. GeneralizedRCNNTransform-lite
#    - 비율유지 resize(min_size/max_size)
#    - normalize(Imagenet mean/std)
#    - pad to batch
#    - rois 스케일 보정
# -----------------------------
class SimpleRCNNTransform(nn.Module):
    def __init__(self, min_size=800, max_size=1333,
                 image_mean=(0.485, 0.456, 0.406),
                 image_std=(0.229, 0.224, 0.225)):
        super().__init__()
        # min_size: 짧은 변을 이 값으로 맞춤(비율 유지)
        # max_size: 긴 변이 이 값을 넘지 않도록 스케일 제한
        self.min_size = min_size if isinstance(min_size, (list, tuple)) else (min_size,)
        self.max_size = max_size
        self.register_buffer("mean", torch.tensor(image_mean).view(-1,1,1))
        self.register_buffer("std",  torch.tensor(image_std).view(-1,1,1))

    @torch.no_grad()
    def _get_size(self, h, w, min_size, max_size):
        # torchvision과 동일한 규칙: 짧은변 -> min_size, 긴변은 max_size 초과하지 않게 단일 scale 사용
        scale = min_size / min(h, w)
        if max(h, w) * scale > max_size:
            scale = max_size / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        return (new_h, new_w), scale

    @torch.no_grad()
    def _resize(self, img):
        # img: CxHxW
        C, H, W = img.shape
        # 학습 시 멀티스케일 하고 싶으면 self.min_size에서 랜덤 선택 가능
        min_size = self.min_size[0]
        (new_h, new_w), scale = self._get_size(H, W, min_size, self.max_size)
        if (new_h, new_w) != (H, W):
            img = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
        return img, scale

    @torch.no_grad()
    def _batch_images(self, images, size_divisible=32):
        # 배치 패딩: 모든 이미지를 Hmax×Wmax로 0-padding (32 배수로 맞추면 FPN 등에 유리)
        max_h = max([img.shape[1] for img in images])
        max_w = max([img.shape[2] for img in images])
        # 선택: 32 배수에 맞추기
        max_h = (max_h + size_divisible - 1) // size_divisible * size_divisible
        max_w = (max_w + size_divisible - 1) // size_divisible * size_divisible

        batch = images[0].new_zeros((len(images), images[0].shape[0], max_h, max_w))
        for i, img in enumerate(images):
            c, h, w = img.shape
            batch[i, :, :h, :w] = img
        return batch

    @torch.no_grad()
    def forward(self, images, rois=None):
        """
        images: list[Tensor(C,H,W)]  (크기 제각각 허용)
        rois:   list[Tensor(K_i,4)]  (각 이미지의 xyxy, 이미지 좌표계 기준) 또는 None
        return: ImageList, scaled_rois(list) or None, scales(list)
        """
        out_images = []
        sizes = []
        scales = []
        scaled_rois = [] if rois is not None else None

        for i, img in enumerate(images):
            # float tensor [0,1] 범위 가정. 0~255라면 /255. 해주세요.
            img, scale = self._resize(img)
            img = (img - self.mean) / self.std  # 정규화
            out_images.append(img)
            sizes.append(img.shape[-2:])       # (H', W')
            scales.append(scale)

            if rois is not None:
                boxes = rois[i].clone()
                # 비율 유지 단일 scale → x,y 모두 동일 배율
                boxes *= scale
                scaled_rois.append(boxes)

        batch = self._batch_images(out_images)  # BxCxHmaxxWmax
        return ImageList(batch, sizes), scaled_rois, scales
    
# ----------------------------------------------------------------------------

transform = SimpleRCNNTransform(min_size=300, max_size=600)
transform()


# -----------------------------
# 1. Backbone (간단한 CNN)
# -----------------------------
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU()
        )
    
    def forward(self, x):
        return self.features(x)  # [B, 128, H, W]


# -----------------------------
# Anchor 생성 함수
# -----------------------------
def generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=[8, 16, 32]):
    anchors = []
    for scale in scales:
        for ratio in ratios:
            w = base_size * scale * (ratio ** 0.5)
            h = base_size * scale / (ratio ** 0.5)
            x1 = -w / 2
            y1 = -h / 2
            x2 = w / 2
            y2 = h / 2
            anchors.append([x1, y1, x2, y2])
    return torch.tensor(anchors)  # [num_anchors, 4]

# -----------------------------
# 2. RPN
# -----------------------------
class RPN(nn.Module):
    def __init__(self, in_channels, num_anchors=9, pre_nms_topk=600, post_nms_topk=100, nms_thresh=0.7):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.cls_logits = nn.Conv2d(256, num_anchors, 1)         # objectness
        self.bbox_pred = nn.Conv2d(256, num_anchors * 4, 1)      # Δbbox
        self.anchors = generate_anchors()
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.nms_thresh = nms_thresh

    def forward(self, features, image_size):
        B, _, H, W = features.shape
        t = F.relu(self.conv(features))
        logits = self.cls_logits(t)       # [B, A, H, W]
        bbox_deltas = self.bbox_pred(t)   # [B, A*4, H, W]

        # (1) 모든 위치의 anchors 생성
        shift_x = torch.arange(0, W) * 16
        shift_y = torch.arange(0, H) * 16
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=2)  # [H, W, 4]
        shifts = shifts.reshape(-1, 4)

        anchors_all = (self.anchors[None, :, :] + shifts[:, None, :]).reshape(-1, 4)  # [H*W*A, 4]

        proposals_batch = []
        for b in range(B):
            # (2) 현재 배치의 scores / deltas
            scores = logits[b].permute(1, 2, 0).reshape(-1)  # [H*W*A]
            deltas = bbox_deltas[b].permute(1, 2, 0).reshape(-1, 4)

            # (3) Δbbox 적용
            proposals = self.apply_deltas_to_anchors(deltas, anchors_all)

            # (4) 이미지 경계로 clip
            proposals[:, 0::2] = proposals[:, 0::2].clamp(0, image_size[1])
            proposals[:, 1::2] = proposals[:, 1::2].clamp(0, image_size[0])

            # (5) Pre-NMS top-k
            topk_idx = scores.topk(self.pre_nms_topk)[1]
            proposals = proposals[topk_idx]
            scores = scores[topk_idx]

            # (6) NMS
            keep = nms(proposals, scores, self.nms_thresh)[:self.post_nms_topk]
            proposals_batch.append(proposals[keep])

        return proposals_batch  # list[Tensor[N,4]]

    @staticmethod
    def apply_deltas_to_anchors(deltas, anchors):
        # anchors: [N, 4], deltas: [N, 4]
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights

        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dw = deltas[:, 2]
        dh = deltas[:, 3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = widths * torch.exp(dw)
        pred_h = heights * torch.exp(dh)

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes


# -----------------------------
# 3. Detection Head
# -----------------------------
class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)   # 클래스
        self.bbox_pred = nn.Linear(1024, num_classes*4) # bbox

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        scores = self.cls_score(x)
        bbox = self.bbox_pred(x)
        return scores, bbox


# -----------------------------
# 4. Faster R-CNN 통합
# -----------------------------
class FasterRCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = Backbone()
        self.rpn = RPN(128)
        self.head = DetectionHead(128 * 7 * 7, num_classes)     # RoI Align 후 7x7 feature

    def forward(self, images):
        B, _, H, W = images.shape
        
        # 1. Feature 추출
        features = self.backbone(images)

        # 2. RPN
        proposals = self.rpn(features, image_size=(H, W))  # RPN → proposals

        # 3. RoI Align (여기서는 이미 주어진 rois 사용)
        roi_list = []
        for b_idx, props in enumerate(proposals):
            batch_idx = torch.full((props.size(0), 1), b_idx, dtype=torch.float)
            roi_list.append(torch.cat([batch_idx, props], dim=1))
        rois = torch.cat(roi_list, dim=0)

        roi_features = roi_align(features, rois, output_size=(7, 7), spatial_scale=features.shape[2] / H)
        roi_features = roi_features.view(roi_features.size(0), -1)

        # 4. Detection Head
        cls_scores, bbox_preds = self.head(roi_features)

        return cls_scores, bbox_preds, proposals

a = next(iter(train_loader))

for i in range(10):
    print(dataset[i][0].unsqueeze(0).shape)







for images, targets in train_loader:
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]







model = get_model(num_classes=2).to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 3
for epoch in range(num_epochs):
    loss_avg = train_one_epoch(model, optimizer, train_loader, device)
    lr_scheduler.step()
    print(f"[Epoch {epoch+1}/{num_epochs}] loss: {loss_avg:.4f}")

# 간단 평가/시각화
evaluate_and_visualize(model, dataset_test, device, save_dir="pred_vis")

# 모델 저장
torch.save(model.state_dict(), "fasterrcnn_pennfudan.pth")
print("Saved: fasterrcnn_pennfudan.pth")


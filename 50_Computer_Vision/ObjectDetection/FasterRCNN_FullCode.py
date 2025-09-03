import os
import torch
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets

##################################################################

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # 이미지 파일 목록
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # 이미지와 마스크 경로
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # mask: (H, W), 값이 0(배경), 1~N(사람 인스턴스)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]  # 0은 배경이므로 제외

        # 각 인스턴스별 마스크 생성
        masks = mask == obj_ids[:, None, None]

        # 바운딩 박스 계산
        boxes = []
        for m in masks:
            pos = np.where(m)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # 모든 사람의 라벨은 1 (person)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = torch.tensor([idx])

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

def compute_regression_targets(proposals, gt_boxes):
    px, py, pw, ph = proposals[:, 0], proposals[:, 1], proposals[:, 2], proposals[:, 3]
    gx, gy, gw, gh = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]

    tx = (gx - px) / pw
    ty = (gy - py) / ph
    tw = torch.log(gw / pw)
    th = torch.log(gh / ph)

    targets = torch.stack([tx, ty, tw, th], dim=1)
    return targets

# Bounding Box -> "label img" program? library?

# Region Proposal (후보 박스 생성)
# Feature Extraction (각 박스마다 CNN feature 추출)
# Classification & Regression (각 박스 feature로 클래스/좌표 예측)


# Backbone CNN 
#   Feature 추출

# Region Proposal Network (RPN) 
#   RPN은 feature map에서 작은 윈도우(Anchor)마다 아래 내용을 동시에 예측 
#   1. 여기 물체 있을 확률(objectness)
#   2. anchor를 어떻게 이동/확장하면 더 정확한 box가 될지(bbox regression)

# Proposal 생성
#   RPN의 결과를 바탕으로 "진짜 물체일 것 같은 후보 박스(top-N proposal)"를 뽑음
#   (이후 ROI Pooling, ROI Head에서 class 분류/box refinement 진행)
#   (실제로는 anchor와 proposal의 NMS(Non-Maximum Suppression) 등 후처리도 필요)

# RoI Pooling (또는 RoI Align)

# Detection Head


# bounding box : label img

#############################################################################
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),                  # 0~1로 변환
    # transforms.RandomHorizontalFlip(),      # 좌우 반전
    # transforms.RandomRotation(10),          # 10도 이내로 회전
    # transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 밝기/대비 변화
    # transforms.RandomResizedCrop(224),      # 랜덤 크롭 후 리사이즈
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 정규화
])


dataset_root = r'D:\DataBase\Data_Image\PennFudanPed'
dataset = PennFudanDataset(dataset_root, transforms=train_transform)

scale_h = 256 / 536
scale_w = 256 / 559

img, info = dataset[0]
info.keys()     # ['boxes', 'labels', 'masks', 'image_id']

bboxes_scaled = info['boxes'].clone()
bboxes_scaled[:, [0, 2]] *= scale_w  # x_min, x_max
bboxes_scaled[:, [1, 3]] *= scale_h  # y_min, y_max


plt.imshow(img.permute(1,2,0))
for bbox in bboxes_scaled:
    x_min, y_min, x_max, y_max = bbox.tolist()
    # 네 꼭짓점 좌표
    xs = [x_min, x_max, x_max, x_min, x_min]
    ys = [y_min, y_min, y_max, y_max, y_min]
    plt.plot(xs, ys, color='r', linewidth=2)
plt.show()



###########################################################################################
###########################################################################################
###########################################################################################
import torch.nn as nn

################################################################################
class Backbone(nn.Module):
    def __init__(self, out_channels=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, out_channels//8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(out_channels//8, out_channels//4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(out_channels//4, out_channels//2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(out_channels//2, out_channels, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.out_channels = out_channels
    
    def forward(self, x):
        return self.features(x)  # [B, 128, H, W]


input_img_sample = torch.rand(5,3,256, 256)    # 5,3,256,256
bb_model = Backbone()
features = bb_model(input_img_sample)
features.shape      # 5, 32, 16, 16



################################################################################
scales = [128, 256, 512]
ratios = [0.5, 1, 2]

class SimpleRPN(nn.Module):
    def __init__(self, in_channels, n_anchors):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.cls_logits = nn.Conv2d(64, n_anchors, 1)  # objectness score
        self.bbox_pred = nn.Conv2d(64, n_anchors * 4, 1)  # bbox regression

    def forward(self, x):
        t = torch.relu(self.conv(x))
        logits = self.cls_logits(t)  # (N, n_anchors, H, W)
        bbox_reg = self.bbox_pred(t)  # (N, n_anchors*4, H, W)
        return logits, bbox_reg

rpn_model = SimpleRPN(bb_model.out_channels, n_anchors=len(scales)*len(ratios))
# anchor : anchors는 CNN feature map의 각 위치마다 미리 정의된 다양한 크기와 비율의 후보 박스들이다.

rpn_logit, rpn_bbox = rpn_model(features)
rpn_logit.shape     # 5, n_anchors, 16, 16      # 내가 관심 있는 k개 클래스의 합집합(=object) (1- p_background)의 확률
rpn_bbox.shape      # 5, n_anchors*4, 16, 16    # 해당 anchor의 box 위치좌표 예측 (가장 object가 있을 확률이 높다고 생각되는 rpn_bbox 좌표 )



################################################################################
# (proposal box를 만드는 과정)
# 1. anchors를 만듭니다.  (generate_anchors 함수)
# 2. RPN이 각 anchor마다
#   objectness score(rpn_logit)와
#   bbox regression 값(rpn_bbox)을 예측합니다.
# 3. bbox regression 값(rpn_bbox)을 anchors에 적용해서
#   실제 proposal box(=object에 더 잘 맞는 박스)를 만듭니다.(이때 apply_deltas_to_anchors 함수 사용)

# generate_anchors -> anchors : 아직 아무런 예측 정보가 없는, 기준이 되는 박스
def generate_anchors(H, W, scales, ratios, stride=1):
    anchors = []
    for y in range(H):
        for x in range(W):
            cx = x * stride
            cy = y * stride
            for scale in scales:
                for ratio in ratios:
                    w = scale * (ratio ** 0.5)
                    h = scale / (ratio ** 0.5)
                    anchors.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
    return torch.tensor(anchors)  # (num_anchors, 4)

# Anchor 생성 및 Proposal 생성 (여기선 임의로 100개 proposal 사용)
H, W = features.shape[2:]       # H : feature_h, W : feature_w
anchors = generate_anchors(H, W, scales=scales, ratios=ratios, stride=16)      # stride : 원본크기에 맞추기 위한 parameter
anchors.shape       # (n_anchors * H * W, 4)



########################################################################
from torchvision.ops import nms     # 겹치는 박스 중 가장 점수 높은 것만 남기고 나머지 제거하는 함수
# rpn_logit.shape     # 5, n_ac, 16, 16
# rpn_bbox.shape      # 5, n_ac*4, 16, 16


# 1. RPN 결과 펼치기 (B=1 가정)
B = rpn_logit.shape[0]      # batch_size
n_anchors = rpn_logit.shape[1]      # CNN_feature별 anchor 수
H, W

# objectness score: (B, n_anchors, H, W) -> (B, n_anchors*H*W)
#   ㄴ 각 anchor_box(=anchor 종류 × feature map 위치)마다 object가 있을 확률
objectness = rpn_logit.sigmoid().reshape(B, -1)     


# bbox regression: (B, n_anchors*4, H, W) -> (B, n_anchors, 4, H, W) -> (B, n_anchors*H*W, 4)
bbox_reg = rpn_bbox.reshape(B, n_anchors, 4, H, W).permute(0,1,3,4,2).reshape(B, -1, 4)

# anchors: (n_anchors * H * W, 4) = (anchor_box, 4)


# 2. bbox regression 적용 (Faster R-CNN 공식)
#   ㄴ anchor box + bbox regression 결과 → 실제 proposal box 좌표로 변환
def apply_deltas_to_anchors(anchors, deltas):
    # anchors: (N, 4), deltas: (N, 4)
    widths  = anchors[:, 2] - anchors[:, 0]     # anchor_box w
    heights = anchors[:, 3] - anchors[:, 1]     # anchor_box h
    ctr_x   = anchors[:, 0] + 0.5 * widths      # anchor_box center x
    ctr_y   = anchors[:, 1] + 0.5 * heights     # anchor_box center y

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



# 3. top-N proposal 선택 (예: N=100)
#   anchors.shape       # (n_anchors * H * W, 4)
N = 100
proposals = []
for b in range(B):
    scores = objectness[b]  # (anchor_box,)
    bbox_reg_b = bbox_reg[b]    # (anchor_box, 4)
    boxes = apply_deltas_to_anchors(anchors, bbox_reg_b)  # (anchor_box, 4)

    # top-N 선택
    topk_scores, topk_idx = torch.topk(scores, N)
    topk_boxes = boxes[topk_idx]

    # (선택) NMS 적용
    keep = nms(topk_boxes, topk_scores, iou_threshold=0.7)
    final_boxes = topk_boxes[keep]
    proposals.append(final_boxes)

# proposals: 각 배치별 top-N proposal (NMS 적용 후)
len(proposals)
proposals[0].shape
proposals[1].shape
proposals[2].shape
proposals[3].shape


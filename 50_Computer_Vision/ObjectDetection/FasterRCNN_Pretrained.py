import os
import torch
import torchvision

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import OrderedDict


from tqdm.auto import tqdm

##################################################################
model_path = r'D:\DataScience\강의) 강의자료\강의) [FastCampus] 55. 한 번에 끝내는 컴퓨터비전 초격차 패키지 Online\Code\model'

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


####################################################################
from torch.utils.data import DataLoader

# dataset = PennFudanDataset(dataset_root, transforms=train_transform)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
# collate_fn=lambda x: tuple(zip(*x))
#   ㄴ batch[0] : tuple(img1, img2, ...)
#   ㄴ batch[1] : tuple(info1, info, ...)

for batch in data_loader:
    break
img, info = batch
# torch.stack(img).shape

#################################################################


import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


####################################################################
# 1. ResNet-18 백본 준비 #############################################

# 1) ResNet-18 기본 backbone (pretrained on ImageNet)
resnet18 = torchvision.models.resnet18(weights="IMAGENET1K_V1")

# 2) 마지막 FC layer 제거 후 feature extractor만 사용
# avgpool, fc 제거 (즉, children()[:-2])
modules = list(resnet18.named_children())[:-2]

# 이름을 유지하면서 OrderedDict로 묶기
backbone = nn.Sequential(OrderedDict(modules))

# backbone(torch.rand(5,3,256,256)).shape[1]    # 512
backbone.out_channels = 512  # ResNet-18 마지막 conv layer 출력 채널 수

# print( [name for name, module in backbone.named_children()] )  # Outer Structure



####################################################################
# 2. RPN(Region Proposal Network) Anchor 설정 #######################

# RPN에서 사용할 anchor 생성기
rpn_anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),   # anchor 크기 후보
    aspect_ratios=((0.5, 1.0, 2.0),)    # 세로형, 정사각형, 가로형
)

####################################################################
# 3. RoI Pooling (RoI Align) #######################################
roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0'],  # backbone에서 어떤 feature map을 쓸지
    output_size=7,        # 고정 크기 (7x7)
    sampling_ratio=2
)


####################################################################
# 5. Faster R-CNN 모델 정의
# 이제 backbone, RPN, RoI Pooler를 합쳐 최종 모델을 만듭니다.

####################################################################
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
####################################################################

# Faster R-CNN 모델 구성
model = FasterRCNN(
    backbone,
    num_classes=2,  # person(1) + background(0)
    rpn_anchor_generator=rpn_anchor_generator,
    box_roi_pool=roi_pooler
)

# for name, param in model.backbone.named_children():
#     print(name)

# backbone conv1~layer2 freezing & layer3,4 unfreezing
for name, param in model.backbone.named_parameters():
    # print(name)
    if ('layer3' in name) or ("layer4" in name):
        param.requires_grad = True
    else:
        param.requires_grad = False


model.to(device)

# 옵티마이저
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)




N_EPOCHS = 10
# 학습 루프
for epoch in tqdm(range(N_EPOCHS)):
    model.train()
    
    for images, targets in tqdm(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {losses.item():.4f}")
    
torch.save(model.state_dict(), f'{model_path}/250903_PennFudan_faster-rcnn_pretrained.pth')


# 추론 (Evaluation) ######################################################
model.eval()
with torch.no_grad():
    img, target = dataset[0]
    prediction = model([img.to(device)])
    print(prediction)
    

# 시각화 (Bounding Box Overlay) ##########################################
import matplotlib.pyplot as plt

img = img.permute(1,2,0).cpu().numpy()
plt.imshow(img)

for box in prediction[0]['boxes']:
    x1, y1, x2, y2 = box.tolist()
    plt.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],'r-',linewidth=2)
plt.show()


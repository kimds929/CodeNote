from __future__ import division
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import os
import random
import math
import numpy as np
from collections import OrderedDict

import torch
from torch import nn
import torch.utils.data as data
import torch.nn.functional as F
import torchvision
from torchvision.ops import misc as misc_nn_ops

# path = r'D:\Python\★★Python_POSTECH_AI\Postech_AI 7) Computer_Vision\Dataset\Lecture12_Object detection'
path = r'/home/pirl/data/Lecture12/'
origin_path = os.getcwd()
os.chdir(path)


# =====================================================================================================================================================
### Funtion for visualization 
def draw_bb(img, boxes, color='g', figsize=(8,8)):
    fig,ax = plt.subplots(1, figsize=figsize)
    for box in boxes:
        rect = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],linewidth=1,edgecolor=color,facecolor='none')
        ax.add_patch(rect)
    ax.imshow(img)
    plt.show()

### Fuction for vislualize boudning with two differnt colors
def draw_bb2(img, boxes1, boxes2, color1='g', color2='r'):
    fig,ax = plt.subplots(1)
    for box in boxes1:
        rect = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],linewidth=1,edgecolor=color1,facecolor='none')
        ax.add_patch(rect)
    for box in boxes2:
        rect = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],linewidth=1,edgecolor=color2,facecolor='none')
        ax.add_patch(rect)
    ax.imshow(img)
    plt.show()





class PennFudanDataset(data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # no crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target




# # Define Faster RCNN model ========================================================================
class FASTERCNN(nn.Module):

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(FASTERCNN, self).__init__()
        '''
            backbone (nn.Module): Feature extractor (In this tutorial, we will use ResNet50)
            rpn (nn.Module): Region Proposal Network
            roi_heads (nn.Module): Classifier using RoI Align
            transform (torchvision.transforms): Data transformation
        '''
        
        self.transform = transform
        self.backbone = backbone    # Feature Extractor
        self.rpn = rpn      # Region Proposal Network
        self.roi_heads = roi_heads  # ROI_calssifier


    def forward(self, images, targets=None):
        '''
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image        
        '''
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]
        
        images, targets = self.transform(images, targets)
    
        features = self.backbone(images.tensors)    # Feature Extractor를 통해 feature를 뽑음
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])

        proposals, proposal_losses = self.rpn(images, features, targets)    # 그 Feature를 이용해 RPN으로부터 proposal을 받음.
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)  # proposal을 이용해 detection결과
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        #  During training, it returns a dict[Tensor] which contains the losses.
        if self.training:
            return losses
        # During testing, it returns list[BoxList] contains additional fields like `scores`
        return detections





# ### Image Transform Hyperparameters
from torchvision.models.detection.transform import GeneralizedRCNNTransform

min_size=800
max_size=1333

image_mean=[0.485, 0.456, 0.406]
image_std=[0.229, 0.224, 0.225]        

transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)


# ### TODO 1 : Define Backbone networks
# - Define resnet 50 model pretrained on imagenet (from troch vision)
# - Freeze '~ layer 1'
# - Freeze BatchNormalization
# - Fill the forward function
# - 마지막 fc layer를 사용하지 않고, 이전의 feature가 return 되도록


from torchvision.models import resnet50

### Define MyResnet 50 Networks
class MyResnet50(nn.Module):
    def __init__(self):
        super(MyResnet50, self).__init__()

        # load pretrained resnet 50 model
        self.resnet = resnet50(pretrained=True)

        # Freeze  ~ layer 1 (Layer 2,3,4 만 학습, 전체 BN 은 미학습)
        for name, parameter in self.resnet.named_parameters():
            if ('layer2' not in name) and ('layer3' not in name) and ('layer4' not in name):
                parameter.requires_grad_(False)
            
            # Freeze BN
            if 'bn' in name:
                parameter.requires_grad_(False)

        
        # Define out_channels
        self.out_channels = 2048
    
    def forward(self, x):
        ### Forward thorugh predefined layers till layer4
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x

backbone = MyResnet50()

### Output Channel variable check
print(backbone.out_channels)

### Check number of layers with requiered_grad
nl = []
for name, parameter in backbone.named_parameters():
    if parameter.requires_grad:
        nl.append(name)
        print(name)
print(len(nl))

### Check forward
x = torch.rand(2, 3, 244, 244)
print(backbone(x).shape)






# ### Region Proposal Networks ===========================================================================
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork

### Define Anchor generator
anchor_sizes = ((32,), (64,), (128,), (256,), (512,))    # 5x1
aspect_ratios = ((1.0, 2.0, 3.0),) * len(anchor_sizes) # 5x3

### Reference - rpn.py
# Define multi scale/aspect ratio anchors for each feature map grids
rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
# Anchor box를 만들어주는 class

### Define Region Proposal Head(Classification, Regression)
class RPNHead(nn.Module):
# conv3x3 - relu - conv1x1 - classification
#                           conv1x1 - regression
    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        # Intermediate layer 뽑아주는 conv
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        # classification 1×1 conv (logit → confidence)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)

        # bbox regression 1×1 conv (bbox의 x_min, x_max, y_min, y_max)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for l in self.children():
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:   # x가 list가 들어옴, batch(B, C, H, W)의 list
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg
    

#rpn_head instantiation
out_channels = backbone.out_channels    # out_channels (이전 feature extractor의 output tensor 채널)
rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
# num_anchors_per_location()[0] 은 사용할 anchor의 수 (이 실습에서는 15개)



### RPN Hyperparameters
# NMS param : Non-Maximum Suppression
#   Bounding Box를 Regression 하다보면 겹치는 Bounding Box가 굉장히 많이 생김.
#   그중에서 제일 좋은 Bounding Box를 뽑아줘야함 (Confidence 가 Maximum이 아닌 것은 다 Suppression)
rpn_pre_nms_top_n_train=10000
rpn_pre_nms_top_n_test=6000
rpn_post_nms_top_n_train=2000
rpn_post_nms_top_n_test=300
rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
rpn_nms_thresh=0.7      # Confidence에 대한 Threshold, confidence가 0.7 미만이면 Suppression

# IOU threshold for anchor (교집합 / 합집합: 얼마나 겹쳐있는지?)
rpn_fg_iou_thresh=0.7   # 0.7 이상이면 정답과 겹치는 영역이 많음. → 좋다고 표시
rpn_bg_iou_thresh=0.3   # 0.3 아래면 안좋다고 표시.

# RPN batch info
rpn_batch_size_per_image=256    # Number of anchor that are sampled during training of the RPN
rpn_positive_fraction=0.5

### Define RPN
rpn = RegionProposalNetwork(
    rpn_anchor_generator, rpn_head,
    rpn_fg_iou_thresh, rpn_bg_iou_thresh,
    rpn_batch_size_per_image, rpn_positive_fraction,
    rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)   # NMS, IoU thresholding 해줘서 좋은 bounding box를 proposal 해줌.









# ### RCNN Detectors ==============================================================================================================
# (Classifier)

from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign

### ROI pool(ROI align actually)
# 'ops/Spoolers'
box_roi_pool = MultiScaleRoIAlign(
                featmap_names=[0, 1, 2, 3],
                output_size=7,
                sampling_ratio=2) #number of sampling point in interpolation grids
# ROIAlign (RolPooling Improved version)을 사용해서 Region Proposal 받은 위치의 feature를 pooling 해서 꺼내줌.




### Detector FC
# Region Proposal 받은 bbox로부터 intermediate feature 를 뽑아줌.
# feature의 channel은 1024로 해줌
class RCNNFC(nn.Module):
    def __init__(self, in_channels, representation_size):
        super(RCNNFC, self).__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x

resolution = box_roi_pool.output_size[0]
representation_size = 1024

# region proposal 받은 bbox로부터 feature 뽑는 network 선언
box_head = RCNNFC(
    out_channels * resolution ** 2,
    representation_size)



### Detector Classifier / Regressor
class FastRCNNPredictor(nn.Module):
    ### FC layers for classification and regression
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)        # num_classes 개수 만큼의 class confidence
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)    # region proposal로 받은 bbox의 refinement를 위한 param

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas

num_classes = 2     # 보행자 or 아니냐.
representation_size = 1024
box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)



### Detector Hyperparameters
box_score_thresh=0.05
box_nms_thresh=0.5
box_detections_per_img=30
box_fg_iou_thresh=0.5
box_bg_iou_thresh=0.5
# Detector batch info
box_batch_size_per_image=512
box_positive_fraction=0.25
bbox_reg_weights=None        



### Final Detectors
# 1) Select training samples => foreground / background assign
# 2) Roi pool
# 3) forward box head, predictor
# 4) fastrcnn_loss  ||  Post process
roi_heads = RoIHeads(
    # Box
    box_roi_pool, box_head, box_predictor,
    box_fg_iou_thresh, box_bg_iou_thresh,
    box_batch_size_per_image, box_positive_fraction,
    bbox_reg_weights,
    box_score_thresh, box_nms_thresh, box_detections_per_img)



print(roi_heads)




# ### Instantiate Faster RCNN model

model =FASTERCNN(backbone, rpn, roi_heads, transform)
# print(model)


# ### Train generated model
# get_ipython().system('pip install pycocotools')




# pip install pycocotools
from engine import train_one_epoch, evaluate
import utils

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# define dataset
import transforms as T
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

dataset = PennFudanDataset('./data/PennFudanPed',  get_transform(train=True))
dataset_test = PennFudanDataset('./data/PennFudanPed', get_transform(train=False))

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

### define training and validation data loaders
### collate_fn - documentation
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=lambda batch: tuple(zip(*batch)))
#     collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=lambda batch: tuple(zip(*batch)))

# move model to device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]     # ★★★ Requires True인것들만 optimizer에 넣어줌
# training loop를 짜다보면 parameter에 requires_grad를 Fals로 해뒀는데,
# True로 바뀌어 있는 경우가 종종 있기 때문에, 걸러서 넣어주어야 함.

optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)






# ### TODO.2 : Train network
# - Fill the forward path to calculate losses


##### TODO.3 

# let's train it for 10 epochs
num_epochs = 10
# print loss every 10 mini-batches
print_step = 10

##### TODO
# Loop over num_epochs
for epoch in range(num_epochs):
    
    ## enumerate through data_loader
    for batch_idx, (images, targets) in enumerate(data_loader):
        ### Train mode
        #todo
        model.train()
        
        images = list(image.to(device) for image in images)        
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        '''
            -------------
            #   TO DO   #
            -------------
            model input : image (tensor list), target (dict list)
            model output : losses (dict - key: (string) each loss name, value: (tensor) each loss value
        '''
        # Fill this blank
        loss_dict = model(images, targets)
        losses = sum([loss for loss in loss_dict.values()])
        loss_value = losses.item()
        
        
        ### BackPropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        ### print itermedicate outputs
        if(batch_idx % print_step == 0):
            print('Epoch: [{}]  [{}/{}]   loss: {}  '.format(epoch, batch_idx//print_step, len(dataset)//(2*print_step), loss_value))

print('Training Done!')



# predefined evaluation function
evaluate(model, data_loader_test, device=device)








# ### Visulization - test data =================================================================================================

# eval mode
model.eval();

data_it = iter(data_loader_test)



data_test = next(data_it)
img_test = data_test[0][0]
bbox_test = data_test[1][0]['boxes']
prediction = model(list([data_test[0][0].cuda()]))
scores = prediction[0]['scores']
print(scores)
threshold = (scores>0.5).sum().cpu().detach().item()
### transform for visulization
P = torchvision.transforms.ToPILImage()
img_test_PIL = P(img_test)
prediction_bbox_np = prediction[0]['boxes'].cpu().detach().numpy()[:threshold]
draw_bb2(img_test_PIL, bbox_test, prediction_bbox_np)






# ### Visualization - user input image -------------------------------------------------------------------------------------

# read image
img_demo = Image.open('./data/PedSample/sample1_easy.jpg').convert("RGB")
img_demo





# Forward through trained model
boxes_demo = model([torchvision.transforms.ToTensor()(img_demo).cuda()])[0]['boxes']
scores_demo = model([torchvision.transforms.ToTensor()(img_demo).cuda()])[0]['scores']
boxes_demo_np = boxes_demo.cpu().detach().numpy()
print(scores_demo)
threshold = (scores_demo>0.5).sum().cpu().detach().item()
boxes_demo_np = boxes_demo.cpu().detach().numpy()[:threshold]
print(boxes_demo_np.shape)
# Draw bounding boxes
draw_bb(img_demo, boxes_demo_np, 'r', (12,12))





# =====================================================================================================
# 1. Why faster?
# - RPN을 통해서 Region Proposal을 위한 시간을 단축 (deep learning 기반이라 빠름)
# 
# 2. Region Proposal
# - Anchor box라는 개념을 통해 pre-defined 된 box를 먼저 regress 하고 그 뒤에 refine 하는 형태로 접근
#
# 3. Non-Maximum Suppression
# - 여러개 Bounding Box가 겹쳐있는 경우에 가장 좋은 Bounding Box를 구하는 방법
#
# 4. IoU
# - BBox를 평가하기 위한 방법. (교집합 / 합집합)
#
# 5. Faster RCNN
# - Feature Extracor → RPN → Classifier
#   → 2 step object detection
# ※ 1step object detection: YOLO (You Only Look Once)
# =====================================================================================================
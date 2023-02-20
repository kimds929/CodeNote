import os
import time
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2 as cv
import PIL
from PIL import Image
import PIL.Image as pil_image
# import collections
# import os.path as osp
# import scipy
# import scipy.misc
# import imageio
import h5py
from tqdm import tqdm


import torch
import torch.nn as nn11
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

import torchvision


# path = r'D:\Python\★★Python_POSTECH_AI\Postech_AI 7) Computer_Vision\Dataset'
# path = r'D:/Python/★★Python_POSTECH_AI/Postech_AI 7) Computer_Vision/Dataset/Lecture08_Image restoration/'
path = r'/home/pirl/data/Lecture08_Image restoration/'
origin_path = os.getcwd()
os.chdir(path)
# os.listdir()

seed = 1
no_cuda = False
torch.manual_seed(seed)
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory':True} if use_cuda else {}
torch.cuda.empty_cache()
# print(torch.cuda.memory_summary())
# torch.cuda.empty_cache()    # 메모리 초기화


# Train Dataset 불러오는 class
class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


# Evaluation Dataset 불러오는 class
class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])



# # Model definition
# 
# ## TO DO - Define the model by following the given architecture
# ### SRCNN
# conv1: (in_channel) num_channels, (out_channel) 64, (kernel_size) 9, (stride) 1, (padding) 4
# conv2: (in_channel) 64, (out_channel) 32, (kernel_size) 5, (stride) 1, (padding) 2
# conv3: (in_channel) 32, (out_channel) num_channels, (kernel_size) 5, (stride) 1, (padding) 2
# 
# Each layers should have ReLU activation but the last conv layer should not have any activation layer.



# SRCNN **** ============================================================================
class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, 9, 1, 4)
        self.conv2 = nn.Conv2d(64, 32, 5, 1, 2)
        self.conv3 = nn.Conv2d(32, num_channels, 5, 1, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = self.conv3(x)
        return x


# # Util functions
def convert_rgb_to_y(img):
    if type(img) == np.ndarray:
        return 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        return 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
    else:
        raise Exception('Unknown Type', type(img))


def convert_rgb_to_ycbcr(img):
    if type(img) == np.ndarray:
        y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def convert_ycbcr_to_rgb(img):
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
        g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
        b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# # Training loop
train_file = "data/91-image_x3.h5"
eval_file = "data/Set5_x3.h5"
outputs_dir = "outputs/"

scale = 3
lr = 1e-4
batch_size = 16
num_workers = 8
seed = 123

outputs_dir = os.path.join(outputs_dir, 'x{}'.format(scale))

if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)

cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(seed)

model = SRCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam([
    {'params': model.conv1.parameters()},
    {'params': model.conv2.parameters()},
    {'params': model.conv3.parameters(), 'lr': lr * 0.1}
], lr=lr)

train_dataset = TrainDataset(train_file)

train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True)
eval_dataset = EvalDataset(eval_file)
eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)


# plt.imshow(a[0][1,0,:], 'gray')
# plt.imshow(a[1][1,0,:], 'gray')

# ======================================================================
# Input Size: 33 × 33 // 31 × 31
best_weights = copy.deepcopy(model.state_dict())
best_epoch = 0
best_psnr = 0.0

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_losses = AverageMeter()

    with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size)) as t:
        t.set_description('epoch: {}/{}'.format(epoch, num_epochs - 1))

        for data in train_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            preds = model(inputs)

            loss = criterion(preds, labels)

            epoch_losses.update(loss.item(), len(inputs))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            t.update(len(inputs))

    torch.save(model.state_dict(), os.path.join(outputs_dir, 'epoch_{}.pth'.format(epoch)))

    model.eval()
    epoch_psnr = AverageMeter()

    for data in eval_dataloader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(inputs).clamp(0.0, 1.0)

        epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

    print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

    if epoch_psnr.avg > best_psnr:
        best_epoch = epoch
        best_psnr = epoch_psnr.avg
        best_weights = copy.deepcopy(model.state_dict())

print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
torch.save(best_weights, os.path.join(outputs_dir, 'best.pth'))















# SRGAN **** ============================================================================
# 1. SCRCNN, Image size에 상관없이 가능
#   Gaussian filter 적용하면 blur, image size와 무관
#   이와 비슷하게, SRCNN에서 구해진 conv filter는 super-resolution하는 filter

# 2. Perceptual Loss
# Idea : pixel값이 비슷하다고 우리 눈에 비슷하게 보이지만은 않는다
# 우리 눈에도 비슷하게 보이게 하도록 강제 하고 싶음.
# 우리눈 == Pretrained model (ImageNet Dataset에 train되어 있음)
# pretrained model의 앞부분 feature extractor를 사용하면 좋은 low-level feature를 얻을 수 있음
# pretrained model은 주로 VGG19를 많이 사용함

# feature를 가져와서 쓸때는 반드시 Relu Activation 다음에 있는 featuref를 쓰는게 좋음
# ... feature가 conv다음에 있을 수도있고, mapool 다음에도 있고, Relu 다음에도 있음

# 3. 왜 Pixelshuffle layer를 쓰는지? (deconv가 아니라)
# Deconv Layer를 쓸 경우 image generation task에서는 checkboard artifact가 생김, 이를 방지하기 위해 pixelshuffle layer를 사용 혹은 이후 upsampling

# 4. low Resolution dataset은 synthetic dataset
# High Resolution 이미지를 통해 bicubic downsampling을 통해 low Resolution image로 만들어줌


import glob
import random

from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torchvision.models import vgg19

import argparse
import math
import itertools
import sys


# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


# Image Dataset -----------------------------------------------------------------------------------
class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
            transforms.Resize((hr_height//4, hr_width//4), Image.BICUBIC),  # Synthetic low resolution data를 만듦
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ]
        )
        self.hr_transform = transforms.Compose(
            [
            transforms.Resize((hr_height, hr_width), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)





# Feature Extractor  -----------------------------------------------------------------------------------
class FeatureExtractor(nn.Module):
    '''
    input.shape: (batch, 3, width, hegiht)
    output.shape: (batch, 256, , width/4, hegiht/4)
    '''
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.feature_extractor(img)

# vgg19 = vgg19(pretrained=True)
# feature_extractor = nn.Sequential(*list(vgg19.features.children())[:18])

# Sequential(
#   (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (1): ReLU(inplace=True)
#   (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (3): ReLU(inplace=True)
#   (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (6): ReLU(inplace=True)
#   (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (8): ReLU(inplace=True)
#   (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (11): ReLU(inplace=True)
#   (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (13): ReLU(inplace=True)
#   (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (15): ReLU(inplace=True)
#   (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (17): ReLU(inplace=True)
# )
# a = FeatureExtractor()
# b = torch.rand((1,3,16,16))
# a(b).shape


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)


# Generator ResNet -----------------------------------------------------------------------------------
class GeneratorResNet(nn.Module):
    '''
    input.shape: (batch, 3, width, hegiht)
    output.shape: (batch, 3, , width * 4, hegiht * 4)
    '''
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(GeneratorResNet, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8))

        # Upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),      # Transpose Convolution과 동일한 역할 (PixelShuffle이 이미지를 만들때 더 안정적인)
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4), nn.Tanh())

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out

# https://m.blog.naver.com/PostView.nhn?blogId=go21&logNo=221147971473&proxyReferer=https:%2F%2Fwww.google.com%2F
# nn.PixelShuffle(upscale_factor=3)     # upscale_factor = uf
# input.shape: (batch, channel, width, hegith) 
#   → output.shape: (batch, channel/(uf**2), width*uf, height*uf)

# a = nn.PixelShuffle(upscale_factor=3)
# b = torch.rand((2,36,8,8))
# a(b).shape
# dir(a)

# a = GeneratorResNet()
# b = torch.rand((2, 3, 16, 16))
# a(b).shape




# Discriminator ResNet: Real? vs Fake? -----------------------------------------------------------------------------------
class Discriminator(nn.Module):     # real or fake // binary classification 마지막 activation으로 sigmoid를 사용해주어야함 (Vanilla GAN)
    def __init__(self, input_shape):    # loss 도 BCELoss를 사용해야함 → GAN Train시 불안정하여 MSE Loss를사용(LSGAN 구조)
        super(Discriminator, self).__init__()   # output size가 1×1이 아님, Patch로 되어 있어서 PatchGAN구조이다.

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)  # patch_size:  h / 16, w / 16
        self.output_shape = (1, patch_h, patch_w)   # Patch 형태로 되어있음.

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))  # ReLU vs. LeakyReLU  // LeakyReLU: 실험적으로 Discriminator에 Leaky ReLU를 써야 학습이 잘됨
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

# b = torch.rand((2, 3, 64, 64))
# a = Discriminator(b.shape[1:])
# a(b).shape










os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

epoch = 0       # epoch to start training from
n_epochs = 10   # number of epochs of training

dataset_name = 'img_align_celeba' # name of dataset

batch_size = 4      # size of the batches
lr = 2e-4           # learning rate
b1 = 0.5            # decay of first order momentum of gradient
b2 = 0.999          # decay of first order momentum of gradient
decay_epoch = 100   # epoch from which to start lr decay

hr_height = 256
hr_width = 256
channels = 3
sample_interval = 100
checkpoint_interval = 10

cuda = torch.cuda.is_available()
if cuda:
    n_cpu = 8 # number of cpu threads to use during batch generation
else:
    n_cpu = 0


hr_shape = (hr_height, hr_width)    # (256, 256)

# Initialize generator and discriminator
generator = GeneratorResNet()
# generator
discriminator = Discriminator(input_shape=(channels, *hr_shape))
# discriminator
feature_extractor = FeatureExtractor()  # perceptual loss
# feature_extractor


# Set feature extractor to inference mode
feature_extractor.eval()    # dropout & batchnorm - training 단계에서만 On되어야함, Test일때 off

# Losses Function 정의
criterion_GAN = torch.nn.MSELoss()      # Loss Function
criterion_content = torch.nn.L1Loss()   # L1 Loss 큰 차이는 없고, 그래프가 V자형 직선 / L2 Loss 그래프가 U자형 곡선: 미분시 차이가 있음, U자형은 0에가까울수록 기울기가 0에 가까워짐


if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    feature_extractor = feature_extractor.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()

if epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/generator_%d.pth"))          # (nn.Module) Load_State_Dict(torch.load(file_path))
    # torch.load하면, dict 형태로 load해줌, 그리고 이 dict에 정의된 weight를 모델에 정의해주는게 load_state_dict
    # ['cov1': weight (torch.cuda.Tensor), ...]
    # Key와 이름이 같아야 weight를 적용.
    discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth"))  # 

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))       # Generator
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))   # Discriminator

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor   # cuda일때는 cuda tensor가 되도록


dataloader = DataLoader(
    ImageDataset("data/%s" % dataset_name, hr_shape=hr_shape),
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_cpu,
)

# Sample Image
data_example = next(iter(dataloader))
len(data_example)
data_example.keys()
data_example['lr'].shape        # low resolution
data_example['hr'].shape        # high resolution

sample_index = 1
plt.figure(figsize=(10,20))
plt.subplot(121)
plt.imshow(np.transpose(data_example['lr'].numpy(), [0,2,3,1])[sample_index,:])
plt.subplot(122)
plt.imshow(np.transpose(data_example['hr'].numpy(), [0,2,3,1])[sample_index,:])
plt.show()


# Test Code
def tensor_to_image(tensor):
    return np.transpose(tensor.detach().to('cpu').numpy(), [0, 2, 3, 1])

a = data_example["lr"][[0],:]
b = data_example["hr"][[0],:]
g_a = generator(a.to(device))

plt.figure(figsize=(10,30))
plt.subplot(131)
plt.imshow(tensor_to_image(a).squeeze())
plt.subplot(132)
plt.imshow(tensor_to_image(b).squeeze())
plt.subplot(133)
plt.imshow(tensor_to_image(g_a).squeeze())
plt.show()


# d_a = discriminator(g_a)
# d_a




# ----------
#  Training
# ----------
# n_epochs = 1
for epoch in range(epoch, n_epochs):
    for i, imgs in enumerate(dataloader):

        # Configure model input
        # imgs_lr = imgs["lr"]    # shape: (4, 3, 64, 64)
        # imgs_hr = imgs["hr"]    # shape: (4, 3, 256, 256)
        imgs_lr = Variable(imgs["lr"].type(Tensor))     # 요즘 torch에서는 Variable로 안감싸도됩니다.
        imgs_hr = Variable(imgs["hr"].type(Tensor))     # require_grad를 위해서 Variable 해줬던 것...

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)  # MSE Loss

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())  
        # detach(): tensor가 network돌면서 computational graph 만듦.
        #           뒤에 real_features에서는 gradient계산이 필요없기 때문에 detach() 해줌

        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN     # (hyper-parameter) 1:1e-3, (MSE Loss가 빠져있는데 이 Loss를 넣어도됨)

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)
        # generator 거친 computarional graph는 필요없음. 왜냐면 discriminator만 update할 거라서
        # back prop시 discriminator를 업데이트 할꺼면 discriminator까지만 가면됨. 더 뒤로갈 필요없음(더 뒤의 computaional그래프는 버리면됨)

        # (Forward) image → Generator → Discriminator → logit
        # (Backward) logit → Discriminator → 필요 ×

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]\n"
            % (epoch, n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % sample_interval == 0:
            # Save image grid with upsampled inputs and SRGAN outputs
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)        # Resize해서 HR이미지와 같이 visualize 하는 부분
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            img_grid = torch.cat((imgs_lr, gen_hr), -1)
            save_image(img_grid, "images/%d.png" % batches_done, normalize=False)

    if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)             # Generator 모델 저장
        torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)     # Discriminator 모델 저장
        # (Load) torch.load → model.load_state_dict()
        # (Save) torch.state_dict() → torch.save
        # ※ model.load_state_dict(): weight를 dictionary형태로 뽑아주는 method


# load_generator = torch.load( "saved_models/generator_%d.pth" % epoch)
torch.save(generator, 'saved_models/SRGAN_generator.pt')
torch.save(discriminator, 'saved_models/SRGAN_discriminator.pt')




# SRCNN TEST ================================================================================================
weights_file = "checkpoint/srcnn_x3.pth"
image_file = "sample/zebra.bmp"
# image_file = "sample/ppt3.bmp"
scale = 3

cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = SRCNN().to(device)

state_dict = model.state_dict()
for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
    if n in state_dict.keys():
        state_dict[n].copy_(p)
    else:
        raise KeyError(n)

model.eval()

image = pil_image.open(image_file).convert('RGB')

image_width = (image.width // scale) * scale
image_height = (image.height // scale) * scale
image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
image = image.resize((image.width // scale, image.height // scale), resample=pil_image.BICUBIC)
image = image.resize((image.width * scale, image.height * scale), resample=pil_image.BICUBIC)
image.save(image_file.replace('.', '_bicubic_x{}.'.format(scale)))

image = np.array(image).astype(np.float32)
ycbcr = convert_rgb_to_ycbcr(image)

y = ycbcr[..., 0]
y /= 255.
y = torch.from_numpy(y).to(device)
y = y.unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    preds = model(y).clamp(0.0, 1.0)

psnr = calc_psnr(y, preds)
print('PSNR: {:.2f}'.format(psnr))

preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
output = pil_image.fromarray(output)
output.save(image_file.replace('.', '_srcnn_x{}.'.format(scale)))







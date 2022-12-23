import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import tensorflow as tf
import torch
import torchvision

import PIL.Image


# Image Preprocessing =========================================================================================================
# !wget https://fox28spokane.com/wp-content/uploads/2020/04/AprilShoe-768x832.jpg

# numpy → PIL.Image
img = Image.fromarray(plt.imread('AprilShoe-768x832.jpg'))
print(np.array(img).shape)

plt.imshow(img)
plt.show()


# https://pytorch.org/docs/stable/torchvision/transforms.html

# Tensor Transform ------------------------------------------------------------------------------
# Transform on tensor
tensor_img = torchvision.transforms.ToTensor()(img)     # Normalize까지 자동으로 됨
tensor_img_array = torch.FloatTensor(np.array(img)).permute(2,0,1)
print(tensor_img.shape, tensor_img_array.shape)
print(tensor_img.min() , tensor_img.max())
print(tensor_img_array.min() , tensor_img_array.max())
tensor_img
plt.imshow(tensor_img.permute(1,2,0))
plt.show()

# Normalize
tensor_norm_img = torchvision.transforms.Normalize(mean=tensor_img_array.mean(), std=tensor_img_array.std(), inplace=False)(tensor_img_array)    # 채널별로 평균, 편차 적용하여 Normalize
print(tensor_norm_img.shape)
print(tensor_norm_img.min() , tensor_norm_img.max())
# img_array = (np.array(img) - np.mean(np.array(img))) / np.std(np.array(img))
# print(np.min(img_array), np.max(img_array))
tensor_norm_img

plt.imshow(tensor_norm_img.permute(1,2,0).int())
plt.show()


# random_erasing
img_randomErasing = torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)(tensor_img_array)
plt.imshow(img_randomErasing.permute(1,2,0).int())
plt.show()
np.array(img).transpose(2,0,1)


# PIL.Image Transform ------------------------------------------------------------------------------
# resize
    # interpolation : 1이나 p로 주면 nearest로 받는다
    #                 Image.NEAREST
    #                 Image.BILNEAR
    #                 Image.BICUBIC
    #                 Image.LANCZOS
img_resize_down = torchvision.transforms.Resize(size=(100,100))(img)
img_resize_down

img_resize_up = torchvision.transforms.Resize(size=(1000,1000), interpolation=1)(img)
img_resize_up

# center_crop ****
img_center = torchvision.transforms.CenterCrop(size=(300,300))(img)
img_center
print(np.array(img_center).shape)

# Color_jitter ****
img_colorjitter = torchvision.transforms.ColorJitter(brightness=0, contrast=1, saturation=0, hue=0)(img)
# 1로 지정한 값에 대해서는 random하게 바꿔줌
img_colorjitter

# Five_Crop ****
img_fivecrop = torchvision.transforms.FiveCrop(size=(400,400))(img)
img_fivecrop
for fivecrop in img_fivecrop:
    plt.imshow(fivecrop)
    plt.show()

# Random_Crop ****
img_randomCrop = torchvision.transforms.RandomCrop(size=(400,400))(img)
img_randomCrop

# GrayScale ****
img_gray = torchvision.transforms.Grayscale(num_output_channels=1)(img)
img_gray

# Random_GrayScale
img_randomGray = torchvision.transforms.RandomGrayscale(p=0.5)(img)
img_randomGray

# add_padding ****
img_addpadding = torchvision.transforms.Pad(padding=(20,20), fill=(255,200,200), padding_mode='constant')(img)
img_addpadding = torchvision.transforms.Pad(padding=(20,20), fill=0, padding_mode='constant')(img)
img_addpadding

# rotation ****
img_rotation = torchvision.transforms.RandomAffine(degrees=90, fillcolor=(0,0,0))(img)
img_rotation

# random_horizontal_flip ****
img_randomHorizontalFlip = torchvision.transforms.RandomHorizontalFlip(p=0.5)(img)
img_randomHorizontalFlip

# random_vertical_flip ****
img_randomVerticalFlip = torchvision.transforms.RandomVerticalFlip(p=0.5)(img)
img_randomVerticalFlip





# 여러개의 transform 방법을 list에 담아줌
transforms = [
    torchvision.transforms.CenterCrop(size=(300,300)),
    torchvision.transforms.ColorJitter(brightness=0, contrast=1, saturation=0, hue=0),
    torchvision.transforms.Pad(padding=(20,20), fill=(255,200,200), padding_mode='constant'),
    torchvision.transforms.RandomAffine(degrees=90, fillcolor=(0,0,0))
    ]

# RandomAlpply : transform에 담겨있는 image_transform을 일정 확률(p=..)로 모두 적용하거나 안하거나 하는 함수
img_randomApply = torchvision.transforms.RandomApply(transforms, p=0.5)(img)
img_randomApply

# RandomChoice : transform에 담겨있는 image_transform중에 랜덤하게 적용하는 함수
img_randomChoice = torchvision.transforms.RandomChoice(transforms)(img)
img_randomChoice







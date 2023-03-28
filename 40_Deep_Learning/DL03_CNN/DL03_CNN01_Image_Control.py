import os 
from glob import glob
from PIL import Image   # PIL는 이미지를 load 할 때 이용
import time
from IPython.display import clear_output

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Postech_AI 9) Team Project/Image/'
os.listdir(absolute_path)

image_path = absolute_path + 'Train/apple/'
os.listdir(image_path)

image_path2 = absolute_path + 'Train/orange/'
os.listdir(image_path2)

# Image Load
imag1 = plt.imread(image_path + '0_100.jpg')
imag2_jpg = Image.open(image_path + '0_100.jpg')
imag2 = np.array(imag2_jpg)


# Image show
plt.imshow(imag1)

imag2_jpg
plt.imshow(imag2_jpg)
plt.imshow(imag2)


# 흑백으로 열기
imag2_kb_jpg = Image.open(image_path + '0_100.jpg').convert('L')
imag2_kb_jpg

imag2_kb = np.array(imag2_kb_jpg)
imag2_kb.shape
plt.imshow(imag2_kb_jpg, 'gray')


plt.imshow(imag2_kb_jpg, 'RdBu')    # 색깔지정하기
plt.imshow(imag2_kb_jpg, 'jet')    # gray반전
plt.colorbar()


# 두번째 이미지 열기
imag3_jpg = Image.open(image_path2 + '0_100.jpg')
imag3_jpg
imag3 = np.array(imag3_jpg)
imag3



# Image Resize
import cv2
imag3_shrink = cv2.resize(imag3, (50,50))
imag3_expand = cv2.resize(imag3, (200,200))
imag3_resize = cv2.resize(imag3, (100,50))
print(imag3_shrink.shape, imag3_expand.shape)


plt.imshow(imag3_shrink)
plt.imshow(imag3_expand)
plt.imshow(imag3_resize)








# Image Agmentation ---------------------------------------
from tensorflow.keras.preprocessing.image import ImageDataGenerator
Image_transform_comparison(inputs, image_gen)


# (주로 사용하는 기능들)
# - width_shift_range       # 가로방향 shift
# - height_shift_range      # 세로방향 shift
# - brightness_range  
# - zoom_range              # 상하좌우로 늘림
# - horizontal_flip         # 세로방향 반전
# - vertical_flip           # 가로방향 반전
# - rescale                 #
# - preprocessing_function  # (Customizing) 어떤함수를 넣어서 적용가능


# - width_shift_range   : 가로방향 shift
gen2 = ImageDataGenerator(width_shift_range=0.3)
image_gen2 = next(iter(gen2.flow(inputs)))
Image_transform_comparison(inputs, image_gen2)

# - height_shift_range : 세로방향 shift
gen3 = ImageDataGenerator(height_shift_range=0.3)
image_gen3 = next(iter(gen3.flow(inputs)))
Image_transform_comparison(inputs, image_gen3)

# - brightness_range  
gen4 = ImageDataGenerator(brightness_range=[1,2])
image_gen4 = next(iter(gen4.flow(inputs)))
Image_transform_comparison(inputs, image_gen4)

# - zoom_range  : 위아래 좌우 늘림
gen5 = ImageDataGenerator(zoom_range=0.3)
image_gen5 = next(iter(gen5.flow(inputs)))
Image_transform_comparison(inputs, image_gen5)

# - zoom_range
gen6 = ImageDataGenerator(rescale=1/255)
image_gen6 = next(iter(gen6.flow(inputs)))
Image_transform_comparison(inputs, image_gen6)


# Rescale시 주의 사항: Train_set에 해주면 Test_set에도 반드시 해주어야함
Train_data_gen = ImageDataGenerator(zoom_ragne=0.7, rescale=1/255)
Test_data_gen = ImageDataGenerator(rescale=1/255)


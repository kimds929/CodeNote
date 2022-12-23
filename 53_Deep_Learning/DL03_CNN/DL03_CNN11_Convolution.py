import os

import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv
import torch

path = r'C:\Users\Admin\Desktop\DataScience\Reference8) POSTECH_AI_Education\Dataset'

origin_path = os.getcwd()
os.chdir(path)
# cv.copyMakeBorder(img_gray, )

img_gray = cv.imread('AprilShoe-768x832.jpg', cv.IMREAD_GRAYSCALE)
plt.imshow(img_gray)

img_small = cv.resize(img_gray, dsize=None, fx=0.1, fy=0.1)
plt.imshow(img_small, 'gray')


image = img_small.copy()
# Box_Kernel
kernel = np.array([[1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]])

# Gaussian_Kernel
kernel = np.array([[1, 2, 1],
                [2, 4, 2],
                [1, 2, 1]])


# Vertical_Kerenl (dx, 가로선)
kernel = np.array([[1, 1, 1],
                [0, 0, 0],
                [-1, -1, -1]])
# Horizontal_Kerenl (dy, 세로선)
kernel = np.array([[1, 0, -1],
                [1, 0, -1],
                [1, 0, -1]])

# Sobel_Vertical_Kerenl (dx, 가로선)
kernel = np.array([[1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]])

# Sobel_Horizontal_Kerenl (dy, 세로선)
kernel = np.array([[1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]])


kernel = np.array([[1, 0, 1],
                [0, 1, 0],
                [1, 0, 1]])

def image_filter(image, kernel, padding=0, kernel_type=None):
    cutting_shape = np.array(np.array(kernel.shape, dtype=np.float32)//2 * 2, dtype=int)
    image_padding = image.copy()
    result_image = np.zeros(np.array(image_padding.shape) - cutting_shape)

    if kernel_type == 'convolution':
        image_filter = kernel[:,::-1]
    else:
        image_filter = kernel.copy()
    
    for h in range(image_padding.shape[1] - cutting_shape[1]):
        for w in range(image_padding.shape[0] - cutting_shape[0]):
            image_cut = image_padding[w:w+cutting_shape[0]+1, h:h+cutting_shape[1]+1]
            # after_filtering = int(np.round(np.sum(image_cut * image_filter),0))
            after_filtering = np.correlate(image_cut.ravel(), image_filter.ravel())

            result_image[w, h] = after_filtering
    return result_image


kernel2 = np.zeros((5,5))
kernel2[:,0] = 1
kernel2[:,-1] = 1
kernel2

plt.imshow(img_small, 'gray')

a = image_filter(img_small, kernel=kernel)
plt.imshow(a, 'gray')

np.min(a)
np.max(a)
c = cv.filter2D(img_small, -1, kernel)
plt.imshow(c, 'gray')

plt.imshow(image_filter(img_small, kernel=kernel), 'gray')

a = image_filter(img_small, kernel=kernel)

img_small.shape
plt.imshow(a, 'gray')
plt.imshow(a.astype('uint8'), 'gray')
plt.imshow(cv.equalizeHist(a.astype('uint8')), 'gray')
plt.imshow(cv.filter2D(img_small, ddepth=-1, kernel=kernel), 'gray')

# plt.imshow(np.concatenate([img_small, result_image], axis=1), 'gray')

a1 = img_small.ravel()
b1 = kernel.ravel()
b2 = kernel2.ravel()


a1
d = np.correlate(a1,b1)
d.shape
d.shape[0]

d2 = np.correlate(a1,b2)
d2.shape

np.product(img_small.shape) - d
np.product(img_small.shape) - d2






# Convolution ###################################################################################################
r = np.array(
    [[255, 155, 85, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 100, 120, 0]]
    )

g = np.array(
    [[0, 0, 0, 0],
    [250, 135, 75, 15],
    [0, 0, 0, 0],
    [0, 110, 0, 140]]
    )

b = np.array(
    [[0, 0, 0, 0],
    [0, 0, 0, 0],
    [245, 175, 105, 0],
    [0, 0, 130, 150]]
    )

# * matplotlib/Pillow/cv2: (4, 4, 3)


img_rgb = cv2.merge((r,g,b))    # H,W,C


img_rgb.transpose(2,0,1)    # C,H,W

b_img_rgb = img_rgb[np.newaxis, ...]/255    # B,H,W,C
b_img_rgb.shape



conv_layer = tf.keras.layers.Conv2D(1, (3,3))
result = conv_layer(b_img_rgb)
result

weights = conv_layer.weights[0]


# convolution (image_filter)
def image_filter(image, kernel, padding=0, kernel_type=None):
    cutting_shape = np.array(np.array(kernel.shape, dtype=np.float32)//2 * 2, dtype=int)
    image_padding = image.copy()
    result_image = np.zeros(np.array(image_padding.shape) - cutting_shape)

    if kernel_type == 'convolution':
        image_filter = kernel[:,::-1]
    else:
        image_filter = kernel.copy()
    
    for h in range(image_padding.shape[1] - cutting_shape[1]):
        for w in range(image_padding.shape[0] - cutting_shape[0]):
            image_cut = image_padding[w:w+cutting_shape[0]+1, h:h+cutting_shape[1]+1]
            # after_filtering = int(np.round(np.sum(image_cut * image_filter),0))
            after_filtering = np.correlate(image_cut.ravel(), image_filter.ravel())

            result_image[w, h] = after_filtering
    return result_image


a1 = image_filter(b_img_rgb[0,:,:,0], kernel=weights[:,:,0,0].numpy(), padding=0)
a2 = image_filter(b_img_rgb[0,:,:,1], kernel=weights[:,:,1,0].numpy(), padding=0)
a3 = image_filter(b_img_rgb[0,:,:,2], kernel=weights[:,:,2,0].numpy(), padding=0)


result[0,:,:,0]
a1+a2+a3 # Channel R + G + B
############################################################################################################






import tensorflow as tf
# img = cv.imread('tsukuba_l.png', cv.IMREAD_GRAYSCALE)
# plt.imshow(img, 'gray')
plt.imshow(img_small, 'gray')

kernel = np.array([[1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]])
plt.imshow(kernel, 'coolwarm')
plt.colorbar()
plt.show()


conv = tf.keras.layers.Conv2D(1, kernel_size=kernel.shape, padding='valid', use_bias=False)
filterd_conv = conv(img_small[np.newaxis,...,np.newaxis].astype(np.float32))

conv.set_weights([kernel[...,np.newaxis,np.newaxis].astype(np.float32)])
conv.weights
# plt.imshow(conv.weights[0][:,:,0,0].numpy(), 'coolwarm')
filterd_conv = conv(img_small[np.newaxis,...,np.newaxis].astype(np.float32))[0,:,:,0].numpy()
plt.imshow(filterd_conv, 'gray')
plt.show()

filtered_func = image_filter(img_small, kernel=kernel)
plt.imshow(filtered_func, 'gray')
plt.show()

filtered_cv = cv.filter2D(img_small, ddepth=-1, kernel=kernel)
plt.imshow(filtered_cv, 'gray')
plt.show()

# (filterd_conv != filtered_func).sum()
# plt.imshow(cv.threshold(filtered_func, thresh=-100, maxval=255, type=cv.THRESH_BINARY_INV)[1], 'gray')







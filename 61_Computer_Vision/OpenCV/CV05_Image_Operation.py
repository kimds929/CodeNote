import os
import time
import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv

path = r'D:\Python\★★Python_POSTECH_AI\Dataset'
origin_path = os.getcwd()
os.chdir(path)




# ## Image Addition -------------------------------
x = np.uint8([250])
y = np.uint8([10])

print(cv.add(x,y) ) # 250+10 = 260 => 255
print( x+y )          # 250+10 = 260 % 256 = 4









# ## Bitwise Operations ------------------------------
# Load two images
img1 = cv.imread('messi5.jpg')
img2 = cv.imread('opencv-logo-white.png')

plt.imshow(img1[:,:,::-1])
plt.show()
plt.imshow(img2[:,:,::-1])
plt.show()

# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols]


# Now create a mask of logo and create its inverse mask also
img2gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)

plt.imshow(mask, 'gray')
plt.show()

plt.imshow(mask_inv, 'gray')
plt.show()



# Now black-out the area of logo in ROI
img1_bg = cv.bitwise_and(roi, roi, mask = mask_inv)
plt.imshow(img1_bg)
plt.show()

# Take only region of logo from logo image.
img2_fg = cv.bitwise_and(img2, img2, mask = mask)
plt.imshow(img2_fg[:,:,::-1])
plt.show()



# Put logo in ROI and modify the main image
dst = cv.add(img1_bg, img2_fg)
img1[0:rows, 0:cols ] = dst
# a = roi.copy()
# a[img2_fg != 0] = img2_fg[img2_fg!=0]
# plt.imshow(roi[:,:,::-1])
# plt.imshow(a[:,:,::-1])
# img1[0:rows, 0:cols ] = a

# dst = cv.hconcat(mask,img1)
img1 = cv.cvtColor(img1,cv.COLOR_BGR2RGB)
plt.subplot(1,2,2),plt.imshow(mask,'gray'), plt.title('Mask')
plt.subplot(1,2,1),plt.imshow(img1), plt.title('Image')




import os

import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv
import torch

path = r'D:\Python\★★Python_POSTECH_AI\Postech_AI 7) Computer_Vision\Dataset'
origin_path = os.getcwd()
os.chdir(path)


# image 열기  -----------------------------------------
# Read an image (read "starry_night.jpg")
img = cv.imread('orange.jpg')
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()

# check if the image was loaded correctly.
if img is None:
    raise RuntimeError("Could not read the image.")

assert os.path.exists('tmp.png')


# matplotlib으로 image표시하기 -----------------------------------------
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))



img = cv.imread('messi5.jpg')
px = img[100,100]
print( px )

# accessing only blue pixel
blue = img[100,100,0]
print( blue )


plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))


# shape, size, dtype -----------------------
print( img.shape )
print( img.size )       # np.product(img.shape)
print( img.dtype )


ball = img[280:340, 330:390]
img[273:333, 100:160] = ball
plt.imshow(cv.cvtColor(ball, cv.COLOR_BGR2RGB))
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# plt.imshow(img[:,:,::-1])

b,g,r = cv.split(img)
img_rgb = cv.merge((r,g,b))
plt.imshow(img_rgb)

plt.figure(figsize=(20,10))
plt.imshow(np.concatenate([b,g,r], axis=1), 'gray')
plt.show()




# ## Making Borders for Images (Padding)
BLUE = [255,0,0]

img1 = cv.imread('opencv-logo-white.png')
img1 = cv.cvtColor(img1,cv.COLOR_BGR2RGB)
plt.imshow(img1)
plt.show()


# * Forms a border around an image.
#     * `dst = cv.copyMakeBorder(src, top, bottom, left, right, borderType[, dst[, value]])`
#     * src - input image
#     * top, bottom, left, right - border width in number of pixels in corresponding directions
#     * borderType - Flag defining what kind of border to be added. It can be following types:
#         * cv.BORDER_CONSTANT - Adds a constant colored border. The value should be given as next argument.
#         * cv.BORDER_REFLECT - Border will be mirror reflection of the border elements, like this : fedcba|abcdefgh|hgfedcb
#         * cv.BORDER_REFLECT_101 or cv.BORDER_DEFAULT - Same as above, but with a slight change, like this : gfedcb|abcdefgh|gfedcba
#         * cv.BORDER_REPLICATE - Last element is replicated throughout, like this: aaaaaa|abcdefgh|hhhhhhh
#         * cv.BORDER_WRAP - wrap around border, it will look like this : cdefgh|abcdefgh|abcdefg
#     * value - Color of border if border type is cv.BORDER_CONSTANT


replicate = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REPLICATE)
reflect = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT)
reflect101 = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT_101)
wrap = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_WRAP)

edge_color = [255,255,0]
constant= cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_CONSTANT, value=edge_color)

plt.subplots_adjust(top=1.2)   # 위아래, 상하좌우 간격
plt.subplot(231),plt.imshow(img1),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant),plt.title('CONSTANT')
plt.show()





# ## Image Addition -------------------------------
x = np.uint8([250])
y = np.uint8([10])

print( cv.add(x,y) ) # 250+10 = 260 => 255
print( x+y )          # 250+10 = 260 % 256 = 4

plt.imshow(img1)
plt.imshow(cv.add(img1,img1))




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




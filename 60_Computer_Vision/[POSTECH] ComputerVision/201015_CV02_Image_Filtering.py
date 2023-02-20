import os
import cv2 as cv # OpenCV
import numpy as np
from matplotlib import pyplot as plt

path = r'D:\Python\★★Python_POSTECH_AI\Postech_AI 7) Computer_Vision\Dataset'
origin_path = os.getcwd()
os.chdir(path)


img = cv.imread('building.jpg')
plt.imshow(img[:,:,::-1])
plt.show()

# make kernel
kernel = np.array([[1,1,1,1,1],
                  [1,1,1,1,1],
                  [1,1,1,1,1],
                  [1,1,1,1,1],
                  [1,1,1,1,1],],dtype=np.float32) / 25

# Filtering ----------------------------------------------------------
# cv.filter2D?
#     * `dst = cv.filter2D( src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]])`
#     * The function does actually compute correlation, not the convolution. If you need a real convolution, flip the kernel.
#     * ddepth: bit depth of outout, set ddepth to -1 to retain bit depth of input

# Convolution ----------------------------------------------------------
#     *There are no convolution function in OpenCV
#     *The filter (or the image) is flipped.
#     *If you need convolution, you need to flip kernel and use filter2D function.


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


# filtered1 = cv.filter2D(src=gray, ddepth=-1, kernel=kernel)
# plt.imshow(filtered1, 'gray')
# plt.show()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
filtered2 = image_filter(image=gray, kernel=kernel*25)
plt.imshow(filtered2, 'gray')
plt.show()







filtered = cv.filter2D(src=img, ddepth=-1, kernel=kernel)

plt.figure(figsize=(20,10))
plt.subplot(1,2,1),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)),plt.title('Original')
plt.subplot(1,2,2),plt.imshow(cv.cvtColor(filtered, cv.COLOR_BGR2RGB)),plt.title('Averaged')
plt.show()


ref = cv.imread("box5.png")
mse = np.mean((ref - filtered)**2)
print(mse)
assert mse < 1
# jpg : uint8 손실압축
# png : uint8 비손실압축


# ## Box filter -------------------------------------------------------------
img = cv.imread('building.jpg')

N = 15
kernel = np.ones((N,N),np.float32)/(N*N)

avg = cv.filter2D(img,-1,kernel)

plt.figure(figsize=(20,10))
plt.subplot(1,2,1),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)),plt.title('Original')
plt.subplot(1,2,2),plt.imshow(cv.cvtColor(avg, cv.COLOR_BGR2RGB)),plt.title('Averaged')
plt.show()


# compare image with the test image
ref = cv.imread("box15.png")
mse = np.mean((ref - avg)**2)
print(mse)
assert mse < 1


# ## Other filters
img = cv.imread('building.jpg')

kernel = np.zeros((3,3),dtype=np.float)
kernel[1,1] = 1.0

# image filtering
unchanged = cv.filter2D(img,-1,kernel) 

plt.figure(figsize=(20,10))
plt.subplot(1,2,1),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)),plt.title('Original')
plt.subplot(1,2,2),plt.imshow(cv.cvtColor(unchanged, cv.COLOR_BGR2RGB)),plt.title('Unchanged')
plt.show()



# * 이미지를 왼쪽으로 10 픽셀을 이동시키는 필터 (shift to left)
#     * 커널의 중심으로 부터 10 픽셀 떨어진 위치에 1을 할당 (커널의 중심은 보통 커널의 중간 지점이다.)
img = cv.imread('building.jpg')

# 이미지를 왼쪽으로 10 픽셀을 이동시키는 필터 구현하기

kernel = np.zeros((1,21))
kernel[0,-1] = 1


# image filtering (
# 이미지 보더 부분은 0으로 채워지게 설정 (use borderType=cv.BORDER_CONSTANT)
shifted = cv.filter2D(img,-1,kernel,borderType=cv.BORDER_CONSTANT)

plt.figure(figsize=(20,10))
plt.subplot(1,2,1),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)),plt.title('Original')
plt.subplot(1,2,2),plt.imshow(cv.cvtColor(shifted, cv.COLOR_BGR2RGB)),plt.title('shift to left')
plt.show()



# Sharpening Filter -------------------------
# make kernel
N = 7  # 박스 필터의 크기는 7으로 할 것
kernel = -np.ones((N,N),np.float32)/(N*N)
kernel[3,3] = 2 + kernel[3,3]

# image filtering
sharpned = shifted = cv.filter2D(img,-1,kernel)

plt.figure(figsize=(20,10))
plt.subplot(1,2,1),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)),plt.title('Original')
plt.subplot(1,2,2),plt.imshow(cv.cvtColor(sharpned, cv.COLOR_BGR2RGB)),plt.title('Sharpened')
plt.show()

# compare image with the test image
ref = cv.imread("sharpned7.png")
mse = np.mean((ref - sharpned)**2)
print(mse)
assert mse < 1





# gaussian filter를 구현 ---------------------------------------
# cv.GaussianBlur 를 사용하지 마세요.

# make kernel
sigma = 2.0 # sigma는 2.0로 할것
N = 7 # 필터 크기를 7로 할것

# 커널의 x와 y 좌표에 대해 gaussian function을 구한다.
idx = np.arange(start=-(N//2),stop=(N//2)+1)
x_idx, y_idx = np.meshgrid(idx, idx)

kernel = 1/(2*np.pi*(sigma**2)) * np.exp(-((x_idx**2)+(y_idx**2))/(2 * (sigma**2)))

# 커널의 합(np.sum(kernel))이 1이 되야 함 (np.sum 을 사용)
kernel = kernel/np.sum(kernel)

# image filtering
blurred = cv.filter2D(img,-1,kernel)

plt.figure(figsize=(20,10))
plt.subplot(1,2,1),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)),plt.title('Original')
plt.subplot(1,2,2),plt.imshow(cv.cvtColor(blurred, cv.COLOR_BGR2RGB)),plt.title('Blurred')
plt.show()


# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# filtered_gaussian = image_filter(image=gray, kernel=kernel)
# plt.imshow(filtered_gaussian, 'gray')
# plt.show()








# # ### Sobel filter and Image Gradient
img = cv.imread('starry_night.jpg')
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY).astype(np.float) / 255

# Sobel filter를 구현하여 image gradient 구하기
# horizontal derivative, vertical derivative, gradient amplitude를 구하기

# make kernel
fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float)
fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float)

# image filtering
# compute horizontal derivative, vertical derivative 
dx = cv.filter2D(img,-1,fx)
dy = cv.filter2D(img,-1,fy)

# compute amplitude 
amp = np.sqrt(dx**2 + dy**2)

# --------------------------------

# convert data type 
dx  = (255*np.clip(dx,0.0,1.0)).astype(np.uint8)
dy = (255*np.clip(dy,0.0,1.0)).astype(np.uint8)
amp = (255*np.clip(amp,0.0,1.0)).astype(np.uint8)

plt.figure(figsize=(20,20))
plt.subplot(2,2,1),plt.imshow(img, cmap='gray'),plt.title('Original')
plt.subplot(2,2,2),plt.imshow(dx, cmap='gray'),plt.title('horizontal derivative')
plt.subplot(2,2,3),plt.imshow(dy, cmap='gray'),plt.title('vertical derivative')
plt.subplot(2,2,4),plt.imshow(amp, cmap='gray'),plt.title('gradient amplitude')
plt.show()

np.min(fx*255)
np.max(fx*255)
# Cumstom Filter
starry_night_gray = cv.imread('starry_night.jpg', cv.IMREAD_GRAYSCALE).astype(np.float) / 255
sobel_dx = image_filter(image=starry_night_gray, kernel=fx)
sobel_dy = image_filter(image=starry_night_gray, kernel=fy)
sobel_amp = np.sqrt(sobel_dx**2 + sobel_dy**2)
# sobel_amp = np.sqrt(sobel_dx.astype(float)**2 + sobel_dy.astype(float)**2).astype('uint8')

plt.figure(figsize=(20,20))
plt.subplot(2,2,1),plt.imshow(starry_night_gray, cmap='gray'),plt.title('Original')
plt.subplot(2,2,2),plt.imshow(sobel_dx.astype('uint8'), cmap='gray'),plt.title('horizontal derivative')
plt.subplot(2,2,3),plt.imshow(sobel_dy.astype('uint8'), cmap='gray'),plt.title('vertical derivative')
plt.subplot(2,2,4),plt.imshow(sobel_amp.astype('uint8'), cmap='gray'),plt.title('gradient amplitude')
plt.show()


# compare image with the test image
ref = cv.imread("dx.png", cv.IMREAD_UNCHANGED)
mse = np.mean((ref - dx)**2)
print(mse)
assert mse < 1

ref = cv.imread("dy.png", cv.IMREAD_UNCHANGED)
mse = np.mean((ref - dy)**2)
print(mse)
assert mse < 1

ref = cv.imread("amp.png", cv.IMREAD_UNCHANGED)
mse = np.mean((ref - amp)**2)
print(mse)
assert mse < 1


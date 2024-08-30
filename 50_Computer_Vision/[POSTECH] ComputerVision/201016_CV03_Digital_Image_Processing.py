import os
import cv2 as cv # OpenCV
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [16, 9]

path = r'D:\Python\★★Python_POSTECH_AI\Postech_AI 7) Computer_Vision\Dataset'
origin_path = os.getcwd()
os.chdir(path)


# #### Histogram Calculation in OpenCV -----------------------------------------------------
# So now we use `cv.calcHist()` function to find the histogram.
# * `hist = cv.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])`
#     * Calculates a histogram of a set of arrays.
#     * images : it is the source image of type uint8 or float32. it should be given in square brackets, ie, "[img]".
#     * channels : it is also given in square brackets. It is the index of channel for which we calculate histogram.
#     For example, if input is grayscale image, its value is [0].
#     For color image, you can pass [0], [1] or [2] to calculate histogram of blue, green or red channel respectively.
#     * mask : mask image. To find histogram of full image, it is given as "None".
#     But if you want to find histogram of particular region of image, you have to create a mask image for that and give it as mask.
#     * histSize : this represents our BIN count. Need to be given in square brackets. For full scale, we pass [256].
#     * ranges : this is our RANGE. Normally, it is [0,256].

img = cv.imread('home.jpg',0)

plt.imshow(img, 'gray')
plt.show()
plt.hist(img.ravel(), bins=256, range=(0,256), edgecolor='gray', color='skyblue')

# calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]) -> hist
n_bins = 10
hist = cv.calcHist(images=img[np.newaxis,...], channels=[0], mask=None, 
                histSize=[n_bins], ranges=(0,256))

hist = hist[:,0] # change shape. (256)
bins = np.array(list(range(0,n_bins))) # create bins. (256)

plt.subplot(2,2,1), plt.imshow(img, 'gray')
plt.subplot(2,2,2), plt.bar(bins, hist)
plt.subplot(2,2,3), plt.stem(bins, hist)
plt.subplot(2,2,4), plt.plot(bins, hist)
plt.show()


# Mask 적용하기 -----------------------------------------------------
img = cv.imread('home.jpg',0)
plt.imshow(img, 'gray')
plt.show()
img.shape

mask = np.zeros_like(img)
mask[100:300, 100:400] = 255
plt.imshow(mask, 'gray')

hist = cv.calcHist(images=img[np.newaxis,...], channels=[0], mask=None, 
                histSize=[256], ranges=(0,256))
hist_mask = cv.calcHist(images=img[np.newaxis,...], channels=[0], mask=mask, 
                histSize=[256], ranges=(0,256))

masked_img = cv.bitwise_and(img, img, mask=mask)

hist = hist[:,0] # change shape. (256)
bins = np.array(list(range(0,n_bins))) # create bins. (256)

plt.subplot(2,2,1), plt.imshow(img, 'gray')
plt.subplot(2,2,2), plt.imshow(mask,'gray')
plt.subplot(2,2,3), plt.imshow(masked_img, 'gray')
plt.subplot(2,2,4), plt.plot(hist, label='full'), plt.plot(hist_mask, label='mask'), plt.legend()
plt.xlim([0,256])
plt.show()




# numpy로 histogram 구하기 ---------------------------------------------------
# #### Histogram Calculation in Numpy
# Numpy also provides you a function, `np.histogram()`. So instead of `cv.calcHist()` function.
# 
# * `numpy.histogram(a, bins=10, range=None, normed=None, weights=None, density=None)`
#     * a: Input data. The histogram is computed over the flattened array.
#     * bins: If bins is an int, it defines the number of equal-width bins in the given range (10, by default).
#     If bins is a sequence, it defines a monotonically increasing array of bin edges, including the rightmost edge, allowing for non-uniform bin widths.
#     * range: The lower and upper range of the bins. If not provided, range is simply (a.min(), a.max()).
#     * Returns:
#         * hist: The values of the histogram. See density and weights for a description of the possible semantics.
#         * bin_edges: Return the bin edges (length(hist)+1).
# 
# `hist` is same as we calculated before. But `numpy.histogram` returns edge of bins `bin_edges` which have 257 elements.


n_bins = 256
hist_np, bin_edge = np.histogram(img, bins=n_bins, range=(0,256))
bins = ((bin_edge[1:] + bin_edge[:-1]) / 2).astype(int).astype(str)

plt.subplot(2,2,1), plt.imshow(img, 'gray')
plt.subplot(2,2,2), plt.bar(bins, hist_np)
plt.subplot(2,2,3), plt.stem(bins, hist_np)
plt.subplot(2,2,4), plt.plot(bins, hist_np)
plt.show()





# #### Compute CDF ---------------------------------------------------
# You can get histogram from CDF.
# * CDF
#     * Like histogram, you can get intuition about contrast, brightness, intensity distribution etc of that image by exmining CDF.
#     * CDF can be created by summing histogram from the first bin to last bin and collecting all intermediate sum values.
#         * `np.cumsum`: sum array from the first element to last element and collect all intermediate sum values.
img = cv.imread('wiki.jpg',0)
plt.imshow(img, 'gray')
plt.show()

hist_np, bin_edge = np.histogram(img, bins=n_bins, range=(0,256))
bins = (bin_edge[1:] + bin_edge[:-1]) / 2

cdf = np.cumsum(hist_np)
cdf_ = cdf * float(hist_np.max()) / cdf.max()      # for visualization

# display
plt.subplot(2,1,1), plt.imshow(img, 'gray')
plt.subplot(2,1,2)
plt.plot(cdf_, color = 'black')
plt.bar(bins, hist_np, color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()









# ### Histogram Equalization  ---------------------------------------------------
# In this section, we will learn the concepts of histogram equalization and use it to improve the contrast of our images.
# Consider an image whose pixel values are confined to some specific range of values only.
# For eg, brighter image will have all pixels confined to high values.
# But a good image will have pixels values from full range.
# So you need to stretch this histogram to either ends and that is what Histogram Equalization does (in simple words).
# This normally improves the contrast of the image.


# ### Histograms Equalization in OpenCV ---------------------------------------------------
# OpenCV has a function to do this, `cv.equalizeHist()`.
# Its input is just grayscale image and output is our histogram equalized image.
# `dst= cv.equalizeHist(src[, dst])`
# 
# Below is a simple code snippet showing its usage for same image we used.
# We can take different images with different light conditions, equalize it and check the results.
# The Histogram of original image lies in brighter region.
# The Histogram of equalized image lies in full range.
# 
# Histogram equalization is good when histogram of the image is confined to a particular region.
# It won't work good in places where there is large intensity variations where histogram covers a large region, ie both bright and dark pixels are present.


img = cv.imread('wiki.jpg',0)
plt.imshow(img, 'gray')
plt.show()

equ = cv.equalizeHist(img)
plt.imshow(cv.equalizeHist(img), 'gray')
plt.show()



# display
plt.subplot(2,2,1),plt.imshow(img, 'gray')
plt.subplot(2,2,2),plt.imshow(equ, 'gray')

hist,bins_edges = np.histogram(img,256,[0,256])
cdf = np.cumsum(hist, dtype=hist.dtype)
cdf_ = cdf * float(hist.max()) / cdf.max() # for visualization
plt.subplot(2,2,3), plt.plot(cdf_, color = 'black'), plt.bar(bins, hist, color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')

hist_equalized, bins_edges = np.histogram(equ,256,[0,256])
cdf_equalized = np.cumsum(hist_equalized, dtype=hist.dtype)
cdf_ = cdf_equalized * float(hist_equalized.max()) / cdf_equalized.max() # for visualization
plt.subplot(2,2,4), plt.plot(cdf_, color = 'black'), plt.bar(bins, hist_equalized, color = 'r')

plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()






# ### Histograms Equalization in Numpy ------------------------------------------------------------------------
# To compute Histograms Equalization, we need a transformation function which maps the input pixels in brighter region to output pixels in full region.
# 
# How to compute Histogram Equalization
# 1. Compute histogram of the image
# 2. Compute CDF
#     * Use `np.cumsum`
# 3. Normalize CDF
#     * Find minimum, maximum of te CDF (excluding zeros)
#     * Normalize: `cdf_normalized[i] = (cdf[i] - min)*255.0/(max-min)` (excluding zeros)
# 4. Change pixel values using normalized CDF
#     * Assign new pixel values using numpy (transform pixel value using CDF)


img = cv.imread('wiki.jpg',0)

# 1. compute histogram
hist_np, bin_edge = np.histogram(img, bins=n_bins, range=(0,256))

# 2. compute cdf
cdf = np.cumsum(hist_np)

# 3. normalize cdf
# find min, max of cdf (excluding zeros)
cdf_nonzero = cdf[cdf!=0]
cdf_min = np.min(cdf_nonzero)
cdf_max = np.max(cdf_nonzero)

cdf_normalized = np.zeros_like(cdf)
cdf_normalized = (cdf - cdf_min)*255.0/(cdf_max - cdf_min)
# cdf_normalized[i] = (cdf[i] - cdf_min)*255.0/(cdf_max - cdf_min)
cdf_normalized[cdf==0] = 0
cdf_normalized


# 4. Change pixel values
img_equ = np.zeros_like(img)
img_equ = cdf_normalized[img]
# for i in range(len(cdf_normalized)):
#     img_equ[img == i] = cdf_normalized[i]



# visualize
plt.subplot(2,2,1),plt.imshow(img, 'gray'), plt.title('Original')
plt.subplot(2,2,2),plt.imshow(img_equ, 'gray'), plt.title('Equalized')

hist_equalized, bins_edges = np.histogram(img_equ.flatten(),256,[0,256])
cdf_equalized = np.cumsum(hist_equalized, dtype=hist.dtype)
cdf_ = cdf_equalized * float(hist_equalized.max()) / cdf.max() # for visualization
plt.subplot(2,2,(3,4)), plt.plot(cdf_, color = 'black'), plt.bar(bins, hist_equalized, color = 'r')

plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()






# ### CLAHE (Contrast Limited Adaptive Histogram Equalization)
# The first histogram equalization we just saw, considers the global contrast of the image.
# In many cases, it is not a good idea.
# For example, below image shows an input image and its result after global histogram equalization.
img = cv.imread('tsukuba_l.png',0)

# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

# display
plt.subplot(2,2,1),plt.imshow(img, 'gray'), plt.title('Original')
plt.subplot(2,2,2),plt.imshow(cl1, 'gray'), plt.title('Equalized')
plt.show()



# It is true that the background contrast has improved after histogram equalization.
# But compare the face of statue in both images.
# We lost most of the information there due to over-brightness.
# It is because its histogram is not confined to a particular region as we saw in previous cases.
# 
# So to solve this problem, adaptive histogram equalization is used.
# An image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV).
# Then each of these blocks are histogram equalized as usual.
# So in a small area, histogram would confine to a small region (unless there is noise).
# If noise is there, it will be amplified.
# To avoid this, contrast limiting is applied.
# If any histogram bin is above the specified contrast limit (by default 40 in OpenCV),
# those pixels are clipped and distributed uniformly to other bins before applying histogram equalization.
# After equalization, to remove artifacts in tile borders, bilinear interpolation is applied.
# 
# Below code snippet shows how to apply CLAHE in OpenCV:

img = cv.imread('tsukuba_l.png',0)

# Histogram Equalization
equ = cv.equalizeHist(img)

# CLAHE
# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

# display
plt.subplot(1,3,1),plt.imshow(img, 'gray'), plt.title('Original')
plt.subplot(1,3,2),plt.imshow(equ, 'gray'), plt.title('Histogram Equalization')
plt.subplot(1,3,3),plt.imshow(cl1, 'gray'), plt.title('CLAHE')
plt.show()











# =============================================================================================
# # Image thresholding
# Image thresholding is a case of binary segmentation.
# Image thresholding assigns white / black to each pixel according to its intensity.
# If the pixel value is smaller than the threshold, it is set to 0, otherwise it is set to a maximum value.
# 
# The function `cv.threshold` is used to apply the thresholding.
# 
# * `retval, dst = cv.threshold( src, thresh, maxval, type[, dst])`
#     * Applies a fixed-level threshold to each array element.
#     * `thresh`:	threshold value which is used to classify the pixel values.
#     * `maxval`: maximum value which is assigned to pixel values exceeding the threshold.
#     * `type`: OpenCV provides different types of thresholding which is given by the fourth parameter of the function.
#     Basic thresholding as described above is done by using the type `cv.THRESH_BINARY`.


# create gradient image 
img = np.arange(0,256,1)
img = np.vstack([img]*256).astype(np.uint8)

# extract mask using thresholding
ret, mask = cv.threshold(img, 50, 255, cv.THRESH_BINARY)
# mask = ((50 < img)*255).astype(np.uint8)                  # chage dtype

# display
plt.subplot(1,2,1),plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB)), plt.title('Image')
plt.subplot(1,2,2),plt.imshow(mask,'gray'), plt.title('Mask')



# Smooth_Mask
# sigmoid = 1/ (1 + np.exp(-x))
img_ = img.astype('float')
mask = 50 < img_
mask = (1/ (1 + np.exp(-(img_-50))) *255).astype('uint8')

# display
plt.subplot(1,2,1),plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB)), plt.title('Image')
plt.subplot(1,2,2),plt.imshow(mask,'gray'), plt.title('Mask')







# # Color ------------------------------------------------------------------------------------------------------
# * Color Spaces
#     * Color space coversion
# * Color Detection
# * Splash of color

# ## Color spaces

# ### Changing Color-space
# 
# You will learn how to convert images from one color-space to another, like BGR $\leftrightarrow$ Gray, BGR $\leftrightarrow$ HSV etc.
# 
# For color conversion, we use the function `cv.cvtColor(input_image, flag)` where flag determines the type of conversion.
# 
# * `dst = cv.cvtColor( src, code[, dst[, dstCn]])`
#     * Converts an image from one color space to another.
#     * `code`: color space conversion code
#         * For BGR $\rightarrow$ Gray conversion we use the flags cv.COLOR_BGR2GRAY.
#         * Similarly for BGR $\rightarrow$ HSV, we use the flag cv.COLOR_BGR2HSV.
# 
# There are more than 150 color-space conversion methods available in OpenCV.
# (https://docs.opencv.org/4.4.0/d8/d01/group__imgproc__color__conversions.html#ga4e0972be5de079fed4e3a10e24ef5ef0).
# 
# But we will look into only two which are most widely used ones, BGR $\leftrightarrow$ Gray, BGR $\leftrightarrow$ HSV.
# To get other flags, just run following commands in Python:

flags = [i for i in dir(cv) if i.startswith('COLOR_')]
print(np.array(flags))



# ### Gray --------------------------------------------------------------
# 
# For BGR $\rightarrow$ Gray conversion we use the flags cv.COLOR_BGR2GRAY.
# 
# #### Convert a image to grayscale using OpenCV
img = cv.imread('Gretag-Macbeth_ColorChecker.jpg')
plt.imshow(img[:,:,::-1])
plt.show()

# split channels
b, g, r = cv.split(img)

# convert
# gray_rgb = cv.add(b, g, r)
gray_rgb = 0.299*r + 0.587*g + 0.144*b

# change dtype
# gray_rgb = gray_rgb.astype('uint8')
gray_rgb = np.clip(gray_rgb, 0, 255).astype('uint8')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# display
plt.subplot(2,2,1), plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB)), plt.title('Original')
plt.subplot(2,2,3), plt.imshow(gray,'gray'), plt.title('Grayscale from Function')
plt.subplot(2,2,4), plt.imshow(gray_rgb,'gray'), plt.title('Grayscale from RGB')
plt.show()

# display
plt.subplot(1,4,1), plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB)), plt.title('Original')
plt.subplot(1,4,2), plt.imshow(b,'gray'), plt.title('B')
plt.subplot(1,4,3), plt.imshow(g,'gray'), plt.title('G')
plt.subplot(1,4,4), plt.imshow(r,'gray'), plt.title('R')
plt.show()




# ### HSV ---------------------------------------------------------------------------------
# 
# In HSV, it is more easier to represent a color than RGB color-space.
# 
# * HSV
#     * The HSV color space has the following three components
#         * H – Hue ( Dominant Color, 색상).
#         * S – Saturation ( Purity / shades of the color / 채도 ).
#         * V – Value ( Intensity/ brightness / 명도 ).
#     * It uses only one channel to describe color (H), making it very intuitive to specify color.
# 
# For BGR $\rightarrow$ HSV, we use the flag cv.COLOR_BGR2HSV.
# 
# * Note
#     * In OpenCv, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255].
#     * Different softwares use different scales.
#     So if you are comparing OpenCV values with them, you need to normalize these ranges.

img = cv.imread('Gretag-Macbeth_ColorChecker.jpg')
plt.imshow(img[:,:,::-1])
plt.show()

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

h, s, v = cv.split(hsv)


plt.subplot(1,4,1), plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB)), plt.title('Original')
plt.subplot(1,4,2), plt.imshow(h,'gray'), plt.title('H')
plt.subplot(1,4,3), plt.imshow(s,'gray'), plt.title('S')
plt.subplot(1,4,4), plt.imshow(v,'gray'), plt.title('V')
plt.show()






# ## Color detection ---------------------------------------------------------------------------------
# We will create an application which extracts a colored object.
# In our application, we will try to extract a blue colored object.
# 
# So here is the method:
# 1. Convert from BGR to HSV color-space
# 2. We threshold the HSV image for a range of blue color
# 3. Now extract the blue object alone, we can do whatever on that image we want.


img = cv.imread('smarties.png')
plt.imshow(img[:,:,::-1])
plt.show()

# Convert BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# define range of blue color in HSV
lower_blue = np.array([100,50,50]) #  blue
upper_blue = np.array([160,255,255]) #  blue

# Threshold the HSV image to get only blue colors
mask = cv.inRange(hsv, lower_blue, upper_blue)

# Bitwise-AND mask and original image
res = cv.bitwise_and(img, img, mask=mask)

# display
plt.subplot(1,3,1), plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB)), plt.title('Original')
plt.subplot(1,3,2), plt.imshow(mask,'gray'), plt.title('Mask')
plt.subplot(1,3,3), plt.imshow(cv.cvtColor(res,cv.COLOR_BGR2RGB)), plt.title('Thresholding')
plt.show()



# ### How to find HSV values from RGB values?
# It is very simple and you can use the same function, `cv.cvtColor()`.
# Instead of passing an image, you just pass the BGR values you want.
# For example, to find the HSV value of Green, try following commands in Python
# 
# Now you can take [H-10, 100,100] and [H+10, 255, 255] as lower bound and upper bound for thresholding.
# Apart from this method, you can use any image editing tools or any online converters to find these values, but don’t forget to adjust the HSV ranges.

# BGR green
bgr_green = np.uint8([[[0, 255, 0]]])

# convert
hsv_green = cv.cvtColor(bgr_green, cv.COLOR_BGR2HSV)

print(hsv_green) # [[[ 60 255 255]]]



# ### Multiple Colors -----------------------------------------------------------------------------------------------
# Try to find a way to extract more than one colored objects, for eg, extract blue, green objects simultaneously.
img = cv.imread('smarties.png')
plt.imshow(img[:,:,::-1])
plt.show()

# Convert BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# find HSV
blue = np.uint8([[[246, 144, 52]]])
hsv_blue = cv.cvtColor(blue, cv.COLOR_BGR2HSV) # [[[106 201 246]]]
print(hsv_blue)

# find HSV
green = np.uint8([[[53, 255, 0]]])
hsv_green = cv.cvtColor(green, cv.COLOR_BGR2HSV) # [[[ 66 255 255]]]
print(hsv_green)

# define range of blue color in HSV
lower_blue = np.array([80, 50, 50])
upper_blue = np.array([120, 255, 255])

lower_green = np.array([30, 50, 50])
upper_green = np.array([90, 255, 255])

# Threshold the HSV image to get only blue colors
mask_blue = cv.inRange(hsv, lower_blue, upper_blue)
mask_green = cv.inRange(hsv, lower_green, upper_green)

# Bitwise-OR
mask = cv.bitwise_or(mask_blue, mask_green)

# Bitwise-AND mask and original image
res = cv.bitwise_and(img, img, mask=mask)

# display
plt.subplot(1,3,1), plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB)), plt.title('Original')
plt.subplot(1,3,2), plt.imshow(mask,'gray'), plt.title('Mask')
plt.subplot(1,3,3), plt.imshow(cv.cvtColor(res,cv.COLOR_BGR2RGB)), plt.title('Thresholding')
plt.show()







# ## Splash of Color -----------------------------------------------------------------------------------------------
# 
# The term splash of color refers to the effect of the use of a colored item on an otherwise monochrome image to draw extra attention to the item.
# It has been used frequently in films as a form of emphasis.
# Some commercials will film a portion in black and white, except the product which appears in color.
# 
# * Methods
# 1. Convert from BGR to HSV color-space
# 2. We threshold the HSV image for a range of desired color to get a mask.
# 3. Set saturation to 0 in pixels outside mask.
# 4. Convert from HSV to BGR color-space

img = cv.imread('smarties.png')
plt.imshow(img[:,:,::-1])
plt.show()

# Convert BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# define range of blue color in HSV
# sample blue HSV [141, 224, 142]
lower_blue = np.array([80, 50, 50])
upper_blue = np.array([120, 255, 255])

lower_green = np.array([30, 50, 50])
upper_green = np.array([90, 255, 255])


# Threshold the HSV image to get only blue colors
mask = cv.inRange(hsv, lower_blue, upper_blue)
# mask_blue = cv.inRange(hsv, lower_blue, upper_blue)
# mask_green = cv.inRange(hsv, lower_green, upper_green)
# mask = cv.bitwise_or(mask_blue, mask_green)


# Set saturation to 0 in pixels outside mask.
# blue인 부분은 채도를 0으로 나머지는 그대로
hsv[mask !=255, 1] = 0


# Convert from HSV to BGR color-space
img2 = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

# display
plt.subplot(1,3,1), plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB)), plt.title('Original')
plt.subplot(1,3,2), plt.imshow(mask,'gray'), plt.title('Mask')
plt.subplot(1,3,3), plt.imshow(cv.cvtColor(img2,cv.COLOR_BGR2RGB)), plt.title('Thresholding')
plt.show()
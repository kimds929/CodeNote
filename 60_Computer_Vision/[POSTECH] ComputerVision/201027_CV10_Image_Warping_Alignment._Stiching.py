import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2 as cv

path = r'D:/Python/★★Python_POSTECH_AI/Postech_AI 7) Computer_Vision/Dataset'
# path = r'D:/Python/★★Python_POSTECH_AI/Postech_AI 7) Computer_Vision/Dataset/Lecture08_Image restoration/'
# path = r'/home/pirl/data/Lecture08_Image restoration/'
origin_path = os.getcwd()
os.chdir(path)
# os.listdir()

plt.rcParams['figure.figsize'] = [16, 9]



# ========== Warping =========================================================================
# ## Scaling -------------------------------------------------------------------
# Scaling is just resizing of the image.
# OpenCV comes with a function `cv.resize()` for this purpose.
# The size of the image can be specified manually, or you can specify the scaling factor.
# Different interpolation methods are used.
# Preferable interpolation methods are `cv.INTER_AREA` for shrinking and `cv.INTER_CUBIC` (slow) & `cv.INTER_LINEAR` for zooming.
# By default, the interpolation method `cv.INTER_LINEAR` is used for all resizing purposes.
# 
# * Resizes an image.
#     * `dst = cv.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])`
#         * dsize: output image size
#         * fx: scale factor along the horizontal axis
#         * fy: scale factor along the vertical axis;


img = cv.imread('messi5.jpg')
plt.imshow(img[:, :, ::-1])
plt.show()

# scale image (1.5x width, 2x height)
width, height = img.shape[1], img.shape[0]
# dst = cv.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])`
dst = cv.resize(img, dsize=(round(width*1.5), round(height*2)))

plt.subplot(121),plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB)),plt.title('Input')
plt.subplot(122),plt.imshow(cv.cvtColor(dst,cv.COLOR_BGR2RGB)),plt.title('Output')
plt.show()






# ## Translation -------------------------------------------------------------------
# Translation is the shifting of an object's location.
# If you know the shift in the (x,y) direction and let it be (tx,ty), you can create the transformation matrix M as follows:
# 
# $M = \begin{bmatrix}1 & 0 & t_x\\ 1 & 0 & t_y\end{bmatrix}$
# 
# You can take make it into a Numpy array of type np.float32 and pass it into the cv.warpAffine() function.
# 
# * Applies an affine transformation to an image.
#     * `dst = cv.warpAffine( src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]])`
#     * warning: The third argument of the cv.warpAffine() function is the size of the output image, which should be in the form of **(width, height)**. Remember width = number of columns, and height = number of rows.


img = cv.imread('messi5.jpg',0)

# translate image (translation of (100,50))
rows, cols = img.shape

M = np.array([[1, 0, 100], 
              [0, 1, 50]], dtype=np.float32)    # 왼쪽으로 100, 아래로 50 trainslation
# dst = cv.warpAffine( src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]])
dst = cv.warpAffine(img, M, dsize=(cols, rows))

plt.subplot(121),plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB)),plt.title('Input')
plt.subplot(122),plt.imshow(cv.cvtColor(dst,cv.COLOR_BGR2RGB)),plt.title('Output')
plt.show()






# ## Rotation ----------------------------------------------------------------------------
# Rotation of an image for an angle θ is achieved by the transformation matrix of the form
# 
# $M = \begin{bmatrix} cos\theta & -sin\theta \\ sin\theta & cos\theta \end{bmatrix}$
# 
# But OpenCV provides scaled rotation with adjustable center of rotation so that you can rotate at any location you prefer. The modified transformation matrix is given by
# 
# $M = \begin{bmatrix} \alpha & \beta & (1- \alpha ) \cdot center.x - \beta \cdot center.y \\ - \beta & \alpha & \beta \cdot center.x + (1- \alpha ) \cdot center.y\end{bmatrix}$
# 
# where:
# 
# $ \alpha = scale \cdot \cos \theta , \\ \beta = scale \cdot \sin \theta$
# 
# To find this transformation matrix, OpenCV provides a function, `cv.getRotationMatrix2D`. Check out the below example which rotates the image by 90 degree with respect to center without any scaling.
# 
# * Calculates an affine matrix of 2D rotation.
#     * `retval = cv.getRotationMatrix2D(center, angle, scale)`
# 
# * Applies an affine transformation to an image.
#     * `dst = cv.warpAffine( src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]])`
#     * warning: The third argument of the cv.warpAffine() function is the size of the output image, which should be in the form of **(width, height)**. Remember width = number of columns, and height = number of rows.


img = cv.imread('messi5.jpg',0)

# compute rotation matrix and rotate image (center should be image center, angle=45, scale=1)
rows, cols = img.shape       # cols-1 and rows-1 are the coordinate limits.

# retval = cv.getRotationMatrix2D(center, angle, scale)`
M = cv.getRotationMatrix2D(center=(cols/2, rows/2), angle=45, scale=1)
dst = cv.warpAffine(img, M, dsize=(cols, rows))

plt.subplot(121),plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB)),plt.title('Input')
plt.subplot(122),plt.imshow(cv.cvtColor(dst,cv.COLOR_BGR2RGB)),plt.title('Output')
plt.show()







# ## Affine Transformation  ----------------------------------------------------------------------------
#  affine transformation, all parallel lines in the original image will still be parallel in the output image. To find the transformation matrix, we need three points from the input image and their corresponding locations in the output image. Then cv.getAffineTransform will create a 2x3 matrix which is to be passed to cv.warpAffine.
# 
# Check the below example, and also look at the green points
# 
# * Calculates an affine transform from three pairs of the corresponding points.
#     * `retval = cv.getAffineTransform(src, dst)`
# 
# * Applies an affine transformation to an image.
#     * `dst = cv.warpAffine( src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]])`
#     * warning: The third argument of the cv.warpAffine() function is the size of the output image, which should be in the form of **(width, height)**. Remember width = number of columns, and height = number of rows.

img = cv.imread('drawing.png')

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

# draw points
for p in pts1:
    img = cv.circle(img, tuple(p), radius=0, color=(0, 255, 0), thickness=8)

# compute affine transformation matrix and transform image
rows, cols, ch = img.shape
M = cv.getAffineTransform(pts1, pts2)
dst = cv.warpAffine(img, M, dsize=(cols, rows))


plt.subplot(121),plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB)),plt.title('Input')
plt.subplot(122),plt.imshow(cv.cvtColor(dst,cv.COLOR_BGR2RGB)),plt.title('Output')
plt.show()




# ## Perspective Transformation  ----------------------------------------------------------------------------
# For perspective transformation, you need a 3x3 transformation matrix.
# Straight lines will remain straight even after the transformation.
# To find this transformation matrix, you need 4 points on the input image and corresponding points on the output image.
# Among these 4 points, 3 of them should not be collinear.
# Then the transformation matrix can be found by the function cv.getPerspectiveTransform.
# Then apply cv.warpPerspective with this 3x3 transformation matrix.
# 
# * Calculates a perspective transform from four pairs of the corresponding points.
#     * `retval = cv.getPerspectiveTransform(src, dst[, solveMethod])`
# 
# * Applies a perspective transformation to an image:
#     * `dst = cv.warpPerspective(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]])`

img = cv.imread('sudoku.png')

pts1 = np.float32([[72,87],[490,70],[40,513],[519,519]])
pts2 = np.float32([[0,0],[500,0],[0,500],[500,500]])
# draw points
for p in pts1:
    img = cv.circle(img, tuple(p), radius=0, color=(0, 255, 0), thickness=8)

rows, cols, ch = img.shape
# compute perspective transformation matrix and transform image (dsize=(300,300))

M = cv.getPerspectiveTransform(pts1, pts2)
dst = cv.warpPerspective(img, M, dsize=(cols, rows))

plt.subplot(121),plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB)),plt.title('Input')
plt.subplot(122),plt.imshow(cv.cvtColor(dst,cv.COLOR_BGR2RGB)),plt.title('Output')
plt.show()




















# ========== Alignment =========================================================================
# ## Feature detection
# * Goal - Find points in an image that can be:
#     * Found in other images
#     * Found precisely – well localized
#     * Found reliably – well matched


# read images
img1 = cv.imread('box.png', 0)  # queryImage
img2 = cv.imread('box_in_scene.png', 0)  # trainImage

plt.subplot(1,2,1), plt.imshow(img1, 'gray'), plt.title('img1')
plt.subplot(1,2,2), plt.imshow(img2, 'gray'), plt.title('img2')
plt.show()



# ### Detect and Compute SIFT Features ---------------------------------------------------------------------
# * Create SIFT detector
#     * `cv.SIFT_create([, nfeatures[, nOctaveLayers[, contrastThreshold[, edgeThreshold[, sigma]]]])`
# 
# * Detects keypoints and computes the descriptors:
#     * `kp, sp = sift.detectAndCompute(image, mask[, descriptors[, useProvidedKeypoints]])`


# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT (mask=None)
# TODO
kp1, des1 = sift.detectAndCompute(img1, mask=None)
kp2, des2 = sift.detectAndCompute(img2, mask=None)



# ### Visualize Features
img1_vis = np.zeros_like(img1)
img2_vis = np.zeros_like(img2)
# img1_vis = cv.drawKeypoints(img1, kp1, img1_vis, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img1_vis = cv.drawKeypoints(img1, kp1, img1_vis, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_vis = cv.drawKeypoints(img2, kp2, img2_vis, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.subplot(1,2,1), plt.imshow(img1_vis, 'gray'), plt.title('Detected SIFT Features in img1')
plt.subplot(1,2,2), plt.imshow(img2_vis, 'gray'), plt.title('Detected SIFT Features in img2')
plt.show()




# ## Feature matching  ---------------------------------------------------------------------
# * We know how to detect feature points. Next question: How to match them?
#     * create the Matcher object
#     * compute matches
# 
# We will use FLANN based matcher.
# FLANN stands for Fast Library for Approximate Nearest Neighbors.
# It contains a collection of algorithms optimized for fast nearest neighbor search in large datasets and for high dimensional features.
# 
# * Create FLANN based matcher:
#     * `matcher = cv.FlannBasedMatcher([, indexParams[, searchParams]])`
# 
# 
# * Compute matches (returns k best matches where k is specified by the user.):
#     * `matches = matcher.knnMatch(queryDescriptors, trainDescriptors, k[, mask[, compactResult]])`


# For FLANN based matcher, we need to pass two dictionaries which specifies the algorithm to be used, its related parameters etc.
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5) # First one is IndexParams. For various algorithms, the information to be passed is explained in FLANN docs.
search_params = dict(checks=50) # Second dictionary is the SearchParams. It specifies the number of times the trees in the index should be recursively traversed. Higher values gives better precision, but also takes more time.

# Create FLANN based matcher
# TODO
matcher = cv.FlannBasedMatcher(index_params, search_params)

# compute k best matches (k=2)
matches = matcher.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
# matches.distance - Distance between descriptors. The lower, the better it is.
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance: # Lowe's ratio test.
        good_matches.append(m)

print(len(good_matches))


# * Draw matched points:
#     * `cv.drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg[, matchColor[, singlePointColor[, matchesMask[, flags]]]])`

# draw matches
draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

img_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)

plt.imshow(img_matches, 'gray'), plt.title('Matches')
plt.show()




# ## Find Homography --------------------------------------------------------------------------
# * We have matches now.
#     * Can we estimate a transformation now?
#     * No. Because of outliers (wrongly matched pairs)
# 
# There can be some possible errors while matching which may affect the result.
# To solve this problem, algorithm uses RANSAC or LEAST_MEDIAN (which can be decided by the flags).
# So good matches which provide correct estimation are called inliers and remaining are called outliers.
# 
# 
# * Finds a perspective transformation between two planes:
#     * `retval, mask = cv.findHomography(srcPoints, dstPoints[, method[, ransacReprojThreshold[, mask[, maxIters[, confidence]]]]])`
# 
# We visualize how the images can be transformed using estimated homography.
# 
# * Applies a perspective transformation to an image:
#     * `dst = cv.warpPerspective(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]])`


MIN_MATCH_COUNT = 10
if len(good_matches) > MIN_MATCH_COUNT:
    # find homography (method=cv2.RANSAC, ransacReprojThreshold=5.0)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, method=cv.RANSAC, ransacReprojThreshold=5.0)
    matchesMask = mask.ravel().tolist()

    # perspective transform
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)

    # draw box
    img2_box = cv.polylines(img2.copy(), [np.int32(dst)], True, 255, 3, cv.LINE_AA)
else:
    print("Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT))
    matchesMask = None

# draw matches
draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)

img3 = cv.drawMatches(img1, kp1, img2_box, kp2, good_matches, None, **draw_params)



plt.imshow(img3, 'gray'), plt.title('Matches in mask')
plt.show()





# ### Image Warping --------------------------------------------------------------------------
# warp images using homography matrix M


# warp img1 to img2 space using M
img1_warped = cv.warpPerspective(img1, M, dsize=(img2.shape[1],img2.shape[0])) # (dsize=(img2.shape[1],img2.shape[0]))
# warp img2 to img1 space using inverse of M
M_inv = np.linalg.inv(M)
img2_warped = cv.warpPerspective(img2, M_inv, dsize=(img1.shape[1],img1.shape[0])) # (dsize=(img1.shape[1],img1.shape[0]))

plt.subplot(2,2,1), plt.imshow(img1, 'gray'), plt.title('img1')
plt.subplot(2,2,2), plt.imshow(img2_warped, 'gray'), plt.title('img2 Warped using inv(M)')
plt.subplot(2,2,3), plt.imshow(img2, 'gray'), plt.title('img2')
plt.subplot(2,2,4), plt.imshow(img1_warped, 'gray'), plt.title('img1 Warped using M')
plt.show()




















# ====== Image Stiching ==========================================================================
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

import argparse
import sys

plt.rcParams['figure.figsize'] = [16, 9]


# ## Image Stitching
# * Feature detection
# * Feature matching & homography estimation using RANSAC
# * align two images using estimated homography
# * Stitching
# 
# How can we remove this seam?

# read images
img_names = ['boat1.jpg','boat2.jpg']
img1 = cv.imread(img_names[0])
img2 = cv.imread(img_names[1])

plt.subplot(1,2,1), plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB)), plt.title('img1')
plt.subplot(1,2,2), plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB)), plt.title('img2')
plt.show()


# Initiate SIFT detector  ---------------------------------------------------------------
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT (mask=None)
kp1, des1 = sift.detectAndCompute(img1, mask=None)
kp2, des2 = sift.detectAndCompute(img2, mask=None)




# ### Feature matching (좀더 빠르게 매칭하게 해주기 위함)  ---------------------------------------------------------------
# For FLANN based matcher, we need to pass two dictionaries which specifies the algorithm to be used, its related parameters etc.
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5) # First one is IndexParams. For various algorithms, the information to be passed is explained in FLANN docs.
search_params = dict(checks=50) # Second dictionary is the SearchParams. It specifies the number of times the trees in the index should be recursively traversed. Higher values gives better precision, but also takes more time.

# Create FLANN based matcher
matcher = cv.FlannBasedMatcher(index_params, search_params)

# compute k best matches (k=2)
matches = matcher.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
# matches.distance - Distance between descriptors. The lower, the better it is.
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance: # Lowe's ratio test.
        good.append(m)

print(len(good))




# * Draw matched points:
# draw matches
draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

img_matches = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

plt.imshow(cv.cvtColor(img_matches,cv.COLOR_BGR2RGB)), plt.title('Matches')
plt.show()




# ### Find Homography ---------------------------------------------------------------
MIN_MATCH_COUNT = 10        # 매칭이 적으면 M matrix를 찾는데 어려움이 있기 때문에 최소 count를 지정
if len(good) > MIN_MATCH_COUNT:

    # find homography (method=cv2.RANSAC, ransacReprojThreshold=5.0)
    # TODO
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    # perspective transform
    h, w, c = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst1 = cv.perspectiveTransform(pts, M)

    h, w, c = img2.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst2 = cv.perspectiveTransform(pts, np.linalg.inv(M))

    img2_poly = cv.polylines(img2.copy(), [np.int32(dst1)], True, (255,0,0), 3, cv.LINE_AA)
    img1_poly = cv.polylines(img1.copy(), [np.int32(dst2)], True, (0,0,255), 3, cv.LINE_AA)
else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

# draw matches
draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)

img3 = cv.drawMatches(img1_poly, kp1, img2_poly, kp2, good, None, **draw_params)

plt.imshow(cv.cvtColor(img3,cv.COLOR_BGR2RGB)), plt.title('Matches in mask')
plt.show()







# ### Image stitching -------------------------------------------------------------

# define output image space
out_width = img2.shape[1]*2
out_height = img2.shape[0]*2
I = np.eye(3)
A = np.eye(3)
A[0:2,2] = [out_width//4, out_height//4] # width, height

# define mask
mask1 = np.ones_like(img1)
mask2 = np.ones_like(img2)

# get inv(M)
M_inv = np.linalg.inv(M)







# warp img2 into output space with img1 camera pose (use M=A@M_inv, dsize=(out_width, out_height))
# A( M(x) ) → A @ M
img1_out = cv.warpPerspective(img1, A @ M, dsize=(out_width, out_height))
mask1_out = cv.warpPerspective(mask1, A @ M, dsize=(out_width, out_height))

# stitch images
img2_out = cv.warpPerspective(img2, A, dsize=(out_width, out_height))
mask2_out = None

# alpha blending
img_stitch = img1_out
img_stitch = img1_out * mask1_out + (1 - mask1_out) * img2_out

# display
img_stitch = np.clip(img_stitch,0,255).astype(np.uint8)
plt.imshow(cv.cvtColor(img_stitch, cv.COLOR_BGR2RGB))
plt.show()






# warp img1 into output space with img2 camera pose(use M=M_inv,dsize=(img2.shape[1]*2, img2.shape[0]))
img1_out = cv.warpPerspective(img1, A, dsize=(out_width, out_height))
mask1_out = None

# stitch images
img2_out = cv.warpPerspective(img2, A @ M_inv, dsize=(out_width, out_height))
mask2_out = cv.warpPerspective(mask2, A @ M_inv, dsize=(out_width, out_height))

# alpha blending
img_stitch = img2_out
img_stitch = img2_out * mask2_out + (1 - mask2_out) * img1_out


# display
img_stitch = np.clip(img_stitch,0,255).astype(np.uint8)
plt.imshow(cv.cvtColor(img_stitch, cv.COLOR_BGR2RGB))
plt.show()










# ## High-level stitching API ----------------------------------------------------
# `cv::Stitcher::create` can create stitcher in one of the predefined configurations (argument mode).
# See `cv::Stitcher::Mode` for details.
# These configurations will setup multiple stitcher properties to operate in one of predefined scenarios.
# After you create stitcher in one of predefined configurations you can adjust stitching by setting any of the stitcher properties.


# read input images
img_names = ['boat1.jpg','boat2.jpg']

imgs = []
for img_name in img_names:
    img = cv.imread(cv.samples.findFile(img_name))
    if img is None:
        print("can't read image " + img_name)
        sys.exit(-1)
    imgs.append(img)


# create stitcher and stitch given images
stitcher = cv.Stitcher.create(mode = cv.Stitcher_PANORAMA)
status, pano = stitcher.stitch(imgs)


if status != cv.Stitcher_OK:
    print("Can't stitch images, error code = %d" % status)
    sys.exit(-1)
print('Done')


plt.imshow(cv.cvtColor(pano,cv.COLOR_BGR2RGB))
plt.show()







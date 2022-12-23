import os
import time
import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv

path = r'D:\Python\★★Python_POSTECH_AI\Dataset'
origin_path = os.getcwd()
os.chdir(path)


# 좋은 Edge란?
# 1) 낮은 Error율 (Low Error Rate)
# 2) 정확한 Edge위치 (Good Localization)
# 3) 응답 최소화 (Minimal Response)



img_gray = cv.imread('AprilShoe-768x832.jpg', cv.IMREAD_GRAYSCALE)


# Edge Detector 종류 ------------------------------------------------
img_canny = cv.Canny(img_gray, 50, 150)
img_sobel = cv.Sobel(img_gray, cv.CV_8U, 1, 0, 3)
img_laplacian = cv.Laplacian(img_gray, cv.CV_8U, ksize=3)

cv.imshow("image", img_gray)
cv.imshow("image canny", img_canny)
cv.imshow("image sobel", img_sobel)
cv.imshow("image Laplacian", img_laplacian)

cv.waitKey(0)
cv.destroyAllWindows()






# Canny_Edge_Detector ------------------------------------------------
# cv.Canny(image, threshold1, threshold2, edges=None, apertureSize=None, L2gradient=None)

#   첫번째 argument) image는 입력 이미지입니다.
#   두번째, 세번째 argument) threshold1, threshold2는 최소 스레숄드와 최대 스레숄드입니다. 
#   네번째 argument) edges에 Canny 결과를 저장할 변수를 적어줍니다.  파이썬에선 Canny 함수 리턴으로 받을 수 있기 때문에 필요없는 항목입니다. 
#   다섯번째 argument) apertureSize는 이미지 그레디언트를 구할때 사용하는 소벨 커널 크기입니다. 디폴트는 3입니다. 
#   여섯번째 argument) L2gradient가 True이면 그레디언트 크기를 계산할 때 sqrt{(dI/dx)^2 + (dI/dy)^2}를 사용합니다.
#       False라면 근사값인 |dI/dx|+|dI/dy|를 사용합니다.  디폴트값은 False입니다.

# img_gray = cv.imread('AprilShoe-768x832.jpg', cv.IMREAD_GRAYSCALE)
cv.imshow("image", img_gray)

img_canny = cv.Canny(img_gray, 50, 150)
# canny = cv.Canny(src, 100, 255)
# cv.Canny(원본 이미지, 임계값1, 임계값2, 커널 크기, L2그라디언트)를 이용하여 가장자리 검출을 적용합니다.
#   임계값1은 임계값1 이하에 포함된 가장자리는 가장자리에서 제외합니다.
#   임계값2는 임계값2 이상에 포함된 가장자리는 가장자리로 간주합니다.
#   커널 크기는 Sobel 마스크의 Aperture Size를 의미합니다. 포함하지 않을 경우, 자동으로 할당됩니다.
#   L2그라디언트는 L2방식의 사용 유/무를 설정합니다. 사용하지 않을 경우, 자동적으로 L1그라디언트 방식을 사용합니다.

cv.imshow("Canny Edge", img_canny)

cv.waitKey(0)
cv.destroyAllWindows()


# (Canny_Edge_Detection_Algorithm) ------------------------------------------------------------------------
# ① Remove Noise → ② Edge Gradient 크기와 방향 계산 
# → ③ Non-Maximum Supression → ④ Hysteresis Thresholding

# ① Remove Noise: Gaussian Blur 사용 (이미지상 노이즈 제거, 상세한 부분 단순화, 이미지가 흐릿해짐)

# ② Edge Gradient 크기와 방향 계산 
#     수평방향 1차미분 필터: 수평선 검출
#     Gx = np.array([[-1, 0, +1],
#                     [-2, 0, +2],
#                     [-1, 0, +1]])

#     수직방향 1차미분 필터: 수직선 검출
#     Gy = np.array([[-1, -2, -1],
#                     [ 0,  0,  0],
#                     [+1, +2, +1]])

#     Edge_Gradient(G) = √(Gx² + Gy²)
#     Angle(θ) = tan^(-1)(Gx/Gy)
#         → Edge 방향을 0º, 45º, 90º, 135º로 근사화


# ③ Non-Maximum Supression: 이미지 전체를 스캔하면서 Edge가 될 수 없는 선들을 제거
#     높은 픽셀값영역과 낮은 픽셀값 영역사이에서 Edge가 검출


# ④ Hysteresis Thresholding: 두개의 Threshold 값을 사용하여 이전단계에서 얻어진 Edge중에 진짜 Edge로 보이는 것만 남기고 모두 제거
#    High Threshold 보다 큰 경우: Edge
#    Low ~ High Threshold 사이인 경우: 주변에 Edge로 판정난 이웃이 있을경우에만 Edge
#    Low Threshold 보다작은경우: Edge가 아님
#   * 보통 High Threshold : Low Threshold = 2:1 ~ 3:1 수준을 추천



# Canny_Edge_Detector Using TrackBar ------------------------------------------------
def nothing():
    pass

cv.namedWindow("Canny Edge")
cv.createTrackbar('low threshold', 'Canny Edge', 0, 1000, nothing)
cv.createTrackbar('high threshold', 'Canny Edge', 0, 1000, nothing)

cv.setTrackbarPos('low threshold', 'Canny Edge', 50)
cv.setTrackbarPos('high threshold', 'Canny Edge', 150)

cv.imshow("Original", img_gray)

while True:

    low = cv.getTrackbarPos('low threshold', 'Canny Edge')
    high = cv.getTrackbarPos('high threshold', 'Canny Edge')

    img_canny = cv.Canny(img_gray, low, high)
    cv.imshow("Canny Edge", img_canny)

    if cv.waitKey(1)&0xFF == 27:
        break


cv.destroyAllWindows()






# Sobel Edge Detector -------------------------------------------------------------------------------------
# sobel = cv.Sobel(gray, cv.CV_8U, 1, 0, 3)
# cv.Sobel(그레이스케일 이미지, 정밀도, x방향 미분, y방향 미분, 커널, 배율, 델타, 픽셀 외삽법)를 이용하여 가장자리 검출을 적용합니다.
#     정밀도는 결과 이미지의 이미지 정밀도를 의미합니다. 정밀도에 따라 결과물이 달라질 수 있습니다.
#     x 방향 미분은 이미지에서 x 방향으로 미분할 값을 설정합니다.
#     y 방향 미분은 이미지에서 y 방향으로 미분할 값을 설정합니다.
#     커널은 소벨 커널의 크기를 설정합니다. 1, 3, 5, 7의 값을 사용합니다.
#     배율은 계산된 미분 값에 대한 배율값입니다.
#     델타는 계산전 미분 값에 대한 추가값입니다.
#     픽셀 외삽법은 이미지를 가장자리 처리할 경우, 영역 밖의 픽셀은 추정해서 값을 할당해야합니다.
#     이미지 밖의 픽셀을 외삽하는데 사용되는 테두리 모드입니다. 외삽 방식을 설정합니다.
# Tip : x방향 미분 값과 y방향의 미분 값의 합이 1 이상이여야 하며 각각의 값은 0보다 커야합니다.



# Laplacian Edge Detector  -------------------------------------------------------------------------------------
# laplacian = cv.Laplacian(gray, cv.CV_8U, ksize=3)
# cv.Laplacian(그레이스케일 이미지, 정밀도, 커널, 배율, 델타, 픽셀 외삽법)를 이용하여 가장자리 검출을 적용합니다.
#     정밀도는 결과 이미지의 이미지 정밀도를 의미합니다. 정밀도에 따라 결과물이 달라질 수 있습니다.
#     커널은 2차 미분 필터의 크기를 설정합니다. 1, 3, 5, 7의 값을 사용합니다.
#     배율은 계산된 미분 값에 대한 배율값입니다.
#     델타는 계산전 미분 값에 대한 추가값입니다.
#     픽셀 외삽법은 이미지를 가장자리 처리할 경우, 영역 밖의 픽셀은 추정해서 값을 할당해야합니다.
#     이미지 밖의 픽셀을 외삽하는데 사용되는 테두리 모드입니다. 외삽 방식을 설정합니다.
# Tip : 커널의 값이 1일 경우, 3x3 Aperture Size를 사용합니다. (중심값 = -4)




# [ 픽셀 외삽법 종류 ]
# 속성	의미
# cv.BORDER_CONSTANT	iiiiii | abcdefgh | iiiiiii
# cv.BORDER_REPLICATE	aaaaaa | abcdefgh | hhhhhhh
# cv.BORDER_REFLECT	fedcba | abcdefgh | hgfedcb
# cv.BORDER_WRAP	cdefgh | abcdefgh | abcdefg
# cv.BORDER_REFLECT_101	gfedcb | abcdefgh | gfedcba
# cv.BORDER_REFLECT101	gfedcb | abcdefgh | gfedcba
# cv.BORDER_DEFAULT	gfedcb | abcdefgh | gfedcba
# cv.BORDER_TRANSPARENT	uvwxyz | abcdefgh | ijklmno
# cv.BORDER_ISOLATED	관심 영역 (ROI) 밖은 고려하지 않음


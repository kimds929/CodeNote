import os
import time
import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv

path = r'D:\Python\★★Python_POSTECH_AI\Dataset/'
origin_path = os.getcwd()
os.chdir(path)


# https://076923.github.io/posts/Python-opencv-5/
# !wget https://fox28spokane.com/wp-content/uploads/2020/04/AprilShoe-768x832.jpg


image = cv.imread('AprilShoe-768x832.jpg')
# video = cv.VideoCapture('camera_video.avi')



# 대칭 ------------------------------------------------------------------

img_dst_ud = cv.flip(image, 0)        # 상하대칭
img_dst_lr = cv.flip(image, 1)        # 좌우대칭

cv.imshow('image_original', image)
cv.imshow('image_dst_ud', img_dst_ud)
cv.imshow('image_dst_lr', img_dst_lr)
cv.waitKey(0)
cv.destroyAllWindows()



# 회전 ------------------------------------------------------------------

height, width, channel = image.shape
matrix = cv.getRotationMatrix2D((width/2, height/2), 90, 1)

# matrix에 회전 배열을 생성하여 저장합니다.
# cv.getRotationMatrix2D((중심점 X좌표, 중심점 Y좌표), 각도, 스케일)을 설정합니다.
#   (중심점)은 Tuple형태로 사용하며 회전할 기준점을 설정합니다.
#   (각도)는 회전할 각도를 설정합니다.
#   (스케일)은 이미지의 확대 비율을 설정합니다.

# *matrix를 numpy형식으로 선언하여 warpAffine을 적용하여 변환할 수 있습니다.


img_rotate = cv.warpAffine(image, matrix, (width, height))


cv.imshow('image_original', image)
cv.imshow('image_rotate', img_rotate)
cv.waitKey(0)
cv.destroyAllWindows()





# 확대 / 축소 ------------------------------------------------------------------

height, width, channel = image.shape
image_expand = cv.pyrUp(image, dstsize=(width*2, height*2), borderType=cv.BORDER_DEFAULT);
image_shrink = cv.pyrDown(image, dstsize=(int(width/2), int(height/2)), borderType=cv.BORDER_DEFAULT);


# 너비와 높이를 이용하여 dstsize (결과 이미지 크기)을 설정합니다.
# cv.pyrUp(원본 이미지)로 이미지를 2배로 확대할 수 있습니다.
# cv.pyrUp(원본 이미지, 결과 이미지 크기, 픽셀 외삽법)을 의미합니다.
# 결과 이미지 크기는 pyrUp()함수일 경우, 이미지 크기의 2배로 사용합니다.
# 픽셀 외삽법은 이미지를 확대 또는 축소할 경우, 영역 밖의 픽셀은 추정해서 값을 할당해야합니다.
# 이미지 밖의 픽셀을 외삽하는데 사용되는 테두리 모드입니다. 외삽 방식을 설정합니다.

cv.imshow('image_original', image)
cv.imshow('image_expand', image_expand)
cv.imshow('image_shrink', image_shrink)
cv.waitKey(0)
cv.destroyAllWindows()




# pyrUp()과 pyrDown() 함수에서 결과 이미지 크기와 픽셀 외삽법은 기본값으로 설정된 인수를 할당해야하므로 생략하여 사용합니다.
# 피라미드 함수에서 픽셀 외삽법은 cv.BORDER_DEFAULT만 사용할 수 있습니다.
# 이미지를 1/8배, 1/4배 ,4배, 8배 등의 배율을 사용해야하는 경우, 반복문을 이용하여 적용할 수 있습니다.






# 크기조절 ------------------------------------------------------------------

img_resize_01 = cv.resize(image, dsize=(640, 480), interpolation=cv.INTER_AREA)

# cv.resize(원본 이미지, 결과 이미지 크기, 보간법)로 이미지의 크기를 조절할 수 있습니다.
# 결과 이미지 크기는 Tuple형을 사용하며, (너비, 높이)를 의미합니다. 설정된 이미지 크기로 변경합니다.
# 보간법은 이미지의 크기를 변경하는 경우, 변형된 이미지의 픽셀은 추정해서 값을 할당해야합니다.
# 보간법을 이용하여 픽셀들의 값을 할당합니다.

img_resize_02 = cv.resize(image, dsize=(0, 0), fx=0.3, fy=0.7, interpolation=cv.INTER_AREA)

# cv.resize(원본 이미지, dsize=(0, 0), 가로비, 세로비, 보간법)로 이미지의 크기를 조절할 수 있습니다.
# 결과 이미지 크기가 (0, 0)으로 크기를 설정하지 않은 경우, fx와 fy를 이용하여 이미지의 비율을 조절할 수 있습니다.
# fx가 0.3인 경우, 원본 이미지 너비의 0.3배로 변경됩니다.
# fy가 0.7인 경우, 원본 이미지 높이의 0.7배로 변경됩니다.
# Tip : 결과 이미지 크기와 가로비, 세로비가 모두 설정된 경우, 결과 이미지 크기의 값으로 이미지의 크기가 조절됩니다.


cv.imshow('image_original', image)
cv.imshow('image_resize_01', img_resize_01)
cv.imshow('image_resize_02', img_resize_02)
cv.waitKey(0)
cv.destroyAllWindows()


# interpolation(보간법) 속성
# cv.INTER_NEAREST	이웃 보간법
# cv.INTER_LINEAR	쌍 선형 보간법
# cv.INTER_LINEAR_EXACT	비트 쌍 선형 보간법
# cv.INTER_CUBIC	바이큐빅 보간법
# cv.INTER_AREA	영역 보간법
# cv.INTER_LANCZOS4	Lanczos 보간법

# 기본적으로 쌍 선형 보간법이 가장 많이 사용됩니다.
# 이미지를 확대하는 경우, 바이큐빅 보간법이나 쌍 선형 보간법을 가장 많이 사용합니다.
# 이미지를 축소하는 경우, 영역 보간법을 가장 많이 사용합니다.
# 영역 보간법에서 이미지를 확대하는 경우, 이웃 보간법과 비슷한 결과를 반환합니다.




# 이미지자르기 ------------------------------------------------------------------

image_copy = image.copy()
image_copy

image_cut = image[100:600, 200:700]
image_copy[0:500, 0:500] = image_cut

# image_cut를 생성하여 src[높이(행), 너비(열)]에서 잘라낼 영역을 설정합니다. List형식과 동일합니다.
# 이후, dst[높이(행), 너비(열)] = image_cut를 이용하여 dst 이미지에 해당 영역을 붙여넣을 수 있습니다.

cv.imshow('image_original', image)
cv.imshow('image_cut', image_cut)
cv.imshow('image_cut_copy', image_copy)
cv.waitKey(0)
cv.destroyAllWindows()




# Gray_Scale ------------------------------------------------------------------

image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# cv.cvtcolor(원본 이미지, 색상 변환 코드)를 이용하여 이미지의 색상 공간을 변경할 수 있습니다.
# 색상 변환 코드는 원본 이미지 색상 공간2결과 이미지 색상 공간을 의미합니다.
# 원본 이미지 색상 공간은 원본 이미지와 일치해야합니다.
# Tip : BGR은 RGB 색상 채널을 의미합니다. (Byte 역순)

cv.imshow('image gray', image_gray)
cv.waitKey(0)
cv.destroyAllWindows()

# CV_8U 이미지 값 범위 : 0 ~ 255
# CV_16U 이미지의 값 범위 : 0 ~ 65535
# CV_32F 이미지의 값 범위 : 0 ~ 1


# [ 색상 공간 코드 ]
# 속성	의미	(비고)
# BGR	Blue, Green, Red 채널	(-)
# BGRA	Blue, Green, Red, Alpha 채널	(-)
# RGB	Red, Green, Blue 채널	(-)
# RGBA	Red, Green, Blue, Alpha 채널	(-)
# GRAY	단일 채널	(그레이스케일)
# BGR565	Blue, Green, Red 채널	(16 비트 이미지)
# XYZ	X, Y, Z 채널	(CIE 1931 색 공간)
# YCrCb	Y, Cr, Cb 채널	(YCC (크로마))
# HSV	Hue, Saturation, Value 채널	(색상, 채도, 명도)
# Lab	L, a, b 채널	(반사율, 색도1, 색도2)
# Luv	L, u, v 채널	(CIE Luv)
# HLS	Hue, Lightness, Saturation 채널	(색상, 밝기, 채도)
# YUV	Y, U, V 채널	(밝기, 색상1, 색상2)
# BG, GB, RG	디모자이킹	(단일 색상 공간으로 변경)
# _EA	디모자이킹	(가장자리 인식)
# _VNG	디모자이킹	(그라데이션 사용)





# 역상 : 영상이나 이미지를 반전 된 색상으로 변환하기 위해서 사용 ------------------------------------------------------------------

image_bitwise = cv.bitwise_not(image)

# cv.bitwise_not(원본 이미지)를 이용하여 이미지의 색상을 반전할 수 있습니다.
# 비트 연산을 이용하여 색상을 반전시킵니다.
# Tip : not 연산 이외에도 and, or, xor 연산이 존재합니다.

cv.imshow("image", image)
cv.imshow("image bitwise", image_bitwise)
cv.waitKey(0)
cv.destroyAllWindows()





# 이진화 : 영상이나 이미지를 어느 지점을 기준으로 흑색 또는 흰색의 색상으로 변환하기 위해서 사용 ------------------------------------------------------------------

image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
ret_binary, image_binary = cv.threshold(src=image_gray, thresh=122, maxval=255, type=cv.THRESH_BINARY)

# 이진화를 적용하기 위해서 그레이스케일로 변환합니다.
# ret_binary, image_binary를 이용하여 이진화 결과를 저장합니다. ret_binary에는 임계값이 저장됩니다.

# cv.threshold(그레스케일 이미지, 임계값, 최댓값, 임계값 종류)를 이용하여 이진화 이미지로 변경합니다.
# 임계값은 이미지의 흑백을 나눌 기준값을 의미합니다. 100으로 설정할 경우, 100보다 이하면 0으로, 100보다 이상이면 최댓값으로 변경합니다.
# 임계값 종류를 이용하여 이진화할 방법 설정합니다.

cv.imshow("image", image)
cv.imshow("image binary", image_binary)
cv.waitKey(0)
cv.destroyAllWindows()



# [ 임계값 종류 ]
# 속성	의미
# cv.THRESH_BINARY	임계값 이상 = 최댓값, 임계값 이하 = 0
# cv.THRESH_BINARY_INV	임계값 이상 = 0, 임계값 이하 = 최댓값
# cv.THRESH_TRUNC	임계값 이상 = 임계값, 임계값 이하 = 원본값
# cv.THRESH_TOZERO	임계값 이상 = 원본값, 임계값 이하 = 0
# cv.THRESH_TOZERO_INV	임계값 이상 = 0, 임계값 이하 = 원본값
# cv.THRESH_MASK	흑색 이미지로 변경
# cv.THRESH_OTSU	Otsu 알고리즘 사용
# cv.THRESH_TRIANGLE	Triangle 알고리즘 사용





# 흐림 효과(Blur) ------------------------------------------------------------------

image_blur = cv.blur(image, (9, 9), anchor=(-1, -1), borderType=cv.BORDER_DEFAULT)

# cv.blur(원본 이미지, (커널 x크기, 커널 y크기), 앵커 포인트, 픽셀 외삽법)를 이용하여 흐림 효과를 적용합니다.
# 커널 크기는 이미지에 흐림 효과를 적용할 크기를 설정합니다. 크기가 클수록 더 많이 흐려집니다.
# 앵커 포인트는 커널에서의 중심점을 의미합니다. (-1, -1)로 사용할 경우, 자동적으로 커널의 중심점으로 할당합니다.
# 픽셀 외삽법은 이미지를 흐림 효과 처리할 경우, 영역 밖의 픽셀은 추정해서 값을 할당해야합니다.
# 이미지 밖의 픽셀을 외삽하는데 사용되는 테두리 모드입니다. 외삽 방식을 설정합니다.

cv.imshow("image", image)
cv.imshow("image blur", image_blur)
cv.waitKey(0)
cv.destroyAllWindows()


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

























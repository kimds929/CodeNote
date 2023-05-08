import os
import time
import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv

path = r'D:\Python\★★Python_POSTECH_AI\Dataset'
origin_path = os.getcwd()
os.chdir(path)

# Contour: Image 특징추출 Detecting ------------------------------------------------------------------------
img_color = cv.imread('figure.png')
# img_color = cv.imread('multi_fruits.jpg')
# img_color = cv.resize(img_color, dsize=None, fx=0.3, fy=0.3)
img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
ret, img_binary = cv.threshold(img_gray, 108, 255, 0)

# plt.imshow(img_binary, 'gray')

# contour --------------------------------------------------------------------------------------------------------
# contours, hierarchy = cv.findContours(image, mode, method, contours=None, hierarchy=None, offset=None)
#   contours: object outline list
#   image: binary image
#   hierarchy: detected countour information save with formed hierarchy
#   mode: contour retriver mode (검출된 edge 정보를 계층 또는 list로 저장)
#   method: contour approximation method (contour를 구성하는 point 검출방법을 구성)
#   offset: contour coordinate(좌표) offset

contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# len(contours)         # 추출된 contour의 갯수
# contours
# hierarchy

cv.imshow('img color', img_color)
cv.waitKey(0)
cv.destroyAllWindows()




# contour 그리기: drawContours: contour를 이미지 위에 그리기(이미지 위에 외곽선 그리기) ----------------------------------
# image = cv.drawContours(image, contours, contouridx, color, 
#                 thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None)
#   image: color_image
#   contours: image에 그릴 contour가 저장된 list혹은 vector
#   contouridx: image에 그릴 특정 contour의 index
#   color: contour에 그릴 contour의 색상
#   thickness: contour를 그릴 선의 두께
contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

def nothing(x):
    pass
cv.namedWindow('img_color')
cv.createTrackbar('contour idx', 'img_color', 0, len(contours)-1, nothing)
cv.setTrackbarPos('contour idx', 'img_color', 0)

while True:
    img_feature = img_color.copy()
    contour_idx = cv.getTrackbarPos('contour idx', 'img_color')
    cv.drawContours(img_feature, contours, contour_idx, (0, 255,0), 3)        # index -1인 contour

    cv.imshow('img_color', img_feature)
    # ESC 키누르면 종료
    if cv.waitKey(1) & 0xFF == 27:
        break
cv.destroyAllWindows()





# contour method --------------------------------------------------------------------------------------------------
# simple 꼭지점만 저장
contours_simple, hierarchy_simple = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

# none 모든 좌표 저장
contours_none, hierarchy_none = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

img_contour_simple = img_color.copy()
img_contour_none = img_color.copy()

for cnt in contours_simple:
    for p in cnt:
        cv.circle(img=img_contour_simple, center=(p[0][0], p[0][1]), radius=10, color=(255,0,0), thickness=-1)

for cnt in contours_none:
    for p in cnt:
        cv.circle(img=img_contour_none, center=(p[0][0], p[0][1]), radius=10, color=(255,0,0), thickness=-1)


cv.imshow('img contour simple', img_contour_simple)
cv.imshow('img contour none', img_contour_none)
cv.waitKey(0)
cv.destroyAllWindows()




# contour mode --------------------------------------------------------------------------------------------------
# cv.RETR_TREE
contours_simple, hierarchy_TREE = cv.findContours(img_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
hierarchy_TREE      # contour 계층구조
# [Next, Previous, First_Child, Parent]
# array([[[-1, -1,  1, -1],                 ← 0
#         [ 2, -1, -1,  0],                 ← 1
#         [ 4,  1,  3,  0],                 ← 2
#         [-1, -1, -1,  2],                 ← 3
#         [ 6,  2,  5,  0],                 ← 4
#         [-1, -1, -1,  4],                 ← 5
#         [ 8,  4,  7,  0],                 ← 6
#         [-1, -1, -1,  6],                 ← 7
#         [-1,  6, -1,  0]]], dtype=int32)  ← 8


# cv.RETR_LIST
contours_simple, hierarchy_LIST = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
hierarchy_LIST      # contour 계층구조 X
# [Next, Previous, First_Child, Parent]
# array([[[ 1, -1, -1, -1],                 ← 0
#         [ 2,  0, -1, -1],                 ← 1
#         [ 3,  1, -1, -1],                 ← 2
#         [ 4,  2, -1, -1],                 ← 3
#         [ 5,  3, -1, -1],                 ← 4
#         [ 6,  4, -1, -1],                 ← 5
#         [ 7,  5, -1, -1],                 ← 6
#         [ 8,  6, -1, -1],                 ← 7
#         [-1,  7, -1, -1]]], dtype=int32)  ← 8


# cv.RETR_EXTERNAL
contours_simple, hierarchy_EXTERNAL = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
hierarchy_EXTERNAL      # 가장 외곽의 contour만 return
# [Next, Previous, First_Child, Parent]
# array([[[-1, -1, -1, -1]]], dtype=int32)


# cv.RETR_CCOMP
contours_simple, hierarchy_CCOMP = cv.findContours(img_binary, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
hierarchy_CCOMP      # 모든 contour를 2개의 Level 계층으로 재구성
# [Next, Previous, First_Child, Parent]
# array([[[ 1, -1, -1, -1],                 ← 0
#         [ 2,  0, -1, -1],                 ← 1
#         [ 3,  1, -1, -1],                 ← 2
#         [-1,  2,  4, -1],                 ← 3
#         [ 5, -1, -1,  3],                 ← 4
#         [ 6,  4, -1,  3],                 ← 5
#         [ 7,  5, -1,  3],                 ← 6
#         [ 8,  6, -1,  3],                 ← 7
#         [-1,  7, -1,  3]]], dtype=int32)  ← 8



# contour 특징 사용하기 --------------------------------------------------------------------------------------------------
contours_simple, hierarchy_CCOMP = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
img_contour_feature = img_color.copy()


for cnt in contours:
    area = cv.contourArea(cnt)     # 영역크기

    epsilon = cv.arcLength(cnt, True) 
    appox = cv.approxPolyDP(cnt, epsilon, True)    # 근사화 (곡선 → 직선)

    M = cv.moments(cnt)           # Moment
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    # cv.circle(img_contour_feature, (cx, cy), 10, (0,0,255), -1)    # 무게중심

    x, y, w, h = cv.boundingRect(cnt)      # 경계사각형 (object를 둘러싼 최소 사각형)
    # cv.rectangle(img_contour_feature, (x,y), (x+w, y+h), (0, 255, 0), 3)

    rect = cv.minAreaRect(cnt)     # 도형의 방향을 고려한 경계 사각형
    box = cv.boxPoints(rect)
    box = np.int0(box)
    # cv.drawContours(img_contour_feature, [box], 0, (0, 0, 255), 3)

    hull = cv.convexHull(cnt)  # Convex Hull (경계 블록 다각형 )
    cv.drawContours(img_contour_feature, [hull], 0, (0, 0, 255), 3)

    # print(area)
    # print(epsilon)
    # print(appox)
    # print(M)
    # print(x, y, w, h)
    # print(rect)
    # print()
cv.imshow('img contour', img_contour_feature)
cv.waitKey(0)
cv.destroyAllWindows()







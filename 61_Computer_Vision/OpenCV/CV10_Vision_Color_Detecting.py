import os
import time
import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv


path = r'D:\Python\★★Python_POSTECH_AI\Dataset'
origin_path = os.getcwd()
os.chdir(path)

# Binary ---------------------------------------------------------------------
# Image Load -------------------------
img_color = cv.imread('redball_on_table.jpg')

cv.imshow('red ball', img_color)
cv.waitKey(0)
cv.destroyAllWindows()

img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)

cv.imshow('red ball gray', img_gray)
cv.waitKey(0)
cv.destroyAllWindows()


# Binary -------------------------
# retval, dst = cv.threshold(src, thresh, maxval, type, dst=None)
ret, img_binary = cv.threshold(img_gray, 100, 255, cv.THRESH_BINARY)
cv.imshow('red ball binary', img_binary)
cv.waitKey(0)
cv.destroyAllWindows()

# plt.imshow(img_color[:,:,1], cmap='Reds_r')
# plt.colorbar()



# Track-bar 추가하기 -------------------------
def nothing(x):
    pass

cv.namedWindow('Binary_track_bar')
# createTrackbar(trackbarName, windowName, value, count, onChange) 
cv.createTrackbar('threshold', 'Binary_track_bar', 0, 255, nothing)
cv.setTrackbarPos('threshold', 'Binary_track_bar', 108)    # Track bar 초기값 설정

while True:
    low = cv.getTrackbarPos('threshold', 'Binary_track_bar')
    ret, img_binary_track = cv.threshold(img_gray, low, 255, cv.THRESH_BINARY)

    cv.imshow('Binary_track_bar', img_binary_track)

    if cv.waitKey(1)&0xFF == 27:
        break
cv.destroyAllWindows()

# from ipywidgets import interact
# def trackbar(threshhold):
#     ret, img_binary_track = cv.threshold(img_gray, threshhold, 255, cv.THRESH_BINARY)
#     plt.imshow(img_binary_track, cmap='gray')
#     plt.show()

# interact(trackbar, threshhold=(0, 255, 1))



# Binary를 활용한 Image Filtering Object 검출 -------------------------
def nothing(x):
    pass

cv.namedWindow('Binary_track_bar')
# createTrackbar(trackbarName, windowName, value, count, onChange) 
cv.createTrackbar('threshold', 'Binary_track_bar', 0, 255, nothing)
cv.setTrackbarPos('threshold', 'Binary_track_bar', 108)    # Track bar 초기값 설정

while True:
    low = cv.getTrackbarPos('threshold', 'Binary_track_bar')
    ret, img_binary_track = cv.threshold(img_gray, low, 255, cv.THRESH_BINARY_INV)

    cv.imshow('Binary_track_bar', img_binary_track)

    img_result = cv.bitwise_and(img_color, img_color, mask=img_binary_track)
    cv.imshow('Result', img_result)

    if cv.waitKey(1)&0xFF == 27:
        break
cv.destroyAllWindows()











# HSV 색공간에서 특정색 검출하기 --------------------------------------------------------------

color = [255, 0 ,0]     # [B, G, R]
pixel = np.uint8([[color]])     # 1px image로 변환

hsv = cv.cvtColor(pixel, cv.COLOR_BGR2HSV)
hsv = hsv[0][0]

print(f'bgr: {color}')
print(f'hsv: {hsv}')


img_color_circle = cv.imread('color_circle.jpg')

cv.imshow('color circle', img_color_circle)
cv.waitKey(0)
cv.destroyAllWindows()

# 색 검출
img_circle_hsv = cv.cvtColor(img_color_circle, cv.COLOR_BGR2HSV)

lower_blue = (120-10, 30, 30)
upper_blue = (120+10, 255, 255)
img_circle_mask = cv.inRange(img_circle_hsv, lowerb=lower_blue, upperb=upper_blue)

img_circle_result = cv.bitwise_and(img_color_circle, img_color_circle, mask=img_circle_mask)
cv.imshow('img_circle color', img_color_circle)
cv.imshow('img_circle mask', img_circle_mask)
cv.imshow('img_circle result', img_circle_result)
cv.waitKey(0)
cv.destroyAllWindows()
















# Web Cam에서 특정 색공간 검출하기 ------------------------------------------------------------
hsv = 0
lower_blue1 = 0
upper_blue1 = 0
lower_blue2 = 0
upper_blue2 = 0
lower_blue3 = 0
upper_blue3 = 0

def mouse_callback(event, x, y, flags, param):
    global hsv, lower_blue1, upper_blue1, lower_blue2, upper_blue2, lower_blue3, upper_blue3, threshold

    # 마우스 왼쪽 버튼 누를시 위치에 있는 픽셀값을 읽어와서 HSV로 변환합니다.
    if event == cv.EVENT_LBUTTONDOWN:
        print(img_color[y, x])
        color = img_color[y, x]

        one_pixel = np.uint8([[color]])
        hsv = cv.cvtColor(one_pixel, cv.COLOR_BGR2HSV)
        hsv = hsv[0][0]

        threshold = cv.getTrackbarPos('threshold', 'img_result')
        # HSV 색공간에서 마우스 클릭으로 얻은 픽셀값과 유사한 필셀값의 범위를 정합니다.
        if hsv[0] < 10:
            print("case1")
            lower_blue1 = np.array([hsv[0]-10+180, threshold, threshold])
            upper_blue1 = np.array([180, 255, 255])
            lower_blue2 = np.array([0, threshold, threshold])
            upper_blue2 = np.array([hsv[0], 255, 255])
            lower_blue3 = np.array([hsv[0], threshold, threshold])
            upper_blue3 = np.array([hsv[0]+10, 255, 255])
            #     print(i-10+180, 180, 0, i)
            #     print(i, i+10)

        elif hsv[0] > 170:
            print("case2")
            lower_blue1 = np.array([hsv[0], threshold, threshold])
            upper_blue1 = np.array([180, 255, 255])
            lower_blue2 = np.array([0, threshold, threshold])
            upper_blue2 = np.array([hsv[0]+10-180, 255, 255])
            lower_blue3 = np.array([hsv[0]-10, threshold, threshold])
            upper_blue3 = np.array([hsv[0], 255, 255])
            #     print(i, 180, 0, i+10-180)
            #     print(i-10, i)
        else:
            print("case3")
            lower_blue1 = np.array([hsv[0], threshold, threshold])
            upper_blue1 = np.array([hsv[0]+10, 255, 255])
            lower_blue2 = np.array([hsv[0]-10, threshold, threshold])
            upper_blue2 = np.array([hsv[0], 255, 255])
            lower_blue3 = np.array([hsv[0]-10, threshold, threshold])
            upper_blue3 = np.array([hsv[0], 255, 255])
            #     print(i, i+10)
            #     print(i-10, i)

        print(hsv[0])
        print("@1", lower_blue1, "~", upper_blue1)
        print("@2", lower_blue2, "~", upper_blue2)
        print("@3", lower_blue3, "~", upper_blue3)

def nothing(x):
    pass

cv.namedWindow('img_color')
cv.setMouseCallback('img_color', mouse_callback)

cv.namedWindow('img_result')
cv.createTrackbar('threshold', 'img_result', 0, 255, nothing)
cv.setTrackbarPos('threshold', 'img_result', 30)

cap = cv.VideoCapture(0)

while(True):
    # img_color = cv.imread('2.jpg')
    ret, img_color = cap.read()
    height, width = img_color.shape[:2]
    img_color = cv.resize(img_color, (width, height), interpolation=cv.INTER_AREA)

    # 원본 영상을 HSV 영상으로 변환합니다.
    img_hsv = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)

    # 범위 값으로 HSV 이미지에서 마스크를 생성합니다.
    img_mask1 = cv.inRange(img_hsv, lower_blue1, upper_blue1)
    img_mask2 = cv.inRange(img_hsv, lower_blue2, upper_blue2)
    img_mask3 = cv.inRange(img_hsv, lower_blue3, upper_blue3)
    img_mask = img_mask1 | img_mask2 | img_mask3

    # Noise 제거
    kernel = np.ones((11, 11), np.uint8)
    img_mask = cv.morphologyEx(img_mask, cv.MORPH_OPEN, kernel)        # opening
    img_mask = cv.morphologyEx(img_mask, cv.MORPH_CLOSE, kernel)        # clossing

    # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 영상 부분을 획득합니다.
    img_result = cv.bitwise_and(img_color, img_color, mask=img_mask)

    # 물체 위치 추적 ---------------------------------------------------------------------------------
    numOfLabels, img_label, stats, centroids = cv.connectedComponentsWithStats(img_mask)
    
    for idx, centroids in enumerate(centroids):
        if stats[idx][0] == 0 and stats[idx][1] == 0:
            continue

        if np.any(np.isnan(centroids)):
            continue

        x, y, width, height, area = stats[idx]
        centerX, centerY = int(centroids[0]), int(centroids[1])

        if area > 100:
            cv.circle(img=img_color, center=(centerX, centerY), radius=10, color=(0,0,255), thickness=10)
            cv.rectangle(img=img_color, pt1=(x,y), pt2=(x+width, y+height), color=(0,0,255))
    # ------------------------------------------------------------------------------------------------

    cv.imshow('img_color', img_color)
    cv.imshow('img_mask', img_mask)
    cv.imshow('img_result', img_result)

    # ESC 키누르면 종료
    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()
# dst = cv.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C, dst=None)
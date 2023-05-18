# (Python) Computer Vision 01 (230509)

# (pip library 설치)
# python get-pip.py     # pip라는 package 받기

# (pip upgrade 설치)
# pip install --upgrade pip
# pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade pip
# 
# ※ 다른방법
#   C:\Users\poscoedu\AppData\Local\Programs\Python\Python39\Lib\site-packages\pip\_vendor\requests
#   sessions.py 내용 중에 'self.verify = True'==> False 로 바꾼다


# (환경변수설정)
# 탐색 >> 고급 시스템 설정 보기 >> 환경변수 >> 사용자변수(or 시스템변수) >> 추가
#    Path : C:\Users\poscoedu\AppData\Local\Programs\Python\Python39\Scripts\ 

# (pip library 확인)
# pip list

# (pip install)
# (matplotlib) pip install matplotlib --trusted-host pypi.org --trusted-host files.pythonhosted.org 
# (matplotlib) pip install opencv-python --trusted-host pypi.org --trusted-host files.pythonhosted.org 
# (matplotlib) pip install opencv-contrib-python --trusted-host pypi.org --trusted-host files.pythonhosted.org 

# (GPU 확인)
# nvidia-smi

# (cuda 11.6 - torch)
# pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 --extra-index-url http://download.pytorch.org/whl/cu102 --trusted-host pypi.org --trusted-host download.pytorch.org 

# (Python Folder path)
# C:\Users\poscoedu\AppData\Local\Programs\Python\Python39

# (Jupyter 실행)
# jupyter notebook

# (Git : CV실습 Git)
# https://github.com/hrdkdh

# (Git: ADP Study Code)
# https://github.com/hrdkdh/adp-study

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from PIL import ImageGrab

import sys
from IPython.display import Image

path = 'd:\\python\\openCV-master\\00. 강사용\\M1. 컴퓨터비전 개요'
os.chdir(path)

np.set_printoptions(suppress=True)

# import torch
# import tensorflow as tf

print(np.__version__)
print(cv2.__version__)

# img_pil = ImageGrab.grabclipboard()
# img_pil

## 【 M1 : Introduction of Computer Vision 】 ########################################################
# (이미지 불러오기) -----------------------------------------------
# folder_path = 'd:\\python\\openCV-master\\00. 강사용'
# os.chdir(folder_path)


cv2.imread("/image/black_dot.png", cv2.IMREAD_GRAYSCALE)
src = cv2.imread("../images/black_dot.png", cv2.IMREAD_GRAYSCALE)
print(type(src))
print(src)
src.shape

# np.fromfile(f"{folder_path}/image/black_dot.png", dtype='uint8')
Image("../images/black_dot.png")

# numpy train_test split ------------------------------------------
a = np.arange(10)
train_idx = np.random.choice(a, 8, replace=False)
test_idx = np.setdiff1d(range(len(a)), train_idx)      # 나머지 꺼내기
# ------------------------------------------------------------------


img = cv2.imread("../images/lenna.bmp")
img_gray = cv2.imread("../images/lenna.bmp", cv2.IMREAD_GRAYSCALE)
# plt.imshow(img[:,:,::-1])     # BGR → RGB
# b, g, r = cv2.split(img)
# plt.imshow(cv2.merge([r,g,b]))


if img is None:
    print('image load fail.')
    sys.exit()

img2 = ~img     # 색상 반전

# 이미지 show
# cv2.nameedWindow()        # 창열기
cv2.imshow('img0', img_gray)
cv2.imshow('img1', img)
cv2.imshow('img2', img2)
cv2.waitKey()               # key 입력 기다리기
# cv2.destroyWindow('img0')     # 특정창만 닫기

while True:
    # if cv2.waitKey() == ord('q'):
    if cv2.waitKey() == 13:
        break
cv2.destroyAllWindows()     # 모든 창 닫기



# (색깔 이미지 생성) ----------------------------------------------------------------
img010 = np.zeros((240, 320), dtype=np.uint8)*255
img011 = np.ones((240, 320), dtype=np.uint8)*255
img012 = np.full((240,320,3), fill_value=(0,255,255), dtype=np.uint8)  # fill value: b g r

# cv2.imwrite('black_img.jpg', img010)     # 이미지 쓰기

# plt.imshow(img011, 'gray')
cv2.imshow('img_black',img010)
cv2.imshow('img_white',img011)
cv2.imshow('img_color',img012)
cv2.waitKey()
cv2.destroyAllWindows()


# (색깔 이미지 생성) ----------------------------------------------------------------
img_020 = cv2.imread('../images/cat.jpg')

img_021 = img_020           # 주소를 공유 (그림수정시 같이 수정됨)
img_022 = img_020.copy()    # 주소를 공유하지 않음

img_021.fill(255)           # 흰색으로 채우기
img_023 = img_022[200:300, 200:300]


# cv2.imshow('original', img_020)
# cv2.imshow('white', img_021)
cv2.imshow('copy', img_022)
cv2.imshow('slicing', img_023)
cv2.waitKey()
cv2.destroyAllWindows()


# (도형 그리기) ----------------------------------------------------------------
img025 = np.full((400,800,3), 255, np.uint8)

# (line)
# ?cv2.line : line(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) 
#   img : 그림그릴 대상
#   pt1 : 시작점
#   pt2 : 끝점
#   color : 색깔
#   thickness : 두께

cv2.line(img025, (50, 50), (200,50), (0,0,255), 5)

# cv2.rectangle : rectangle(img, rec, color[, thickness[, lineType[, shift]]]) 
#   img : 그림그릴 대상
#   pt1 : 시작점(좌상단)
#   pt2 : 끝점(우하단)
#   color : 색깔
#   thickness : 두께
cv2.rectangle(img025, (50, 200,150,100), (0, 255, 0), -1)
# thickness : -1 (내부 채우기)


# (circle)
# cv2.circle? : circle(img, center, radius, color[, thickness[, lineType[, shift]]])
#   img : 그림그릴 대상
#   center : 중심위치
#   radius : 원의반지름
#   color : 색깔
cv2.circle(img025, (300,100), 30, (255, 0, 0), 3, cv2.LINE_AA)
# cv2.LINE_AA : AntiAlias - 곡선을 부드럽게 보정해주는 기술(default) 
# cv2.LINE_4 : 4점 이웃연결
# cv2.LINE_8 : 8점 이웃연결


# (text)
# ?cv2.putText : putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
text = 'this is computer vision'
cv2.putText(img025, text, (50, 350), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

cv2.imshow('canvas', img025)
cv2.waitKey()
cv2.destroyAllWindows()




# (카메라 이용하기) ----------------------------------------------------------------
# (camera capture)
cap = cv2.VideoCapture(0)       # 0 : 1 번 카메라
# cap2 = cv.VideoCapture(1)  # 2대의 camera 필요시


ret, img_cap = cap.read()

cv2.imshow('color', img_cap)
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()





# (video)
cap = cv2.VideoCapture(0)       # 카메라에서 캡쳐
# cap = cv2.VideoCapture('../images/Butterfly.mp4')       # 동영상 재생

if cap.isOpened() is False:
    print('Camera open failed!')
    sys.exit()

# camera property
v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))       # 가로
v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))      # 세로
v_fps = int(cap.get(cv2.CAP_PROP_FPS))               # 초당 프레임수



# ?cv2.VideoWriter: VideoWriter(file명, codec, frame수, (해상도W, 해상도H))
fourcc = cv2.VideoWriter_fourcc(*'DIVX')    # 'D', 'I', 'V', 'X'
# fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
out = cv2.VideoWriter('out.avi', fourcc, v_fps, (v_width, v_height))

video = []  # video

text = 'this is computer vision'


n = 0
while True:
    # print(n, end=' ')
    ret, frame = cap.read()     # ret : 읽다가 문제가 생기면 False return

    frame_save = frame[:,::-1,:].copy()
    cv2.putText(frame_save, text, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame_save)
    out.write(frame_save)
    video.append(frame_save)

    if ret is False:
        break

    if cv2.waitKey(10) == 27:       # ESC key
        break
    if n > 100:
        break
    n +=1

cv2.waitKey(0)
cv2.destroyAllWindows()
out.release()
cap.release()   # 카메라 자원 해제




video = np.stack(video)     # video
video.shape
plt.imshow(video[0,:,:,::-1])       # 0번 idx frame 보기
plt.imshow(video[50,:,:,::-1])      # 50번 idx frame 보기
plt.imshow(video[100,:,:,::-1])     # 100번 idx frame 보기







#
img_030 = cv2.imread('../images/dog.jpg')
cv2.imshow('img', img_030)

while True:
    keycode = cv2.waitKey()
    if keycode == ord('i') or keycode == ord('I'):
        img_030 = ~img_030
        cv2.imshow('img', img_030)

    elif keycode == 27:
        break

cv2.destroyAllWindows()




### 마우스 이벤트
drag = False

def onMouse(event, x, y, flags, param):
    global drag
    if  event == cv2.EVENT_LBUTTONDOWN:
        drag = True
        cv2.circle(img_035, (x,y), 10, (100,100,0), -1)
        cv2.imshow('mouse_event', img_035)
    if event == cv2.EVENT_MOUSEMOVE:
        if drag == True:
            cv2.circle(img_035, (x,y), 10, (100,100,0), -1)
            cv2.imshow('mouse_event', img_035)
    if event == cv2.EVENT_LBUTTONUP:
        if drag == True:
            drag = False
    
    if event == cv2.EVENT_RBUTTONDOWN:
        print('RB')
        img_035[:] = 255
        cv2.imshow('mouse_event', img_035)

img_035 = np.ones((480,640, 3), dtype=np.uint8)*255

cv2.imshow('mouse_event', img_035)
cv2.setMouseCallback('mouse_event', onMouse)
cv2.waitKey()
cv2.destroyAllWindows()


# track-bar
def onChange(pos):
    value = pos * 16
    if value >= 255:
        value = 255
    img_037[:] = value
    cv2.imshow('win', img_037)

img_037 = np.zeros((480,640), dtype=np.uint8)

cv2.imshow('win', img_037)
cv2.createTrackbar('mytb', 'win', 0, 16, onChange)
cv2.waitKey()
cv2.destroyAllWindows()


























## 【 M2: Image Processing 】 #################################################################
img_040 = cv2.imread("../images/candy.jpg")
# plt.imshow(img_040[:,:,::-1])
# plt.imshow(img_040[:,:,[2,1,0]])

# split channel
b, g, r = cv2.split(img_040)

cv2.imshow('img', img_040)
cv2.imshow('img_b', b)
cv2.imshow('img_g', g)
cv2.imshow('img_r', r)
cv2.waitKey()
cv2.destroyAllWindows()





# convert to hsv
img_040_hsv = cv2.cvtColor(img_040, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(img_040_hsv)

cv2.imshow('img', img_040)
cv2.imshow('img_hsv', img_040_hsv)
cv2.imshow('img_h', h)
cv2.imshow('img_s', s)
cv2.imshow('img_v', v)
cv2.waitKey()
cv2.destroyAllWindows()


# 특정 색 추출
img_042 = cv2.imread('../images/candy2.png')
# plt.imshow(img_042[:,:,::-1])

img_042_hsv = cv2.cvtColor(img_042, cv2.COLOR_BGR2HSV)
img_042_mask = cv2.inRange(img_042_hsv, (20,150,0), (40,255,255))
img_042_out = cv2.copyTo(img_042, img_042_mask)

# def on_trackbar(pos):   # 하나의 창에 여러 트랙바를 사용할 경우 callback함수는 사용하지 않음
#     pass

cv2.imshow('original', img_042)
cv2.imshow('mask', img_042_mask)
cv2.imshow('out', img_042_out)
cv2.waitKey()
cv2.destroyAllWindows()



# saturation

img_043 = img_042 + 200
img_044 = cv2.add(img_042, 200)

cv2.imshow('original', img_042)
cv2.imshow('np+200', img_043)
cv2.imshow('cv+200', img_044)
cv2.waitKey()
cv2.destroyAllWindows()


# 합성
img_045 = cv2.imread('../images/plane.jpg')
img_046 = cv2.imread('../images/cloud.jpg')
img_045_046 = cv2.addWeighted(img_045, 0.5, img_046, 0.5, 0)

img_045_mask = cv2.inRange(cv2.cvtColor(img_045,cv2.COLOR_BGR2HSV), (0, 0, 0), (255, 30,255) )
img_045_mask < 10
# cv2.cvtColor(img_045_046, cv2.COLOR_HSV2BGR)

cv2.imshow('img_plane', img_045)
cv2.imshow('img_plane_mask', img_045_mask)
cv2.imshow('img_cloud', img_046)
cv2.imshow('concat', img_045_046)
cv2.waitKey()
cv2.destroyAllWindows()



# (화질개선)
# 명암비 조정
img_048 = cv2.imread('../images/hawkes.jpg', cv2.IMREAD_GRAYSCALE)

img_048_norm = cv2.normalize(img_048, None, 0, 255, cv2.NORM_MINMAX)    # normalization
img_048_equal = cv2.equalizeHist(img_048)   # 평활화 (equalizer)

cv2.imshow('hawkes', img_048)
cv2.imshow('norm', img_048_norm)
cv2.imshow('equal', img_048_equal)
cv2.waitKey()
cv2.destroyAllWindows()

# 컬러영상
img_050 = cv2.imread('../images/field.bmp')
img_050_ycrbc = cv2.cvtColor(img_050, cv2.COLOR_BGR2YCrCb)
y, cr, cb = cv2.split(img_050_ycrbc)

y_dst = cv2.equalizeHist(y)
dst_ycrcb = cv2.merge([y_dst, cr, cb])
img_050_color = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2BGR)


cv2.imshow('field',img_050)
cv2.imshow('YcrCb',img_050_color)
cv2.waitKey()
cv2.destroyAllWindows()




# 역투영 예제
# 입력 영상에서 ROI를 지정하고, 히스토그램 계산

img_055 = cv2.imread('../images/field.jpg')

if img_055 is None:
    print('Image load failed!')
    sys.exit()

x, y, w, h = cv2.selectROI(img_055)

img_055_ycrcb = cv2.cvtColor(img_055, cv2.COLOR_BGR2YCrCb)
crop = img_055_ycrcb[y:y+h, x:x+w]

channels = [1, 2]
cr_bins = 128
cb_bins = 128
histSize = [cr_bins, cb_bins]
cr_range = [0, 256]
cb_range = [0, 256]
ranges = cr_range + cb_range

hist = cv2.calcHist([crop], channels, None, histSize, ranges)
hist_norm = cv2.normalize(cv2.log(hist+1), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# 입력 영상 전체에 대해 히스토그램 역투영

backproj = cv2.calcBackProject([img_055_ycrcb], channels, hist, ranges, 1)
dst = cv2.copyTo(img_055, backproj)

cv2.imshow('backproj', backproj)
cv2.imshow('hist_norm', hist_norm)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()





# (필터링 및 기하학적 변환) ----------------------------------------------------------

# 필터
# img_060 = cv2.imread('../images/rose.jpg', cv2.IMREAD_GRAYSCALE)
img_060 = cv2.imread('../images/noise.bmp', cv2.IMREAD_GRAYSCALE)

# Average filter
img_060_a3 = cv2.blur(img_060, (3,3))
img_060_a5 = cv2.blur(img_060, (5,5))
# img_060_f7 = cv2.blur(img_060, (7,7))

# Gaussian filter
img_060_g3 = cv2.GaussianBlur(img_060, (0,0), 3)
img_060_g5 = cv2.GaussianBlur(img_060, (0,0), 5)

# Median filter
img_060_m3 = cv2.medianBlur(img_060, 3)
img_060_m5 = cv2.medianBlur(img_060, 5)

# Soble(Differential) filter
img_060_sr3 = cv2.Sobel(img_060, -1,1,0, 3, delta=128)  # 가로 방향 미분, delta=128(색상반전)
img_060_sr5 = cv2.Sobel(img_060, -1,1,0, 5, delta=128)  # 가로 방향 미분, delta=128(색상반전)
img_060_sc3 = cv2.Sobel(img_060, -1,0,1, 3, delta=128)  # 세로 방향 미분, delta=128(색상반전)
img_060_sc5 = cv2.Sobel(img_060, -1,0,1, 5, delta=128)  # 세로 방향 미분, delta=128(색상반전)


cv2.imshow('original', img_060)
cv2.imshow('filter_a3', img_060_a3)
# cv2.imshow('filter_a5', img_060_a5)
# cv2.imshow('filter_f7', img_060_f7)
cv2.imshow('filter_g3', img_060_g3)
# cv2.imshow('filter_g5', img_060_g5)
cv2.imshow('filter_m3', img_060_m3)
# cv2.imshow('filter_m5', img_060_m5)
cv2.imshow('filter_sr3', img_060_sr3)
# cv2.imshow('filter_sr5', img_060_sr5)
cv2.imshow('filter_sc3', img_060_sc3)
# cv2.imshow('filter_sc5', img_060_sc5)
cv2.waitKey()
cv2.destroyAllWindows()





# 영상이동, 회전
img_065 = cv2.imread('../images/dog.jpg')

# M = [[ cosθ, sinθ, x_s],
#      [-sinθ, cosθ, y_s]]

M065_1 = np.array([[1,0, 50], [0,1, 50]], dtype=np.float32)
img_065_s1 = cv2.warpAffine(img_065, M065_1, (0,0))

M065_2 = np.array([[1,1, 50], [0,1, 50]], dtype=np.float32)
img_065_s2 = cv2.warpAffine(img_065, M065_2, (0,0))


def M_mat(x, y, rad):
    return np.array([[np.cos(rad), np.sin(rad), x], [-np.sin(rad), np.cos(rad), y]])

theta = 45
rad = theta * np.pi / 180
M065_3 = M_mat(100, 50, theta)
cp_065 = (img_065.shape[1]/2, img_065.shape[0]/2)

rot_065 = cv2.getRotationMatrix2D(cp_065, theta, 1)
img_065_s3 = cv2.warpAffine(img_065, rot_065, (0,0))

cv2.imshow('original', img_065)
cv2.imshow('shift_1', img_065_s1)
cv2.imshow('shift_2', img_065_s2)
cv2.imshow('shift_3', img_065_s3)
cv2.waitKey()
cv2.destroyAllWindows()




# 크기 확대, 축소
img_067 = cv2.imread('../images/cat.jpg')

img_067_r1 = cv2.resize(img_067, (0,0), fx=2, fy=2)    # 2배로 확대
img_067_r2 = cv2.resize(img_067, (800,600))    # (800, 600) 으로 확대
img_067_r3 = cv2.resize(img_067, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)    # (800, 600) 으로 확대

cv2.imshow('original', img_067)
cv2.imshow('resize_1', img_067_r1)
cv2.imshow('resize_2', img_067_r2)
cv2.imshow('resize_3', img_067_r3)
cv2.waitKey()
cv2.destroyAllWindows()




# 투시변환
img_068 = cv2.imread('../images/perspective_transform_sample.jpg')

dot_list = [(612,214),(865,227),(866,634),(618,658)]
srcQuard = np.array(dot_list, np.float32)

w_068 = dot_list[1][0] - dot_list[0][0]
h_068 = dot_list[2][1] - dot_list[1][1]

dstQuard = np.array([[0,0], [w_068,0], [w_068,h_068], [0,h_068]], np.float32)

M068 = cv2.getPerspectiveTransform(srcQuard, dstQuard)
img_068_p1 = cv2.warpPerspective(img_068, M068, (w_068, h_068))



cv2.imshow('original', img_068)
cv2.imshow('projection', img_068_p1)
cv2.waitKey()
cv2.destroyAllWindows()







# (특징 추출) ----------------------------------------------------------
# ★★ Canny Edge 
img_070 = cv2.imread('../images/house.jpg')
img_070_c1 = cv2.Canny(img_070, 50, 150)
img_070_c2 = cv2.Canny(img_070, 10, 200)


cv2.imshow('original', img_070)
cv2.imshow('canny_01', img_070_c1)
cv2.imshow('canny_02', img_070_c2)
cv2.waitKey()
cv2.destroyAllWindows()



# Hough transform (허프변환)
# Line
img_072 = cv2.imread('../images/building.jpg', cv2.IMREAD_GRAYSCALE)
theta = 1

edges_072 = cv2.Canny(img_072, 50, 150)
lines_072 = cv2.HoughLinesP(edges_072, 1, theta*np.pi/180, 200, minLineLength=30, maxLineGap=50)
img_072_color = cv2.cvtColor(edges_072, cv2.COLOR_GRAY2BGR)

if lines_072 is not None:
    for line in lines_072:
        xs, ys, xe, ye = line.reshape(-1)
        cv2.line(img_072_color, (xs, ys), (xe, ye), (0,0,255), 2, cv2.LINE_AA)

cv2.imshow('original', img_072)
cv2.imshow('hough', img_072_color)
cv2.waitKey()
cv2.destroyAllWindows()


# Circle
img_073 = cv2.imread('../images/pipe.jpg', cv2.IMREAD_GRAYSCALE)
img_073_g = cv2.GaussianBlur(img_073, (0,0), 1)
img_073_color = cv2.cvtColor(img_073, cv2.COLOR_GRAY2BGR)


img_073_circle = cv2.HoughCircles(img_073_g, cv2.HOUGH_GRADIENT, 1, 50
                                  ,minRadius=5 ,maxRadius=150
                                ,param1=50, param2=100)

if img_073_circle is not None:
    for circle in img_073_circle[0]:
        xc, yc, r = circle.astype(int)
        cv2.circle(img_073_color, (xc, yc), r, (0,0,255), 2, cv2.LINE_AA)


cv2.imshow('original', img_073)
cv2.imshow('gaussian_filter', img_073_g)
cv2.imshow('circle', img_073_color)
# cv2.imshow('hough', img_072_color)
cv2.waitKey()
cv2.destroyAllWindows()








# (이진 영상처리) ----------------------------------------------------------
# Manual Binary
img_080 = cv2.imread('../images/cells.jpg', cv2.IMREAD_GRAYSCALE)

thresh_080, img_080_b = cv2.threshold(img_080, 160, 255, cv2.THRESH_BINARY)


cv2.imshow('original', img_080)
cv2.imshow('binary', img_080_b)
cv2.waitKey()
cv2.destroyAllWindows()


# Auto Binary
img_081 = cv2.imread('../images/leaf.jpg', cv2.IMREAD_GRAYSCALE)
thresh_081, img_081_b = cv2.threshold(img_081, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow('original', img_081)
cv2.imshow('binary', img_081_b)
cv2.waitKey()
cv2.destroyAllWindows()


# Adaptive Binary
img_082 = cv2.imread('../images/sudoku.jpg', cv2.IMREAD_GRAYSCALE)
img_082_b = cv2.adaptiveThreshold(img_082, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)

cv2.imshow('original', img_082)
cv2.imshow('binary', img_082_b)
cv2.waitKey()
cv2.destroyAllWindows()




# 침식과 팽창 ------------------------------------------------------
img_083 = cv2.imread('../images/circuit.jpg', cv2.IMREAD_GRAYSCALE)
# img_083 = cv2.imread('../images/noise.bmp', cv2.IMREAD_GRAYSCALE)

img_083_se = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
img_083_ero = cv2.erode(img_083, img_083_se)
img_083_dil = cv2.dilate(img_083, None)
img_083_ero_dil = cv2.dilate(img_083_ero, None)


cv2.imshow('original', img_083)
cv2.imshow('erode', img_083_ero)
cv2.imshow('dilate', img_083_dil)
cv2.imshow('erode_dilate', img_083_ero_dil)
cv2.waitKey()
cv2.destroyAllWindows()



# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# Labeling ★★ ------------------------------------------------------
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

img_085 = cv2.imread('../images/number.jpg', cv2.IMREAD_GRAYSCALE)

_, img_085_b = cv2.threshold(img_085, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(img_085_b)

img_085_color = cv2.cvtColor(img_085, cv2.COLOR_GRAY2BGR)

if len(stats) > 0:
    for i in range(1, len(stats)):  # 0: background
        x, y, w, h, area = stats[i]

        if area < 20:   # noise filtering
            continue
        cv2.rectangle(img_085_color, (x, y, w, h), (0, 0, 255))

        plt.figure(figsize=(0.5,1.5))
        plt.imshow(img_085[y:y+h,x:x+w],'gray')
        plt.show()


cv2.imshow('original', img_085)
cv2.imshow('bounding_box', img_085_color)
cv2.waitKey()
cv2.destroyAllWindows()




# Outline ★★ ------------------------------------------------------
# img_088 = cv2.imread('../images/blocks.png', cv2.IMREAD_GRAYSCALE)
img_088 = cv2.imread('../images_contour/shape.png', cv2.IMREAD_GRAYSCALE)

ret, img_088_b = cv2.threshold(img_088, 0,255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
contours, hier = cv2.findContours(img_088_b, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)     # 외곽선 검출

img_088_color = cv2.cvtColor(img_088, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img_088_color, contours, -1, (0,0,255), 2, cv2.LINE_AA, hier)

cv2.imshow('original', img_088)
cv2.imshow('outline', img_088_color)
cv2.waitKey()
cv2.destroyAllWindows()





# Contour 
fig1 = img_088_color.copy()
fig2 = img_088_color.copy()


for i in contours:
    for j in i:
        cv2.circle(fig1, tuple(j[0]), 1, (255,0,0), -1)

for contour in contours:
    # bounding box
    x,y,w,h = cv2.boundingRect(contour)
    cv2.rectangle(fig2,(x,y), (x+w, y+h), (255,255,0), 1)

    # 도형 종류
    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    for i in approx:
        cv2.circle(fig2, tuple(i[0]), 4, (255, 0, 0), -1)

    vtc = len(approx)

    area = cv2.contourArea(contour)
    _, radius = cv2.minEnclosingCircle(contour)
    ratio = area / (radius*radius*np.pi)

    if ratio > 0.9:
    # if np.allclose(ratio, 1, rtol=1e-1, atol=1e-1): 
        print('circle')
    elif vtc == 3:
        print('triangle')
    elif vtc == 4:
        print('rectangle')
    elif vtc == 5:
        print('pentagon')
    else:
        print('else')

    print(cv2.isContourConvex(contour))

cv2.imshow('fig1', fig1)
cv2.imshow('fig2', fig2)
cv2.waitKey()
cv2.destroyAllWindows()



    














## 【 M3: Obejct Detection 】 #################################################################



# Template Matching --------------------------------------------------------
# template matching
img_090 = cv2.imread('../images/circuit_board_resized.jpg')
img_090_gray = cv2.cvtColor(img_090, cv2.COLOR_BGR2GRAY)
img_090_temp = cv2.imread('../images/circuit_board_template.jpg')
img_090_temp_gray = cv2.cvtColor(img_090_temp, cv2.COLOR_BGR2GRAY)


res_090 = cv2.matchTemplate(img_090_gray, img_090_temp_gray, cv2.TM_CCOEFF_NORMED)

img_090_gray.shape, img_090_temp_gray.shape, res_090.shape
# 648+153-1, 1081+158-1


# correlation max position
_, maxv, _, maxloc = cv2.minMaxLoc(res_090)

# res_090.max()
# maxloc = np.argmax(res_090.max(0)), np.argmax(res_090.max(1))
# res_090[maxloc[::-1]]

cv2.rectangle(img_090
            ,maxloc
            ,np.array(maxloc) +np.array(img_090_temp_gray.shape[::-1])
            ,(255,255,0), 3)

cv2.imshow('original', img_090)
cv2.imshow('original_gray', img_090_gray)
cv2.imshow('template', img_090_temp)
cv2.imshow('template_gray', img_090_temp_gray)
cv2.waitKey()
cv2.destroyAllWindows()



# corner detecting
img_092 = cv2.imread('../images/building.jpg')
img_092_gray = cv2.cvtColor(img_092, cv2.COLOR_BGR2GRAY)

img_092_fast = img_092.copy()
fast_092 = cv2.FastFeatureDetector_create(30)
keypoints_092 = fast_092.detect(img_092_gray)

len(keypoints_092)
for keypoint in keypoints_092:
    point = tuple(map(int, keypoint.pt))
    cv2.circle(img_092_fast, point, 3, (0,0,255), 1)

# img_092_good = img_092.copy()
# corner_092 = cv2.goodFeaturesToTrack(img_092_gray, 400, 0.01, 10)


cv2.imshow('original', img_092)
cv2.waitKey()
cv2.destroyAllWindows()






# Object Detection ------------------------------------------------------------------

cap_100 = cv2.VideoCapture('../images/PETS2000.avi')

_, back_100 = cap_100.read()
back_100_gray = cv2.cvtColor(back_100, cv2.COLOR_BGR2GRAY)
back_100_blur = cv2.GaussianBlur(back_100_gray, None, 1)

# cv2.imshow('background', back_100_blur)
# cv2.waitKey()
# cv2.destroyAllWindows()

frame_back = back_100_gray.copy()
while True:
# for _ in range(1):
    ret, frame = cap_100.read()
    if ret is False:
        break
    

    # 첫배경과 비교
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray, None, 1)
    frame_diff = cv2.absdiff(back_100_blur, frame_blur)

    # bounding box
    _, diff_threshold = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(diff_threshold)
    
    if len(stats) > 0:
        for i in range(1, len(stats)):  # 0: background
            x, y, w, h, area = stats[i]

            if area < 100:   # noise filtering
                continue
            cv2.rectangle(frame, (x, y, w, h), (0, 0, 255))
            cv2.rectangle(frame_diff, (x, y, w, h), (255, 255, 255))

    cv2.imshow('video', frame)
    cv2.imshow('video_back', frame_diff)

    # back_ground갱신
    frame_diff_before = cv2.absdiff(frame_back, frame_blur)

    cv2.imshow('video_before', frame_diff_before)
    # frame_back = frame_blur.copy()
     

    if cv2.waitKey(10)==27:
        break

cv2.waitKey()
cv2.destroyAllWindows()
cap_100.release()




# mean shift
cap = cv2.VideoCapture("../images/woman_who_is_running.avi")

ret, frame = cap.read()
cv2.imshow("selectROI", frame) #ROI 선택을 위해 영상 첫번째 프레임을 불러와 출력
x, y, w, h = cv2.selectROI("selectROI", frame)  # 사용자가 선택한 정보를 받아줌
# ROI선택함수 실행, ROI 선택 인터페이스 출력

rc = (x, y, w, h) #선택한 ROI영역의 좌표정보를 튜플로 rc 변수에 저장
roi = frame[y:y+h, x:x+w] #선택한 ROI영역을 frame에서 크롭(crop)한 후 roi변수에 저장
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) #색상 히스토그램 생성을 위해 크롭한 영역을 HSV색공간으로 변환
channels = [0, 1] #히스토그램 생성시 사용할 색정보 채널을 H, S로 선택
ranges = [0, 180, 0, 256] #H의 범위는 0~179, S의 범위는 0~255로 설정

hist = cv2.calcHist([roi_hsv], channels, None, [200, 200], ranges) #색상 히스토그램 생성하여 hist 변수에 저장

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1) #mean shift 알고리즘 종료기준 설정
while True: #동영상 실시간 처리
    ret, frame = cap.read()
    if ret is False:
        break
    #불러온 frame을 hist를 이용, 역투영하기 위해 HSV색공간으로 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #frame에 hist를 역투영하여 부합하는 색정보만 남기고 제거, backproj 변수에 저장
    backproj = cv2.calcBackProject([hsv], channels, hist, ranges, 1)
    _, rc = cv2.meanShift(backproj, rc, term_crit) #backproj에 mean shift 실행

    cv2.rectangle(frame, rc, (0, 0, 255), 2) #mean shift된 window 좌표에 사각형 생성
    cv2.imshow("frame", frame)
    if cv2.waitKey(25) == 27:
        break
        
cap.release()
cv2.destroyAllWindows()




# cam shift
cap = cv2.VideoCapture("../images/woman_who_is_running.avi")

ret, frame = cap.read()
cv2.imshow("selectROI", frame) #ROI 선택을 위해 영상 첫번째 프레임을 불러와 출력
x, y, w, h = cv2.selectROI("selectROI", frame) #ROI선택함수 실행, ROI 선택 인터페이스 출력

rc = (x, y, w, h) #선택한 ROI영역의 좌표정보를 튜플로 rc 변수에 저장
roi = frame[y:y+h, x:x+w] #선택한 ROI영역을 frame에서 크롭(crop)한 후 roi변수에 저장
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) #색상 히스토그램 생성을 위해 크롭한 영역을 HSV색공간으로 변환
channels = [0, 1] #히스토그램 생성시 사용할 색정보 채널을 H, S로 선택
ranges = [0, 180, 0, 256] #H의 범위는 0~179, S의 범위는 0~255로 설정

hist = cv2.calcHist([roi_hsv], channels, None, [200, 200], ranges) #색상 히스토그램 생성하여 hist 변수에 저장

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1) #mean shift 알고리즘 종료기준 설정
while True: #동영상 실시간 처리
    ret, frame = cap.read()
    if ret is False:
        break
    #불러온 frame을 hist를 이용, 역투영하기 위해 HSV색공간으로 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #frame에 hist를 역투영하여 부합하는 색정보만 남기고 제거, backproj 변수에 저장
    backproj = cv2.calcBackProject([hsv], channels, hist, ranges, 1)
    ret_cam, rc = cv2.CamShift(backproj, rc, term_crit) #backproj에 cam shift 실행
    cv2.ellipse(frame, ret_cam, (0, 255, 0), 2) #타원형 윈도우 정보로 타원 생성
    cv2.rectangle(frame, rc, (0, 0, 255), 2) #mean shift된 window 좌표에 사각형 생성
    cv2.imshow("frame", frame)
    if cv2.waitKey(25) == 27:
        break
        
cap.release()
cv2.destroyAllWindows()




## 【 Machine Learning & CNN 】 #################################################################
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from PIL import ImageGrab

import sys
from IPython.display import Image

path = 'd:\\python\\openCV-master\\00. 강사용\\M1. 컴퓨터비전 개요'
os.chdir(path)

import torch
import tensorflow as tf

# DNN
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train.shape, y_train.shape, x_test.shape, y_test.shape

idx = 8
plt.figure()
plt.title(f"label : {y_test[idx]}")
plt.imshow(x_test[idx], 'gray')
plt.show()

x_train_exp = tf.expand_dims(tf.constant(x_train), 1)/255
x_test_exp = tf.expand_dims(tf.constant(x_test), 1)/255




class MnistModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.seqential = tf.keras.Sequential([
            # tf.keras.layers.Conv2D(64, 3, padding='same')
            
            tf.keras.layers.Flatten()
            ,tf.keras.layers.Dense(128, activation='sigmoid')  
            ,tf.keras.layers.Dense(10, activation='softmax')  
        ])

    def __call__(self, x, training=True):
        return self.seqential(x)

l1 = tf.keras.layers.Conv2D(64, 3, padding='same')



mdl_tf = MnistModel()
mdl_tf.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
mdl_tf.fit(x_train_exp, y_train, batch_size=64, epochs=10)

tf.argmax(mdl_tf.predict(x_test_exp[:5]), axis=1)
y_test[:5]

mdl_tf.evaluate(x_test_exp, y_test, verbose=2)



# torch model

train_set = torch.utils.data.TensorDataset(torch.Tensor(x_train/255), torch.Tensor(y_train))
test_set = torch.utils.data.TensorDataset(torch.Tensor(x_test/255), torch.Tensor(y_test))

train_loader = torch.utils.data.DataLoader(train_set, batch_size=10)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=10)

# no_cuda = False
# use_cuda = not no_cuda and torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")


class MnistTorch(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.sequential = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, 1, 1)
            ,torch.nn.MaxPool2d(2)
            ,torch.nn.Conv2d(64, 128, 3, 1, 1)
            ,torch.nn.MaxPool2d(2)
            ,torch.nn.Flatten()
            ,torch.nn.Linear(128*7*7, 128)
            ,torch.nn.Sigmoid()
            ,torch.nn.Linear(128,10)
            ,torch.nn.Softmax()
        )
        # self.sequential = torch.nn.Sequential(
        #     torch.nn.Flatten()
        #     ,torch.nn.Linear(28*28, 128)
        #     ,torch.nn.Sigmoid()
        #     ,torch.nn.Linear(128,10)
        #     ,torch.nn.Softmax()
        # )

    def forward(self, x):
        return self.sequential(x)

mdl_torch = MnistTorch()
# mdl_torch(torch.rand(3,1,28,28)).shape


optimizer = torch.optim.Adam(mdl_torch.parameters())
loss_function = torch.nn.CrossEntropyLoss()

epochs = 5
mdl_torch.train()
for i in range(epochs):
    print(i)
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        pred = mdl_torch(batch_X.unsqueeze(1))
        loss = loss_function(pred, torch.tensor(batch_y, dtype=torch.long))          
        loss.backward()
        optimizer.step()




with torch.no_grad():
    mdl_torch.eval()
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)/255
    test_pred = mdl_torch(x_test_tensor.unsqueeze(1)[:20])
    print(y_test[:20])
    print(torch.argmax(test_pred, axis=1))






# - 학습모델 활용 ------------------------------------------------------------------------------
# http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
# https://pjreddie.com/media/files/yolov3.weights
# http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel


# bvlc_googlenet.caffemodel     # GoogleNet
# pose_iter_440000.caffemodel   # SSD : Single Shot Multibox Detector
# yolov3.weights                # YOLO


import numpy as np
import tensorflow as tf

# img = cv2.imread('../images/space_shuttle.jpg')
# img = cv2.imread('../images/scooter.jpg')
# img = cv2.imread('../images/building.jpg')
img = cv2.imread('../images/dog.jpg')
# plt.pyplot(img[:,:,::-1])

# cv2.imshow('img', img)
# cv2.waitKey()
# cv2.destroyAllWindows()

model_path =  '../models/bvlc_googlenet.caffemodel'
config_path = '../models/googlenet/deploy.prototxt'

net = cv2.dnn.readNet(model_path, config_path)
if net.empty():
    print('netowrk load error')

class_names = []
with open('../models/googlenet/classification_classes_ILSVRC2012.txt', 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')
# np.array(class_names)

blob = cv2.dnn.blobFromImage(img, 1, (224,224), (104, 117, 123))
net.setInput(blob)
prob = net.forward()
# prob.shape    # (1,1000)
class_names[np.argmax(prob)]

cv2.putText(img, class_names[np.argmax(prob)], (10,30), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,255), 1)
cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()





# Face Detection ------------------------------------------------------------
# SSD : Single Shot Multibox Detector
# res10_300x300_ssd_iter_140000_fp16.caffemodel     # SSD(Single Shot Multibox Detector) Face detection

model_path =  '../models/ssd/res10_300x300_ssd_iter_140000_fp16.caffemodel'
config_path = '../models/ssd/deploy.prototxt'

net = cv2.dnn.readNet(model_path, config_path)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    h,w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1, (300,300), (104,177,123))
    net.setInput(blob)
    out = net.forward()     # (1,1,200,7) : 200개 후보군, 7은 박스정보
    detect = out[0,0,:,:]    # ?, ?, confience, x1_ratio, y1_ratio, x2_ratio, y2_ratio
    # print(detect.shape)

    for i in range(detect.shape[0]):
        confidence = detect[i,2]

        if confidence < 0.5:
            break
        
        # x1 = int(detect[i,3]*w)
        # y1 = int(detect[i,4]*h)
        # x2 = int(detect[i,5]*w)
        # y2 = int(detect[i,6]*h)
        x1, y1 = (np.array(frame.shape[:2])[::-1] * detect[i,3:5]).astype(int)
        x2, y2 = (np.array(frame.shape[:2])[::-1] * detect[i,5:7]).astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))

    cv2.imshow('frame', frame[:,::-1,:])
    if cv2.waitKey(10) == 27:
        break
# detect[0]
cv2.destroyAllWindows()
cap.release()

# plt.figure()
# plt.barh(range(200), detect[:,2])
# plt.show()







# Skeleton Detection ------------------------------------------------------------


# pose_iter_440000.caffemodel
model_path =  '../models/openPose/pose_iter_440000.caffemodel'
config_path = '../models/openPose/pose_deploy_linevec.prototxt'

# img_file = '../images/pose1.jpg'
img_file = '../images/pose2.jpg'
img = cv2.imread(img_file)
h, w = img.shape[:2]

# img.shape # (794, 540, 3)
# cv2.imshow('img', img)
# cv2.waitKey()
# cv2.destroyAllWindows()

net = cv2.dnn.readNet(model_path, config_path)
blob = cv2.dnn.blobFromImage(img, 1/255, (368,368))
net.setInput(blob)
out = net.forward()
# out.shape   # (1, 57, 46, 46)    
# 57 : 18   keypointconfidence maps (skeleton point 18개)
#                                    0: 코, 1: 목, ...
#       1   background
#      19*2 part affinity maps
pos_name = {0: 'Nose', 1:'Neck', 2:'L_Shoulder', 3:'L_Elbow', 4:'L_Ankle',
             5:'R_Shoulder', 6:'R_Elbow', 7:'R_Ankle', 8:'L_Hip', 9:'L_Knee',
             10:'L_Wrist', 11:'R_Hip', 12:'R_Knee', 13:'R_Wrist',
             14:'L_Eye', 15: 'R_Eye', 16: 'L_Ear', 17:'R_Ear', 18:'Background'}

pose_pairs = [(1,2), (2,3), (3,4),  # 왼팔
              (1,5), (5,6), (6,7),  # 오른팔
              (1,8), (8,9), (9,10),  # 왼쪽다리
              (1,11), (11,12), (12,13),  # 오른쪽다리
              (1,0), (0,14), (14,16), (0,15), (15,17)   #얼굴
              ]

points = {}
for i in range(18):
    heat_map = out[0, i, :, :]

    _, conf, _, point = cv2.minMaxLoc(heat_map)
    # point = np.argmax(heat_map.max(0)), np.argmax(heat_map.max(1))
    # conf = heat_map[point[::-1]]

    x = int(w * point[0] / out.shape[3])
    y = int(h * point[1] / out.shape[2])

    points[i] = (x,y) if conf > 0.1 else None

    # skeleton point
    # cv2.circle(img, (x, y), 3, (0,0,255), -1)
    cv2.putText(img, pos_name[i], (x,y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,255), 1)

# skeleton line
for ps, pe in pose_pairs:
    if points[pe] is not None:
        cv2.line(img, points[ps], points[pe], (0, 255, 0), 3, cv2.LINE_AA)
    cv2.circle(img, points[ps], 3, (0,0,255), -1)
    cv2.circle(img, points[pe], 3, (0,0,255), -1)

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()



plt.figure()
plt.contourf(heat_map, cmap='gray')
plt.show()








# YOLO ----------------------------------------------------------
# yolov3.weights                # YOLO

class_labels = '../models/yolo/coco.names'
model_path =  '../models/yolo/yolov3.weights'
config_path = '../models/yolo/yolov3.cfg'


img_file = '../images/dog1.jpg'
# img_file = '../images/pose2.jpg'
img = cv2.imread(img_file)
h, w = img.shape[:2]

# img.shape # (576, 768, 3)
# cv2.imshow('img', img)
# cv2.waitKey()
# cv2.destroyAllWindows()


class_names = []
with open('../models/yolo/coco.names', 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')
# np.array(class_names)
colors = np.random.uniform(0, 255, size=(len(class_names), 3)).astype(int)
# colors.shape    # (80, 3) : 80 class, rgb

layer_names = net.getLayerNames()   # 24개 layer
# len(layer_names)
output_layers = net.getUnconnectedOutLayersNames()

net = cv2.dnn.readNet(model_path, config_path)
blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), swapRB=True)
# image_size: (320, 320), (416, 416), (608, 608) 중에 선택가능

net.setInput(blob)
outs = net.forward(output_layers)
# outs[0].shape, outs[1].shape, outs[2].shape
# ((300, 85), (1200, 85), (4800, 85))       # 1st layer rst, 2nd layer rst, 3rd layer rst
# 85 : x, y, w, h, (object_score), class_score 1~80

# outs_concat = np.vstack([*outs])

conf_threshold = 0.5
nms_threshold = 0.4

np.max(outs[0][:, 5:])
np.max(outs[1][:, 5:])
np.max(outs[2][:, 5:])

boxes = []
confidences = []
class_ids = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        # print(confidence)

        if confidence > conf_threshold:  
            cx = int(detection[0]*w)
            cy = int(detection[1]*h)
            bw = int(detection[2]*w)
            bh = int(detection[3]*w)
            bx = int(cx - bw/2)
            by = int(cy - bh/2)
            
            class_ids.append(class_id)
            confidences.append(confidence)
            boxes.append((bx, by, bw, bh))
            # cv2.rectangle(img, (bx, by), (bx+bw, by+bh), (0,0,255), 1, cv2.LINE_AA)

# 중복박스 제거
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
for i in indices:
    nbx, nby, nbw, nbh = boxes[i]
    label = f'{class_names[class_ids[i]]}: {confidences[i]:.2}'
    cv2.rectangle(img, (nbx, nby, nbw, nbh), (0,0,255), 2)
    cv2.putText(img, label, (nbx, nby-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 1)

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()






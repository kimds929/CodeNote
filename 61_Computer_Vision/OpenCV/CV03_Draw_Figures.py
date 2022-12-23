import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv

path = r'D:\Python\★★Python_POSTECH_AI\Dataset'
origin_path = os.getcwd()
os.chdir(path)

# 선 (line) ----------------------------------------------------------
# cv.line(img,        선분이 그려질 이미지 
#         (x1, y1),   선분의 시작점
#         (x2, y2),   선분의 끝점
#         color,      선분의 색( B, G, R )
#         thickness,  선굵기(디폴트값 1)
#         lineType,   디폴트값 cv.LINE_8(=8-connected line)
#         shift )     디폴트값 0

# 컬러 이미지를 저장할 넘파이 배열을 생성합니다.
width = 500
height = 500
bpp = 3

img_line = np.zeros((height, width, bpp), np.uint8)

# 화면 중앙을 가로질러 선굵기 3인 대각선을 두개 그려 교차하도록합니다.  
cv.line(img_line, (width-1, 0), (0, height-1), (255, 0, 0), 3)
cv.line(img_line, (0, 0), (width-1, height-1), (0, 0, 255), 3) 

cv.imshow("result", img_line)
cv.waitKey(0)
cv.destroyAllWindows()





# 사각형 (rectangle) ----------------------------------------------------------
# cv.rectangle(img,        사각형이 그려질 이미지 
#             (x1, y1),        사각형의 시작점
#             (x2, y2),        시작점과 대각선에 있는 사각형의 끝점
#                 color,        사각형의 색 ( B, G , R )
#             thickness,        선굵기(디폴트값 1), -1 이면 사각형 내부가 채워짐
#             lineType,        디폴트값 cv.LINE_8(=8-connected line)
#                 shift )       디폴트값 0


# 컬러 이미지를 저장할 넘파이 배열을 생성합니다.
width = 500
height = 500
bpp = 3

img_rect = np.zeros((height, width, bpp), np.uint8)

# 선굵기 3인 빨간색 사각형을 그립니다.  
cv.rectangle(img_rect, (50, 50),  (450, 450), (0, 0, 255), 3)

# 내부가 파란색으로 채워진 사각형을 그립니다. 
cv.rectangle(img_rect, (150, 200), (250, 300), (255, 0, 0), -1)

cv.imshow("result", img_rect)
cv.waitKey(0)
cv.destroyAllWindows()




# 원 (circle) ----------------------------------------------------------
# cv.circle(img,        원이 그려질 이미지 
#         center,     원의 중심 좌표 ( x, y )
#         radius,     원의 반지름
#         color,      원의 색( B, G, R )
#         thickness,  선굵기(디폴트값 1)
#         lineType,   디폴트값 cv.LINE_8(=8-connected line)
#         shift )     디폴트값 0

# 컬러 이미지를 저장할 넘파이 배열을 생성합니다.
width = 500
height = 500
bpp = 3

img_circle = np.zeros((height, width, bpp), np.uint8)

# (250,250)이 중심인 반지름 10인 파란색으로 채워진 원을 그립니다.
cv.circle(img_circle, (250, 250), 10, (255, 0, 0), -1)

# (250,250)이 중심인 반지름이 100인 선굵기가 1인 빨간색 원을 그립니다.  
cv.circle(img_circle, (250, 250), 100, (0, 0, 255), 1)

cv.imshow("result", img_circle)
cv.waitKey(0)
cv.destroyAllWindows()



# 타원 (ellipse) ----------------------------------------------------------
# cv.ellipse(img,        타원이 그려질 이미지 
#         center,        중심 좌표(x, y)
#         axes,        메인 축 방향의 반지름
#         angle,        회전각
#         startAngle,        호의 시작각도
#         endAngle,        호의 끝각도
#         color,        타원의 색( B, G, R )
#         thickness,        선굵기(디폴트값 1), -1이면 내부가 채워집니다.
#         lineType,        디폴트값 cv.LINE_8(=8-connected line)
#         shift )       디폴트값 0


# 컬러 이미지를 저장할 넘파이 배열을 생성합니다.
width = 500
height = 500
bpp = 3

img_eplise = np.zeros((height, width, bpp), np.uint8)
center = (int(height/2), int(width/2))

# center를 중심으로 하는 3개의 원을 그립니다.
# x축 방향 반지름 길이 200, y축 방향 반지름 길이 10인 파란색 타원을 그립니다.   
cv.ellipse(img_eplise, center, (200, 10), 0, 0, 360, (255, 0, 0), 3 ) 

# x축 방향 반지름 길이 10, y축 방향 반지름 길이 200인 녹색 타원을 그립니다. 
cv.ellipse(img_eplise, center, (10, 200), 0, 0, 360, (0, 255, 0), 3 )  
# x축 방향 반지름 길이 200, y축 방향 반지름 길이 200인 빨간색 타원을 그립니다. 

# 반지름 200인 원이 그려집니다.  
cv.ellipse(img_eplise, center, (200, 200), 0, 0, 360, (0, 0, 255), 3 ) 

# 타원을 시계방향으로 45도 회전하여 그립니다. 
cv.ellipse(img_eplise, center, (10, 200), 45, 0, 360,  (0, 255, 255), 3 ) 

# 타원을 반시계방향으로 45도 회전하여 그립니다.  
cv.ellipse(img_eplise, center, (10, 200), -45, 0, 360,  (255, 255, 0), 3 ) 

# 타원을 시계방향으로 0도에서 90도까지만 그립니다.  
cv.ellipse(img_eplise, center, (100, 100), 0, 0, 90,  (255, 0, 255), 3 ) 


cv.imshow("result", img_eplise)
cv.waitKey(0)
cv.destroyAllWindows()







# 다각형 (Polygon) ---------------------------------------------------------------------------
# cv.polylines:  이미지(ndarray 객체)  img에 폴리곤 외곽선을 draw
# cv.polylines(img,   폴리곤이 그려질 이미지 
#             pts,   폴리곤을 구성하는 정점(veryes)이 저장되어 있는 배열     
#             isClosed,   pts의 첫번째 정덤과 마지막 정점을 연결할지 여부  
#             color,   폴리곤의 색
#             thickness,   디폴트값 1
#             lineType,   디폴트값 cv.LINE_8(=8-connected line)
#             shift )  디폴트값 0

# cv.fillPoly:  이미지(ndarray 객체)  img에 내부가 채워진 폴리곤을 draw
# cv.fillPoly(img,   폴리곤이 그려질 이미지 
#             pts,   폴리곤을 구성하는 정점(veryes)이 저장되어 있는 배열     
#             color,   폴리곤의 색
#             lineType,   디폴트값 cv.LINE_8(=8-connected line)
#             shift,   디폴트값 0
#             offset )  모든 정점에 적용되는 오프셋

width = 640
height = 640
bpp = 3

img_polygon = np.zeros((height, width, bpp), np.uint8)

red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
white = (255, 255, 255)
yellow = (0, 255, 255)
cyan = (255, 255, 0)
magenta = (255, 0, 255)


thickness = 2 

pts = np.array([[315, 50], [570, 240], [475, 550], [150, 550], [50, 240]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv.polylines(img_polygon, [pts], False, green, thickness)  

pts = np.array([[315, 160], [150, 280], [210, 480], [420, 480], [480, 280]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv.polylines(img_polygon, [pts], True, green, thickness)  

pts = np.array([[320, 245],[410,315],[380,415],[265,415], [240, 315]], np.int32)
pts = pts.reshape((-1,1,2))
cv.fillPoly(img_polygon, [pts], yellow) 

cv.imshow("result", img_polygon)
cv.waitKey(0)
cv.destroyAllWindows()






# 글자 (Letter) ---------------------------------------------------------------------------
# cv.putText(img,   문자열을 그릴 이미지
#         text,   문자열             
#         org,   문자열의 왼쪽아래 좌표
#         fontFace,   폰트 타입
#         fontScale,   폰트 기본 크기에 곱해질 폰트 스케일 팩터(Font scale factor)
#         color,   글자 색
#         thickness,   디폴트값 1
#         lineType,   디폴트값 cv.LINE_8(=8-connected line)
#         bottomLeftOrigin)   디폴트값 false

img_w = 640
img_h = 480
bpp = 3

img_letter = np.zeros((img_h, img_w, bpp), np.uint8)

red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
white = (255, 255, 255)
yellow = (0, 255, 255)
cyan = (255, 255, 0)
magenta = (255, 0, 255)

center_x = int(img_w / 2.0)
center_y = int(img_h / 2.0)

thickness = 2 

location = (center_x - 200, center_y - 100)
font = cv.FONT_HERSHEY_SCRIPT_SIMPLEX  # hand-writing style font
fontScale = 3.5
cv.putText(img_letter, 'OpenCV', location, font, fontScale, yellow, thickness)

location = (center_x - 150, center_y + 20)
font = cv.FONT_ITALIC  # italic font
fontScale = 2
cv.putText(img_letter, 'Tutorial', location, font, fontScale, red, thickness)

location = (center_x - 250, center_y + 100)
font = cv.FONT_HERSHEY_SIMPLEX  # normal size sans-serif font
fontScale = 1.5
cv.putText(img_letter, 'webnautes.tistory.com', location, font, fontScale, blue, thickness)

location = (center_x - 130, center_y + 150)
font = cv.FONT_HERSHEY_COMPLEX  # normal size serif font
fontScale = 1.2
cv.putText(img_letter, 'webnautes', location, font, fontScale, green, thickness)

cv.imshow("drawing", img_letter)
cv.waitKey(0)
cv.destroyAllWindows()

# [ OpenCV에서 사용할 수 있는 폰트 리스트 ]
# cv.FONT_HERSHEY_SIMPLEX	        normal size sans-serif font
# cv.FONT_HERSHEY_PLAIN	        small size sans-serif font
# cv.FONT_HERSHEY_DUPLEX	        normal size sans-serif font (more complex than FONT_HERSHEY_SIMPLEX)
# cv.FONT_HERSHEY_COMPLEX	        normal size serif font
# cv.FONT_HERSHEY_TRIPLEX	        normal size serif font (more complex than FONT_HERSHEY_COMPLEX)
# cv.FONT_HERSHEY_COMPLEX_SMALL	smaller version of FONT_HERSHEY_COMPLEX
# cv.FONT_HERSHEY_SCRIPT_SIMPLEX	hand-writing style font
# cv.FONT_HERSHEY_SCRIPT_COMPLEX	more complex variant of FONT_HERSHEY_SCRIPT_SIMPLEX
# cv.FONT_ITALIC	                flag for italic font

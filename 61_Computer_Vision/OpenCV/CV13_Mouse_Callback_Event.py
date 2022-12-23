import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv

path = r'D:\Python\★★Python_POSTECH_AI\Dataset'
origin_path = os.getcwd()
os.chdir(path)

img_color = cv.imread('AprilShoe-768x832.jpg')


def mouse_callback(event, x, y, flags, param):
    # 마우스 왼쪽 버튼 Event
    if event == cv.EVENT_LBUTTONDOWN:
        print(f'Left Button: coordinate ({x},{y}), {flags}, {param}')
    elif event == cv.EVENT_RBUTTONDOWN:
        print(f'Right Button: coordinate ({x},{y}), {flags}, {param}')


cv.imshow('img color', img_color)
cv.setMouseCallback('img color', mouse_callback)
cv.waitKey(0)
cv.destroyAllWindows()




# -------------------------------------------------------------------------------------------------------------------
# 1. 마우스 콜백 함수 등록 방법
# OpenCV에서 생성한 윈도우 위에서 마우스 이벤트 발생시 호출되는 마우스 콜백 함수를 등록하는 간단한 예제입니다.
# 마우스 왼쪽/오른쪽 버튼을 누르거나 더블클릭, 휠 스크롤 등을 등을 할 때마다 “마우스 이벤트 발생”이라는 메시지가 출력됩니다. 

# 1. 마우스 이벤트 발생시 호출될 함수를 정의합니다. 
def mouse_callback(event, x, y, flags, param):
    print("마우스 이벤트 발생")


img = np.zeros((512, 512, 3), np.uint8)
cv.namedWindow('image')  # 2. 마우스 이벤트를 감지할 윈도우를 생성합니다.  


# 3. 이름이 image인 윈도우에서 마우스 이벤트가 발생하면 mouse_callback 함수가 호출되게 됩니다. 
cv.setMouseCallback('image', mouse_callback)  


cv.imshow('image',img)
cv.waitKey(0)

cv.destroyAllWindows() 




# -------------------------------------------------------------------------------------------------------------------
# 2. 마우스 이벤트 테스트
# OpenCV 윈도우 위에서 발생한 모든 마우스 이벤트를 출력하는 예제입니다.  
# 하지만 빠진 조합 이벤트가 있을 수도 있습니다.
#   cv.EVENT_MOUSEMOVE       - 마우스 이동할 때 발생
#   cv.EVENT_LBUTTONDOWN   - 왼쪽 마우스 버튼 누르고 있을 때 발생
#   cv.EVENT_LBUTTONUP          - 누르고 있던 왼쪽 마우스 버튼을 떼면 발생
#   cv.EVENT_LBUTTONDBLCLK - 왼쪽 마우스 버튼을 더블 클릭시 발생

#   cv.EVENT_MOUSEWHEEL      - 수직 방향으로 휠 스크롤시 발생
#   cv.EVENT_MOUSEHWHEEL    - 수평 방향으로 휠 스크롤시 발생

#   cv.EVENT_FLAG_LBUTTON, cv.EVENT_FLAG_RBUTTON, cv.EVENT_FLAG_MBUTTON
# 왼쪽 버튼, 오른쪽 버튼, 중간 버튼 관련 이벤트시 발생

#   cv.EVENT_FLAG_CTRLKEY, cv.EVENT_FLAG_SHIFTKEY, cv.EVENT_FLAG_ALTKEY
# 마우스 이벤트 중 Ctrl, Shift, Alt 키를 누르고 있으면 발생


mouse_event_types = { 0:"EVENT_MOUSEMOVE", 1:"EVENT_LBUTTONDOWN", 2:"EVENT_RBUTTONDOWN", 3:"EVENT_MBUTTONDOWN",
                 4:"EVENT_LBUTTONUP", 5:"EVENT_RBUTTONUP", 6:"EVENT_MBUTTONUP",
                 7:"EVENT_LBUTTONDBLCLK", 8:"EVENT_RBUTTONDBLCLK", 9:"EVENT_MBUTTONDBLCLK",
                 10:"EVENT_MOUSEWHEEL", 11:"EVENT_MOUSEHWHEEL"}

mouse_event_flags = { 0:"None", 1:"EVENT_FLAG_LBUTTON", 2:"EVENT_FLAG_RBUTTON", 4:"EVENT_FLAG_MBUTTON",
                8:"EVENT_FLAG_CTRLKEY", 9:"EVENT_FLAG_CTRLKEY + EVENT_FLAG_LBUTTON",
                10:"EVENT_FLAG_CTRLKEY + EVENT_FLAG_RBUTTON", 11:"EVENT_FLAG_CTRLKEY + EVENT_FLAG_MBUTTON",

                16:"EVENT_FLAG_SHIFTKEY", 17:"EVENT_FLAG_SHIFTKEY + EVENT_FLAG_LBUTTON",
                18:"EVENT_FLAG_SHIFTLKEY + EVENT_FLAG_RBUTTON", 19:"EVENT_FLAG_SHIFTKEY + EVENT_FLAG_MBUTTON",

                32:"EVENT_FLAG_ALTKEY", 33:"EVENT_FLAG_ALTKEY + EVENT_FLAG_LBUTTON",
                34:"EVENT_FLAG_ALTKEY + EVENT_FLAG_RBUTTON", 35:"EVENT_FLAG_ALTKEY + EVENT_FLAG_MBUTTON"}

 
# 1. 마우스 이벤트 발생시 호출될 함수를 정의합니다. 
def mouse_callback(event, x, y, flags, param):

    print(flags)
    print( '( '+ str(x) + ' ' + str(y), ')' + ' ' + mouse_event_types[event], end=" " )

    if event == 10: 
        if flags > 0:
            print("forward scrolling")
        else:
            print("backward scrolling")
    elif event == 11:
        if flags > 0:
            print("right scrolling")
        else:
            print("left scrolling")
    else:
        print( mouse_event_flags[flags])



img = np.zeros((512, 512, 3), np.uint8)
cv.namedWindow('image')  # 2. 마우스 이벤트를 감지할 윈도우를 생성합니다.  


# 3. 이름이 image인 윈도우에서 마우스 이벤트가 발생하면 mouse_callback 함수가 호출되게 됩니다. 
cv.setMouseCallback('image', mouse_callback)  


cv.imshow('image',img)
cv.waitKey(0)

cv.destroyAllWindows() 






# -------------------------------------------------------------------------------------------------------------------
# 3. 마우스로 원 / 사각형 그리기
# 앞에서 테스트해본 마우스 이벤트를 이용하여 원과 사각형을 그리는 예제입니다. 

# 마우스 왼쪽 버튼을 누른 후 이동했다가 놓으면 그리기 모드에 따라 내부가 채워진 원 또는 사각형이 그려집니다.  
# 새로운 도형을 그릴때마다 랜덤으로 색을 생성하여 사용합니다.
# m 키를 누르면 원 또는 사각형 그리기 모드가 변경됩니다.


mouse_is_pressing = False   # 왼쪽 마우스 버튼 상태 체크를 위해 사용
drawing_mode = True       # 현재 그리기 모드 선택을 위해 사용 ( 원 / 사각형 )
start_x, start_y = -1, -1   # 최초로 왼쪽 마우스 버튼 누른 위치를 저장하기 위해 사용
color = (255, 255, 255)   # 도형 내부 채울때 사용할 색 지정시 사용 ( 초기값은 흰색 )


def mouse_callback(event,x,y,flags,param):

    global color, start_x, start_y, drawing_mode, mouse_is_pressing


    if event == cv.EVENT_MOUSEMOVE:
        if mouse_is_pressing == True: # 마우스 왼쪽 버튼을 누른 채 이동하면 

            if drawing_mode == True: # 이동된 마우스 커서 위치를 반영하여 사각형/윈을 그림
                cv.rectangle(img,(start_x,start_y),(x,y),color,-1)
            else:
                cv.circle(img, (start_x,start_y), max(abs(start_x - x), abs(start_y - y)), color, -1)

    elif event == cv.EVENT_LBUTTONDOWN:
        # 랜덤으로 (blue, green, red)로 사용될 색을 생성
        color = (random.randrange(256), random.randrange(256), random.randrange(256)) 
        mouse_is_pressing = True     # 왼쪽 마우스 버튼을 누른 것 감지 
        start_x, start_y = x, y     # 최초로 왼쪽 마우스 버튼 누른 위치를 저장 


    elif event == cv.EVENT_LBUTTONUP: 

        mouse_is_pressing = False    # 왼쪽 마우스 버튼에서 손뗀 것을 감지   

        if drawing_mode == True:  # 최종 위치에 마우스 커서 위치를 반영하여 사각형/윈을 그림
            cv.rectangle(img,(start_x,start_y),(x,y),color,-1)
        else:
            cv.circle(img, (start_x,start_y), max(abs(start_x - x), abs(start_y - y)), color, -1)



img = np.zeros((512, 512, 3), np.uint8) 
cv.namedWindow('image')   
cv.setMouseCallback('image', mouse_callback) 


while(1):

    cv.imshow('image',img)

    k = cv.waitKey(1) & 0xFF

    if k == ord('m'): # m 누르면 그리기 모드 변경( 사각형 / 원 ) 
        drawing_mode = not drawing_mode

    elif k == 27: 
        break

cv.destroyAllWindows()
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# conda install -c conda-forge opencv
# conda install opencv
# pip install opencv-python
# import cv2 as cv                # 이미지 resize 등 제공
# https://076923.github.io/posts/Python-opencv-8/
import cv2 as cv


path = r'D:\Python\★★Python_POSTECH_AI\Dataset/'
origin_path = os.getcwd()
os.chdir(path)


# https://webnautes.tistory.com/1219?category=791376        # webnautes Blog
# https://076923.github.io/posts/Python-opencv-1/           # OpenCV Blog
# !wget https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.shutterstock.com%2Fvideo%2Fclip-20693977-snooker-stun-shot-on-red-ball&psig=AOvVaw1U078WZN1-yPNDPYVs8xx3&ust=1602652036994000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCIjPnZ_msOwCFQAAAAAdAAAAABAb


# [Lecture 01] Image Control ======================================================
# Image Load
img_color = cv.imread('redball_on_table.jpg', cv.IMREAD_COLOR)
img_gray = cv.imread('redball_on_table.jpg', cv.IMREAD_GRAYSCALE)
img_unchange = cv.imread('redball_on_table.jpg', cv.IMREAD_UNCHANGED)

# img_color.shape     # (heigt, width, channel)
# heigt, width, channel = img_color.shape


# * mode
# cv.IMREAD_UNCHANGED : 원본 사용
# cv.IMREAD_GRAYSCALE : 1 채널, 그레이스케일 적용
# cv.IMREAD_COLOR : 3 채널, BGR 이미지 사용
# cv.IMREAD_ANYDEPTH : 이미지에 따라 정밀도를 16/32비트 또는 8비트로 사용
# cv.IMREAD_ANYCOLOR : 가능한 3 채널, 색상 이미지로 사용
# cv.IMREAD_REDUCED_GRAYSCALE_2 : 1 채널, 1/2 크기, 그레이스케일 적용
# cv.IMREAD_REDUCED_GRAYSCALE_4 : 1 채널, 1/4 크기, 그레이스케일 적용
# cv.IMREAD_REDUCED_GRAYSCALE_8 : 1 채널, 1/8 크기, 그레이스케일 적용
# cv.IMREAD_REDUCED_COLOR_2 : 3 채널, 1/2 크기, BGR 이미지 사용
# cv.IMREAD_REDUCED_COLOR_4 : 3 채널, 1/4 크기, BGR 이미지 사용
# cv.IMREAD_REDUCED_COLOR_8 : 3 채널, 1/8 크기, BGR 이미지 사용

    # show Image at window
cv.namedWindow('show Image')
cv.imshow('show Image', img_color)
cv.waitKey(0)              # time마다 키 입력상태를 받아옵니다. 0일 경우, 지속적으로 검사하여 해당 구문을 넘어가지 않습니다.
cv.destroyAllWindows()     # 모든 윈도우창을 닫습니다.



# color → gray
img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)

cv.namedWindow('show Image')
cv.imshow('show Image', img_color)
cv.waitKey(0)
cv.imshow('show Image', img_gray)
cv.waitKey(0)
cv.destroyAllWindows()



# Image Save
cv.imwrite('save_gray_image.jpg', img_gray)





# ★★★ BGR → RGB ★★★ ----------------------------------------------------------------------
# img_color = cv.imread('redball_on_table.jpg')

# Method_1 : rgb = bgr[:,:,::-1]
# plt.imshow(img_color[:,:,::-1)

# Method_2
# image_color_rgb =cv.cvtColor(img_color, cv.COLOR_BGR2RGB)
# plt.imshow(image_color_rgb)

# Method_3
# b, g, r = cv.split(img_color)
# image_color_rgb =cv.merge([r,g,b])
# plt.imshow(image_color_rgb)























# [Lecture 02] Video Control ======================================================

# Camera capture ------------------------------------------------------------
cap = cv.VideoCapture(0)   
# cap2 = cv.VideoCapture(1)  # 2대의 camera 필요시

ret, img_cap = cap.read()
#ret은 카메라의 상태가 저장되며 정상 작동할 경우 True를 반환합니다. 작동하지 않을 경우 False를 반환합니다.

cv.imshow('color', img_cap)
cv.waitKey(0)
cap.release()
cv.destroyAllWindows()



# Camera Video Capture ------------------------------------------------------------
video_cap = cv.VideoCapture(0)

while True:
    ret, video_cap_color = video_cap.read()

    if ret == False:
        continue
    
    # 영상
    cv.imshow('video', video_cap_color)

    # Gray Scale 영상
    video_cap_gray = cv.cvtColor(video_cap_color, cv.COLOR_BGR2GRAY)
    cv.imshow('video2', video_cap_gray)

    if cv.waitKey(1)&0xFF == 27:
        break
    
video_cap.release()
cv.destroyAllWindows()




# Video Save ------------------------------------------------------------
fourcc = cv.VideoWriter_fourcc(*'XVID')
video_writer = cv.VideoWriter('camera_video.avi', fourcc, 30, (640, 480))
# cv.VideoWriter(file명, codec, frame수, (해상도W, 해상도H))

video_cap = cv.VideoCapture(0)

while True:
    ret, video_cap_color = video_cap.read()

    if ret == False:
        continue
    
    # 영상
    cv.imshow('video', video_cap_color)
    video_writer.write(video_cap_color)

    if cv.waitKey(1)&0xFF == 27:
        break

video_cap.release()
video_writer.release()
cv.destroyAllWindows()



# Video Play ------------------------------------------------------------
video_play = cv.VideoCapture('camera_video.avi')
# video_play.get(cv.CAP_PROP_POS_FRAMES)     # cv.CAP_PROP_POS_FRAMES는 현재 프레임 개수를 의미합니다.
# video_play.get(cv.CAP_PROP_FRAME_COUNT)    # cv.CAP_PROP_FRAME_COUNT는 총 프레임 개수를 의미합니다.

while True:
    # 재생이 끝나면 반복 -------------------
    if(video_play.get(cv.CAP_PROP_POS_FRAMES) == video_play.get(cv.CAP_PROP_FRAME_COUNT)):
        cv.waitKey(0)
        video_play.open("camera_video.avi")
    # --------------------------------------

    ret, video_play_color = video_play.read()

    # if ret == False:    # vedio 종료여부
    #     break

    cv.imshow('video', video_play_color)

    if cv.waitKey(1)&0xFF == 27:
        break
    time.sleep(0.03)
video_play.release()
cv.destroyAllWindows()



# capture = cv.VideoCapture('camera_video.avi')
#     capture.get(속성) : VideoCapture의 속성을 반환합니다.
#     capture.grab() : Frame의 호출 성공 유/무를 반환합니다.
#     capture.isOpened() : VideoCapture의 성공 유/무를 반환합니다.
#     capture.open(카메라 장치 번호 또는 경로) : 카메라나 동영상 파일을 엽니다.
#     capture.release() : VideoCapture의 장치를 닫고 메모리를 해제합니다.
#     capture.retrieve() : VideoCapture의 프레임과 플래그를 반환합니다.
#     capture.set(속성, 값) : VideoCapture의 속성의 값을 설정합니다.


# < VideoCapture 속성 >
# 속성	의미	(비고)
# cv.CAP_PROP_FRAME_WIDTH	프레임의 너비	(-)
# cv.CAP_PROP_FRAME_HEIGHT	프레임의 높이	(-)
# cv.CAP_PROP_FRAME_COUNT	프레임의 총 개수	(-)
# cv.CAP_PROP_FPS	프레임 속도	(-)
# cv.CAP_PROP_FOURCC	코덱 코드	(-)
# cv.CAP_PROP_BRIGHTNESS	이미지 밝기	(카메라만 해당)
# cv.CAP_PROP_CONTRAST	이미지 대비	(카메라만 해당)
# cv.CAP_PROP_SATURATION	이미지 채도	(카메라만 해당)
# cv.CAP_PROP_HUE	이미지 색상	(카메라만 해당)
# cv.CAP_PROP_GAIN	이미지 게인	(카메라만 해당)
# cv.CAP_PROP_EXPOSURE	이미지 노출	(카메라만 해당)
# cv.CAP_PROP_POS_MSEC	프레임 재생 시간	(ms 반환)
# cv.CAP_PROP_POS_FRAMES	현재 프레임	(프레임의 총 개수 미만)
# CAP_PROP_POS_AVI_RATIO	비디오 파일 상대 위치	(0 = 시작, 1 = 끝)





# Video List Append ------------------------------------------------------------
video_cap = cv.VideoCapture(0)
frames = []

while True:
    ret, video_cap_color = video_cap.read()
    cv.imshow('video', video_cap_color)
    frames.append(video_cap_color)

    if cv.waitKey(1)&0xFF == 27:
        break
video_cap.release()
cv.destroyAllWindows()
frames = np.array(frames)

frames.shape
# frames.__sizeof__()





# Video Save ------------------------------------------------------------
fourcc = cv.VideoWriter_fourcc(*'XVID')
video_writer = cv.VideoWriter('camera_video.avi', fourcc, 30, frames[0].shape[:-1][::-1])
# cv.VideoWriter(file명, codec, frame수, (해상도W, 해상도H))

for f in frames:
    video_writer.write(f)
video_writer.release()



# Video Load ------------------------------------------------------------
frames =[]
video_load = cv.VideoCapture('camera_video.avi')


while True:
    ret, frame = video_load.read()
    
    if ret:
        frames.append(frame)
    else:
        break
video_load.release()
frames = np.array(frames)






# Video Play ------------------------------------------------------------
for f in frames:
    cv.imshow('video_play', f)

    time.sleep(0.03)
    if cv.waitKey(1)&0xFF == 27:
        break
cv.waitKey(0)
cv.destroyAllWindows()



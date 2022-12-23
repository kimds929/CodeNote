# https://pyautogui.readthedocs.io/en/latest/index.html
# https://wikidocs.net/book/4706

# 2.1 마우스 자동화 - pyautogui 사용법 (1) =============================================
import pyautogui

# 좌표 객체 얻기
pos = pyautogui.position()
pos
x, y = pos

# 화면 전체 크기 확인하기
print(pyautogui.size())

# x, y 좌표
print(pos.x, pos.y)

pyautogui.onScreen(x, y)  # True if x & y are within the screen.
pyautogui.PAUSE = 2.5

# 마우스 이동 (x 좌표, y 좌표)
pyautogui.moveTo(500, 500)

# 마우스 이동 (x 좌표, y 좌표 2초간)
pyautogui.moveTo(100, 100, 2) 

# 마우스 이동 ( 현재위치에서 )
pyautogui.moveRel(200, 300, 2)

# 마우스 클릭
pyautogui.click()

# 2초 간격으로 2번 클릭
pyautogui.click(clicks= 2, interval=2)
pyautogui.click('Calc.png') # Find where button.png appears on the screen and click it.

# 더블 클릭
pyautogui.doubleClick()

# 오른쪽 클릭
pyautogui.click(button='right')


# 클릭
moveToX, moveToY = 70, 70
pyautogui.rightClick(x=moveToX, y=moveToY)
pyautogui.middleClick(x=moveToX, y=moveToY) # middle key click (scroll mark)
pyautogui.doubleClick(x=moveToX, y=moveToY)
pyautogui.tripleClick(x=moveToX, y=moveToY)

# Individual button down and up events can be called separately
pyautogui.mouseDown(x=moveToX, y=moveToY, button='left')    # keeping press of mouse-button
pyautogui.mouseUp(x=moveToX, y=moveToY, button='left')    # seperate from mouse-button


# 스크롤하기
pyautogui.scroll(10)

# 드래그하기
pyautogui.drag(0, 300, 1, button='left')

num_seconds = 1
pyautogui.dragTo(x, y, duration=num_seconds)  # drag mouse to XY

xOffset, yOffset = 50, 50
pyautogui.dragRel(xOffset, yOffset, duration=num_seconds)  # drag mouse relative to its current position

pyautogui.write('hello world!') # 괄호 안의 문자를 타이핑 합니다.
pyautogui.write('hello world!', interval=0.25) # 각 문자를 0.25마다 타이핑합니다.

pyautogui.typewrite(['a', 'b', 'cde', 'left', 'backspace', 'enter'], interval=0.25)



# 2.2 키보드 자동화 - pyautogui 사용법 (2) =============================================
import pyperclip

pyperclip.copy("안녕하세요") # 클립보드에 텍스트를 복사합니다.

pyautogui.hotkey('ctrl', 'v') # 붙여넣기 (hotkey 설명은 아래에 있습니다.)

pyautogui.press('shift') # shift 키를 누릅니다.
pyautogui.press('ctrl') # ctrl 키를 누릅니다.

pyautogui.keyDown('ctrl') # ctrl 키를 누른 상태를 유지합니다.
pyautogui.press('c') # c key를 입력합니다.
pyautogui.keyUp('ctrl') # ctrl 키를 뗍니다.

pyautogui.press(['left', 'left', 'left']) # 왼쪽 방향키를 세번 입력합니다.
pyautogui.press('left', presses=3) # 왼쪽 방향키를 세번 입력합니다.
pyautogui.press('enter', presses=3, interval=3) # enter 키를 3초에 한번씩 세번 입력합니다.

# 여러 키를 동시에 입력해야 할 때 keyDown과 keyUp을 사용하면 상당히 불편해요.
# 그걸 편하게 해주는 함수가 바로, hotkey() 함수입니다.
pyautogui.hotkey('ctrl', 'c') # ctrl + c 키를 입력합니다.


# ['\t', '\n', '\r', ' ', '!', '"', '#', '$', '%', '&', "'", '(',
# ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7',
# '8', '9', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`',
# 'a', 'b', 'c', 'd', 'e','f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
# 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~',
# 'accept', 'add', 'alt', 'altleft', 'altright', 'apps', 'backspace',
# 'browserback', 'browserfavorites', 'browserforward', 'browserhome',
# 'browserrefresh', 'browsersearch', 'browserstop', 'capslock', 'clear',
# 'convert', 'ctrl', 'ctrlleft', 'ctrlright', 'decimal', 'del', 'delete',
# 'divide', 'down', 'end', 'enter', 'esc', 'escape', 'execute', 'f1', 'f10',
# 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f2', 'f20',
# 'f21', 'f22', 'f23', 'f24', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9',
# 'final', 'fn', 'hanguel', 'hangul', 'hanja', 'help', 'home', 'insert', 'junja',
# 'kana', 'kanji', 'launchapp1', 'launchapp2', 'launchmail',
# 'launchmediaselect', 'left', 'modechange', 'multiply', 'nexttrack',
# 'nonconvert', 'num0', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6',
# 'num7', 'num8', 'num9', 'numlock', 'pagedown', 'pageup', 'pause', 'pgdn',
# 'pgup', 'playpause', 'prevtrack', 'print', 'printscreen', 'prntscrn',
# 'prtsc', 'prtscr', 'return', 'right', 'scrolllock', 'select', 'separator',
# 'shift', 'shiftleft', 'shiftright', 'sleep', 'space', 'stop', 'subtract', 'tab',
# 'up', 'volumedown', 'volumemute', 'volumeup', 'win', 'winleft', 'winright', 'yen',
# 'command', 'option', 'optionleft', 'optionright']



# 2.3 메세지 박스 - pyautogui 사용법(3) =============================================
import pyautogui as pg

a = pg.alert(text='내용입니다', title='제목입니다', button='OK')
print(a)

a = pg.confirm(text='내용입니다', title='제목입니다', buttons=['OK', 'Cancel'])
print(a)

a = pg.prompt(text='내용입니다', title='제목입니다', default='입력하세요')
print(a)

a = pg.password(text='내용입니다', title='제목입니다', default='입력하세요', mask='*')
print(a)



# 2.4 이미지로 좌표찾기 - pyautogui 사용법 (4) =============================================
import os
os.getcwd()
import matplotlib.pyplot as plt

im1 = pyautogui.screenshot()
im2 = pyautogui.screenshot('screenshot.png')        # screenshot 파일로 저장
im = pyautogui.screenshot(region=(0,0, 300, 400))       # 특정영역만 저장

# find all of objects using generator
printA_generator = pyautogui.locateAllOnScreen('print_a.png')

for i in printA_generator:
    print(i)


plt.imshow(plt.imread('Calc.png'))
plt.imshow(plt.imread('7.png'))

b7 = pg.locateOnScreen('7.png')
b7
b7 = pg.locateOnScreen('7.png')
b7
b5 = pg.locateOnScreen('mail.png')
b5

button5location = pg.locateOnScreen('Calc.png') # 이미지가 있는 위치를 가져옵니다.
print(button5location)

button5location = pg.locateOnScreen('Calc.png')
point = pg.center(button5location) # Box 객체의 중앙 좌표를 리턴합니다.
print(point)

button5location.left
button5location.top
button5location.width

# Grayscale Matching
pg.locateOnScreen('Calc.png', grayscale=True)
im.getpixel((100, 200))

im.getpixel((10,10))
im1.getpixel((29,107))

import numpy as np
a = np.ones(shape=(3, 10,10))
a[0,:,:] = 41/255
a[1,:,:] = 128/255
a[2,:,:] = 185/255

b = np.ones(shape=(10,10,3))*1

plt.imshow()

plt.imshow(a.transpose(1,2,0))
plt.imshow(b)

plt.imread('7.png')



import collections
from collections import namedtuple

def get_position_color(img=True):
    result_object = namedtuple('PointColorObject', ['position', 'rgb', 'img'])
    x, y = pg.position()
    screen_shot = pyautogui.screenshot()
    r, g, b = screen_shot.getpixel((x,y))

    color_map = np.ones(shape=(3, 5,5 ))
    color_map[0,:,:] = r/255
    color_map[1,:,:] = g/255
    color_map[2,:,:] = b/255

    result = result_object(position=(x, y), rgb=(r,g,b), img=None)
    if img:
        fig = plt.figure(figsize=(1,1))
        plt.imshow(color_map.transpose(1,2,0))
        plt.axis('off')
        plt.show()
        result = result._replace(img=fig)

    return result

get_position_color()




# 활성화된 창 정보 얻기 =============================================
import time, ctypes

time.sleep(3)    # 타이틀을 가져오고자 하는 윈도우를 활성화 하기위해 의도적으로 3초 멈춥니다.

lib = ctypes.windll.LoadLibrary('user32.dll')
handle = lib.GetForegroundWindow()    # 활성화된 윈도우의 핸들얻음
handle

buffer_title = ctypes.create_unicode_buffer(255)    # 타이틀을 저장할 버퍼
lib.GetWindowTextW(handle, buffer_title, ctypes.sizeof(buffer_title))    # 버퍼에 타이틀 저장

print(buffer_title.value)    # 버퍼출력 (이름)



# 활성화된 창 정보 얻기 =============================================
import win32ui

from typing import Optional
from ctypes import wintypes, windll, create_unicode_buffer
import os
from PIL import Image
# from PIL import Image, ImageFilter, ImageGrab  # imports the library
# import matplotlib.pyplot as plt
from io import StringIO,  BytesIO
import win32clipboard
# img2 = ImageGrab.grabclipboard()        # ClipBoard에 있는 이미지를 변수에 넣기
# fig.savefig('abc')                    # fig 변수에 저장된 이미지를 'abc.png'파일로 저장하기

def fun_Send_To_Clipboard(clip_type, data):
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(clip_type, data)
    win32clipboard.CloseClipboard()

def fun_Img_To_Clipboard(fig):
    '''
    fig: pyplot figure
    '''
    fig.savefig('pyplot_temper_img', bbox_inches='tight')    # png파일로저장
    PIL_img = Image.open('pyplot_temper_img.png').copy()   #png파일 PIL image형태로 불러오기
    os.remove('pyplot_temper_img.png')  # png파일 지우기
    output = BytesIO()
    PIL_img.convert("RGB").save(output, "BMP")
    data = output.getvalue()[14:]
    output.close()
    fun_Send_To_Clipboard(win32clipboard.CF_DIB, data)
    

# img_copy(fig)


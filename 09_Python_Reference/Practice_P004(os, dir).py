#-*- coding: UTF-8 -*-
print("Start Making Folder")
import os
import getpass

username = getpass.getuser()    # 사용자명 얻기
folder_name = 'pypi'

dir_folder = 'C:\\Users' + '\\'+ username + '\Desktop' + '\\' + folder_name
#dir_file = 'C:\\Users' + '\\'+ username + '\Desktop' + '\\' + folder_name + '\\' + file_name
os.path.isdir(dir_folder)       # 해당경로에 파일/폴더가 있는지 확인

if not os.path.isdir(dir_folder):
    os.mkdir(dir_folder)  # 폴더 만들기

print("Making Folder Success!")
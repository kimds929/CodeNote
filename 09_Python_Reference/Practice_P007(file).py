import numpy as np
import pandas as pd
import pyperclip
import os
# color_df = pd.read_clipboard()

# color_dict = color_df.to_dict('record')
# pyperclip.copy(str(color_dict))
os.getcwd()
file_list = os.listdir('./')    # 현재폴더 전체 파일리스트
file_list_py = [file for file in file_list if file.endswith(".py")]     # 확장자가 '.py' 로 끝나는 파일 리스트
file_list_csv = [file for file in file_list if file.endswith(".csv")]     # 확장자가 '.py' 로 끝나는 파일 리스트

file_list_csv
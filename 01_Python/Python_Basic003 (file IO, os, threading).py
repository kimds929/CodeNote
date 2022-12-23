import os
import getpass

username = getpass.getuser()    # 사용자명 얻기
folder_name = 'pypi'

os.listdir('.')     # 현디렉토리 파일목록
dir_folder = 'C:\\Users' + '\\'+ username + '\Desktop' + '\\' + folder_name
#dir_file = 'C:\\Users' + '\\'+ username + '\Desktop' + '\\' + folder_name + '\\' + file_name
os.path.isdir(dir_folder)       # 해당경로에 파일/폴더가 있는지 확인

if not os.path.isdir(dir_folder):
    os.mkdir(dir_folder)  # 폴더 만들기

print("Making Folder Success!")



# color_df = pd.read_clipboard()

# color_dict = color_df.to_dict('record')
# pyperclip.copy(str(color_dict))
os.getcwd()
file_list = os.listdir('./')    # 현재폴더 전체 파일리스트
file_list_py = [file for file in file_list if file.endswith(".py")]     # 확장자가 '.py' 로 끝나는 파일 리스트
file_list_csv = [file for file in file_list if file.endswith(".csv")]     # 확장자가 '.py' 로 끝나는 파일 리스트

file_list_csv


# ---------------------------------------------------------------------------
import webbrowser
webbrowser.open('http://google.com')        # 해당 웹사이트 주소가 열림
webbrowser.open_new('http://naver.com')        # 해당 웹사이트 주소가 열림


# ---------------------------------------------------------------------------
import time
import threading


def long_task(): # 3초의 시간이 걸리는 함수
    for i in range(3):
        time.sleep(1)   # 1초 대기
        print('working: %s sec' %(i+1))

# 일반작업시
start_1 = time.time()   # start

long_task()

end_1 = time.time()   # end

print('총 걸린시간 : %s sec' %(end_1 - start_1))



# thread Multi 작업시
start_2 = time.time()   # start

threads = []
for i in range(3):
    t = threading.Thread(target=long_task)  # 스레드 생성
    threads.append(t)

for t in threads:
    t.start()   # 스레드 실행

end_2 = time.time() # end

print('총 걸린시간 : %s sec' %(end_2 - start_2))





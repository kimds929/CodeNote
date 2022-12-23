# https://m.blog.naver.com/jhkang8420/221291682151      # Regression

import os
from IPython.display import clear_output

# 현재 Directory에 있는 'ipynb' file을 'py' file로 converting 해주는 함수
def fun_Convert_ipynb_to_py():
    ''' # 현재 Directory에 있는 'ipynb' file을 'py' file로 converting 해주는 함수 '''
    print(f'현재경로 : {os.getcwd}')
    file_list = os.listdir()    # 현재 경로 파일 List
    ipynb_list = [fl for fl in file_list if os.path.splitext(fl)[1] == '.ipynb']    # ipynb file List

    print(f'ipynb file List : {ipynb_list}')
    if ipynb_list:
        n=0
        sucess_file_list = []
        fail_file_list = []
        for ipynb in ipynb_list:
            n+=1
            print(f'{round(n/len(ipynb_list)*100,1)} %')
            try:
                os.system('jupyter nbconvert --to script "' + ipynb + '"')
                sucess_file_list.append(ipynb)
                print(f'{ipynb} file convert Success!')
                # clear_output(wait=True)
            except:
                fail_file_list.append(ipynb)
                print('{ipynb} file convert Fail')
            # clear_output(wait=True)
        if fail_file_list:
            print(f'{fail_file_list} file convert Fail')

# 지정 주소(address)에 있는 'ipynb' file을 'py' file로 converting 해주는 함수
def fun_Convert_ipynb_to_py_dir(address):
    '''
    # 지정 주소(address)에 있는 'ipynb' file을 'py' file로 converting 해주는 함수
    
    < Input >
    address (str) : 주소
    '''
    origin_wd = os.getcwd()     # 첫 경로 저장
    print(f'○ 기존경로 : {origin_wd}')
    try:
        os.chdir('./'+address)
        print(f'요청경로 접근결과 : {os.getcwd()}')
        fun_Convert_ipynb_to_py()       # 현재 Directory에 있는 'ipynb' file을 'py' file로 converting 해주는 함수
        os.chdir(origin_wd) # 처음있었던 경로로 되돌아가기
    except:
        print(f'지정된 경로를 찾을 수 없습니다.')

# os.getcwd()
# os.chdir('../')
# fun_Convert_ipynb_to_py_dir('./')
import sys
sys.version # Python Version
# '3.6.10 |Anaconda, Inc.| (default, May  8 2020, 02:54:21) \n[GCC 7.3.0]'

# Kernel 등록
# ipython kernel install --user --name=py36 



# map, reduce --------------------------------------------------------------------
from functools import reduce
a = [1,2,3,4]
list(map(str, a))

    # reduce 함수
reduce(lambda x,y: x-y, a)
reduce(abc1, a)

def abc1 (x,y):
    print(f'x: {x}')
    print(f'y: {y}')
    print(f'x-y: {x-y}')
    print('--------')
    return x-y

    # reduce 함수: 초기값 설정
reduce(lambda x,y: x-y, a, 10)
def abc2 (x,y):
    print(f'x: {x}')
    print(f'y: {y}')
    print(f'x-y: {x-y}')
    print('--------')
    return x-y

reduce(abc2, a, 10)

# ---------------------------------------------------------------------

dir(object) # Object내 사용할 수 있는 command 및 Method List

# 사용할 수 있는 Method 및 설명을 출력해주는 함수
def get_methods(object, spacing=20): 
  methodList = [] 
  for method_name in dir(object): 
    try: 
        if callable(getattr(object, method_name)): 
            methodList.append(str(method_name)) 
    except: 
        methodList.append(str(method_name)) 
  processFunc = (lambda s: ' '.join(s.split())) or (lambda s: s) 
  for method in methodList: 
    try: 
        print(str(method.ljust(spacing)) + ' ' + 
              processFunc(str(getattr(object, method).__doc__)[0:90])) 
    except: 
        print(method.ljust(spacing) + ' ' + ' getattr() failed') 




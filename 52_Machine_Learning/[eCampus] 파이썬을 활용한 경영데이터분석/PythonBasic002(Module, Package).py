import random
random.randint(1,100)

[random.randint(1,100) for i in range(10)]

# 만든 모듈 호출
from PythonBasic003_MakeModule import covert_c_to_f as cf
print(cf(10))

# 시간관련 모듈
import time
import datetime
time.localtime()

datetime.datetime.strptime('21/12/2008', '%d/%m/%Y')
datetime.datetime.strptime('21/12/2008', '%d/%m/%Y').strftime('%Y-%m-%d')

# 웹페이지 소스 모듈
import urllib.request
response = urllib.request.urlopen("http://www.google.co.kr")
response.read()

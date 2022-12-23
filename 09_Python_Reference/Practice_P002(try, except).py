# -*- coding: UTF-8 -*-

# 예외처리 방법 ------------------------------------------------------------------------------------
    # try / except
a = [89, "90점", 100, NA, "우"]
for i in a :
    try :       # 정상시 실행
        score = int(i)
    except:     # 예외, 오류 발생시 실행
        score = str(i) + ": 예외발생"
    print(score)
print("프로그램 종료")

# 예외처리 : while / try
while True :
    a = input("점수입력 :")
    try:
        score = int(a)
        print("입력점수 : ", score)
        break
    except:
        print("입력된 점수 형식이 잘못되었습니다. (입력값: ",a,")\n")
print("프로그램 종료")


    # 예외종류 / as e : Error 메세지 출력
a = 89
try:
    score = int(a)
    print(score)
    a[5]
except ValueError as e:
    print("숫자형식이 잘못되었습니다. ",e)
except IndexError as e:
    print("첨자 범위를 벗어났습니다. ", e)
except NameError as e:
    print("명칭미발견 ", e)
except TypeError as e:
    print("Type 에러 ", e)
except:
    print("기타에러발생 ", e)
print("프로그램 종료")


    # 복수개의 예외 한꺼번에 처리
a = "89"
try:
    score = int(a)
    print(score)
    a[5]
except (ValueError, TypeError) as e:
    print("값의 형식이나 Type이 잘못되었습니다. ",e)
except (IndexError, NameError) as e:
    print("첨자 범위를 벗어났거나 명칭 미발견했습니다. ", e)
except:
    print("기타에러발생 ", e)
print("프로그램 종료")


    # Raise : 고의로 예외 발생
def calcsum(n):
    if (n < 0):
        raise ValueError    # 고의로 예외 발생
    sum = 0
    for i in range(n+1):
        sum = sum+i
    return sum


a_sum = input("1부터 몇까지 더할까요? : ")
try:
    print(calcsum(int(a_sum)))
except TypeError:
    print("입력형식이 잘못되었습니다.")
except ValueError:
    print("입력값이 잘못되었습니다.")


    # 자원정리 finally : 오류여부와 관계없이 반드시 실행
try:
    print("네트워크 접속")
    a=2/0
    print(a)
    print("네트워크 통신 수행")
finally:        #오류여부와 관계없이 반드시 실행 후 전체 블록을 빠져나옴
    print("접속 해제")
print("작업완료")

    # assert 조건, 메세지 : 프로그램의 현재 상태에서 오류가 있는지 확인
score = 128
assert score <100, "점수는 100이하여야 합니다."
print(score)

#  -- 예외종류
NameError : 명칭이 발견되지 않는다. 초기화하지 않은 변수를 사용할때 발생
ValueError : 타입은 맞지만 값의 형식이 잘못되었다.
ZeroDivisionError : 0으로 나누었다
IndexError : 첨자(글자 갯수 이외의 수를 세려고 할때)가 범위를 벗어났다.
TypeError : 타입이 맞지 않다. 숫자가필요한 곳에 문자열을 사용한경우 등

#----------------------------------------------------------------------------------------------

a = "3+4"
eval(a) # 수식을 실행
b='korea'
repr(b)  # 객체로부터 문자열 표현식 생성
c = 'value=3'
d = exec(c)     # 파이썬 코드를 실행
value

    # compile(source, filename, mode)
        # ( source = 문자열코드 ) or ( filename = 스크립트파일 or 코드: '"<string>' )
        # mode = 표현식: 'eval' / 실행: 'exec'
code = compile("""
for i in range(5):
    print(i, end=', ')
print()
""", '"<string>', 'exec')
exec(code)



#----------------------------------------------------------------------------------------------
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from numpy import nan as NA
import copy
import datetime

    # Date
a = datetime.date(2017,3,12)
b = datetime.date(2019,3,26)
print(b-a)

datetime.date(year, month, day)
datetime.time(hour, minute, second)
datetime.datetime(year, month, day, hour, minute, second)

print( datetime.date.today() )          #현재일 추출
print( datetime.datetime.now() )		#현재일, 시각 추출

    # calendar (달력)
import calendar as cald
print( cald.calendar(2019, m=3) )
print( cald.month(2019, 3) )
print( cald.monthrange(2019, 3) )       # 각 월의 시작요일과


# 날짜
t = datetime.date.today()           #현재일 추출
type(t)
tstr = t.strftime("%Y%m%d")        #날짜 → 문자형으로 변경
datetime.datetime.strptime('tstr','%Y%m%d')


# data set : date_range
pd.date_range(start = '20180101',end ='20180531', freq ='MS') # 1/1 ~ 5/31 매달 첫날 구간으로 Data 산출
pd.date_range(start = '20180101', periods =10, freq ='M') # 1/1부터 매달 마지막날기준으로 10개 Data 산출
    # date shift
Series( pd.date_range(start = '20180101', periods =10, freq ='MS') ).shift(1)   # Data열을 기준으로 1칸 Shift
Series( pd.date_range(start = '20180101', periods =10, freq ='MS').shift(1) )   # 날짜 를 1구간 Shift 후 Data열생성
Series( pd.date_range(start = '20180101', periods =10, freq ='MS').shift(1, freq = 'w') )   # 날짜 를 1구간(freq = shift 구간설정) Shift 후 Data열생성


mts = Series( pd.date_range(start = '20180101',end ='20180601', freq ='MS').strftime("%y%m") )
mts2 = Series( pd.date_range(start = '20180901',end ='20190201', freq ='MS').strftime("%y%m") )
mtd = pd.to_datetime(mts, format = '%y%m')      # 문자 → 날짜 형식
mtd.dt.strftime("%Y%m")                 # 날짜 → 문자 형식

    # data set : 문자 → 날짜
mt = DataFrame({'M1' : pd.to_datetime(mts,format = '%y%m'),'M2' : pd.to_datetime(mts2,format = '%y%m'),
                'M3' : pd.to_datetime(mts,format = '%y%m').dt.to_period(freq = 'm'),
                'M4' : pd.to_datetime(mts2,format = '%y%m').dt.to_period(freq = 'm')})
mtm = mt['M2']-mt['M1']
mtm2 = mt['M4']-mt['M3']
(mt['M2'].dt.year - mt['M1'].dt.year)*12 + (mt['M2'].dt.month - mt['M1'].dt.month)

print( np.timedelta64(1,"M") )
mtm/ np.timedelta64(1,"M")

mtstr = mt['M2'].dt.strftime("%Y%m")

# -------- 날짜 Data Resampling
n = 100
d1 = pd.DataFrame({'D0' : pd.date_range(start='20180101', periods=n, freq='w'), 'D1' : np.random.randn(n) })
d1.index = d1['D0']
    # Down - sampling : 기존데이터보다 데이터를 줄이는경우 (기간을 묶는경우)
d1['D1'].resample('m').mean()
d1['D1'].resample('m').first()
d1['D1'].resample('m').last()
d1['D1'].resample('m', closed='left').mean()            # closed : Left 경계값을 계산에 포함, Right 경계값은 미포함
d1['D1'].resample('m', closed='right').mean()           # closed : (Default) Right 경계값을 계산에 포함, Left 경계값은 미포함
d1['D1'].resample('m').ohlc()                           # ohlc() : 요약 * open(=start), high(=max), low(=min), close(=end)
    # Up - sampling : 기존데이터보다 데이터를 늘리는경우 (기간을 푸는경우)
d1['D1'].head(20)
d1['D1'].resample('d').ffill().head(20)                 # ffill 이전행 값으로 치환 (앞(위)쪽 데이터로 치환)
d1['D1'].resample('d').bfill().head(20)                 # bfill 이후행 값으로 치환 (뒤(아래)쪽 데이터로 치환)

d1['D1'].to_clipboard()        #Clipboard로 내보내기


# date_range :  freq 인수로 특정한 날짜만 생성되도록 할 수도 있다. 많이 사용되는 freq 인수값은 다음과 같다.
•s: 초
•T: 분
•H: 시간
•D: 일(day)
•B: 주말이 아닌 평일
•W: 주(일요일)
•W-MON: 주(월요일)
•M: 각 달(month)의 마지막 날
•MS: 각 달의 첫날
•BM: 주말이 아닌 평일 중에서 각 달의 마지막 날
•BMS: 주말이 아닌 평일 중에서 각 달의 첫날
•WOM-2THU: 각 달의 두번째 목요일
•Q-JAN: 각 분기의 첫달의 마지막 날
•Q-DEC: 각 분기의 마지막 달의 마지막 날


    # calendar (달력)
import calendar as cald

print( cald.calendar(2019, m=3) )
print( cald.month(2019, 3) )
print( cald.monthrange(2019, 3) )       # 각 월의 시작요일과
# -*- coding: UTF-8 -*-
import re

#----------- DataFrame 관련(Pandas)  # https://blog.naver.com/townpharm/220971613200 #
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from numpy import nan as NA
import copy
import datetime

import pyperclip        #Clipboard 관련 Package

import os
import getpass

a=[[1,2,3,4],[5,6,7,8]]
a=pd.DataFrame(a)
a = pd.read_clipboard()  #Clipboard로 입력하기

a.to_clipboard(sep = ' ')        #Clipboard로 내보내기, sep = ' 구분자'

a['직번']
a.직번
list(a.index)       # index 확인하기

a.sample(n=3)   # Table내 임의갯수 Sampling Display
a.head(n=3)     # Table내 앞부분 Sampling Display
a.tail(n=3)     # Table내 뒷부분 Sampling Display

a.fillna(0)      # DataFrame안의 NA값을 0으로 치환
a['공장/섹션'].fillna(0)      # Data-Frame의 '공장/섹션'열에 NA값을 0으로 치환
a.isna().sum()      # Column별로 NA값의 갯수를 구함
a.shape             # DataFrame의 크기 표현
len(list(a.columns))    # Data Column갯수
len(a)              # Data Row갯수

a.loc[0:3,['근무조','직책명']]    # R언어처럼 But, Row, Column명으로
a.iloc[1:3,3:5]     # R언어처럼 DataFrame Slice
a.iloc[:,3]
a.iloc[3,:]

a.columns       # Column명
a.values        # 각각의 값명
b=list(a.columns)
a[b[3]]
b



#------------ Data-Frame 연산
minmax_a = DataFrame( [a['직번'].min(), a['직번'].max()])
minmax_a = minmax_a.T
minmax_a.columns = ['A','B']

minmax_a.to_csv("D:/작업방/test.csv")
pd.read_csv("D:/작업방/test.csv")

a.groupby("공장/섹션").mean()       # sum:합계, count:갯수, cumsum(누적합계)
a.describe()  # Table내 숫자항목에 대한 Summary
a.info()    # Table 정보 : 클래스, 입력값의 수, 각 칼럼별 null값이 아닌 입력 값 수, 데이터 형태, 메모리 할당량

b=['직번','근속년']

a['직번'].mean(0)

a['공장/섹션'].value_counts()

c=['인사파트장단위명칭','인사직원직능자격명']
a[c+['직번']].groupby(c).sum()
a.groupby(c).mean()['직번']

df_a = a
df_a['Plus']=  list(range(0,len(a)))

df_ga = a.groupby(c).mean()[['직번','Plus']]


#------------ Data-Frame  다루기
        import time
        start_time = time.time()        # System 시간측정 Start

a = DataFrame( { 'A1': ['a0','a1','a2','a3','a3','a4'], 'A2': [NA,1,3,3,1, NA], 'A3' : [0,NA,2,NA,0,5] } )
b = DataFrame( { 'B1': ['b0','b1','a3','a4','c1','a3'], 'B2': [5,NA,1,5,NA,3], 'B3' : [NA,9,NA,6,5,8] } )
a2 = DataFrame( { 'A0': ['a0','a0','a0','a0','a1','a1','a1'], 'A1' : ['b1','b2','b1','b0','b2','b2','b0'], 'A2': [1,NA,4,NA,5,NA,3], 'A3' : [NA,9,NA,6,NA,8,NA] } )
c
c['A0'] = c['A2']+c['A3']
c
    # 정렬
#c.sort_index(axis=0, ascending=Flase)              # DataFrame내 Column명 정렬 / ascending = True 오름차순, False 내림차순
c.sort_values(by=['A2','A3'], ascending=True)      # 각 Column내 값의 정렬 / ascending = True 오름차순, False 내림차순
c[0:3]
    # Shift (Data Lag, Lead)
c.shift(2)      # n칸 뒤로 이동
c.shift(-2)     # n칸 앞으로 이동

    # Table 잇기 (rbind, cbind)
c
pd.concat( [c[1:3], c[4:6]] , axis = 0)     # Table 위아래로 잇기 (rbind, UNION ALL)  (Default)
pd.concat( [c[1:3], c[4:6]] , axis = 1)     # Table 옆으로 잇기 (cbind)

    # Merge ( Key가 같은경우 : on,  다른경우 : left_on, right_on )
a
b
pd.merge(a, b, left_on='A1', right_on='B1', how='inner')           # how=inner : inner join (Default)
pd.merge(a, b, left_on='A1', right_on='B1', how='outer')           # how=outer : outer join
pd.merge(a, b, left_on='A1', right_on='B1', how='left')         # how=Left : Left Table기준
pd.merge(a, b, left_on='A1', right_on='B1', how='right')           # how=right : right Table기준
        # 2개 이상 Key일경우
pd.merge(a, b, left_on=['A1','A2'], right_on=['B1','B2'], how='inner')           # how=inner : inner join (Default)

        end_time = time.time()              # System 시간측정 End
        print(end_time - start_time)

        #단일값 산출
Series(a['A2'].unique())

        #Index 설정
d =a
d.index = ['a','b','c','d','e','f']             # a table도 같이 바뀜



#----------------- NA, NULL값 처리
import numpy
from numpy import nan as NA

c = pd.read_clipboard()
c=DataFrame({ 'a': [NA, NA, 1,2, NA, NA,3,NA] , 'b' : [NA, 9,8,NA,7,6,NA,NA] })
c

    # 확인
c.isnull()      # NA, NULL값 검사
c.isnull().sum()    # NA값 갯수 확인
c.isnull().all()    # 모든값이 NA값인지 확인
c.isnull().any()    # NA값이 1개라도 있는지 확인

    # 삭제
c.dropna()      # NA값 하나라도 포함된 행 전체 삭제
c.dropna(how='all')      # 모든 행이 NA값인 행만 삭제
c.dropna(thresh =0)     # 같은 열 에서 NA가 아닌 값이 n개인 행 출력

    # 치환
c.fillna(0)         # 모든 NA 값을 0으로 치환
c.fillna(method = 'ffill', limit=1)      # NA값을 이전행 값으로 치환 / limit : 이후 행 몇개까지 적용할껀지
c.fillna(method = 'bfill', limit=1)      # NA값을 이후행 값으로 치환 / limit : 이전 행 몇개까지 적용할껀지
c.fillna(method = 'ffill').fillna(method='bfill')       # NA값을 앞뒤값으로 치환

    # 연산 : 일반적 연산의 경우 Na값 무시하고 자동연산
c.mean(skipna=False)        # skipna = False 설정시, Na를 연산에 포함 ( Na포함시 연산결과 Na로 리턴)



        # Data Frame내 Column별로 서로 묶기
lab =list(a['A1']) + list(b['B1'])
a['A1'] + " / " + b['B1']
e = DataFrame({"E" : Series(range(1,7))})
a['A1'] + " / " + b['B2'].astype(str) + " / " + e['E'].astype(str)

f = a
f['Index'] = f.index
a.index = range(1,7)
range( 1, Series(a.columns).count() )
a.iloc[:,1:3]

list(pd.date_range(start='2019-01-01', periods=7))
f['rank']=f['Index'].rank()

        # groupby + transform
a['A4'] = a.groupby("A1").transform("index")

a['A5']=Series( [1]*len(a) ).cumsum()
a['A5'] = a[['A1','A5']].groupby('A1').rank(ascending=False).astype(int)        # ascending = True : 오름차순, False : 내림차순

        #groupby + transform
a2.groupby('A0').shift(-1)
a2['A2'].groupby('A0').rank()
a2['A5']=Series( [1]*len(a2) ).cumsum()
a2['A6'] = a2[['A0','A1','A5']].groupby(['A0','A1']).rank(ascending=True).astype(int)        # ascending = True : 오름차순, False : 내림차순
a2.sort_values(by=['A0','A1'], ascending=True)
a2.groupby('A0').cumsum()
key = 'A0'
a2.groupby('A0').fillna(method='bfill', limit = 2)
a2.groupby('A0').fillna(method='ffill', limit = 2)
a2['A5']=Series( [1]*len(a2)


Series( a2['A0'].unique() )
a2.groupby('A0').sum()

# ------------------- 이동평균 구하기
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from numpy import nan as NA
import copy

a = pd.read_clipboard()  # Clipboard로 입력하기
a2 = DataFrame({'A0': ['a0', 'a0', 'a0', 'a0', 'a1', 'a1', 'a1'], 'A1': ['b1', 'b2', 'b1', 'b0', 'b2', 'b2', 'b0'],
                'A2': [1, NA, 4, NA, 5, NA, 3], 'A3': [NA, 9, NA, 6, NA, 8, NA]})

a2['A2'].rolling(window=3, min_periods=1).mean()
# window = 구간,   min_periods = 계산하는 최소 Data갯수,   center= 중심값기준
# skipna = False 설정시, Na를 연산에 포함 ( Na포함시 연산결과 Na로 리턴))
a2['A2'].rolling(window=3, min_periods=2).mean()
a2['A2'].rolling(window=3, min_periods=1, center=True).mean()
# win_type='gaussian', win_type='triang

b1 = copy.copy(a2)
b2 = a2['A3'].fillna(method="ffill").rolling(window=3, min_periods=1).mean()  # name = 'A3'
b3 = Series(a2['A3'].fillna(method="ffill").rolling(window=3, min_periods=1).mean(), name='B2')  # name = 'B2'

b1['B3'] = a2['A3'].fillna(method="ffill").rolling(window=3, min_periods=1).mean()
b0 = pd.concat([b1, b2, b3], axis=1)
b0



    # Module
import Module_P001 as md
md.my_funciton()
%run Module_P001.py     # 직접실행


    # Dir
import os
import getpass
os.getcwd()
os.chdir('D:\작업방\Python')
os.path.exists('파이썬 기본문법2.txt')
os.path.exists('Module_P001.py')
os.chdir('D:\Python')
dir(md)
type(md)

	# 데이터 형변환
a=int("2")  # 문자 → 숫자(정수형)
b=float("2") # 문자 → 숫자(실수형)
c=str(2)	# 숫자 → 문자
e=str( datetime.date.today())
e.replace("-","")


f_list = os.listdir('.')     # 현재디렉토리 파일 목록
 #os.mkdir()

os.getcwd()
origin_adr = os.getcwd() #'D:\\Python\\Python_Pjt001'

# Directory 변경
working_adr =  'C:\\Users' + '\\'+ username + '\Desktop'
os.chdir(working_adr)

username = getpass.getuser()    # 사용자명 얻기
folder_name = 'pypi'
file_name = 'python'

#dir_folder = 'C:\\Users' + '\\'+ username + '\Desktop' + '\\' + folder_name
#dir_file = 'C:\\Users' + '\\'+ username + '\Desktop' + '\\' + folder_name + '\\' + file_name
os.path.isdir(dir_folder)       # 해당경로에 파일/폴더가 있는지 확인
os.path.isdir(dir_file)       # 해당경로에 파일/폴더가 있는지 확인

if not os.path.isdir(folder_name):
    os.mkdir(folder_name)  # 폴더 만들기

os.chdir( '.\\'+ folder_name)       # 새로만든 폴더 안으로 경로변경
os.getcwd()

os.chdir(origin_adr)
os.getcwd()

# 해당폴더안에 원하는 파일명이 들어가있는지
f_list = os.listdir('.')     # 현재디렉토리 파일 목록
f_list
 #os.mkdir()
f_list.index("Practice_P001.py")        # List 내 원소의 위치(주소)
f_list.count("Practice_P001.py")        # List 내 원소의 갯수

name1 = 'Practice'
name2 = '001'

f_count0 = [f for f in f_list if f.count(name1) * f.count(name2) > 0]   # name1, name2 를 모두 포함하고 있는 파일명
len(f_count0)

f_count1 = ''
for f in f_list :
    if (f.count(name1) * f.count(name2) > 0) :
        f_count1 = 1
        break
    else : f_count1 = 0

name0 = 'ice_P001'
f_loc = [f_list.index(f) for f in f_list if f.count(name0)]    # name0를 포함하는 파일의 List내 위치
f_list[f_loc[0]]    # name0를 포함하는 파일의 명칭




    # Random
import random as rd
rd.randint(1,10)



#---------------- Numpy Library
import numpy as np
data = [0.1,2,3,4,0.5]
data_ar = np.array(data)
type(data)
type(data_ar)

data
data_ar
np.arange(12)
np.linspace(1,10,10)

#---------------- Pandas Library
import pandas as pd

data = pd.Series([1,2,3,4,5])
data
type(data)

a.lt(b)

#----------- Plot 관련 Package (matplotlib)
import numpy as np
from numpy import nan as NA
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

plt.close('all')

ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()







#------------------------------------------------------------------------------------------------------------
 #DB구축
#DBMS다운 : http://www.sqlite.org

 #------------------------------------------------------------------------------------------------------------

【 파이썬 Pandas Read, Write 함수 】
** Read Function**                    | ** Write Function **
 --------------------------------------+-------------------------------------
 read_csv                              | to_csv
 read_excel                            | to_excel
 read_hdf                              | to_hdf
 read_sql                              | to_sql
 read_json                             | to_json
 read_msgpack (experimental)           | to_msgpack
 read_html                             | to_html
 read_gbq (experimental)               | to_gbq
 read_stata                            | to_stata
 read_sas                              |
 read_clipboard                        | to_clipboard
 read_pickle                           | to_pickle


【 파이썬 Pandas DataFrame 함수 】
** view data **
df.head(n)    // 첫 데이터부터 일련의 n개의 데이터를 보여줍니다.
df.tail(n)    // 마지막 데이터부터 일련의 n개의 데이터를 보여줍니다.
df.index      // df 의 인덱스를 보여줍니다.
df.columns    // df 의 columns 보여주기
df.values     // df 를 구성하는 데이터들을 보여주기
df.describe() // df 의 통계적인 기초 요약들을 보여주기(count, mean, std, min, 25%, 50%, 75%, max)
 df.T          // (transpose) index와 columns 의 축을 바꿉니다.
df.sort_index(axis=1, ascending=False) // index 의 값을 기준으로 내림차순으로 정렬
df.sort_values(by='B')                 // column='B' 의 값을 기준으로 정렬


** selection **
df['A'] == df.A // column 'A' 의 값을 series로 반환합니다.
df[0:3]         // df 의 0번 row 부터 2번 row 까지 잘라서 df 로 반환합니다.
df['20130102':'20130105']
     // index 가 고유값을 가지고 있다면 그를 직접 지정해서 잘라 df로 반환합니다.
df.loc[:, [...]]
     // label을 통해서 범위를 선택할 수 있습니다. 앞은 row-label, 뒤는 column-label-list
     // row-label ( 여기서는 dates 로 series이므로 dates[n]으로 지정하거나, index 고유 값을 직접 지정할수 있습니다. )
     // column-label ( 선택하고자 하는 column label을 지정해 리스트로 만들어줍니다. )
df.iloc[:, :]
     // 값이 아닌 row 위치와 column 위치를 integet index로 지정하여 선택합니다.
     // 위치를 slicing 하거나 선택하기 원하는 위치를 리스트로 지정해 넣어줄수 있습니다.
df.at[labeling system], df.iat[positioning system]
     // .loc, .iloc 과 같으나 하나의 값만을 선택하는 방법

** boolean indexing **
 df[df.A > 0]         // column 'A' 의 값이 0 이상인 row 만 선택
df[df > 0]           // 내부의 값중에 0이상인 값만 보여주고 나머진 NaN 으로 보여준다
df[df.A.isin([...])] // 값이 고유값이라면 그값들의 유무로 선택을 결정한다

** Settimg **
 df['Column'] = Series([...], index=[...])
     // 새로운 column에 시리즈로 만든 배열을 index에 맞추어 data를 배치합니다.
df.at[label_index, 'column'] = value
     // at[], iat[]을 이용 위치를 직접 지정하여 수정, 대치합니다.
df.iat[row_num, column_num] = value
 df.loc[:, ['column']] = 배열           // 지정한 row에 배열내 값을 변경합니다.
df[boolean selection] = value         // boolean 에 맞추어 값을 변경합니다.

 ** Stats **
 df.mean(axis) // (): 각 column별 평균을 구합니다. (axis): 축에 따른 각 row별로 평균 구하기
df.min()      // column 별로 최소값을 구합니다.
df.max()      // column 별로 최대값을 구합니다.
df.std()      // column 별로 표준편차를 구합니다.
df.sub(s, axis=) // df 에서 s를 빼준다. (s 길이가 같은 series나 list, 그저 일반 수가 될수 있다)

 ** Histogramming **
 Series.value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True)
     // 히스토그램은 시리즈함수로 내부의 같은 결과값의 누적을 보여주는 함수입니다.
     // 한데.. 꼭 시리즈에 넣을 필요 없이
    // pandas.value_counts(data) 를 통해 그 값을 동일하게 반환 받을수 있습니다.
     // data는 numpy의 배열이 될수도 있고, 리스트가 될수도 있습니다.
Series.mode()  // series 내에서 가장 빈번하게, 혹은 많이 나온 값을 반환합니다.


 ** apply **
 df.apply(np.cumsum)   // numpy 가 가지는 함수를 끌어와서 계산하거나
df.apply(lambda x:)   // 파이썬 자체의 람다함수를 끌어와서 커스텀 함수를 넣어줄 수 있습니다.

 ** string methods **
 Obj.str.lower()  // obj 은 series, dataFrame 이 될수 있음
Obj.str.upper()
 Obj.str.len()
 Obj.str.strip()  // lstrip(), rstrip() left-, right-
Obj.str.replace('a', 'b')
 Obj.str.split(',') // rsplit('') r- : reverse


** group function **
# https://www.tutorialspoint.com/python_pandas/python_pandas_window_functions.htm
# [ .rolling() Function ] 	
# This function can be applied on a series of data. 
# Specify the window=n argument and apply the appropriate statistical function on top of it.
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(10, 4),
   index = pd.date_range('1/1/2000', periods=10),
   columns = ['A', 'B', 'C', 'D'])
print df.rolling(window=3).mean()


# [.expanding() Function]
# This function can be applied on a series of data. 
# Specify the min_periods=n argument and apply the appropriate statistical function on top of it.
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(10, 4),
   index = pd.date_range('1/1/2000', periods=10),
   columns = ['A', 'B', 'C', 'D'])
print df.expanding(min_periods=3).mean()


# [.ewm() Function]
# ewm is applied on a series of data.
# Specify any of the com, span, halflife argument and apply the appropriate statistical function on top of it. It assigns the weights exponentially.
import pandas as pd
import numpy as np
 
df = pd.DataFrame(np.random.randn(10, 4),
   index = pd.date_range('1/1/2000', periods=10),
   columns = ['A', 'B', 'C', 'D'])
print df.ewm(com=0.5).mean()



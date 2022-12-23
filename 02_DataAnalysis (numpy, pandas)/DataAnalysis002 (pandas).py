# -*- coding: UTF-8 -*-
#----------- pd.DataFrame 관련(Pandas)  # https://blog.naver.com/townpharm/220971613200 #
import numpy as np
import pandas as pd
from numpy import nan as NA
import copy
import re


# http://www.cheat-sheets.org/#Python
# https://sinxloud.com/python-cheat-sheet-beginner-advanced/    # 파이썬 주요 Cheet-Sheet
# https://dataitgirls2.github.io/10minutes2pandas/      # pandas 10분완성

 # JupitorNoteBook
# %who   # 현재까지 입력된 변수 보기
# del 변수    # 변수 삭제
# %reset   #모든변수삭제
# exit # ipython 종료

# pd.DataFrame에서 숫자항목을 string (object) → numeric (int, float) 로 변환하는 함수
def df_numerical (df, cols, type='float'):
    df_num = pd.DataFrame()     # 반환할 DF 정의
    for i in df.columns:
        if i in cols:
            try:
                df_num[i] = df[i].astype(type)
            except:
                try:
                    df_num[i] = df[i].astype('float')
                except:
                    df_num[i] = df[i]
        else:
            df_num[i] = df[i]
    return df_num

df_list = [[1,2,3,4],[5,6,7,8]]
df = pd.DataFrame(df_list)
df = pd.read_clipboard()  #Clipboard로 입력하기
df.to_clipboard()        #Clipboard로 내보내기


# pandas 표기법
pd.options.display.float_format = '{:.2f}'.format   # 과학적 표기법(Scientific notation)을 사용하지 않는 경우
pd.set_option('display.float_format', '{:.2e}'.format)  # 과학적 표기법(Scientific notation)
pd.set_option('display.float_format', '{:.2f}'.format) # 항상 float 형식으로
pd.set_option('display.float_format', '{:.2e}'.format) # 항상 사이언티픽
pd.set_option('display.float_format', '${:.2g}'.format)  # 적당히 알아서
pd.set_option('display.float_format', None) #지정한 표기법을 원래 상태로 돌리기: None


df['col1']
df.col1
list(df.index)       # index 확인하기


    # 데이터셋 정보
df.info()
df.shape    # number of (row, column)

    # 데이터 column별 타입 확인
df.dtypes
df.dtypes.value_counts()

    # 데이터셋 숫자형변수 summary
df.describe()


    # 샘플데이터 확인
df.sample(n=3)   # Table내 임의갯수 Sampling Display
df.head(n=3)     # Table내 앞부분 Sampling Display
df.tail(n=3)     # Table내 뒷부분 Sampling Display


# 데이터셋 변환
df_np = df.to_numpy() # 자동 형변환 int > float > object 으로 통일됨
    # 일반적으로 data_type을 동일한 set을 numpy로 바꾸어야 함



df.fillna(0)      # DataFrame안의 NA값을 0으로 치환
df['col_null'].fillna(0)      # Data-Frame의 '공장/섹션'열에 NA값을 0으로 치환
df.isna().sum()      # Column별로 NA값의 갯수를 구함
df.shape             # DataFrame의 크기 표현
len(list(df.columns))    # Data Column갯수
len(df)              # Data Row갯수

df.loc[0:3,['col1','col2']]    # R언어처럼 But, Row, Column명으로
df.iloc[1:3,3:5]     # R언어처럼 pd.DataFrame Slice
df.iloc[:,3]
df.iloc[3,:]

df.columns       # Column명
df.values        # 각각의 값명
df2=list(df.columns)
df[df2[3]]


# index, column name지정하기
df.index.name = 'index_name'
df.columns.name = 'columns_name'

#------------ Data-Frame 연산
minmax_a = pd.DataFrame( [df['col1'].min(), df['col1'].max()])
minmax_a = minmax_a.T
minmax_a.columns = ['df','df2']

minmax_a.to_csv("D:/작업방/test.csv")
pd.read_csv("D:/작업방/test.csv")


# ----------- Data indexing
df['col2']
df.loc[0:2, 'col3':'col6']
df.loc[0:2]
df.loc[:,'col3':'col6']

idx1 = df['col5'] >= 1
idx1
df[idx1]

idx4 = df.dtypes == 'object'   #  data type이 'object'인 것만 가져오기
df.iloc[:, idx4.array]         #
df.iloc[:, (df.dtypes == 'object').array]


# -------------------- Data indexing + 연산
df.groupby("col1").mean()       # sum:합계, count:갯수, cumsum(누적합계)
df.describe()  # Table내 숫자항목에 대한 Summary
df.info()    # Table 정보 : 클래스, 입력값의 수, 각 칼럼별 null값이 아닌 입력 값 수, 데이터 형태, 메모리 할당량

df2=['col1','col2']

df['col1'].mean(0)

df['col2'].value_counts()

df3=['col3','col4']
df[df3+['col5']].groupby(df3).sum()
df.groupby(df3).mean()['col6']

df_a = df
df_a['Plus']=  list(range(0,len(df)))

df_ga = df.groupby(df3).mean()[['col3','col4']]

# 평균계산 (전체, column별, row별)
df.mean().mean()    #  데이터 전체 평균 (nan무시)
df.mean(0)          # column별 평균 (nan무시)
    # df_contin.mean()
df.mean(1)          # row별 평균 (nan무시)



# mode() the value that appears most often. (최빈값)
# DataFrame.mode(self, axis=0, numeric_only=False, dropna=True) 
df['column'].mode()

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mode.html
df = pd.DataFrame([('bird', 2, 2),
                ('mammal', 4, np.nan),
                ('arthropod', 8, 0),
                ('bird', 2, np.nan)],
                index=('falcon', 'horse', 'spider', 'ostrich'),
                columns=('species', 'legs', 'wings'))

    #            species  legs  wings
    # falcon        bird     2    2.0
    # horse       mammal     4    NaN
    # spider   arthropod     8    0.0
    # ostrich       bird     2    NaN

    # By default, missing values are not considered, and the mode of wings are both 0 and 2. 
    # The second row of species and legs contains NaN, because they have only one mode, 
    # but the DataFrame has two rows.

df.mode()
    #   species  legs  wings
    # 0    bird   2.0    0.0
    # 1     NaN   NaN    2.0


    # Setting dropna=False NaN values are considered and they can be the mode (like for wings).
df.mode(dropna=False)
    #   species  legs  wings
    # 0    bird     2    NaN


    # Setting numeric_only=True, only the mode of numeric columns is computed, 
    # and columns of other types are ignored.
df.mode(numeric_only=True)
    #    legs  wings
    # 0   2.0    0.0
    # 1   NaN    2.0


    # To compute the mode over columns and not rows, use the axis parameter:
df.mode(axis='columns', numeric_only=True)
    #            0    1
    # falcon   2.0  NaN
    # horse    4.0  NaN
    # spider   0.0  8.0
    # ostrich  2.0  NaN






#------------ Data-Frame  다루기
# 공백제거 ----------------------------------------------------------------------------------------------------------------
a = [' 1',' a ',' c d ','e']

pd.Series(a).str.strip()    # 양쪽공백제거
pd.Series(a).str.lstrip()    # 왼쪽공백제거
pd.Series(a).str.rstrip()    # 오른쪽공백제거

#----------------------------------------------------------------------------------------------------------------------

import time
start_time = time.time()        # System 시간측정 Start

a = pd.DataFrame( { 'A1': ['a0','a1','a2','a3','a3','a4'], 'A2': [NA,1,3,3,1, NA], 'A3' : [0,NA,2,NA,0,5] } )
b = pd.DataFrame( { 'B1': ['b0','b1','a3','a4','c1','a3'], 'B2': [5,NA,1,5,NA,3], 'B3' : [NA,9,NA,6,5,8] } )
a2 = pd.DataFrame( { 'A0': ['a0','a0','a0','a0','a1','a1','a1'], 'A1' : ['b1','b2','b1','b0','b2','b2','b0'],
                'A2': [1,NA,4,NA,5,NA,3], 'A3' : [NA,9,NA,6,NA,8,NA] } )
c['A0'] = c['A2']+c['A3']

     # 정렬
#c.sort_index(axis=0, ascending=Flase)              # DataFrame내 Column명 정렬 / ascending = True 오름차순, False 내림차순
c.sort_values(by=['A2','A3'], ascending=True)      # 각 Column내 값의 정렬 / ascending = True 오름차순, False 내림차순
c.sort_values(by=['A2','A3'], ascending=[False, True])
c[0:3]
     # Shift (Data Lag, Lead)
c.shift(2)      # n칸 뒤로 이동
c.shift(-2)     # n칸 앞으로 이동

    # Table 잇기 (rbind, cbind)
pd.concat( [c[1:3], c[4:6]] , axis = 0)     # Table 위아래로 잇기 (rbind, UNION ALL)  (Default)
pd.concat( [c[1:3], c[4:6]] , axis = 1)     # Table 옆으로 잇기 (cbind)

     # Merge ( Key가 같은경우 : on,  다른경우 : left_on, right_on )
pd.merge(a, b, left_on='A1', right_on='B1', how='inner')           # how=inner : inner join (Default)
pd.merge(a, b, left_on='A1', right_on='B1', how='outer')           # how=outer : outer join
pd.merge(a, b, left_on='A1', right_on='B1', how='left')         # how=Left : Left Table기준
pd.merge(a, b, left_on='A1', right_on='B1', how='right')           # how=right : right Table기준
        # 2개 이상 Key일경우
pd.merge(a, b, left_on=['A1','A2'], right_on=['B1','B2'], how='inner')           # how=inner : inner join (Default)

end_time = time.time()              # System 시간측정 End
print(end_time - start_time)

         #단일값 산출
pd.Series(a['A2'].unique())

         #Index 설정
d = a  # Table 참조
d.index = ['a','b','c','d','e','f']             # a table도 같이 바뀜


#----------------- NA, NULL값 처리
import numpy
from numpy import nan as NA

c = pd.read_clipboard()
c=pd.DataFrame({ 'a': [NA, NA, 1,2, NA, NA,3,NA] , 'b' : [NA, 9,8,NA,7,6,NA,NA] })


     # 확인
c.isnull()      # NA, NULL값 검사
c.isnull().sum()    # NA값 갯수 확인
c.isnull().all()    # 모든값이 NA값인지 확인
c.isnull().any()    # NA값이 1개라도 있는지 확인

    # 결측치 확인
c.isnull().sum()  # column별로 결측치 갯수를 나타내줌
c.isnull().sum(1) # row별로 결측치 갯수를 나타내줌

    # 삭제
c.dropna()      # NA값 하나라도 포함된 행 전체 삭제
c.dropna(how='all')      # 모든 행이 NA값인 행만 삭제
c.dropna(thresh =0)     # 같은 열 에서 NA가 아닌 값이 n개인 행 출력

    # 치환
c.fillna(0)         # 모든 NA 값을 0으로 치환
c.fillna(method = 'ffill', limit=1)      # NA값을 이전행 값으로 치환 / limit : 이후 행 몇개까지 적용할껀지
c.fillna(method = 'bfill', limit=1)      # NA값을 이후행 값으로 치환 / limit : 이전 행 몇개까지 적용할껀지
c.fillna(method = 'ffill').fillna(method='bfill')       # NA값을 앞뒤값으로 치환


    # 연속형 변수의 NULL값을 평균으로 채우기
idx_contin = df.dtypes=='float64'   # 연속형 편수 선택
df_contin=df.loc[:,idx_contin]
print(df_contin)
df_contin_fill = df_contin.fillna(df_contin.mean())
print(df_contin_fill)

    # categorical 변수를 mode(최빈값)으로 채우기
idx_cat = df.dtypes=='object'
idx_cat                                 
df_cat = df.loc[:,idx_cat]
print(df_cat)
df_cat


df['col1'].value_counts()

    # mode() the value that appears most often. (최빈값)
    # DataFrame.mode(self, axis=0, numeric_only=False, dropna=True) 
cat_mode = df_cat.mode().iloc[0]
df_cat.fillna(cat_mode)



    # 연산 : 일반적 연산의 경우 Na값 무시하고 자동연산
c.mean(skipna=False)        # skipna = False 설정시, Na를 연산에 포함 ( Na포함시 연산결과 Na로 리턴)


# 특정값으로의 치환
# df.at(치환조건에 만족하는 index, 치환column) = '치환내용'
df.at[df[df['치환할Column'] == i].index, '치환할Column'] = '치환내용'


         # Data Frame내 Column별로 서로 묶기
lab =list(a['A1']) + list(b['B1'])
a['A1'] + " / " + b['B1']
e = pd.DataFrame({"E" : pd.Series(range(1,7))})
a['A1'] + " / " + b['B2'].astype(str) + " / " + e['E'].astype(str)

f = a
f['Index'] = f.index
a.index = range(1,7)
range( 1, pd.Series(a.columns).count() )
a.iloc[:,1:3]

list(pd.date_range(start='2019-01-01', periods=7))
f['rank']=f['Index'].rank()

        # groupby + transform
a['A4'] = a.groupby("A1").transform("index")

a['A5']=pd.Series( [1]*len(a) ).cumsum()
a['A5'] = a[['A1','A5']].groupby('A1').rank(ascending=False).astype(int)        # ascending = True : 오름차순, False : 내림차순

    #groupby + transform
a2.groupby('A0').shift(-1)
a2['A2'].groupby('A0').rank()
a2['A5']=pd.Series( [1]*len(a2) ).cumsum()
a2['A6'] = a2[['A0','A1','A5']].groupby(['A0','A1']).rank(ascending=True).astype(int)        # ascending = True : 오름차순, False : 내림차순
a2.sort_values(by=['A0','A1'], ascending=True)
a2.groupby('A0').cumsum()
key = 'A0'
a2.groupby('A0').fillna(method='bfill', limit = 2)
a2.groupby('A0').fillna(method='ffill', limit = 2)
a2['A5']=pd.Series( [1]*len(a2))

pd.Series( a2['A0'].unique() )
a2.groupby('A0').sum()

# ------------------- 이동평균 구하기

a = pd.read_clipboard()  # Clipboard로 입력하기
a2 = pd.DataFrame({'A0': ['a0', 'a0', 'a0', 'a0', 'a1', 'a1', 'a1'], 'A1': ['b1', 'b2', 'b1', 'b0', 'b2', 'b2', 'b0'],
                'A2': [1, NA, 4, NA, 5, NA, 3], 'A3': [NA, 9, NA, 6, NA, 8, NA]})

a2['A2'].rolling(window=3, min_periods=1).mean()
# window = 구간,   min_periods = 계산하는 최소 Data갯수,   center= 중심값기준
# skipna = False 설정시, Na를 연산에 포함 ( Na포함시 연산결과 Na로 리턴))
a2['A2'].rolling(window=3, min_periods=2).mean()
a2['A2'].rolling(window=3, min_periods=1, center=True).mean()

b1 = copy.copy(a2)
b2 = a2['A3'].fillna(method="ffill").rolling(window=3, min_periods=1).mean()  # name = 'A3'
b3 = pd.Series(a2['A3'].fillna(method="ffill").rolling(window=3, min_periods=1).mean(), name='B2')  # name = 'B2'

b1['B3'] = a2['A3'].fillna(method="ffill").rolling(window=3, min_periods=1).mean()
b0 = pd.concat([b1, b2, b3], axis=1)


# 데이터 형변환
a=int("2")  # 문자 → 숫자(정수형)
b=float("2") # 문자 → 숫자(실수형)
c=str(2)# 숫자 → 문자
e=str( datetime.date.today())
e.replace("-","")

    # Random
import random as rd
rd.randint(1,10)

 
 
 # 파일 읽기 쓰기 ---------------------------------------------------------------------------------------------------------------------------
# Pandas 한글로 csv문서 읽기쓰기
# encoding = 'cp949'
a.to_csv(fileName, index=False, encoding='cp949')
b = pd.read_csv(fileName, encoding='cp949')
# encoding = 'ms949'
a.to_csv(fileName, index=False, encoding='ms949')
pd.read_csv(fileName, engine='python', encoding='ms949')

    # Random
import random as rd
rd.randint(1,10)


#----------- 파일로 쓰고 읽기 (csv)
import os

os.getcwd()
os.listdir('.')     # 현재디렉토리 파일 목록

dataset = [{'col1':'a','col2':1},{'col1':'b','col2':2}]
dataset2 = [{'col1':'c','col2':3},{'col1':'d','col2':4}]
df_dataset = pd.DataFrame(dataset)
df_dataset2 = pd.DataFrame(dataset2)

df_dataset.to_csv('df_dataset.csv',index=False)        # csv파일로 쓰기

read_df = pd.read_csv('df_dataset.csv')

df_dataset.append(df_dataset2)

 

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
import seaborn as sns
from plotnine import *
matplotlib.style.use('ggplot')

plt.close('all')

ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
x = [x for x in range(1,11)]
y = [x for x in range(0,19,2)]
z = [x**2 for x in y]

plt.plot(x, y, color='b', marker='o', linestyle='')

flights = sns.load_dataset('flights')
flights

(ggplot(flights, aes(x='year', y='passengers'))\
+geom_point())
















#------------------------------------------------------------------------------------------------------------
"""
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


【 파이썬 Pandas pd.DataFrame 함수 】 
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
df['Column'] = pd.Series([...], index=[...]) 
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
pd.Series.value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True)
      // 히스토그램은 시리즈함수로 내부의 같은 결과값의 누적을 보여주는 함수입니다. 
      // 한데.. 꼭 시리즈에 넣을 필요 없이 
    // pandas.value_counts(data) 를 통해 그 값을 동일하게 반환 받을수 있습니다.
      // data는 numpy의 배열이 될수도 있고, 리스트가 될수도 있습니다.
pd.Series.mode()  // series 내에서 가장 빈번하게, 혹은 많이 나온 값을 반환합니다.


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


** Formating **
코드	설명
%s	문자열(String)
%c	문자 1개(character)
%d	정수(Integer)
%f	부동소수(floating-point)
%o	8진수
%x	16진수
%%	Literal % (문자 % 자체)
"""
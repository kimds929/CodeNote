# 【 Install Module 】-----------------------------------------------------------------------------------------------------
# $ conda config --set ssl_verify no

# $ conda install ipykernel
# $ conda install ipython
# $ conda install numpy
# $ conda install pandas
# $ conda install matplotlib
#     $ conda install -c conda-forge matplotlib
# $ conda install seaborn
# $ conda install scipy
# $ conda install statsmodels

# $ conda install -c conda-forge missingno
# $ conda install -c conda-forge pandas-profiling

# $ conda install six

# -----------------------------------------------------------------------------------------------------


# import numpy as np
# import pandas as pd

# import missingno as msno
# import pandas_profiling as pd_report

# import matplotlib.pyplot as plt
# import seaborn as sns

# import scipy
# import statsmodels.api as sm

# from six.moves import cPickle

data_url = 'https://raw.githubusercontent.com/kimds929/CodeNote/main/00_DataAnalysis_Basic/'


# 【 Python Basic 】 ==========================================================================
# print --------------------------------------------------------------------------------
print('Hello, Python')


# variable --------------------------------------------------------------------------------
a = 20
print(a)
a

b = 6
print(b)

a + b
print(a+b)

c = a + b
print(c)


w = 'Hello'
print(w)

p = 'Python'

print(w + p)
# a + w   # error


# f-string : 변수들로 이루어진 문자를 사용자가 쉽게 표현하기 위한 방법  ★★★★★
# (여러 포매팅 방법 중 f-string이 가장 쉽고 강력함)
print(w + ' ' + p)
print(f'{w} {p}')

# print(a + ' ' + w)
print(f'{a} {w}')





# operation (사칙연산) --------------------------------------------------------------------------------
a + b

a - b

a * b

a / b

a // b      # 몫

a % b       # 나머지

a ** 2      # 제곱
a ** b

# ========================================================================================
# (list, dictionary 기초 응용 및 실습  *pandas, matplotlib 연계) ==========================
test_df

test_df['x1'] + test_df['x3']

test_df['x1'] % test_df['x3']
test_df['x1'] // test_df['x3']

test_df['x1'] ** 2

# =========================================================================================



# data-structure --------------------------------------------------------------------------------
# numeric: 숫자형 / string: 문자형  (object: 문자형 집합체)
#   numeric → int: 정수, float: 소수

# numeric → string

a1 = 10
type(a1)

a2 = 20.3
type(a2)

a3 = 'Hello'
type(a3)

a4 = '1'
type(a4)


# list : 여러개의 자료를 하나의 변수에 넣는 방식
l1 = [1,2,3]
l1
print(l1)

l2 = ['a', 1, 2]
l2
print(l2)

# list 원소(데이터) 접근 방법 (일반적으로 프로그래밍은 0부터 시작)
l1[0] 
l1[1]
l1[2]
l1[0:2]


# 원소추가                # ★★★★★
l2 = [3,4,5]
l2

l2.append(6)
l2

# List 이어 붙이기                # ★★
l1 + l2


# tuple : 값수정이 불가능한 list
l3 = [1,7,4]
l3[0] = 3
l3

t1 = (1,7,4)
t1[0] = 3
t1

# tuple 원소(데이터) 접근 방법
t1[0]
t1[0:2]






# dictionary: 여러개의 자료에 key와 value를 지정하여 저장하는 방식
# key를 통해서만 데이터 접근이 가능
d1 = {'a':1, 'b':3, 'c':'AAA'}
d1

# dictionary 원소(데이터) 접근 방법
d1['a']
d1['b']
d1['c']

# d1['AAA']     # error


d2 = {'a': [1,2,3], 'b':['A', 'B','C']}
d2
d2['a']
d2['b']

# 원소추가                # ★★★★★
d2['c'] = [9,8,7]
d2

# Key / Value                # ★★★★
d2.keys()
d2.values()
d2.items()



# ========================================================================================
# (list, dictionary 기초 응용 및 실습  *pandas, matplotlib 연계) ==========================

# list → Pandas Data
l1
pd.Series( l1 )

plt.hist(l1)        # histogram



[l1, l3]
pd.DataFrame( [l1, l3] )

plt.scatter(l1, l3)     # scatterplot


# dictionary → Pandas Data
d1
pd.Series(d1)

d2
pd.DataFrame(d2)

plt.scatter(d2['a'], d2['c'])     # scatterplot
plt.xticks(d2['a'], d2['b'])    # x축 label변경

# =========================================================================================






# 제어문 # ★★★★★--------------------------------------------------------------------------------
abc = 10
if abc >= 20:
    print('abc: 20 이상')
elif abc >= 10:
    print('abc: 10 ~ 19')
else:
    print('abc: 10미만')


# 루프문 # ★★★★★--------------------------------------------------------------------------------
for i in [10, 15, 20]:
    print(i)


# 루프문 강제 탈출
for i in [10, 15, 20]:
    if i > 17:
        break
    else:
        print(i)


# 함수 # ★★★★★ --------------------------------------------------------------------------------
def add_f(x, y):
    return x + y

add_f(10, 20)

def hello_printing():
    print('Hello')

hello_printing()


#초기값설정
def add_f2(x, y=10):
    return x+y

add_f2(10, 20)
add_f2(10)



# Cpk 함수 만들기
# (상하한이 모두 있을경우) Cpk = min(usl - mean, mean - lsl) / (3 * std)
def cpk(mean, std, lsl, usl):
    cpk_value = min(usl - mean, mean - lsl) / (3 * std)
    return cpk_value

# 함수실행
cpk(mean = 15, std=3.5, lsl=10, usl=18)
cpk(mean = 15, std=3.5, lsl=12, usl=19)



# if문 응용 실습
# (상한만 있을경우)      cpk_value = (usl - mean) / (3 * std) 
# (하한만 있을경우)      cpk_value = (mean - lsl) / (3 * std)
# (상하한 둘다 있을경우) cpk_value = min(usl - mean,  mean - lsl) / (3 * std)
def cpk(mean, std, lsl=None, usl=None):
    if lsl is None:
        cpk_value = (usl - mean) / (3 * std) 
    elif usl is None:
        cpk_value = (mean - lsl) / (3 * std)
    else:
        cpk_value = min(usl - mean,  mean - lsl) / (3 * std)

    return cpk_value


# 초기값 지정
def cpk(mean, std, lsl=-np.inf, usl=np.inf):
    return min(usl - mean, mean - lsl) / (3 * std)


# 함수실행
cpk(mean = 15, std=3.5, lsl=10, usl=18)
cpk(mean = 15, std=3.5, lsl=11)
cpk(mean = 15, std=3.5, usl=21)




# ==============================================================================================
# (if, for, function 응용 및 실습  *pandas, matplotlib 연계) =====================================
# function 실습 ***
# 어떤 series(s)가 주어졌을때, 해당 series의 평균을 구하는 함수를 만들자 (함수명: series_mean)
def series_mean(s):
    return s.mean()

# 어떤 series(s)가 주어졌을때, 해당 series의 3sigma Range를 구하는 함수를 만들자 (함수명: series_sigma)
def series_sigma(s):
    s_mean = s.mean()
    s_std = s.std()
    
    simga_plus3 = round(s_mean + 3 * s_std,1)
    simga_minus3 = round(s_mean - 3 * s_std,1)
    return f"{simga_minus3} ~ {simga_plus3}"

steel_df = pd.read_csv(data_url + "steel_simple.csv")
steel_df.shape      # (8, 5)
steel_df


# 위에서 구한 함수를 가지고 steel_df 데이터의 YP값의 평균을 구해보자
series_mean(steel_df['YP'])

# 위에서 구한 함수를 가지고 steel_df 데이터의 생산공장별 YP값의 평균을 구해보자
steel_df.groupby('생산공장')['YP'].mean()
steel_df.groupby('생산공장')['YP'].agg('mean')
steel_df.groupby('생산공장')['YP'].agg(series_mean)

# 위에서 구한 함수를 가지고 steel_df 데이터의 생산공장별 YP값의 평균,편차, 3sigma을 구해보자
steel_df.groupby('생산공장')['YP'].agg(['mean','std', series_sigma])




# function / if문 실습 ***
# 어떤값(x)가 주어졌을때, 
#    . 그값이 10보다 크거나 같으면: True
#    . 그값이 10보다 작으면: False
# 를 리턴(return)하는 함수를 만들어보자 (함수명 is_over_10)

def is_over_10(x):
    if x >= 10:
        return True
    else:
        return False


# for문 실습 ***
# itterrows 함수를 사용하여 test_df의 x3열의 각 행마다 값이 
# 10보다 크거나 같은수인지(True, False)를 저장하는 ①list, ②dictionary(key: index, value:x3값)를 생성해라

for row in test_df.iterrows():
    print(row)
    # break
row
row[0]  # index
row[1]  # row_series_data


result_list = []     # 결과 저장용 빈 list
result_dict = {}     # 결과 저장용 빈 dictionary
for row in test_df.iterrows():
    row_index = row[0]
    row_data = row[1]
    # print(row_data['x3'])
    
    # print(is_over_10( row_data['x3'] ))
    result_10 = is_over_10( row_data['x3'] )
    
    result_list.append(result_10)
    result_dict[row_index] = result_10


result_list
result_dict
# pd.Series(result_dict)
# test_df['x3'] > 10


# apply함수 + lambda식  # ★★★★★
test_df['x3']
test_df['x3'].apply(lambda x: is_over_10(x) )



# Cpk 함수 만들기 ---------------------
# 어떤 Sereis 를 입력받아서 공정능력지수(cpk)를 리턴하는 함수를 만들어라 (기존 cpk 함수 활용가능)
# test_df의 x3열에 대한 cpk를 구해보자 (lsl, usl 마음대로 대입)

def cpk_series(x, lsl=-np.inf, usl=np.inf):
    x_mean = x.mean()
    x_std = x.std()
    
    return cpk(mean=x_mean, std=x_std, lsl=lsl, usl=usl)

cpk_series(test_df['x3'], lsl=5)
cpk_series(test_df['x3'], usl=13)
cpk_series(test_df['x3'], lsl=5, usl=13)
cpk_series(test_df['x3'], lsl=5, usl=11)


# Cpk_series함수 응용
# steel_df의 생산공장별 YP공정능력(하한: 140, 상한: 210)을 구해서 Series형태로 구해보자
result_dict = {}
for fac in ['1공장', '2공장']:
    df_fac = steel_df[steel_df['생산공장'] == fac]
    cpk_fac = cpk_series(df_fac['YP'], lsl=140, usl=210)
    
    result_dict[fac] = cpk_fac
pd.Series(result_dict)

steel_df.groupby(['생산공장'])['YP'].apply(lambda x: cpk_series(x, lsl=140, usl=210))
# =========================================================================================












# Class ★★★ --------------------------------------------------------------------------------
class MakeCar():
    def __init__(self, oil):
        self.tank = oil
       
    def go(self, distance):
        self.tank = self.tank - distance

c1 = MakeCar(100)
c1.tank
c1.go(60)
c1.tank

c2 = MakeCar(100)
c2.tank


c1.go(60)
c1.tank
c2.tank



class MakeCar():
    def __init__(self, oil):
        self.tank = oil
       
    def go(self, distance):
        if self.tank - distance < 0:
            print('기름이 없어 앞으로 갈 수 없습니다. 기름을 충전해주세요. 남은기름량:',self.tank)
        else:
            self.tank = self.tank - distance
    
    def charge(self, oil):
        if self.tank + oil > 100:
            print('100을 초과하여 충전할 수 없습니다. 충전량은 100으로 고정됩니다.')
            self.tank = 100
        else:
            self.tank = self.tank + oil

c1 = MakeCar(100)
c1.tank
c1.go(60)
c1.tank
c1.go(60)
c1.tank

c1.charge(60)
c1.tank
c1.charge(60)
c1.tank



# Cpk Class만들기
class CPK():
    def __init__(self, lsl=None, usl=None):
        self.lsl = lsl
        self.usl = usl
    
    def calculate(self, mean, std):
        if self.lsl is None:
            cpk_result = (self.usl - mean) / (3 * std) 
        elif self.usl is None:
            cpk_result = (mean - self.lsl) / (3 * std)
        else:
            cpk_result = min(self.usl - mean, mean - self.lsl) / (3 * std)
        
        self.result = cpk_result
        return self.result


cpk1 = Cpk(lsl=10, usl=18)
cpk1.calculate(mean=15, std=3.5)
cpk1.result

cpk2 = Cpk(lsl=11)
cpk2.calculate(mean=15, std=3.5)
cpk2.result

cpk3 = Cpk(usl=21)
cpk3.calculate(mean=15, std=3.5)
cpk3.result


# function?
# dir()

# ========================================================================================
# (class) 응용 및 실습  *pandas, matplotlib 연계) =========================================
# CpkSeries 클래스 만들기
#  ① CpkSeries 클래스를 활용해 steel_df의 YP값의 cpk를 구하자 (lsl = 140, usl=210)
#  ① CpkSeries 클래스를 활용해 steel_df의 생산공장별 YP값의 cpk를 구하자 (lsl = 140, usl=210)
class CPK_Analysis:
    def __init__(self, lsl=-np.inf, usl=np.inf):
       pass
    
    def cpk(self, x):
        pass
        # return cpk_value
    
    def plot(self, x):
        pass
        # return cpk_plot
        

class CPK_Analysis:
    def __init__(self, lsl=-np.inf, usl=np.inf):
       self.lsl = lsl
       self.usl = usl
    
    def cpk(self, x):
        self.x_mean = x.mean()
        self.x_std = x.std()
        cpk_result = min(self.usl - self.x_mean, self.x_mean - self.lsl) / (3 * self.x_std)
        
        return cpk_result

    def plot(self, x):
        line_df = cpk_line(x)
        
        cpk_plot = plt.figure()
        plt.hist(x)
        plt.plot(line_df.iloc[:,0], line_df.iloc[:,1], color='blue')
        
        for l in [self.lsl, self.usl]:
            plt.axvline(l, color='red', ls='--', alpha=0.5)
        plt.close()

        return cpk_plot
    
cs = CPK_Analysis(lsl = 140, usl=210)
cs.cpk(steel_df['YP'])
cs.plot(steel_df['YP'])


steel_df.groupby(['생산공장'])['YP'].apply(lambda x: cs.cpk(x))
steel_df.groupby(['생산공장'])['YP'].agg(cs.cpk)
steel_df.groupby(['생산공장'])['YP'].agg(cs.plot)
# ===============================================================================================








# Module(Library Import) ---------------------------------------------------------------

# import module
import math

# module 사용
#   ※ dot(.): 라이브러리의 변수 또는 함수를 호출하여 사용할때 사용하는 명령어
math.pi               # math Module의 pi라는 '변수'에 접근하여 사용
math.log10(100)     # math Module의 log10 이라는 '함수'에 접근하여 해당 함수를 사용


# module alias
import math as mt

mt.log10(1000)



# Library에서 특정 Module / 함수 / 변수만을 불러서 쓰고 싶을때
# log10(1/10)   # error

from math import log10
log10(1/10)



# Install Module
# conda install ....
# pip install ....

# conda install pandas
# pip install pandas




# More About Python...
# https://wikidocs.net/book/1
# https://www.kaggle.com/learn/overview






# 【 Pandas Basic 】 ==========================================================================
# import pandas as pd

# pandas? : DataFrame을 다루는데 특화된 Python Library
#   pandas is a fast, powerful, flexible and easy to use open source data analysis and
#   manipulation tool, built on top of the Python programming language.


# Import Modules ------------------------------------------------------------------------------------------
import pandas as pd


# ● Series 생성
# Series 생성 - List로부터 생성하기
l1 = ['a', 'b','c']
l2 = [1, 2, 3]
t1 = (2,3,4)


pd.Series(l1)        # index 자동부여
pd.Series(l1)        # index 자동부여

s1 = pd.Series(l1, name='ABC')
s1
s1[0]

s2 = pd.Series(l1, name='ABC', index=l2)
s2
s2[0]
s2[1]

s3 = pd.Series(t1, name='ABC', index=l2)
s3

# Series 생성 - Dictionary로부터 생성하기
di1 = {'a': 8, 'b':9, 'c': 0}
di1
s4 = pd.Series(di1)       # key: index, value:value
s4

di2 = {1: 'e', 2:'f', 3: 'g'}
s5 = pd.Series(di2)
s5



# ● DataFrame 생성
# DataFrame - List로부터 생성하기
l3 = [[1,2,3],[4,5,6]]
pd.DataFrame(l3)
pd.DataFrame(l3, columns=['A1','A2','A3'])
pd.DataFrame(l3, columns=['A1','A2','A3'], index=['a','b'])

# DataFrame - Dictionary로부터 생성하기
di2 = {'A1':[1,2,3], 'A2':[4,5,6]}
pd.DataFrame(di2)

di3 = {'A1':{'a':1, 'b':2}, 'A2':{'a':3, 'b':4}}
pd.DataFrame(di3)



# Data Save/Load from file ---------------------------------------------
# csv
df = pd.read_csv(data_url + 'test_data.csv')
df.to_csv(path + '/test_df2.csv', index=False)
df.shape
df

# ※ Excel
# df.to_excel("output.xlsx")
# df.to_excel("output.xlsx", sheet_name='Sheet_name_1')  

# pd.read_excel('tmp.xlsx', sheet_name='Sheet1', header=None, index_col=0)  
# pd.read_excel(open('tmp.xlsx', 'rb'), sheet_name='Sheet3')

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html


df_wine = pd.read_csv(data_url + 'wine_aroma.csv')
df_wine.shape
df_wine
# df_wine = pd.read_clipboard()



# Data Paste / Copy -----------------------------------------------------
# Data Load from clipboard
# df = pd.read_clipboard()
#   ※ dot(.): 라이브러리의 함수를 호출하여 사용

# Data copy to clipboard
# df.to_clipboard()
# df.to_clipboard(index=False)

df


# Data Explore -----------------------------------------------------
# info
df.info()

# shape
df.shape

# date type
df.dtypes

# numeric variable summary
df.describe()
df.describe().T


# sample data display
df.sample(3)

df.head()
df.head(3)

df.tail()
df.tail(3)

df.nunique()



# Column_selection ------------------------------------------------------
df
    # One column selection
df['x1']        # Series
df[['x1']]      # DataFrame

c1 = df['x1']
c1



    # Multi column selection
df[['x1', 'x3']]
c2 = df[['x1', 'x3']]
c2


# index만 추출하기
df.index

# column명만 추출하기
df.columns




# ※ Pandas 이해하기 ----------------------------------
# Series ?
pd.Series([1,2,3,4,5], name='abc')
series = pd.Series([1,2,3,4,5], name='abc')

# DataFrame ?
series.to_frame()
pd.DataFrame([1,2,3,4,5], columns=['abc'])


# DataFrame → Series 
d1 = df[['x1']]
d1

d1['x1']

# Series → DataFrame
s1 = df['x1']
s1

s1.to_frame()


# Series Selection
s1[0]
s1[3]
s1[0:3]



# Calculation (Operation) ------------------------------------------------------------
df
# Column끼리 연산
df['x1'] + df['x3']

# 연산된 값으로 새로운 열추가
df2 = df.copy()
df2

df2['x5'] = df2['x1'] + df2['x3']
df2


# Operation
df['x1']
df['x1'].count()
df['x1'].sum()
df['x1'].mean()
df['x1'].std()

df['x1'].median()

df['x1'].agg('mean')
df['x1'].agg(['count', 'mean', 'std'])

sum(df['x1'])


# apply함수
df['x2']
df['x2'].apply(lambda x: x=='a')

test_df.apply(lambda x: x['x2'],axis=1)
test_df.apply(lambda x: x['x1']+ x['x3'],axis=1)










# Filtering ------------------------------------------------------------
df

df['x2']=='a'
df[df['x2']=='a']

df['x1'] > 2
df[df['x1'] > 2]
df[df['x1'] > 2][['x1', 'x2']]

filter1 = df[df['x1'] > 2]
filter1[filter1['x1']<5]

filter2 = filter1[filter1['x1']<5]
filter2

filter2[['x4']]


# and: &
df[(df['x1'] > 2) & (df['x1'] < 5)]

# or: |
df[(df['x1'] < 3) | (df['x1'] > 4)]

# not: ~
df[~(df['x4'] == 'g3')]
df[df['x4'] != 'g3']

# in:
df[df['x4'].isin(['g1', 'g2'])]
df[~df['x4'].isin(['g1', 'g2'])]      # not in

# like:
df['x4'].str.contains('g1')
df['x4'].str.contains('g2')
df['x4'].str.contains('g')

df[df['x4'].str.contains('g')]
df[df['x4'].str.contains('1')]
df[~df['x4'].str.contains('1')]


# query filter
df.query("x4 == 'g1'")
df.query("x1 > 2 & x4== 'g1'")




# Sort -----------------------------------------------------------------
df.sort_values('x4')                   # 오름차순
df.sort_values('x4', ascending=False)  # 내림차순

# 복수조건
df.sort_values(['x4', 'x3'])

df.sort_values(['x4', 'x3'], ascending=[True, False])


# index
df.sort_index(ascending=False) 




# reset_index
df_y = df.set_index('y')
df_y.reset_index()
df_y.reset_index(drop=True)

# set_index
df.set_index('y')





# Concat & Merge --------------------------------------------------------
dict1 = {'A':[1,2,3], 'B':['a','b','c']}
dict2 = {'A':[9,8,7], 'B':['b','c','a']}
dict3 = {'B':['a','c','d'], 'C':[9,8,7]}
dict4 = {'D':['b','d','a'], 'C':[4,3,6]}

df1 = pd.DataFrame(dict1)
df2 = pd.DataFrame(dict2)
df3 = pd.DataFrame(dict3)
df4 = pd.DataFrame(dict4)


# concat
pd.concat([df1, df2], axis=0)
pd.concat([df1, df2], axis=1)

pd.concat([df1, df3], axis=0)
pd.concat([df1, df3], axis=1)


# merge (join)
pd.merge(df1, df2, on='B')

pd.merge(df1, df3, on='B')
pd.merge(df1, df3, on='B', how='left')
pd.merge(df1, df3, on='B', how='right')
pd.merge(df1, df3, on='B', how='outer')

pd.merge(df1, df4, left_on='B', right_on='D', how='inner')
pd.merge(df1, df4, left_on='B', right_on='D', how='outer')




# Groupby (pivot-table) ----------------------------------------------------------------
df_group = df.groupby('x2')

df_group.mean()
df_group[['x1']].mean()
df_group[['x1','x3']].mean()

df_group.agg(['mean', 'std']).to_clipboard()
df_group.agg({'x1':'mean', 'x3':'std', 'y':['min','max']})



# unstack, stack
df.groupby(['x2', 'x4'])['X1'].count()
df.groupby(['x2', 'x4'])['X1'].count().unstack('x2')


# groupby + for문  ***
q_dict = {}
for gi, gv in df.groupby(['x2']):
    q_dict[gi] = gv['x1'].quantile(0.5)
q_dict
pd.Series(q_dict)


q_dict = {}
for gi, gv in df.groupby(['x2']):
    q10 =  gv['x1'].quantile(0.1)
    q50 =  gv['x1'].quantile(0.5)
    q90 =  gv['x1'].quantile(0.9)
    
    q_dict[gi] = [q10, q50, q90]
q_dict
pd.DataFrame(q_dict, index=['q10','q50','q90'])
    

q_dict = {}
for gi, gv in df.groupby(['x2']):
    q_dict[gi] = []
    for q in [0.1, 0.5, 0.9]:
        q_ =  gv['x1'].quantile(q)
        q_dict[gi].append(q_)
q_dict
pd.DataFrame(q_dict, index=['q10','q50','q90'])



# More About pandas...
# https://dataitgirls2.github.io/10minutes2pandas/










# 【 EDA(Exploratory Data Analysis, 탐색적 데이터 분석) 】=============================================
# import numpy as np
# import missingno as msno
# import pandas_profiling as pd_report

import pandas as pd
import numpy as np
# numpy : 수학적 연산을 다루는데 특화된 Library

test_dict2 = {'y': [10, 13, 20, 7, 15],
            'x1': [2, np.nan, 5, 2, np.nan],
            'x2': ['a', 'a', 'b', 'b', 'b'],
            'x3': [np.nan, np.nan, 5, 12, np.nan],
            'x4': ['g1', 'g2', 'g1', 'g2', 'g3']}

test_df2 = pd.DataFrame(test_dict2)
test_df2.to_csv("test_data2(na).csv", index=False, encoding='utf-8-sig')
df = test_df2.copy()

df = pd.read_csv(data_url + 'titanic.csv')
df.shape        # (1310, 8)
df

# Dataset Information --------------------------------------------------------------------
# data information
df.info()

df.shape
df.dtypes

# 숫자형 Data Summary
df.describe()
df.describe(include='all')

df['x1'].plot.hist()    # visualization

# 문자형 Data Summary
df['x4'].value_counts()
df['x4'].value_counts().plot.bar()    # visualization
df['x4'].value_counts().plot.barh()    # visualization


# Sample Data --------------------------------------------------------------------
df.head()
df.head(3)

df.tail()
df.tail(3)

df.sample(4)


# 결측치 --------------------------------------------------------------------
df.isna()
df.isna().sum()         # column별 결측치 확인
df.isna().sum().sum()   # 데이터셋 결측치 갯수 확인

(~df.isna()).sum()      # column별 결측치가 아닌 것의 갯수

df.isna().sum().plot.bar()      # 결측치 갯수
(~df.isna()).sum().plot.bar()      # 결측치가 아닌것의 갯수


# missingno library
import missingno as msno
# msno : pandas 기반 결측치를 다루는 Library

msno.matrix(df)
msno.bar(df)





# Dataset Summary Library ------------------------------------------------------------
# pandas_profiling library
# import pandas_profiling as pd_report
# pandas_profiling : pandas 기반 DataFrame Summary를 시각적으로 제공해주는 Library
# pd_report.ProfileReport(df)



# 데이터 저장 로드 (pkl 확장자) -----------------------------------
# conda install six
# 아래에서 사용된 cPickle 은 Python 자료형으로 데이터를 저장하고 불러오는 패키지이다.

from six.moves import cPickle

# path = r'D:\작업방\업무 - 자동차 ★★★\Worksapce_Python\Model'
# cPickle.dump(변수명, open('경로/저장할_파일명.pkl', 'wb'))
cPickle.dump(df, open(path + '/test_data.pkl', 'wb'))

# Loading = cPickle.load(open('경로/로드할_파일명.pkl', 'rb'))
df_load = cPickle.load(open(path + '/test_data.pkl', 'rb'))












#【 Visulization : Matplotlib, seaborn 】==========================================================================================================
# import matplotlib.pyplot as plt
# import seaborn as sns


import pandas as pd
# Example Data
test_dict = {'y': [10, 13, 20, 7, 15],
            'x1': [2, 4, 5, 2, 4],
            'x2': ['a', 'a', 'b', 'b', 'b'],
            'x3': [10, 8, 5, 12, 7],
            'x4': ['g1', 'g2', 'g1', 'g2', 'g3']}

test_df = pd.DataFrame(test_dict)
df = test_df.copy()
# df = pd.read_clipboard()


# 【 matplotlib 】=====================================================

import matplotlib.pyplot as plt
# matplotlib: python에서 Graph를 쉽게 그려주는 Python의 대표적인 시각화 Library
# < 사용법 >
# plt.figure()              # 캔버스 생성
# plt.~~~                   # Graph 드로잉
# plt.~~~                   # 추가옵션 (label, axis, limit ...)
# plt.show() / plt.close    # 종료조건

# matplotlib 한글폰트사용
from matplotlib import font_manager, rc
# font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/gothic.ttf").get_name()
rc('font', family=font_name)



# line ----------------------------------------------------------
plt.plot(df['x1'])
plt.show()

plt.plot(df['x1'], df['y'])
plt.show()


x2_a = df[df['x2'] =='a']
x2_b = df[df['x2'] =='b']

# group별
plt.plot(x2_a['x1'])
plt.plot(x2_b['x1'])
plt.show()



# group별 + marker
plt.plot(x2_a['x1'], marker='o')
plt.plot(x2_b['x1'], marker='o')
plt.show()

# group별 + marker + ls
plt.plot(x2_a['x1'], marker='o', ls='')
plt.plot(x2_b['x1'], marker='o', ls='')
plt.show()


# group별 + marker + ls + x,y값 지정
plt.plot(x2_a['x1'], x2_a['y'], marker='o', ls='')
plt.plot(x2_b['x1'], x2_b['y'], marker='o', ls='')
plt.show()



# scatter-plot ----------------------------------------------------------
plt.scatter(df['x1'], df['y'])
plt.show()


plt.scatter(x2_a['x1'], x2_a['y'], label='a')
plt.scatter(x2_b['x1'], x2_b['y'], label='b')
plt.legend()
plt.show()


# 데코레이션 ----------------------------------------
    # linestyle
        # -	solid line style
        # --	dashed line style
        # -.	dash-dot line style
        # :	dotted line style

    # style
        # color(c)	선 색깔
        # linewidth(lw)	선 굵기
        # linestyle(ls)	선 스타일
        # marker		마커 종류
        # markersize(ms)	마커 크기
        # markeredgecolor(mec)	마커 선 색깔
        # markeredgewidth(mew)	마커 선 굵기
        # markerfacecolor(mfc)	마커 내부 색깔


# barplot ----------------------------------------------------------
df['x2'].value_counts()
df['x2'].value_counts().plot.bar()
# df['x2'].value_counts().plot.barh()
df['x2'].value_counts().plot.bar(color='skyblue', edgecolor='grey')


bar_data = df['x2'].value_counts()
bar_data

# bar plot
plt.bar(x=bar_data.index, height=bar_data.values)
plt.show()

# barh plot
plt.barh(y=bar_data.index, width=bar_data.values)
plt.show()


# string_counts = df['x2'].value_counts()
# plt.bar(string_counts.index, string_counts)
# plt.barh(string_counts.index, string_counts)




# boxplot ----------------------------------------------------------
plt.boxplot(df['x1'])
plt.show()

box1 = df[df['x2']=='a']['x1']
box2 = df[df['x2']=='b']['x1']
plt.boxplot([box1, box2], labels=['a', 'b'])

df.plot.box()

# df.boxplot(column='x1', by='x2')
# plt.show()



# Other Technique =========================================================
# Sub-line ----------------------------------------------------------
# vertical-line
plt.hist(df['x1'])
plt.axvline(3, color='red')     # 보조선(세로)
plt.show()


# horizontal-line
plt.hist(df['x1'])
plt.axhline(1.5, color='red')     # 보조선(가로)
plt.show()


# Title / Axis_Name ----------------------------------------------------------
plt.hist(df['x1'])
plt.title('Histogram')       # title
plt.ylabel('y_Value')       # label
plt.xlabel('x_Value')       # label
plt.show()


# Axis Scale ----------------------------------------------------------
plt.hist(df['x1'])
plt.axis(xmin=-5, xmax=15, ymin=-3, ymax=10)
plt.show()






# 【 seaborn 】=====================================================
import seaborn as sns
# seaborn: matplotlib 기반하여 만들어진 graph를 예쁘게 그려주는 시각화 Library

# boxplot ----------------------------------------------------------
sns.boxplot(x=df['x2'], y=df['x1'])
sns.boxplot(x='x2', y='x1', data=df)


# strip plot
sns.stripplot(df['x2'], df['x1'], jitter=True)
sns.stripplot(x='x2', y='x1', data=df)


# joint plot ----------------------------------------------------------
sns.jointplot(df['x1'], df['x3'])
sns.jointplot(x='x1', y='x3', data=df)

sns.jointplot(df['x1'], df['x3'], kind='kde')
sns.jointplot(x='x1', y='x3', data=df, kind='kde')


# distplot ----------------------------------------------------------
sns.distplot(df['x1'])
plt.show()

sns.distplot(df['x1'], kde=False)
plt.show()

# Gaussian Line Drawing
import scipy as sp
sns.distplot(df['x1'], kde=False, fit=sp.stats.norm)
plt.show()




# pair-plot ------------------------------------------------------------
sns.pairplot(df)
sns.pairplot(df, hue='x2')
plt.show()


# correlaction ----------------------------------------------------------
df.corr()

plt.matshow(df.corr())     # pyplot
plt.colorbar()
plt.show()

sns.heatmap(df.corr())     # seaborn
sns.heatmap(df.corr(), annot=True)     # seaborn with annotate
sns.heatmap(df.corr(), annot=True, cmap='Reds')     # seaborn with red-colors
sns.heatmap(df.corr(), annot=True, cmap='Blues')     # seaborn with blue-colors
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')     # seaborn with blue-colors
plt.show()







# More About Matplotlib...
# https://matplotlib.org/
# http://pythonstudy.xyz/python/article/407-Matplotlib-%EC%B0%A8%ED%8A%B8-%ED%94%8C%EB%A1%AF-%EA%B7%B8%EB%A6%AC%EA%B8%B0

# More About Seaborn...
# https://seaborn.pydata.org/
# https://datascienceschool.net/view-notebook/4c2d5ff1caab4b21a708cc662137bc65/






























#【 Statistics 】----------------------------------------------------------------------------------------------------------
# import scipy
# scipy?
#   SciPy는 파이썬을 기반으로 하여 과학, 분석, 그리고 엔지니어링을 위한 과학(계산)적 컴퓨팅 영역의 여러 기본적인 작업을 위한 라이브러리(패키지 모음)
#   SciPy는 수치적분 루틴과 미분방정식 해석기, 방정식의 근을 구하는 알고리즘, 표준 연속/이산 확률분포와 다양한 통계관련 도구 등을 제공

import numpy as np
import pandas as pd


# Example Data
test_dict = {'y': [10, 13, 20, 7, 15],
            'x1': [2, 4, 5, 2, 4],
            'x2': ['a', 'a', 'b', 'b', 'b'],
            'x3': [10, 8, 5, 12, 7],
            'x4': ['g1', 'g2', 'g1', 'g2', 'g3']}

test_df = pd.DataFrame(test_dict)
df = test_df.copy()



# 【 scipy 】 ================================================================
import scipy as sp

# t-test: '두집단의 평균이 같은지?'를 비교하는 모수적 통계방법
df.agg(['mean', 'std'])

# ○ 1 Sample t ---------------------------------------------------------------
sp.stats.ttest_1samp(df['x1'], 4)   # x1 Column의 평균이 4와 같은가?
sp.stats.ttest_1samp(df['x1'], 6)   # x1 Column의 평균이 6와 같은가?

# visualization
sns.distplot(df['x1'], fit=sp.stats.norm, kde=False)
plt.axvline(df['x1'].mean(), color='blue')
plt.axvline(4, alpha=0.5, color='orange')
plt.axvline(6, alpha=0.5, color='orange')
plt.show()


# ○ 2 Sample t ---------------------------------------------------------------
t1_data = df[df['x2']=='a']['x1']
t2_data = df[df['x2']=='b']['x1']

sp.stats.ttest_ind(t1_data, t2_data, equal_var=False)   # t_test : (t_value, p-value)

# visualization
sns.distplot(t1_data, fit=sp.stats.norm, kde=False, fit_kws={'color':'steelblue'})
plt.axvline(t1_data.mean(), color='steelblue')
sns.distplot(t2_data, fit=sp.stats.norm, kde=False, fit_kws={'color':'orange'})
plt.axvline(t2_data.mean(), color='orange')
plt.show()

t1_data.mean()
t2_data.mean()



# ANOVA ---------------------------------------------------------------
sp.stats.f_oneway(t1_data, t2_data) 

a1_data = df[df['x4']=='g1']['x1']
a2_data = df[df['x4']=='g2']['x1']
a3_data = df[df['x4']=='g3']['x1']

sp.stats.f_oneway(a1_data, a2_data, a3_data) 


# visualization
sns.boxplot(data=df, x='x4', y='x1')
plt.plot([a1_data.mean(), a2_data.mean(), a3_data.mean()], marker='o', ls='-', color='red')
plt.show()


# import matplotlib
# matplotlib.style.use('seaborn-whitegrid')
# matplotlib.style.available





































# 【 Machine_Learning  】==========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
import seaborn as sns

# 【 sklearn 】 
#   Machine learning module for Python
#   sklearn is a Python module integrating classical machine learning algorithms 
#   in the tightly-knit world of scientific Python packages (numpy, scipy, matplotlib).



# 【 Regressor : LinearRegression 】--------------------------------------------

# --- (Load Dataset) ------------------------------------------------
data_url = 'https://raw.githubusercontent.com/kimds929/CodeNote/main/00_DataAnalysis_Basic'
df_wine = pd.read_csv(data_url + '/wine_aroma.csv')
df_wine.shape        # (25, 10)
df_wine
# ----------------------------------------------------------------


X = df_wine[['Sr']]
y = df_wine['Aroma']


plt.scatter(X, y)
plt.show()

from sklearn.linear_model import LinearRegression
# (nst order) y = β0 + β1·x1 + β2·x2 + ... + βn·xn            * LinearCombination
# (1st order) y = β0 + β1·x1

LR = LinearRegression()
LR.fit(X, y)

LR.coef_
LR.intercept_


# (predict)
LR_pred = LR.predict(X)

LR_tb = pd.DataFrame(LR_pred, columns=['pred'])
LR_tb['true'] = y
LR_tb


# (graph)
Xp = np.linspace(X.min(), X.max(), 10)

plt.figure()
plt.scatter(X, y)
plt.plot(Xp, LR.predict(Xp), color='red')
plt.show()



# (Evaluate Regressor) --------------------------------
from sklearn.metrics import r2_score, mean_squared_error

r2_score(y_true=y, y_pred=LR_pred)          # R2_score
mean_squared_error(y_true=y, y_pred=LR_pred)        # MSE
np.sqrt( mean_squared_error(y_true=y, y_pred=LR_pred) )     # RMSE
# ------------------------------------------------------------






# 【 Classifier : Logistic Regression 】 --------------------------------------------

# --- (Load Dataset) ------------------------------------------------
# clf_dict = {'X' :[3.8, 4. , 3. , 1.3, 3.5, 4.3, 2.1, 3.6, 2.4, 5.2, 3.8, 2.1, 2.8,
#                  4.8, 3.2, 1.7, 4.2, 3.6, 4.3, 2.3, 5.6, 6. , 6.6, 3.8, 5.1, 5.7,
#                 5.5, 6. , 5.3, 5.9, 3.7, 5.4, 7.1, 4.1, 6.6, 3.9, 5.4, 5.3, 6.2, 4.5],
#             'y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#                 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
#                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ]}
# df_clf = pd.DataFrame(clf_dict)
df_clf = pd.read_csv(data_url + "/test_data3(clf).csv", encoding='utf-8-sig')
df_clf.shape    # (40,2)
df_clf.head(10)
# ----------------------------------------------------------------


X = df_clf[['X']]
y = df_clf['y']

plt.scatter(X, y)
plt.show()

# (modeling)
from sklearn.linear_model import LogisticRegression
# (LinearRegression)  y = β0 + β1·x1 + β2·x2 + ... + βn·xn

# y = p = sigmoid(z) = sigmoid(β0 + β1·x1 + β2·x2 + ... + βn·xn)            * y means 'p (probability)'
#       * p = sigmoid(z)  ↔  log( p/(1-p) ) = z
#           ㄴ sigmoid(z) = 1 / (1 + exp(-z))
#           ㄴ z = β0 + β1·x1 + β2·x2 + ... + βn·xn           * LinearCombination



# (sigmoid function) -----------------
def sigmoid(z):
    return 1 / (1+ np.exp( -1 * z))

Xp = np.linspace(-10,10)

# (graph)
plt.plot(Xp, sigmoid(Xp))
plt.show()
# ----------------------------------

LRC = LogisticRegression()
LRC.fit(X, y)


# (predict)
LRC_pred = LRC.predict(X)
LRC_pred

LRC_pred_proba = LRC.predict_proba(X)
LRC_pred_proba


# (Graph)
Xp = np.linspace(X.min(), X.max(), 10)
LRC.predict_proba(Xp)
pd.DataFrame(LRC.predict_proba(Xp))
LRC_pred_proba_Xp = pd.DataFrame(LRC.predict_proba(Xp))[1]

plt.figure()
plt.scatter(X, y)
plt.plot(Xp, LRC_pred_proba_Xp, color='red')
plt.axhline(0.5, color='orange', ls='--', alpha=0.5)
plt.show()


# (coef_, intercept_)
LRC.coef_       # 1.5778
LRC.intercept_

# Interpret Coeficient : X값이 1증가할수록 exp(계수)배 만큼 확률이 증가
np.exp(LRC.coef_)   # 4.844 : X가 1증가할때 1일 확률이 4.844배 증가



# (Decision-Boundary)
# log( p/(1-p) ) =  β0 + β1·x1 + β2·x2 + ... + βn·xn
#    (1st order)  log( p/(1-p) ) = logit =  β0 + β1·x1
#                  → x1 = (logit - β0) / β1
threshold = 0.5
logit = np.log(threshold / (1-threshold))
decision_boundary = ( (logit - LRC.intercept_) / LRC.coef_ )[0][0]
decision_boundary

# LRC_pred_proba[:,1] > threshold
LRC_proba_X = (LRC_pred_proba[:,1] > threshold).astype(int)
LRC_proba_X



# (Graph with Decision-Boundary)
Xp = np.linspace(X.min(), X.max(), 100)
LRC_pred_proba_Xp = LRC.predict_proba(Xp)[:,1]


plt.figure()
plt.scatter(X, y)       # data
plt.plot(Xp, LRC_pred_proba_Xp, color='red')    # Probability Graph
plt.axhline(threshold, color='orange', ls='--', alpha=0.5)      # Threshold
plt.axvline(decision_boundary, color='orange',ls='--', alpha=0.5)    # Decision Boundary
plt.show()








# (Evaluate Classifier) -------------------------------------------------
# accuracy
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy = accuracy_score(y, LRC_pred)
accuracy

# confusion_matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, LRC_pred)
cm

[[TN, FP], [FN, TP]] = cm
#           (pred 0) (pred 1)
# (real 0) [[ 18,       2],      [[ TN (True Negative),  FP (False Positve) ]
# (real 1)  [  4,      16]]       [ FN (False Negative), TP (True Positive) ]]
# 
#  * True / False : 예측값과 실제값이 같은경우 True
#  * Positive / Negative : 예측값이 1인경우 Positive



cm_frame = pd.DataFrame(cm, index=['Real_T', 'Real_F'], columns=['Pred_T', 'Pred_F'])
cm_frame


# Other evaluation metrics
from sklearn.metrics import precision_score, recall_score

# accuracy = (TN + TP) / (cm.sum())
accuracy
(TN + TP) / (cm.sum())

# precision = TP / (TP + FP)            #* 예측 1에 대한 정확도 (예측값이 얼마나 정확한가)
precision_score(y, LRC_pred)
TP / (TP + FP)


# recall = TP / (TP + FN)               #* 실제 1에 대한 예측도 (=sensitivity) (실제정답을 예측이 얼마나 맞췄는냐?)
# 1이라고 예측하는 것이 중요할때
recall_score(y, LRC_pred)
TP / (TP + FN)


# * precision ↔ recall : 서로 trade-off 관계


from sklearn.metrics import f1_score
# f1_score = 2*(precision * recall ) / (precision + rec)
# Harmonic_Mean of the precision and recall. (precision, recall의 조화평균)
# Imbalance Data의경우 반드시 f1_score를 확인해주어야 함

f1_score(y, LRC_pred)


# classification_report
from sklearn.metrics import classification_report
# . macro avg : 단순 평균값
# . weighted avg : 각 class에 속하는 표본의 갯수로 가중평균한 값
#   * Imbalance Dataset의 경우 f1-score를 반드시 고려해야 함
print(classification_report(y, LRC_pred))



# ROC-curve, AUC
# ROC-curve와 AUC를 사용하면 분류문제에서 여러 임계값 설정에 대한 모델의 성능을 구할 수 있게 된다.

#           (pred 0) (pred 1)
# (real 0) [[ 18,       2],      [[ TN (True Negative),  FP (False Positve) ]
# (real 1)  [  4,      16]]       [ FN (False Negative), TP (True Positive) ]]

# FPR(False Positive Ratio) =  FP / (FP + TN)               # 실제 0의 예측 실패도
FPR = FP / (FP + TN)

# TPR(True Positive Ratio) = recall = TP / (TP + FN)        # 실제 1의 예측 정확도
TPR = TP / (TP + FN)



from sklearn.metrics import roc_curve
# ROC curve : 임계값에 대한 TPR, FPR의 변화를 곡선으로 나타낸 것
#             X축에 FPR, Y축에 TPR을 두어 최적의 임계값을 찾는 것

LRC_pred_proba = LRC.predict_proba(X)
roc_curve(y, LRC_pred_proba[:, 1])
FPRs, TPRs, Thresholds = roc_curve(y, LRC_pred_proba[:, 1])
# 실제 0의 예측정확도, 실제 1의 예측정확도


# 최적 threshold 찾기
# . 0의 예측 실패도는 최소화하면서, 1의 예측도는 최대화 : maximize(TPR - FPR)
optimal_threshold = Thresholds[np.argmax(TPRs - FPRs)]

# ROC_Curve
plt.figure()
plt.plot(FPRs,TPRs, label='ROC')
plt.plot([0,1], [0,1], label='STR')     #  가운데 직선: ROC 곡선의 최저값
plt.xlabel('FPR')   # 실제 0의 예측정확도
plt.ylabel('TPR')   # 실제 1의 예측정확도
plt.legend()
plt.show()


# AUC area
from sklearn.metrics import roc_auc_score
# AUC(Area Under the ROC Curve) : ROC그래프의 하부영역 (클수록 예측능력이 좋음)
# 1 완벽 / 0.9~1 매우 정확 / 0.7~0.9 정확 / 0.5~0.7 덜정확

LRC_pred = LRC.predict(X)
roc_auc_score(y, LRC_pred)

# --------------------------------------------------------------------------------------------------











####### 【 Decision_Tree 】##################################################################
data_url = 'https://raw.githubusercontent.com/kimds929/CodeNote/main/00_DataAnalysis_Basic'
df_titanic = pd.read_csv(data_url + '/titanic_simple.csv')
df_titanic.shape    # (50,8)
df_titanic.head()
# Pclass : 1 = 1등석, 2 = 2등석, 3 = 3등석
# Survived : 0 = 사망, 1 = 생존
# Sex : male = 남성, female = 여성
# Age : 나이
# SibSp : 타이타닉 호에 동승한 자매 / 배우자의 수
# Parch : 타이타닉 호에 동승한 부모 / 자식의 수
# Fare : 승객 요금
# Embarked : 탑승지, C = 셰르부르, Q = 퀸즈타운, S = 사우샘프턴


# (Decision_Tree Classifier) - descrete X -----------------------------------------------------------------
#  . X : pclass, sex_class
#  . y : survived

df_titanic['sex_class'] = (df_titanic['sex'] == 'male').astype(int)

X = df_titanic[['pclass', 'sex_class']]
y = df_titanic['survived']

df_titanic.groupby(['pclass', 'sex_class','survived']).size()
df_titanic.groupby(['pclass', 'sex_class','survived']).size().unstack('survived')

df_titanic.groupby(['pclass','survived']).size().unstack('survived')
df_titanic.groupby(['sex_class','survived']).size().unstack('survived')


from sklearn.tree import DecisionTreeClassifier
# gini = n1/N * gini_g1 + n2/N * gini_g2
#       ㄴ gini_g1 = 1 - p(g1_class0)**2 - p(g1_class1)**2
#       ㄴ gini_g2 = 1 - p(g2_class0)**2 - p(g2_class1)**2

DT_clf = DecisionTreeClassifier()
# DT_clf = DecisionTreeClassifier(criterion='gini')
DT_clf.fit(X, y)


# (predict)
DT_clf_pred = DT_clf.predict(X)
DT_clf_pred

DT_clf_pred_proba = DT_clf.predict_proba(X)
DT_clf_pred_proba

DT_clf_pred_proba_df = pd.DataFrame(DT_clf_pred_proba)
DT_clf_pred_proba_df

pd.concat([X, DT_clf_pred_proba_df], axis=1).to_clipboard()


# (tree plot)
from sklearn import tree
tree.plot_tree(DT_clf)
tree.plot_tree(DT_clf, feature_names=X.columns)
tree.plot_tree(DT_clf, feature_names=X.columns, filled=True)   # class의 쏠림에 따라 색상을 부여
tree.plot_tree(DT_clf, feature_names=X.columns, filled=True, max_depth=2)  # max_depth부여

DT_clf.cost_complexity_pruning_path(X, y)   # 변화가 생기는 alpha값 list 및 그때의 불순도

# (Feature Importance)
DT_clf.feature_importances_
plt.barh(X.columns, DT_clf.feature_importances_)


# (Evaluate Decision Tree Classifier)
from sklearn.metrics import accuracy_score
accuracy_score(y, DT_clf_pred)

from sklearn.metrics import roc_auc_score
roc_auc_score(y, DT_clf_pred)


# (Logistic Regression과 비교) ---
from sklearn.linear_model import LogisticRegression
LRC1 = LogisticRegression()
LRC1.fit(X,y)

LRC1_pred = LRC1.predict(X)

accuracy_score(y, LRC1_pred)
roc_auc_score(y, LRC1_pred)




# (Decision_Tree Classifier) - continuous X -----------------------------------------------------------------
#  . X : fare, age
#  . y : survived
X = df_titanic[['fare', 'age']]
y = df_titanic['survived']


# (modeling)
from sklearn.tree import DecisionTreeClassifier
DT_clf2 = DecisionTreeClassifier()
# DT_clf2 = DecisionTreeClassifier(max_depth=3)
# DT_clf = DecisionTreeClassifier(criterion='gini')
DT_clf2.fit(X, y)


# (predict)
DT_clf2_pred = DT_clf2.predict(X)
DT_clf2_pred

DT_clf2_pred_proba = DT_clf2.predict_proba(X)
DT_clf2_pred_proba

DT_clf2_pred_proba_df = pd.DataFrame(DT_clf2_pred_proba)
DT_clf2_pred_proba_df

pd.concat([X, DT_clf2_pred_proba_df], axis=1).to_clipboard()


# (tree plot)
from sklearn import tree
tree.plot_tree(DT_clf2)
tree.plot_tree(DT_clf2, feature_names=X.columns)
tree.plot_tree(DT_clf2, feature_names=X.columns, filled=True)   # class의 쏠림에 따라 색상을 부여
tree.plot_tree(DT_clf2, feature_names=X.columns, filled=True, max_depth=3)  # max_depth부여

DT_clf2.cost_complexity_pruning_path(X, y)   # 변화가 생기는 alpha값 list 및 그때의 불순도

# (Feature Importance)
DT_clf2.feature_importances_
plt.barh(X.columns, DT_clf2.feature_importances_)


# (Evaluate Decision Tree Classifier)
from sklearn.metrics import accuracy_score
accuracy_score(y, DT_clf2_pred)

from sklearn.metrics import roc_auc_score
roc_auc_score(y, DT_clf2_pred)


# (Logistic Regression과 비교) ---
from sklearn.linear_model import LogisticRegression
LRC2 = LogisticRegression()
LRC2.fit(X,y)

LRC2_pred = LRC2.predict(X)

accuracy_score(y, LRC2_pred)
roc_auc_score(y, LRC2_pred)









# (Decision_Tree Regressor) - descrete X -----------------------------------------------------------------
#  . X : pclass, sex_class
#  . y : fare
df_titanic['sex_class'] = (df_titanic['sex'] == 'male').astype(int)

X = df_titanic[['pclass', 'sex_class']]
y = df_titanic['fare']


# (modeling)
from sklearn.tree import DecisionTreeRegressor
DT_reg = DecisionTreeRegressor()
DT_reg.fit(X, y)


# (predict)
DT_reg_pred = DT_reg.predict(X)
DT_reg_pred

DT_reg_pred_df = pd.Series(DT_reg_pred).to_frame()
DT_reg_pred_df

pd.concat([X, DT_reg_pred_df], axis=1).to_clipboard()


# (tree plot)
from sklearn import tree
tree.plot_tree(DT_reg)
tree.plot_tree(DT_reg, feature_names=X.columns)
tree.plot_tree(DT_reg, feature_names=X.columns, filled=True)   # class의 쏠림에 따라 색상을 부여
tree.plot_tree(DT_reg, feature_names=X.columns, filled=True, max_depth=2)  # max_depth부여

DT_reg.cost_complexity_pruning_path(X, y)   # 변화가 생기는 alpha값 list 및 그때의 불순도

# (Feature Importance)
DT_reg.feature_importances_
plt.barh(X.columns, DT_reg.feature_importances_)


# (Evaluate Decision Tree Regressor)
from sklearn.metrics import r2_score, mean_squared_error
r2_score(y, DT_reg_pred)
mean_squared_error(y, DT_reg_pred)
np.sqrt(mean_squared_error(y, DT_reg_pred))


# (Linear Regression과 비교) ---
from sklearn.linear_model import LinearRegression
LR1 = LinearRegression()
LR1.fit(X,y)

LR1_pred = LR1.predict(X)

r2_score(y, LR1_pred)
mean_squared_error(y, LR1_pred)
np.sqrt(mean_squared_error(y, LR1_pred))




# (Decision_Tree Regressor) - Continuous X -----------------------------------------------------------------
#  . X : pclass, age
#  . y : fare

X = df_titanic[['pclass', 'age']]
y = df_titanic['fare']

df_titanic.groupby(['pclass', 'sex_class'])['fare'].mean()
df_titanic.groupby(['pclass', 'sex_class'])['fare'].mean().unstack('sex_class')


# (modeling)
from sklearn.tree import DecisionTreeRegressor
DT_reg2 = DecisionTreeRegressor()
DT_reg2.fit(X, y)


# (predict)
DT_reg2_pred = DT_reg2.predict(X)
DT_reg2_pred

DT_reg2_pred_df = pd.Series(DT_reg2_pred).to_frame()
DT_reg2_pred_df

pd.concat([X, DT_reg2_pred_df], axis=1).to_clipboard()


# (tree plot)
from sklearn import tree
tree.plot_tree(DT_reg2)
tree.plot_tree(DT_reg2, feature_names=X.columns)
tree.plot_tree(DT_reg2, feature_names=X.columns, filled=True)   # class의 쏠림에 따라 색상을 부여
tree.plot_tree(DT_reg2, feature_names=X.columns, filled=True, max_depth=2)  # max_depth부여

DT_reg2.cost_complexity_pruning_path(X, y)   # 변화가 생기는 alpha값 list 및 그때의 불순도


# (Feature Importance)
DT_reg.feature_importances_
plt.barh(X.columns, DT_reg2.feature_importances_)


# (Evaluate Decision Tree Regressor)
from sklearn.metrics import r2_score, mean_squared_error
r2_score(y, DT_reg2_pred)
mean_squared_error(y, DT_reg2_pred)
np.sqrt(mean_squared_error(y, DT_reg2_pred))


# (Linear Regression과 비교) ---
from sklearn.linear_model import LinearRegression
LR2 = LinearRegression()
LR2.fit(X,y)

LR2_pred = LR2.predict(X)

r2_score(y, LR2_pred)
mean_squared_error(y, LR2_pred)
np.sqrt(mean_squared_error(y, LR2_pred))

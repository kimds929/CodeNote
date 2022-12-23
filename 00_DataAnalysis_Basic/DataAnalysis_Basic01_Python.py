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
# $ conda uninstall -c conda-forge pandas-profiling

# -----------------------------------------------------------------------------------------------------


# import numpy as np
# import pandas as pd

# import missingno as msno
# import pandas_profiling as pd_report

# import matplotlib.pyplot as plt
# import seaborn as sns

# import scipy
# import statsmodels.api as sm

# from DataAnalysis_Module import Cpk, cpk_line, fun_Decimalpoint, distboxplot, sm_LinearRegression
# from DataAnalysis_Module import *
# -----------------------------------------------------------------------------------------------------


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


# operation (사칙연산) --------------------------------------------------------------------------------
a + b

a - b

a * b

a / b

a // b      # 몫

a % b       # 나머지

a ** 2      # 제곱
a ** b


# data-structure --------------------------------------------------------------------------------
# numeric: 숫자형 / string: 문자형  (object: 문자형 집합체)
#   numeric → int: 정수, float: 소수

# string → string

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


# 원소추가
l2 = [3,4,5]
l2

l2.append(6)
l2



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

# 원소추가
d2['c'] = [9,8,7]
d2



# 제어문 --------------------------------------------------------------------------------
abc = 10
if abc >= 20:
    print('abc: 20 이상')
elif abc >= 10:
    print('abc: 10 ~ 19')
else:
    print('abc: 10미만')


# 루프문 --------------------------------------------------------------------------------
for i in [10, 15, 20]:
    print(i)


# 루프문 강제 탈출
for i in [10, 15, 20]:
    if i > 17:
        break
    else:
        print(i)


# 함수 --------------------------------------------------------------------------------
def function_add(x, y):
    add = x + y
    return add

function_add(10, 20)

def hello_printing():
    print('Hello')

hello_printing()




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


# Example Data
test_dict = {'y': [10, 13, 20, 7, 15],
            'x1': [2, 4, 5, 2, 4],
            'x2': ['a', 'a', 'b', 'b', 'b'],
            'x3': [10, 8, 5, 12, 7],
            'x4': ['g1', 'g2', 'g1', 'g2', 'g3']}

test_df = pd.DataFrame(test_dict)
test_df

df = test_df.copy()



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
df['x1'].sum()
df['x1'].mean()
df['x1'].std()

df['x1'].median()

sum(df['x1'])





from DataAnalysis_Module import Cpk
# Cpk (Cumstomizing Function)
cpk_calc = Cpk()

cpk_calc.usl = 10
cpk_calc.decimal = 2
# cpk_calc.lean = True
cpk_calc.reset()
cpk_calc

cpk_calc(df['x1'], lsl=5, usl=10)
cpk_calc.decimal
cpk_calc


# index만 추출하기
df.index

# column명만 추출하기
df.columns



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




# Sort -----------------------------------------------------------------
df.sort_values('x4')                   # 오름차순
df.sort_values('x4', ascending=False)  # 내림차순

# 복수조건
df.sort_values(['x4', 'x3'])

df.sort_values(['x4', 'x3'], ascending=[True, False])


# index
df.sort_index(ascending=False) 




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


from DataAnalysis_Module import Cpk
# Cpk (Cumstomizing Function)

cpk_calc = Cpk()
cpk_calc.lsl = 10
cpk_calc.usl = 20
cpk_calc.decimal = 4
cpk_calc.lean = True
cpk_calc

df_group['y'].agg(cpk_calc)
df_group['y'].agg(['mean', 'std', cpk_calc])
df_group['y'].agg(lambda x: cpk_calc(x, lsl=12, usl=18))
df_group['y'].describe()





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
df = test_df2.copy()

# path = 'D:\Python\Dataset'
# df = pd.read_csv(path + '\Titanic.csv')
# df

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
import pandas_profiling as pd_report
# pandas_profiling : pandas 기반 DataFrame Summary를 시각적으로 제공해주는 Library

pd_report.ProfileReport(df)



# Dataset Summary Library ------------------------------------------------------------
# 【 Cumstomizing Module 】
from DataAnalysis_Module import SummaryPlot, DF_Summary

# SummaryPlot
sm_plt = SummaryPlot(df)

sm_plt.summary_plot(on=['x1'])
sm_plt.summary_plot(on=['x1', 'x2'])
sm_plt.summary_plot(on=df.columns)
sm_plt.summary_plot(on=df.columns, dtypes='numeric')
sm_plt.summary_plot(on=df.columns, dtypes='object')


# DF_Summary
sm_df = DF_Summary(df)
sm_df.summary
sm_df.summary.to_clipboard()

sm_df.summary_plot()        # Summary Plot in DF_Summary
sm_df.summary_plot(on=['x3', 'x4'])
sm_df.summary_plot(dtypes='numeric')
sm_df.summary_plot(dtypes='object')











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

# line ----------------------------------------------------------
df['x1'].plot()
df['x1'].plot(marker='o')

plt.plot('x1', 'x3', data=df)
plt.plot('x1', 'x3', data=df.sort_values('x1'))
plt.plot('x1', 'x3', data=df.sort_values('x1'), marker='o')
plt.show()

# scatter-plot ----------------------------------------------------------
plt.scatter('x1', 'x3', data=df)
plt.plot('x1', 'x3', data=df.sort_values('x1'))
plt.show()
# plt.plot('x1', 'x3', data=df.sort_values('x1'), marker='o')




# barplot ----------------------------------------------------------
df['x2'].value_counts()
# df['x2'].value_counts().plot.bar()
# df['x2'].value_counts().plot.barh()
df['x2'].value_counts().plot(kind='bar')
# df['x2'].value_counts().plot(kind='barh')

df['x2'].value_counts().plot(kind='bar', color='skyblue', edgecolor='grey')
plt.xticks(rotation=0)
plt.show()


# string_counts = df['x2'].value_counts()
# plt.bar(string_counts.index, string_counts)
# plt.barh(string_counts.index, string_counts)



# histogram kde ----------------------------------------------------------
from DataAnalysis_Module import cpk_line

df['x1'].hist()
# plt.hist('x1', data=df, edgecolor='grey')
plt.plot('x1', 'cpk', data=cpk_line(df['x1']), color='red')
plt.grid(alpha=0.1)
plt.show()




# boxplot ----------------------------------------------------------
df.boxplot(column='x1', by='x2')
# df.boxplot(column='x1', by='x2')
# plt.grid(alpha=0.3)
# plt.show()




# Other Technique =========================================================
# Sub-line ----------------------------------------------------------
# vertical-line
plt.hist(df['x1'])
plt.axvline(3, color='red')
plt.show()


# horizontal-line
plt.hist(df['x1'])
plt.axhline(1.5, color='red')
plt.show()


# Title / Axis_Name ----------------------------------------------------------
plt.hist(df['x1'])
plt.title('Histogram')
plt.ylabel('y_Value')
plt.xlabel('x_Value')
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



# 【 Cumstomizing Module 】
from DataAnalysis_Module import distboxplot

distboxplot(data=df, on='x1')
distboxplot(data=df, on='x1', group='x2')
distboxplot(data=df, on='x1', group='x2', mean_line=True)


# matplotlib 한글화 문제 해결
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)



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
plt.plot([a1_data.mean(), a2_data.mean(), a3_data.mean()], 'o-', color='red')
plt.show()

from DataAnalysis_Module import distboxplot
distboxplot(data=df, on='x1', group='x4')
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


# Example Data
test_dict = {'y': [10, 13, 20, 7, 15],
            'x1': [2, 4, 5, 2, 4],
            'x2': ['a', 'a', 'b', 'b', 'b'],
            'x3': [10, 8, 5, 12, 7],
            'x4': ['g1', 'g2', 'g1', 'g2', 'g3']}

test_df = pd.DataFrame(test_dict)
df = test_df.copy()

y = test_df[['y']]

# X = test_df[['x3']]
X = test_df[['x1','x3']]


# from DataAnalysis_Module import DF_Summary, SummaryPlot
# path = r'C:\Users\USER\Desktop\Python\9) Dataset'
# df = pd.read_csv(path + '\wine_aroma.csv')

# df_summary = DF_Summary(df)
# df_summary.summary

# df_summary.summary_plot()


# y = df[['Aroma']]
# X = df[['Mo', 'Ba', 'Cr', 'Sr', 'Pb', 'B', 'Mg', 'Ca', 'K']]

# y
# X



# 【 Linear_Regression  】==========================================================
import statsmodels.api as sm


# Learning
# X_add = sm.add_constant(X)
# LR = sm.OLS(y, X_add).fit()


# LR.summary()
# LR.model.params

# # predict
# LR_pred = LR.predict(X_add)

# LR_pred_tb = LR_pred.to_frame(name='pred')
# LR_pred_tb['true_y'] = y


# # evaluate
# LR.rsquared
# LR.rsquared_adj
# np.sqrt(LR.mse_resid)





# 【 Cumstomizing Module: Regression 】
from DataAnalysis_Module import sm_LinearRegression

# Learning
# LR = sm_LinearRegression()
# LR.fit(X, y)

LR = sm_LinearRegression()
LR.fit(X,y)

LR.summary()
LR.model
LR.model.params



# predict
LR_OLS_pred = LR.predict(X)

OLS_pred_tb = LR_OLS_pred.to_frame(name='pred')
OLS_pred_tb['true_y'] = y


# evaluate
LR.model.rsquared               # R2
LR.model.rsquared_adj           # R2_adj
LR.model.mse_resid              # MSE (statics)
np.sqrt(LR.model.mse_resid)     # RMSE
# np.sqrt(( (y['y'] - LR_OLS_pred)**2).sum() / 2 )  # RMSE
LR.model.ssr / len(y)           # MSE (ML)


# OLS_wine = sm_LinearRegression().fit(df.iloc[:,:-1], df.iloc[:,-1].to_frame())
# OLS_wine.features_plot(df.iloc[:,:-1], df.iloc[:,-1].to_frame())


# visualization
LR.features_plot(X, y)







# 【 sklearn 】 ===============================================
# Machine learning module for Python
# sklearn is a Python module integrating classical machine learning algorithms 
# in the tightly-knit world of scientific Python packages (numpy, scipy, matplotlib).

from sklearn.linear_model import LinearRegression

LR_sklearn = LinearRegression()
LR_sklearn.fit(X, y)

LR_sklearn.coef_
LR_sklearn.intercept_


# predict
LR_sklearn_pred = LR_sklearn.predict(X)

LR_sklearn_tb = pd.DataFrame(LR_sklearn_pred, columns=['pred'])
LR_sklearn_tb['true'] = y
LR_sklearn_tb


# evaluate
from sklearn.metrics import r2_score, mean_squared_error
import sklearn

r2_score(y_true=y, y_pred=LR_sklearn_pred)
mean_squared_error(y_true=y, y_pred=LR_sklearn_pred)


plt.scatter(LR_sklearn_pred, y)
plt.plot(y.sort_values('y'), y.sort_values('y'), 'r--')











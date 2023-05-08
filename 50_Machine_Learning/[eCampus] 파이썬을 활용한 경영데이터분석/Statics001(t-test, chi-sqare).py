import pandas as pd
import numpy as np
import scipy as sp
import os

from matplotlib import pyplot as plt
import matplotlib
from plotnine import *
import seaborn as sns

# https://datascienceschool.net/notebook/ETC/    # 참고사이트 : 데이터 사이언스 스쿨
# https://sinxloud.com/machine-learning-cheat-sheets-python-math-statistics/?print=pdf      # Machine-Learning Cheet-Sheet


# del(customr_df)       #변수 삭제
df = pd.read_clipboard()  #Clipboard로 입력하기
# df.to_clipboard()        #Clipboard로 내보내기
df = pd.read_csv('Database/supermarket_sales.csv')

pd.set_option('display.float_format', '{:.2f}'.format) # 항상 float 형식으로
pd.set_option('display.float_format', '{:.2e}'.format) # 항상 사이언티픽
pd.set_option('display.float_format', '${:.2g}'.format)  # 적당히 알아서
pd.set_option('display.float_format', None) #지정한 표기법을 원래 상태로 돌리기: None

# Table Info
df.describe()
df.info()


# Factor열의 Level알기(중복제거)
x = 'XB'
df.drop_duplicates(x, keep='first')      # (중복제거) keep : 'first' 첫번째값남김 / 'last' 마지막값 남김
df[x].drop_duplicates().tolist()  # pandas Series to list


# df.groupby(['Payment','Gender']).count()
# df.groupby(['Payment','Gender']).count().iloc[:,0]

# 평균구하기
x1 = 'UnitPrice'
x2 = 'Quantity'
x3 = 'Tax5'
x4 = 'Gender'
np.mean(df[x1]), np.mean(df[x2])
df[x1].mean(), df[x2].mean()

# ggplot 이용 Graph
ggplot(df,aes(x=x1))\
    +geom_point(aes(y=x2), color='blue')\
    +geom_point(aes(y=x3), color='red')

ggplot(df,aes(x=x1))\
    +geom_histogram(aes(fill=x4),bins=15,color='grey')



# 【 상관관계, 상관계수 】 ------------------------------------------------------------------------------------
df_cor = df.corr() # (default) method='pearson'
# df_cor = df.corr(method='pearson')   # 상관관계구하기

# 상관관계 시각화
fig, ax = plt.subplots( figsize=(7,7) ) # 그림 사이즈 지정

    # 상관관계 그래프 시각화 기본
sns.heatmap(data = df_cor, annot=True, fmt = '.2f', linewidths=.5, cmap='Blues')

    # 위아래 상관관계그래프 시각화
sns.clustermap(df_cor, 
               annot = True,      # 실제 값 화면에 나타내기
               fmt = '.2f',
               cmap = 'RdYlBu_r',  # Red, Yellow, Blue 색상으로 표시
               vmin = -1, vmax = 1, #컬러차트 -1 ~ 1 범위로 표시
              )
    # 아래쪽만 상관관계 그래프 그리기
mask = np.zeros_like(df_cor, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True     # 삼각형 마스크를 만든다(위 쪽 삼각형에 True, 아래 삼각형에 False)
sns.heatmap(df_cor, 
            cmap = 'RdYlBu_r', 
            fmt = '.2f',
            annot = True,   # 실제 값을 표시한다
            mask=mask,      # 표시하지 않을 마스크 부분을 지정한다
            linewidths=.5,  # 경계면 실선으로 구분하기
            cbar_kws={"shrink": .5},# 컬러바 크기 절반으로 줄이기
            vmin = -1,vmax = 1   # 컬러바 범위 -1 ~ 1
           ) 
# ----------------------------------------------------------------------------------------------------


# 【 t-test 】   ------------------------------------------------------------------------------------
x1 = '인장방향'
x1_l1 = 'C'
x1_l2 = 'L'

x2 = '초시험_EL'

col_x1_l1 = df[ df[x1]==x1_l1 ][x2]  # col_x1_l1 = df[df['Gender']=='Male']['UnitPrice']
col_x1_l2 = df[ df[x1]==x1_l2 ][x2]  # col_x1_l2 = df[df['Gender']=='Male']['UnitPrice']



# df.groupby(x1).mean()
# df.groupby(x1).mean()[x2]
# df.groupby(x1).std()[x2]
# df.groupby(x1).var()[x2]
df.groupby(x1).describe()[x2]

# ○ 1 Sample t
ax = sns.distplot(col_x1_l1, kde=False, fit=sp.stats.norm, label="x1_l1")
sp.stats.ttest_1samp(col_x1_l1, 70)


# ○ 2 Sample t
    # 확률밀도함수 Plot그리기
col_x1_l1.plot(kind='kde', color='blue')
col_x1_l2.plot(kind='kde', color='red')

    # 히스토그램 및 정규분포 그래프 그리기
ax = sns.distplot(col_x1_l1, kde=False, fit=sp.stats.norm, label="x1_l1")
ax = sns.distplot(col_x1_l2, kde=False, fit=sp.stats.norm, label="x1_l2")
# ax.lines[0].set_linestyle(":")
ax.lines[0].set_color("steelblue")
ax.lines[1].set_color("coral")
plt.legend()
plt.show()

    # ggplot : Box-Plot그리기, histogram그리기
ggplot(df, aes(x=x1, y=x2)) + geom_boxplot(aes(color=x1)) + theme_bw()
ggplot(df, aes(x=x2)) + geom_histogram(aes(fill=x1), bins=10,color='grey', position='identity', alpha=0.3)

    # t-test : use Scipy
t_test_result = sp.stats.ttest_ind(col_x1_l1, col_x1_l2, equal_var=False)   # t_test : (t_value, p-value)
t_test_result
t_test_result[0]        # t_value
t_test_result[1]        # p-value

    # t-test : use Statsmodels
import statsmodels.stats.api as sms
t_test = sms.CompareMeans(sms.DescrStatsW(col_x1_l1), sms.DescrStatsW(col_x1_l2))   # t-test

t_test_result = t_test.ttest_ind(usevar = 'unequal')     # t_test : (t_value, p-value, degree of freedom)  / (usevar) equal : 동일집단, unequal : 서로다른집단
t_test_result
t_test_result[2]        # degree of freedom

# t_test.tconfint_diff(usevar='unequal')  # t값에 대한 95%신뢰구간 : 구간내에 0이 포함되어있는지 확인(?)

    # ※ 대응표본 t검정 : 동일표본에서 前/後 비교 (학생들의 강의 수강 전/후 평균변화 검증) → x1, x2의 데이터 수가 같아야 함
# sp.stats.ttest_rel(x1, x2)
# ----------------------------------------------------------------------------------------------------



# 【 카이제곱검정 (비율 검정) 】  ------------------------------------------------------------------------------------
# ○ 2 × 2
x1 = 'Gender'
x2 = 'CustomerType'
df[x1].drop_duplicates().tolist()
df[x2].drop_duplicates().tolist()


chi_tb = pd.crosstab(df[x1], df[x2])    # 교차표 만들기 (두개의 변수에 관하여 Level에 따른 갯수)
chi_tb      # 교차 Table 결과
chi_result = sp.stats.chi2_contingency(chi_tb)       
chi_result      # chi-square값, p-value,
chi_result[0]       # chi-square값
chi_result[1]       # p-value  # p-value < 0.05 : Level별로 비율이 같지 않다.

sp.stats.chisquare(chi_tb) # x2 의 각 Level별로 Chi-Square 검정  # (chi-square값 list, p-value list)
# ----------------------------------------------------------------------------------------------------


# 【 ANOVA 】   -------------------------------------------------------------------------------------------
# ○ Oneway ANOVA
x1 = 'Gender'   # Cash, Credit card, Ewallet
x1_l1 = 'Male'
x1_l2 = 'Female'

x2 = 'UnitPrice'

ggplot(df, aes(x=x1, y=x2)) + geom_boxplot(aes(color=x1)) + theme_bw()

x1_level = []
x1_x2_dict = {}
for i in df[x1].drop_duplicates().tolist():
    x1_level.append(i)
    x1_x2_dict[i] = df[ df[x1] == i ][x2].to_list()

sp.stats.f_oneway(x1_x2_dict[x1_l1], x1_x2_dict[x1_l2])         # Oneway ANOVA : F-value, p-value
# ----------------------------------------------------------------------------------------------------



    # 3 × 2
# x1 = 'Payment'
# x2 = 'Branch'
# df['Payment'].drop_duplicates().tolist()   # Cash, Credit card, Ewallet
# df['Branch'].drop_duplicates().tolist()    # A, B, C
# df['City'].drop_duplicates().tolist()

# 교차표 만들기
    # df.groupby(['Payment']).count().iloc[:,0]
    # df.groupby(['Branch']).count().iloc[:,0]
    # df.groupby(['Payment','Branch']).count().iloc[:,0]    # A, B, C






































sp.stats.chisquare(chi_tb)
sp.stats.chisquare(chi_tb)


x1 = [4,6,17,16,8,9]        # 관측치
x2 = [10,10,10,10,10,10]    # 기대치
print( pd.Series(x1) / pd.Series(x2) )

chis = sp.stats.chisquare(x1, x2)

chis







# ----------------------------------------------------------------------------------------------------



# 정규성검증  ------------------------------------------------------------------------------------
# ○ Scipy에서 제공하는 정규성검정 명령어
    # - 콜모고로프-스미르노프 검정(Kolmogorov-Smirnov test) : scipy.stats.ks_2samp     
    # - 샤피로-윌크 검정(Shapiro–Wilk test) : scipy.stats.shapiro
    # - 앤더스-달링 검정(Anderson–Darling test) : scipy.stats.anderson
    # - 다고스티노 K-제곱 검정(D'Agostino's K-squared test) : scipy.stats.mstats.normaltest

# ○ StatsModels에서 제공하는 정규성검정 명령어
    # - 콜모고로프-스미르노프 검정(Kolmogorov-Smirnov test) : statsmodels.stats.diagnostic.kstest_normal
    # - 옴니버스 검정(Omnibus Normality test) : statsmodels.stats.stattools.omni_normtest
    # - 자크-베라 검정(Jarque–Bera test) : statsmodels.stats.stattools.jarque_bera
    # - 릴리포스 검정(Lilliefors test) : statsmodels.stats.diagnostic.lillifors

sp.stats.ks_2samp(x1, x2)       #콜모고로프-스미르노프 검정 :정규분포에 국한되지 않고 두 표본이 같은 분포를 따르는지 확인할 수 있는 방법
    # p-value < 0.05 : 두 분포는 서로 다른 분포
# ----------------------------------------------------------------------------------------------------
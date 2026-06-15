import numpy as np
import pandas as pd

data_path = "D:/DataScience/★GitHub_kimds929/CodeNote/02_DataAnalysis (numpy, pandas)/BigData"


########################################################################################################
# 단일표본 검정
########################################################################################################
df_coffee2 = pd.DataFrame({
    'Caffeine(mg)': [
        94.2, 93.7, 95.5, 93.9, 94.0, 95.2, 94.7, 93.5, 92.8, 94.4,
        93.8, 94.6, 93.3, 95.1, 94.3, 94.9, 93.9, 94.8, 95.0, 94.2,
        93.7, 94.4, 95.1, 94.0, 93.6
    ]
})


from scipy.stats import shapiro, ttest_1samp

# 1
df_coffee2['Caffeine(mg)'].mean()

#2
shapiro(df_coffee2['Caffeine(mg)']).pvalue

# 3
result = ttest_1samp(df_coffee2['Caffeine(mg)'], 95, alternative='less')
result

# 4
result.pvalue

# 5
result.pvalue < 0.05 # 기각


########################################################################################################
# 독립표본 검정
########################################################################################################


df_phone = pd.DataFrame({
    '충전기': ['New'] * 10 + ['Old'] * 10,
    '충전시간': [
        1.5, 1.6, 1.4, 1.7, 1.5, 1.6, 1.7, 1.4, 1.6, 1.5,
        1.7, 1.8, 1.7, 1.9, 1.8, 1.7, 1.8, 1.9, 1.7, 1.6
    ]
})

df_phone.head(3)

x1 = df_phone.query("충전기 == 'New'")['충전시간']
x2 = df_phone.query("충전기 == 'Old'")['충전시간']

from scipy.stats import shapiro, levene, ttest_ind

shapiro(x1)    # p > 0.05 → 정규성
shapiro(x2)   # p > 0.05 → 정규성

levene(x1, x2)  # p >0.05 → 등분산성

# ttest-ind(대상, 기존)
result = ttest_ind(x1, x2, equal_var=True, alternative='less')

# 1
result.statistic

# 2
result.pvalue

# 3
"기각"


########################################################################################################
# 대응표본 검정
########################################################################################################
# 데이터
df_study = pd.DataFrame({
    'User': list(range(1, 11)),
    '기존방법': [60.4, 60.7, 60.5, 60.3, 60.8, 60.6, 60.2, 60.5, 60.7, 60.4],
    '새로운방법': [59.8, 60.2, 60.1, 59.9, 59.7, 58.4, 57.0, 60.3, 59.6, 59.8]
})
print(df_study.head(2))

from scipy.stats import shapiro, ttest_rel
x1 = df_study['기존방법']
x2 = df_study['새로운방법']

shapiro(x1)
shapiro(x2)

# 1
(x2 - x1).mean()

# 2
result = ttest_rel(x2-x1, 0, alternative='less')
result.statistic

# 3 
result.pvalue

# 4
"기각"



########################################################################################################
# 일원 분산분석
########################################################################################################

df_math = pd.read_csv(f"{data_path}/part3/ch6/math.csv", encoding='utf-8-sig')
print(df_math.head())


# 1
from scipy.stats import shapiro, levene

groups = []
shapiro_results = {}
for gi, gv in df_math.groupby(['groups']):
    print(gi)
    result = shapiro(gv['scores'])
    shapiro_results[gi[0]] = result
    groups.append(gv)
{k: v.pvalue for k, v in shapiro_results.items()}

# 2 등분산성
levene_result = levene(groups[0]['scores'], groups[1]['scores'], groups[2]['scores'], groups[3]['scores'])
levene_result.pvalue



from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

model = ols("scores ~ C(groups)", df_math).fit()
result = anova_lm(model)
result

# 3
result['PR(>F)'].iloc[0] < 0.05       # 기각


# 4
result['df'].iloc[0]

# 5
result['df'].iloc[-1]

# 6 score의 잔차제곱합 SSE
result['sum_sq'].iloc[0]

# 7 score의 평균 제곱합 SSE
result['mean_sq'].iloc[0]

# 8 F통계량
result['F'].iloc[0]

# 9 socre의 p-value
result['PR(>F)'].iloc[0]





########################################################################################################
# 이원 분산분석
########################################################################################################

df_tomato = pd.read_csv(f"{data_path}/part3/ch6/tomato2.csv", encoding='utf-8-sig')
print(df_tomato.head())

df_tomato['비료유형'].value_counts()
df_tomato['물주기'].value_counts()

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

levene()


model = ols("수확량 ~ C(비료유형) + C(물주기) + C(비료유형):C(물주기)", df_tomato).fit()
result = anova_lm(model)


# 1.
result['F'].iloc[0]

# 2
result['PR(>F)'].iloc[0]

# 3
result['PR(>F)'].iloc[0] < 0.05   # 채택

# 4.
result['F'].iloc[1] 

# 5.
result['PR(>F)'].iloc[1]

# 6.
result['PR(>F)'].iloc[1] < 0.05    # 기각

# 7
result['F'].iloc[2] 

# 8
result['PR(>F)'].iloc[2] 

# 9
result['PR(>F)'].iloc[2] < 0.05     # 채택






########################################################################################################
########################################################################################################
# 카이제곱 검정
########################################################################################################
########################################################################################################


########################################################################################################
# 적합도 검증
########################################################################################################




# 2~4. 적합도 검정

observed = [550, 250, 100, 70, 30]
expected = [1000*0.60, 1000*0.25, 1000*0.08, 1000*0.05, 1000*0.02]



# 1 # 1. 교통사고 5회 이상 경험 비율
observed[-1] / sum(observed)

# 2 
from scipy.stats import chisquare
result = chisquare(observed, expected)
result.statistic

# 3
result.pvalue

# 4
result.pvalue < 0.05    # 기각


########################################################################################################
# 독립성 검증
########################################################################################################

# cross-table data
ct_camp = pd.DataFrame([[50, 30], [60, 40]], index=['빅분기', '정처기'], columns=['등록','등록안함'])
ct_camp

# row data
df_camp = pd.DataFrame({
        '캠프': ['빅분기']*80 + ['정처기']*100,
        '등록여부': ['등록']*50 + ['등록안함']*30 + ['등록']*60 + ['등록안함']*40
})
df_camp.head()

ct_camp = df_camp.groupby(['캠프', '등록여부']).size().unstack(['등록여부'])



from scipy.stats import chi2_contingency

# 1. 서로 독립적인지 검정통계량 구하기
result = chi2_contingency(ct_camp)
result.statistic

# 2. pavlue
result.pvalue

#3 유의수준 0.05 이하에서 귀무가설 채택여부
result.pvalue < 0.05    # 채택 : 독립이다.









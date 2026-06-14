import pandas as pd

data_path = "D:/DataScience/★GitHub_kimds929/CodeNote/02_DataAnalysis (numpy, pandas)/BigData"





########################################################################################################
########################################################################################################
# 가설검정
########################################################################################################
########################################################################################################


########################################################################################################
# 단일표본 검정
########################################################################################################
df_popcone = pd.DataFrame({
    'weights':[122, 121, 120, 119, 125, 115, 121, 118, 117, 127,
           123, 129, 119, 124, 114, 126, 122, 124, 121, 116,
           120, 123, 127, 118, 122, 117, 124, 125, 123, 121],
})

# ---------------------------------------------------------------------------
# shapiro : 정규성 판단 (H_0 : 정규성을 만족)
from scipy.stats import shapiro

result = shapiro(df_popcone['weights'])
result.statistic        # shapiro
result.pvalue           # p-value : p≥0.05 : 정규성 가정 타당,  p < 0.05 정규성 가정 기각

# ---------------------------------------------------------------------------
# t-test : 정규성을 따를 때 
from scipy.stats import ttest_1samp

# two-side
result = ttest_1samp(df_popcone['weights'], 120)
# result = ttest_1samp(df_popcone['weights'], 120, alternative='two-sided')
result.statistic        # statistics
result.pvalue           # p-value

# one-side
result = ttest_1samp(df_popcone['weights'], 120, alternative='greater')
result.statistic        # statistics
result.pvalue           # p-value


result = ttest_1samp(df_popcone['weights'], 120, alternative='less')
result.statistic        # statistics
result.pvalue           # p-value

# ---------------------------------------------------------------------------
# wilcoxon : 정규성을 따르지 않을 때
from scipy.stats import wilcoxon

# two-side
result = wilcoxon(df_popcone['weights'], 120)
result = wilcoxon(df_popcone['weights'], 120, alternative='two-sided')
result.statistic        # statistics
result.pvalue           # p-value


# one-side
result = wilcoxon(df_popcone['weights'], 120, alternative='greater')
result.statistic        # statistics
result.pvalue           # p-value


result = wilcoxon(df_popcone['weights'], 120, alternative='less')
result.statistic        # statistics
result.pvalue           # p-value



########################################################################################################
# 대응표본 검정
########################################################################################################
import pandas as pd
df_rel = pd.DataFrame({
    'before':[85, 90, 92, 88, 86, 89, 83, 87],
    'after':[85.5,89.9,92.6,89.5,85.8,88.8,84.6,87.8]
})

# ---------------------------------------------------------------------------
# shapiro : 정규성 판단 (H_0 : 정규성을 만족)
from scipy.stats import shapiro

result = shapiro(df_rel['before'] - df_rel['after'])
result.statistic        # shapiro
result.pvalue           # p-value : p≥0.05 : 정규성 가정 타당,  p < 0.05 정규성 가정 기각


# ---------------------------------------------------------------------------
# t-test : 정규성을 따를 때 
from scipy.stats import ttest_rel

# two-side
result = ttest_rel(df_rel['before'], df_rel['after'])
# result = ttest_rel(df_rel['before'], df_rel['after'], alternative='two-sided')
result.statistic        # statistics
result.pvalue           # p-value

# one-side
result = ttest_rel(df_rel['before'], df_rel['after'], alternative='greater')
result.statistic        # statistics
result.pvalue           # p-value


result = ttest_rel(df_rel['before'], df_rel['after'], alternative='less')
result.statistic        # statistics
result.pvalue           # p-value


# ---------------------------------------------------------------------------
# wilcoxon : 정규성을 따르지 않을 때
from scipy.stats import wilcoxon

# two-side
result = wilcoxon(df_rel['before'], df_rel['after'])
result = wilcoxon(df_rel['before'], df_rel['after'], alternative='two-sided')
result.statistic        # statistics
result.pvalue           # p-value


# one-side
result = wilcoxon(df_rel['before'], df_rel['after'], alternative='greater')
result.statistic        # statistics
result.pvalue           # p-value


result = wilcoxon(df_rel['before'], df_rel['after'], alternative='less')
result.statistic        # statistics
result.pvalue           # p-value



########################################################################################################
# 독립표본 검정
########################################################################################################

import pandas as pd
class1 = [85, 90, 92, 88, 86, 89, 83, 87]
class2 = [80, 82, 88, 85, 84]


# shapiro : 정규성 판단 (H_0 : 정규성을 만족)
from scipy.stats import shapiro

result1 = shapiro(class1)
result1.statistic        # shapiro
result1.pvalue           # p-value : p≥0.05 : 정규성 가정 타당,  p < 0.05 정규성 가정 기각

result1 = shapiro(class2)
result1.statistic        # shapiro
result1.pvalue           # p-value : p≥0.05 : 정규성 가정 타당,  p < 0.05 정규성 가정 기각


# ---------------------------------------------------------------------------
# 둘다 정규성을 따를 때 → 등분산성 검증 (levene)

from scipy.stats import levene
result = levene(class1, class2)
result.statistic        # statistics
result.pvalue           # p-value : p≥0.05 : 등분산 가정 타당,  p < 0.05 등분산 가정 기각


# ---------------------------------------------------------------------------
# 둘다 정규성 ○ + 등분산 ○ → ttest-ind
from scipy.stats import ttest_ind

result = ttest_ind(class1, class2, alternative='two-sided')
result.statistic        # statistics
result.pvalue           # p-value : p≥0.05 : 등분산 가정 타당,  p < 0.05 등분산 가정 기각


# ---------------------------------------------------------------------------
# 둘다 정규성 ○ + 등분산 × → welch-t
from scipy.stats import ttest_ind

result = ttest_ind(class1, class2, alternative='two-sided', equal_var=False)
result.statistic        # statistics
result.pvalue           # p-value : p≥0.05 : 등분산 가정 타당,  p < 0.05 등분산 가정 기각


# ---------------------------------------------------------------------------
# 둘중 하나라도 정규성 × → Mann-Whitney U
from scipy.stats import mannwhitneyu

result = mannwhitneyu(class1, class2, alternative='two-sided')
result.statistic        # statistics
result.pvalue           # p-value : p≥0.05 : 등분산 가정 타당,  p < 0.05 등분산 가정 기각









########################################################################################################
########################################################################################################
# 분산 분석
########################################################################################################
########################################################################################################

########################################################################################################
# 일원 분산 분석 : 요인의 수가 1개인 경우
########################################################################################################

import pandas as pd
df_anova = pd.DataFrame({
    'A': [10.5, 11.3, 10.8, 9.6, 11.1, 10.2, 10.9, 11.4, 10.5, 10.3],
    'B': [11.9, 12.4, 12.1, 13.2, 12.5, 11.8, 12.2, 12.9, 12.4, 12.3],
    'C': [11.2, 11.7, 11.6, 10.9, 11.3, 11.1, 10.8, 11.5, 11.4, 11.0],
    'D': [9.8, 9.4, 9.1, 9.5, 9.6, 9.9, 9.2, 9.7, 9.3, 9.4]
})
print(df_anova.head(2))

# ---------------------------------------------------------------------------
# scipy
from scipy import stats

print("=== 정규성 검정 ===")
print(stats.shapiro(df_anova['A']))
print(stats.shapiro(df_anova['B']))
print(stats.shapiro(df_anova['C']))
print(stats.shapiro(df_anova['D']))

print("\n === 등분산 검정 ===")
print(stats.levene(df_anova['A'], df_anova['B'], df_anova['C'], df_anova['D']))

print("\n === 일원 분산 분석 ===")
print(stats.f_oneway(df_anova['A'], df_anova['B'], df_anova['C'], df_anova['D']))

# ---------------------------------------------------------------------------
# stats_models ols
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

df_fertilizer = pd.read_csv(f"{data_path}/part3/ch2/fertilizer.csv", encoding='utf-8-sig')

model = ols('성장~C(비료)', df_fertilizer).fit()
result = anova_lm(model)
result


########################################################################################################
# 이원 분산 분석 : 요인의 수가 2개인 경우
########################################################################################################
df_tree = pd.read_csv(f"{data_path}/part3/ch2/tree.csv", encoding='utf-8-sig')
df_tree.head()


from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

model = ols("성장률 ~ C(나무) + C(비료) + C(나무):C(비료)", df_tree).fit()
result = anova_lm(model)
result






########################################################################################################
########################################################################################################
# 카이제곱 검정
########################################################################################################
########################################################################################################

########################################################################################################
# 적합도 검정 : 이 데이터가 특정 분포를 따르나? (관측분포 vs 이론분포)
########################################################################################################

from scipy.stats import chisquare

observed = [150, 120, 30]
expected = [0.5*300, 0.35*300, 0.15*300]
print(chisquare(observed, expected))



########################################################################################################
# 독립성 검정 : 두 변수가 독립인가? (두 범주형 변수)
########################################################################################################

from scipy.stats import chi2_contingency
# H0 : 두 변수는 독립적이다.

df_excercise = pd.DataFrame({'좋아함': [80, 90],
                   '좋아하지 않음': [30, 10]},
                  index=['남자', '여자'])

result = chi2_contingency(df_excercise)      # p < 0.05 : 독립적이지 않다.
print(result)

# row-data
df_excercise_row = pd.DataFrame({
    '성별': ['남자']*110 + ['여자']*100,
    '운동': ['좋아함']*80 + ['좋아하지 않음']*30 + ['좋아함']*90 + ['좋아하지 않음']*10
})


result = chi2_contingency(df_excercise_row.groupby(['성별','운동']).size().unstack('운동'))
# result = chi2_contingency(pd.crosstab(df_excercise_row['성별'], df_excercise_row['운동']))



########################################################################################################
# 동질성 검정 : 두 집단의 범주 분포가 같나? (집단 A 분포 vs 집단 B 분포)
########################################################################################################

df_club = pd.DataFrame([[50, 50], [30, 70]], index=['통계학과','컴퓨터공학과'], columns=['가입','미가입'])
result = chi2_contingency(df_club)      # p < 0.05 : 두 집단은 다르다
print(result)

# row-data
df_club_row = pd.DataFrame({
    '학과': ['통계학과']*100 + ['컴퓨터공학과']*100,
    '동아리가입여부': ['가입']*50 + ['미가입']*50 + ['가입']*30 + ['미가입']*70
})

result = chi2_contingency(df_club_row.groupby(['학과','동아리가입여부']).size().unstack('동아리가입여부'))
# result = chi2_contingency(pd.crosstab(df_club_row['학과'], df_club_row['동아리가입여부']))
result


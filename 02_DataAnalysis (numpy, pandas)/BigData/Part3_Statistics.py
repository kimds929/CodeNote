import pandas as pd

df_popcone = pd.DataFrame({
    'weights':[122, 121, 120, 119, 125, 115, 121, 118, 117, 127,
           123, 129, 119, 124, 114, 126, 122, 124, 121, 116,
           120, 123, 127, 118, 122, 117, 124, 125, 123, 121],
})


########################################################################################################
########################################################################################################
# 가설검정
########################################################################################################
########################################################################################################


########################################################################################################
# 단일표본 검정
########################################################################################################
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
# 일원 분산 분석
########################################################################################################

import pandas as pd
df_anova = pd.DataFrame({
    'A': [10.5, 11.3, 10.8, 9.6, 11.1, 10.2, 10.9, 11.4, 10.5, 10.3],
    'B': [11.9, 12.4, 12.1, 13.2, 12.5, 11.8, 12.2, 12.9, 12.4, 12.3],
    'C': [11.2, 11.7, 11.6, 10.9, 11.3, 11.1, 10.8, 11.5, 11.4, 11.0],
    'D': [9.8, 9.4, 9.1, 9.5, 9.6, 9.9, 9.2, 9.7, 9.3, 9.4]
})
print(df_anova.head(2))


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
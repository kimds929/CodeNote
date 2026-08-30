import numpy as np
import pandas as pd

data_path = "D:/DataScience/★GitHub_kimds929/CodeNote/02_DataAnalysis (numpy, pandas)/BigData"

df = pd.read_csv(f"{data_path}/Practice/practice_high1.csv", encoding='utf-8-sig')

# print(f"< shape {df.shape} >")

df_summary = pd.concat([df.dtypes, df.nunique(), df.isna().sum(axis=0),
                        df.agg(['min','max']).T], axis=1)\
                        .rename(columns={0:'dtypes', 1:'nunique', 2:'isna'})
# print(df_summary)

# < shape (72, 26) >
#             dtypes  nunique  isna     min      max
# id           int64       72     0       1       72
# split          str        2     0    test    train
# caff       float64       65     0   91.87    99.57
# charger        str        2     0     New      Old
# ch_t       float64       49     0    1.18     2.34
# pre        float64       67     0   57.35    63.66
# post       float64       65     0   56.53    64.11
# grp4           str        4     0      G1       G4
# score      float64       70     0    52.3    81.94
# fert           str        2     0      F1       F2
# water          str        3     0      W1       W3
# yield      float64       69     0   31.34    52.64
# acc            str        5     0       0       5p
# camp           str        2     0     BDA      ENG
# reg            str        2     0       N        Y
# disc         int64       29     0       0       30
# temp         int64       25     0      10       34
# ad           int64       69     0     155      946
# ord        float64       72     0  296.44  1221.78
# age          int64       37     0      22       65
# svc          int64        5     0       1        5
# soc          int64        5     0       1        5
# book         int64        5     0       1        5
# target       int64        2     0       0        1
# pred_prob  float64       71     0  0.0099   0.7775
# pred_cls     int64        2     0       0        1




print('-'*100)
## 1. 


print(1)
print(f"Ans : {df['caff'].mean():.4f}")
print('-'*100)


## 2.
from scipy.stats import shapiro

result = shapiro(df['caff'])

print(2)
print(f"Ans : {result.pvalue:.4f}")
print('-'*100)



## 3
from scipy.stats import ttest_1samp

result = ttest_1samp(df['caff'], 95, alternative='less')

print(3)
print(f"Ans : {result.statistic:.4f}")
print('-'*100)


## 4.
print(4)
print(f"Ans : {result.pvalue:.4f}")
print('-'*100)


## 5.
# charger        str        2     0     New      Old


charger_var = df.groupby('charger')['ch_t'].var()

f_cht = charger_var.max() / charger_var.min()

print(5)
print(f"Ans : {f_cht:.4f}")
print('-'*100)


## 6
x_new = df.query("charger=='New'")['ch_t']
x_old = df.query("charger=='Old'")['ch_t']

n_new, n_old = len(x_new), len(x_old)
var_new, var_old = x_new.var(ddof=1), x_old.var(ddof=1)

var_common = ((n_new - 1) * var_new + (n_old - 1) * var_old) / (n_new + n_old -2)

print(6)
print(f"Ans : {var_common:.4f}")
print('-'*100)
 

## 7 
from scipy.stats import levene, ttest_ind

print(levene(x_new, x_old))

result = ttest_ind(x_new, x_old, equal_var=False)

print(7)
print(f"Ans : {result.pvalue:.4f}")
print('-'*100)



## 8 
from scipy.stats import mannwhitneyu

result = mannwhitneyu(x_new, x_old)

print(8)
print(f"Ans : {result.pvalue:.4f}")
print('-'*100)


## 9 
x_pre = df['pre']
x_post =  df['post']

print(9)
print(f"Ans : {(x_pre-x_post).mean():.4f}")
print('-'*100)


## 10
from scipy.stats import shapiro
result = shapiro(x_pre-x_post)

print(10)
print(f"Ans : {result.pvalue:.4f}")
print('-'*100)



## 11
from scipy.stats import ttest_rel
result = ttest_rel(x_pre, x_post, alternative='greater')

print(11)
print(f"Ans : {result.pvalue:.4f}")
print('-'*100)


## 12
# grp4           str        4     0      G1       G4
# score      float64       70     0    52.3    81.94
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
model = ols("score ~ C(grp4)", df).fit()
result = anova_lm(model)

print(12)
print(result)
print(f"Ans : {result.loc['C(grp4)']['F']:.4f}")
print('-'*100)


## 13
print(13)
print(f"Ans : {result.loc['C(grp4)']['PR(>F)']:.4f}")
print('-'*100)



## 14
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

df['yield_'] = df['yield'].copy()
model = ols("yield_ ~ C(fert) + C(water) + C(fert):C(water)", df).fit()
result = anova_lm(model)

# yield      float64       69     0   31.34    52.64
# fert           str        2     0      F1       F2
# water          str        3     0      W1       W3


print(14)
print(result)
print(f"Ans : {result.loc['C(fert)']['PR(>F)']:.4f}")
print('-'*100)


## 15
# acc            str        5     0       0       5p
from scipy.stats import chisquare

n_acc = df['acc'].value_counts().sort_index()
expected_ratio = (0.60, 0.25, 0.08, 0.05, 0.02)
expected_n = len(df) * np.array(expected_ratio)

result = chisquare(n_acc, expected_n)

print(15)
print(f"Ans : {result.statistic:.4f}")
print('-'*100)


## 16 : 독립검증
from scipy.stats import chi2_contingency
ct_mat = pd.crosstab(df['camp'], df['reg'])
result = chi2_contingency(ct_mat)

print(16)
print(f"Ans : {result.pvalue:.4f}")
print('-'*100)


## 17
result = df[['disc','temp']].corr()

print(17)
print(f"Ans : {result.iloc[0,1]:.4f}")
print('-'*100)


## 18.
# ord        float64       72     0  296.44  1221.78
# disc         int64       29     0       0       30
# temp         int64       25     0      10       34
# ad           int64       69     0     155      946

from statsmodels.formula.api import ols
model = ols("ord ~ disc + temp + ad",df).fit()


print(18)
print(f"Ans : {model.rsquared:.4f}")
print('-'*100)

## 19.
pred = model.predict({'disc':10, 'temp':20, 'ad':500})

print(19)
print(f"Ans : {pred.iloc[0]:.4f}")
print('-'*100)


## 20.
df_train = df.query("split == 'train'")
df_test = df.query("split == 'test'")

from statsmodels.formula.api import logit
model = logit("target ~ age + svc + soc + book",df_train).fit()

pred_proba = model.predict(df_test)

from sklearn.metrics import roc_auc_score

result = roc_auc_score(df_test['target'], pred_proba)

print(20)
print(f"Ans : {result:.4f}")
print('-'*100)






# model.params[1:].abs().nlargest(5).index[0]
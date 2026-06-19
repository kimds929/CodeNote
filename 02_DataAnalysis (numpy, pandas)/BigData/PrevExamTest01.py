import numpy as np
import pandas as pd
# D:/DataScience/★GitHub_kimds929/CodeNote/"02_DataAnalysis (numpy, pandas)"/BigData
data_path = "D:/DataScience/★GitHub_kimds929/CodeNote/02_DataAnalysis (numpy, pandas)/BigData"


import warnings
warnings.filterwarnings('ignore')
    
rs = 1

df = pd.read_csv(f"{data_path}/part4/ch1/titanic.csv")

# print(f"< shape {df.shape} >")

df_summary = pd.concat([df.dtypes, df.nunique(), df.isna().sum(axis=0),
                        df.agg(['min','max']).T], axis=1)\
                            .rename(columns={0:'dtypes', 1: 'nunique', 2: 'isna'})

# print(df_summary)
# < shape (1310, 8) >
#            dtypes  nunique  isna     min       max
# pclass    float64        3     1     1.0       3.0
# survived  float64        2     1     0.0       1.0
# sex           str        2     1  female      male
# age       float64       98   264  0.1667      80.0
# sibsp     float64        7     1     0.0       8.0
# parch     float64        8     1     0.0       9.0
# fare      float64      281     2     0.0  512.3292
# embarked      str        3     3       C         S


# 1.
from scipy.stats import chi2_contingency

ct_mat = pd.crosstab(df['sex'], df['survived'])
result = chi2_contingency(ct_mat)
print(1)
# print(result)
print(f"chi2 statistic : {result.statistic:.3f}")
print('-'*100, end='\n')


# 2.
from statsmodels.formula.api import logit
model = logit("survived ~ C(sex) + sibsp + parch + fare", df).fit()

print(2)
# print(model.summary())
print(f"parch coef : {model.params.loc['parch']:.3f}")
print('-'*100, end='\n')


# 3.
print(3)
print(f"sibsp odds_ratio : {np.exp(model.params.loc['sibsp']):.3f}")
print('-'*100, end='\n')

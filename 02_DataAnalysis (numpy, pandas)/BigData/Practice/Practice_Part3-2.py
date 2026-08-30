# 출력을 원할 경우 print() 함수 활용
# 예시) print(df.head())

# getcwd(), chdir() 등 작업 폴더 설정 불필요
# 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가
import numpy as np
import pandas as pd

df = pd.read_csv("data/bcc.csv")

print(df.shape)
# 사용자 코딩
df_summary = pd.concat([df.dtypes, df.nunique(), df.isna().sum(axis=0), df.agg(['min','max']).T], axis=1)
df_summary.columns = ['dtypes', 'nunique', 'isna', 'min', 'max']
print(df_summary)

print()
print(df.head(3))


# log-Resistin
df['log_Resistin'] = np.log(df['Resistin'])
# 해당 화면에서는 제출하지 않으며, 문제 풀이 후 답안제출에서 결괏값 제출

from scipy.stats import f_oneway
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
model = ols("log_Resistin ~ Classification", df).fit()
result = anova_lm(model)
print(result)

print()
print(df.groupby('Classification')['log_Resistin'].var())


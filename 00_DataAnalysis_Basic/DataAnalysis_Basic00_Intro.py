
# Library 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats




###### pandas ###################################################################
# 데이터 불러오기
df = pd.read_clipboard()

# 데이터 내보내기
df.to_clipboard()
df.to_clipboard(index=False)


# 데이터 탐색
df      # 데이터보기
df.sample(3)      # 데이터보기 (3개만보기)


# Colum 선택하기
df['x1']                # 1개만 선택 (Series)

df[['x1', 'x2']]        # 2개이상 선택 (DataFrame)
df[['x1']]        # 1개를 DataFrame 형태로 선택 (DataFrame)


# 필터링
df[df['x2'] == 'a']
df[df['x2'] != 'a']
df[df['y'] > 10]


df[df['x2'] == 'a']['x2']       # 필터링 후 특정 column선택


# Operation (연산)
df['x1']
df['x1'].count()
df['x1'].sum()
df['x1'].mean()
df['x1'].std()

df['x1'].median()

df['x1'].agg('mean')
df['x1'].agg(['count', 'mean', 'std'])


# groupby (Pivot)
df.groupby('x2').sum()
df.groupby('x2')['x1'].sum()
df.groupby(['x2','x4'])['x1'].sum()
df.groupby(['x2','x4'])['x1'].agg(['count','mean','std'])



###### matplotlib.pyplot ###################################################################
# 그래프 그리기
plt.hist(x=df['x1'])      # histogram

plt.scatter(x=df['x1'], y=df['y'])      # scatter plot

plt.plot(df['x1'], df['y'])      # Line Plot


# boxplot
sns.boxplot(data=df, x='x2', y='x1')


box1 = df[df['x2'] == 'a']['x1']
box2 = df[df['x2'] != 'a']['x1']
plt.boxplot([box1, box2], labels=['Class_1', 'Class_2'])


# options
plt.hist(x=df['x1'], bins=30)      # 막대 갯수
plt.hist(x=df['x1'], color='skyblue')      # 채우기색
plt.hist(x=df['x1'], color='skyblue', edgecolor='grey')      # 외곽선
plt.hist(x=df['x1'], color='skyblue', edgecolor='grey', alpha=0.1)      # 투명도


plt.scatter(x=df['x1'], y=df['y'], color='orange')      # 채우기색
plt.scatter(x=df['x1'], y=df['y'], color='orange', edgecolor='blue')      # 외곽선
plt.scatter(x=df['x1'], y=df['y'], color='orange', edgecolor='blue', alpha=0.1)      # 투명도


# 다중줄수행 및 기타 옵션 
plt.figure()
plt.hist(x=df['x1'])
plt.axvline(3, color='red', ls='dashed', alpha=0.7)      # Vertical Line
plt.show()


f = plt.figure()
plt.hist(x=df['x1'])
plt.axhline(1.5, color='orange', ls='dashed', alpha=0.7)      # Horizontal Line
plt.show()

f

f = plt.figure()
plt.hist(x=df['x1'])
plt.axhline(1.5, color='orange', ls='dashed', alpha=0.7)      # Horizontal Line
plt.close()
f




# 함수 활용 ###################################################################
# function calc cpk (process capability index)
def cpk(mean, std, lsl=-np.inf, usl=np.inf, lean=False):
    if np.isnan(lsl) and np.isnan(usl):
        return np.nan
    lsl = -np.inf if np.isnan(lsl) else lsl
    usl = np.inf if np.isnan(usl) else usl

    cpk = min(usl-mean, mean-lsl) / (3 * std)
    if lean:
       sign = 1 if usl-mean < mean-lsl else -1
       cpk = 0.01 if cpk < 0 else cpk
       cpk *= sign
    return cpk

def cpk_line(X, bins=50, density=False):
    X_describe = X.describe()
    X_lim = X_describe[['min', 'max']]
    X_min = min(X_describe['min'], X_describe['mean'] - 3 * X_describe['std'])
    X_max = max(X_describe['max'], X_describe['mean'] + 3 * X_describe['std'])
    x_100Divide = np.linspace(X_min, X_max, 101)   # x 정의
    y_100Norm = (1 / (np.sqrt(2 * np.pi)*X_describe['std'])) * np.exp(-1* (x_100Divide - X_describe['mean'])** 2 / (2* (X_describe['std']**2)) )
    if not density:
        y_rev = len(X)/(bins) * (X_describe['max'] -X_describe['min'])
        y_100Norm *= y_rev
    return pd.DataFrame([x_100Divide,y_100Norm], index=[X.name, 'cpk']).T


# cpk 구하기
cpk(mean=100, std=3.3, lsl=90)
cpk(mean=100, std=3.3, lsl=90, usl=106)


# Histogram + Cpk Line
plt.hist(df['x1'])

cpk_x1 = cpk_line(X=df['x1'])
cpk_x1

plt.hist(df['x1'])
plt.plot(cpk_x1['x1'], cpk_x1['cpk'], color='red', alpha=0.7)
plt.show()











# 【 scipy 】 ================================================================
from scipy import stats

# Example Data
# test_dict = {'y': [10, 13, 20, 7, 15],
#             'x1': [2, 4, 5, 2, 4],
#             'x2': ['a', 'a', 'b', 'b', 'b'],
#             'x3': [10, 8, 5, 12, 7],
#             'x4': ['g1', 'g2', 'g1', 'g2', 'g3']}

# test_df = pd.DataFrame(test_dict)
# df = test_df.copy()

df = pd.read_clipboard()
# t-test: '두집단의 평균이 같은지?'를 비교하는 모수적 통계방법
df.agg(['mean', 'std'])

# ○ 1 Sample t ---------------------------------------------------------------
ttest_4 = stats.ttest_1samp(df['x1'], 4)   # x1 Column의 평균이 4와 같은가?
ttest_6 = stats.ttest_1samp(df['x1'], 6)   # x1 Column의 평균이 6와 같은가?

ttest_4
ttest_6

# visualization
plt.hist(df['x1'], color='skyblue', edgecolor='grey')
plt.plot(cpk_line(df['x1'])['x1'], cpk_line(df['x1'])['cpk'])
plt.axvline(df['x1'].mean(), color='blue', label='mean', alpha=0.5)
plt.axvline(4, alpha=0.7, ls='dashed', color='orange', label='4')
plt.axvline(6, alpha=0.7, ls='dashed', color='brown', label='6')
plt.legend()
plt.show()



# ○ 2 Sample t ---------------------------------------------------------------
t1_data = df[df['x2']=='a']['x1']
t2_data = df[df['x2']=='b']['x1']
df.groupby('x2')['x1'].agg(['mean','std'])

ttest_ind = stats.ttest_ind(t1_data, t2_data, equal_var=False)   # t_test : (t_value, p-value)
ttest_ind


# visualization
plt.hist([t1_data, t2_data], color=['skyblue', 'orange'], edgecolor='grey')
plt.plot(cpk_line(t1_data)['x1'], cpk_line(t1_data)['cpk'], color='skyblue', alpha=0.7)
plt.plot(cpk_line(t2_data)['x1'], cpk_line(t2_data)['cpk'], color='orange', alpha=0.7)
plt.axvline(t1_data.mean(), color='skyblue', label='t1_mean', alpha=0.5)
plt.axvline(t2_data.mean(), color='orange', label='t2_mean', alpha=0.5)
plt.legend()
plt.show()

plt.boxplot([t1_data, t2_data], vert=False, showmeans=True, meanprops={'marker':'o', 'markerfacecolor':'red', 'markeredgecolor':'none'})
plt.show()







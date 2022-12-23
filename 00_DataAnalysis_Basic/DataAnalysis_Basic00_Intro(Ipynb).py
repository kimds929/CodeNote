# %%
""" # 【 DataAnalysis Basic 】 """

# %%
""" ## Pandas (DataFrame Library) """

# %%
""" ○ Pandas Libarary 실행 """

# %%
import pandas as pd

# %%
""" ○ Data Load (데이터불러오기) """

# %%
# 데이터 파일 또는 URL-File에서 불러오기
df = pd.read_csv("https://raw.githubusercontent.com/kimds929/kimds929.github.io/master/_data/test_df.csv")
df

# %%
# 데이터 클립보드에서 불러오기
# df = pd.read_clipboard()
df

# %%
# 직접입력 Data로부터 만들기
test_dict = {'y': [10, 13, 20, 7, 15],
            'x1': [2, 4, 5, 2, 4],
            'x2': ['a', 'a', 'b', 'b', 'b'],
            'x3': [10, 8, 5, 12, 7],
            'x4': ['g1', 'g2', 'g1', 'g2', 'g3']}

df = pd.DataFrame(test_dict)
df


# %%
# 데이터 클립보드로 내보내기
df.to_clipboard()



# %%
""" ○ Display & Indexing (데이터 확인 & 인덱싱) """

# %%
df      # 데이터보기

# %%
df.sample(3)      # 데이터보기 (3개만보기)

# %%
# 1개 Colum 선택하기 (select, indexing)
df['x1']                # 1개만 선택 (Series)

# %%
# 2개이상 Colum 선택하기 (select, indexing)
df[['x1', 'x2']]        # 2개이상 선택 (DataFrame)

# %%
# 1개이상 Colum을 DataFrame형태로 선택하기 (select, indexing)
df[['x1']]        # 1개를 DataFrame 형태로 선택 (DataFrame)


# %%
""" **※ (참고) 기능어 역할** <br>
&nbsp; □.□ : 앞의 대상에 어떤 함수를 사용하거나, 변수를 불러올때 사용 <br>
&nbsp;&nbsp;&nbsp;   pd.read_csv(...) → pandas Library에서  read_csv 함수를 사용 <br>
    <br>
&nbsp; □() : 함수( input → 함수 → output) <br>
&nbsp;&nbsp;&nbsp;   pd.read_csv(...) → pandas Library에서  read_csv함수를 사용하는데 ...이라는 경로에서 불러올 예정 <br>
    <br>
&nbsp; □[] : 대상에 접근, Item추출 <br>
&nbsp;&nbsp;&nbsp;   df['x1'] → df라는 데이터프레임에서 'x1' column에 접근하여 데이터를 추출 <br>
<br>
<br>
"""

# %%
""" ○ Filtering (필터링) """

# %%
# 'equal' filtering
df[df['x2'] == 'a']

# %%
# 'not equal' filtering
df[df['x2'] != 'a']

# %%
# 'inequality' filtering
df[df['y'] > 10]

# %%
# filtering → select 'x2'
df[df['x2'] == 'a']['x2']       # 필터링 후 특정 column선택


# %%
""" ○ Operation (연산) """

# %%
# 'x1' column select
df['x1']

# %%
# 갯수
df['x1'].count()

# %%
# 합계
df['x1'].sum()

# %%
# 평균
df['x1'].mean()

# %%
# 편차
df['x1'].std()

# %%
# 합계함수로 평균값 구하기
df['x1'].agg('mean')

# %%
# 합계함수로 여러개의 통계값(갯수, 평균, 편차) 한꺼번에 구하기
df['x1'].agg(['count', 'mean', 'std'])

# %%
""" ○ groupby (Pivot) """

# %%
# 'x2'로 grouping 후 전체 Column에 대한 합계
df.groupby('x2').sum()

# %%
# 'x2'로 grouping 후 'x1'column에 대한 합계
df.groupby('x2')['x1'].sum()

# %%
# 'x2, x4' 로 grouping 후 'x1'column에 대한 합계
df.groupby(['x2','x4'])['x1'].sum()

# %%
# 'x2, x4' 로 grouping 후 'x1'column에 대한 통계값(갯수, 평균, 편차)
df.groupby(['x2','x4'])['x1'].agg(['count','mean','std'])


# %%
"""
<br>
<br>
<br>
<br>
<br>
"""




# %%
""" ## Matplotlib.pyplot & Seaborn (Graph Library) """

# %%
""" ○ matplotlib, seaborn Libarary 실행 """

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# data 확인
df


# %%
""" ○ Draw Graph """

# %%
# histogram
plt.hist(x=df['x1'])      # histogram

# %%
# scatter-plot
plt.scatter(x=df['x1'], y=df['y'])      # scatter plot

# %%
# line-plot
plt.plot(df['x1'], df['y'])      # Line Plot

# %%
# boxplot (seaborn)
sns.boxplot(data=df, x='x2', y='x1')    # boxplot (seaborn)

# %%
# boxplot (pyplot)
box1 = df[df['x2'] == 'a']['x1']
box2 = df[df['x2'] != 'a']['x1']
plt.boxplot([box1, box2], labels=['Class_1', 'Class_2'])        # boxplot (pyplot)

# %%
""" ○ Draw Option """

# %%
plt.hist(x=df['x1'], bins=30)      # 막대 갯수

# %%
plt.hist(x=df['x1'], color='skyblue')      # 채우기색

# %%
plt.hist(x=df['x1'], color='skyblue', edgecolor='grey')      # 외곽선

# %%
plt.hist(x=df['x1'], color='skyblue', edgecolor='grey', alpha=0.1)      # 투명도

# %%
plt.scatter(x=df['x1'], y=df['y'], color='orange')      # 채우기색

# %%
plt.scatter(x=df['x1'], y=df['y'], color='orange', edgecolor='blue')      # 외곽선

# %%
plt.scatter(x=df['x1'], y=df['y'], color='orange', edgecolor='blue', alpha=0.1)      # 투명도


# %%
# Histogram with Normal Distribution Curve
from scipy import stats
sns.distplot(df['x1'], fit=stats.norm, kde=False)




# %%
""" ○ Graph Draw Full Code"""

# %%
# 다중줄수행 및 기타 옵션 
plt.figure()                    # Canvas 그리기
plt.hist(x=df['x1'])            # Histogram
plt.axvline(3, color='red', ls='dashed', alpha=0.7)      # Vertical Line
plt.show()                      # plot 그리기의 종료(plt.show(), plot.close())

# %%
# Figure의 변수 저장 (plt.show)
f = plt.figure()
plt.hist(x=df['x1'])
plt.axhline(1.5, color='orange', ls='dashed', alpha=0.7)      # Horizontal Line
plt.show()

# %%
f

# %%
# Figure의 변수 저장 (plt.close)
f = plt.figure()
plt.hist(x=df['x1'])
plt.axhline(1.5, color='orange', ls='dashed', alpha=0.7)      # Horizontal Line
plt.close()

# %%
f

# %%
"""
<br>
<br>
<br>
<br>
<br>
"""


# %%
""" ## Scipy (Statistics) """

# %%
""" ○ Scipy Libarary 실행 """
from scipy import stats


# %%
df


# %%
""" ○ t-test <br>
&nbsp;$nbsp; '두집단의 평균이 같은지?'를 비교하는 모수적 통계방법
"""

# %%
df.agg(['mean', 'std'])

# %%
""" ○ 1 Sample t """

# %%
# 1-sample t
stats.ttest_1samp(df['x1'], 4)   # x1 Column의 평균이 4와 같은가?


# %%
# 1-sample t
stats.ttest_1samp(df['x1'], 6)   # x1 Column의 평균이 6와 같은가?

# %%
# # visualization (histogram)
plt.figure()
sns.distplot(df['x1'], bins=3, fit=stats.norm, kde=False, hist=True)
plt.axvline(4, alpha=0.7, ls='dashed', color='orange', label='4')
plt.axvline(6, alpha=0.7, ls='dashed', color='brown', label='6')
plt.legend()
plt.show()


# %%
""" ○ 2 Sample t """

# %%
# gruop나누기
t1_data = df[df['x2']=='a']['x1']
t2_data = df[df['x2']=='b']['x1']

# %%
t1_data

# %%
t2_data


# %%
df.groupby('x2')['x1'].agg(['mean','std'])

# %%
# 2-sample t
ttest_ind = stats.ttest_ind(t1_data, t2_data, equal_var=False)      # 두개 그룹의 평균이 같은가?
ttest_ind # 결과 : (t_value, p-value)

# %%
# visualization (Histogram)
plt.figure()
sns.distplot(t1_data, bins=3, fit=stats.norm, kde=False, hist=True, fit_kws={'color':'steelblue'})
sns.distplot(t2_data, bins=3, fit=stats.norm, kde=False, hist=True, fit_kws={'color':'orange'})
plt.axvline(t1_data.mean(), alpha=0.7, ls='dashed', color='steelblue', label='t1')
plt.axvline(t2_data.mean(), alpha=0.7, ls='dashed', color='orange', label='t2')
plt.legend()
plt.show()

# %%
# visualization (boxplot)
plt.figure()
plt.boxplot([t1_data, t2_data], showmeans=True, meanprops={'marker':'o', 'markerfacecolor':'red', 'markeredgecolor':'none'})
plt.xticks([1,2], ['t1', 't2'])
plt.show()

# %%
# visualization (vertical boxplot)
plt.figure()
plt.boxplot([t1_data, t2_data], vert=False, showmeans=True, meanprops={'marker':'o', 'markerfacecolor':'red', 'markeredgecolor':'none'})
plt.yticks([1,2], ['t1', 't2'])
plt.show()





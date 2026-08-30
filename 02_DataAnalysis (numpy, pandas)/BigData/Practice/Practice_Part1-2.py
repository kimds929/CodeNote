# 출력을 원할 경우 print() 함수 활용
# 예시) print(df.head())

# getcwd(), chdir() 등 작업 폴더 설정 불필요
# 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가

import pandas as pd

df = pd.read_csv("data/employee_performance.csv")

# 사용자 코딩

# 해당 화면에서는 제출하지 않으며, 문제 풀이 후 답안제출에서 결괏값 제출
print(df.shape)
df_summary = pd.concat([df.dtypes, df.nunique(), df.isna().sum(axis=0), df.agg(['min','max']).T], axis=1)
print(df_summary)

print('-' * 100)
# 1 
print('(1)')
df['고객만족도'] = df['고객만족도'].fillna(df['고객만족도'].mean())
print(df.isna().sum(axis=0))
print('-' * 100)
print()

# 2
print('(2)')
df = df[~df['근속연수'].isna()]
print(df.shape)
print(df.isna().sum(axis=0))
print('-' * 100)
print()

# 3
print('(3)')
custom_satisfy_3q = int(df['고객만족도'].describe()['75%'])
print(custom_satisfy_3q)
print('-' * 100)
print()
# 4
print('(4)')
df_mean_salary = df.groupby(['부서'], dropna=False)['연봉'].mean()
print(df_mean_salary.nlargest(5).map(int))
print('-' * 100)
print()


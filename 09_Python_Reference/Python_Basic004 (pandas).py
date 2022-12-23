import numpy as np
import pandas as pd


# Series
s1 = pd.Series([1,2,3,], index=['a', 'b', 'c'], name='ABC', dtype=np.int32)
s1

s1.index
s1.values
s1['b']         # 인덱스를 활용한 값 조회
s1['c'] = 9     # 인덱스를 활용한 값 변경
s1

s1['d']=5       # 값추가

s2 = pd.Series([4,5,6,7], index=s1.index)   # 인덱스 재사용하기
s2



# size : 갯수 반환
# shape : 튜플형대로 shape반환
# unique : 유일한 값만 ndarray로 반환
# count : NaN을 제외한 개수를 반환
# mean : NaN을 제외한 평균을 반환
# value_counts: NaN을 제외한 각 값들의 빈도를 반환
s2 = pd.Series([2,3,2,3,4,5,1,2,np.nan, 4,6,3,2, np.nan])
s2.size
s2.value_counts()
s2[[5,7,8]]



train = pd.read_csv('./Dataset/train.csv')
train.shape
train.info()
train.describe()
train.head()

dir(train['Name'].value_counts())
train['Sex'].value_counts()
train['Embarked'].value_counts()
train['Cabin'].value_counts()
train['Cabin'].isna().sum()

# dict to DataFrame
dict_df = {'a':100, 'b':200, 'c':300}

pd.DataFrame(dict_df, index=[0,1,2])
pd.DataFrame(dict_df, index=range(len(dict_df)))
pd.DataFrame([dict_df])


# csv File load
    # sep : 각 데이터를 구별하기 위한 구분자(seperator) 설정
    # header : header를 무시할경우, None 설정
    # index_col : index로 사용할 Column 선택
    # usecoles : 실제로 DataFrame에 로딩할 Column만 따로 설정
train_data = pd.read_csv('./Dataset/train.csv')
train_data.head()


# DataFrame Slicing, Selection
train_data['Survived']      # Series
train_data[['Survived', 'Name']]    # DataFrame

train_data[0:10]
train_data[:5]          # row Slicing
train_data.iloc[2,]         # 0 Base(순서 Base) Slicing
train_data.loc[2,'Name']    # Index, Column명 Base Slicing


# DataFrame Boolean Selection
train_data[(train_data['Age'] >=30) & (train_data['Age'] < 40) & (train_data['Pclass'] == 1)]


# 새로운 Column추가
train_data['Age'] *2
train_data['Age_Double']  = train_data['Age'] *2        # 새 Column추가
train_data.head()

train_data.insert(3,'Fare10', train_data['Fare']/10)    # 4번째칸(0,1,2,3)에 'Fare10' Column추가

# Column삭제
train_data.drop('Age_Double', axis=1)
train_data.drop(['Fare10','Age_Double'], axis=1)
train_data.drop(['Fare10','Age_Double'], axis=1, inplace=True)

# if문을 통한 새로운 Column생성
# dict to DataFrame
dict_data = [{'a':1, 'b':10, 'c':100},{'a':2, 'b':20, 'c':200}]
dict_df = pd.DataFrame(dict_data)
dict_df

dict_df.apply(lambda x: print(x))
dict_df.apply(lambda x: print(x[1]))
dict_df.apply(lambda x: print(x['b']), axis=1)

dict_df['a']>1
if dict_df['a']>1:
    print(dict_df['a'])

train_data.apply(lambda x: '1stClass' if x['Pclass']==1 else '',axis=1)


# 변수간의 상관관계
import matplotlib.pyplot as plt
train_data.head()

    # 상관관계
train_data.corr()   # 상관관계 Matrix
plt.matshow(train_data.corr())      # 상관관계 시각화


# 결측치 처리 NaN(NA : 결측치)값
train_data.info()
train_data.isna()
train_data.isna().sum()     # Column별 Na값 갯수

    # NA값 데이터에서 삭제(dropna)
train_data.dropna()
train_data.dropna(subset=['Age', 'Cabin'])  # 특정 Column의 Na값이 있는 Row만 삭제
train_data.dropna(axis=1)   # Column을 삭제

    # NA값 치환(fillna)
train_data['Age'].mean()
train_data['Age'].fillna(train_data['Age'].mean())      # 'Age' Column의 결측값을 'Age'Column의 평균값으로 치환

# 생존자 나이 평균
age_survived = train_data[train_data['Survived'] == 1]['Age'].mean()
age_death = train_data[train_data['Survived'] == 0]['Age'].mean()

print(age_survived, age_death)

# 실제 데이터에 적용
train_data.loc[train_data['Survived'] == 1, 'Age'] = train_data[train_data['Survived'] == 1]['Age'].fillna(age_survived)
train_data.loc[train_data['Survived'] == 0, 'Age'] = train_data[train_data['Survived'] == 0]['Age'].fillna(age_death)

# (×) train_data[train_data['Survived'] == 1]['Age'] = train_data[train_data['Survived'] == 1]['Age'].fillna(age_survived)
# (×) train_data[train_data['Survived'] == 0]['Age'] = train_data[train_data['Survived'] == 0]['Age'].fillna(age_death)
# train_data['Age'] = train_data.apply(lambda x: train_data[train_data['Survived']==1]['Age'].mean() if x['Survived']==1 else train_data[train_data['Survived']==0]['Age'].mean(), axis=1)



# 숫자데이터의 범주형 데이터화
import math
train_data = pd.read_csv('./Dataset/train.csv')
train_data.info()
train_data.head()
train_data['Pclass'] = train_data['Pclass'].astype(str)  # 타입바꾸기

def age_categorise(age):
    if math.isnan(age):
        return -1
    return int(np.floor(age/10) * 10)

train_data['Age'].apply(age_categorise)

# Cut을 통한 Group화
train_data['Age'].hist()
train_data['Age'].describe()
age_group = pd.cut(train_data['Age'], np.arange(0,101,10), right=False)
age_group.value_counts()


# 범주형 데이터 전처리 하기 (one-hot encoding)
pd.get_dummies(train_data, columns=['Pclass','Sex','Embarked']).info()
pd.get_dummies(train_data, columns=['Pclass','Sex','Embarked'], drop_first=True).info()     # 첫번째 Level값은 제거



# 그룹핑 이해하기 (Groupby)
class_group = train_data.groupby('Pclass')
class_group.groups   # Group, Index값

gender_group = train_data.groupby('Sex')
gender_group.groups   # Group, Index값

class_group.count()
class_group.sum()
class_group.mean()
class_group.max()


multi_group = train_data.groupby(['Pclass','Sex'])
multi_group.mean()
multi_group.mean().loc[(2,'female')]
multi_group.mean()['Survived']


# index처리 - set_index : index 변경
train_data.head()
train_data.set_index(['Pclass','Sex'], inplace=True)
train_data.head()

# index처리 - reset_index : index 초기화
train_data.head()
train_data.reset_index(inplace=True)
train_data.head()



# index를 이용한 groupby
train_data.set_index('Embarked', inplace=True)
train_data.head()

train_data.groupby(level=0).mean()
train_data.reset_index(inplace=True)

train_data.set_index('Age').groupby(age_categorise).mean()
train_data.set_index(['Pclass','Sex']).groupby(level=[0,1]).mean()


# aggregate 집계함수 사용
train_data.set_index(['Pclass','Sex']).groupby(level=[0,1]).aggregate([np.mean, np.sum, np.max])
train_data.set_index(['Pclass','Sex']).groupby(level=[0,1]).aggregate({'Age':np.mean, 'Fare':[np.mean, np.std]})
# train_data.set_index(['Pclass','Sex']).groupby(level=[0,1]).agg({'Age':np.mean, 'Fare':[np.mean, np.std]})


# Transform 함수 이해
def percentile(column, p=0.3):
    return np.quantile(column, p)

percentile(train_data['Fare'])
train_data.groupby('Pclass').mean()
train_data.groupby('Pclass').agg(percentile)
train_data.groupby('Pclass').transform(np.mean)

train_data.groupby(['Pclass','Sex']).mean()
train_data.groupby(['Pclass','Sex']).transform(np.mean)



# pivot과 pivot Table
df = pd.DataFrame({
    '지역': ['서울', '서울', '서울', '경기', '경기', '부산', '서울', '서울', '부산', '경기', '경기', '경기'],
    '요일': ['월요일', '화요일', '수요일', '월요일', '화요일', '월요일', '목요일', '금요일', '화요일', '수요일', '목요일', '금요일'],
    '강수량': [100, 80, 1000, 200, 200, 100, 50, 100, 200, 100, 50, 100],
    '강수확률': [80, 70, 90, 10, 20, 30, 50, 90, 20, 80, 50, 10]
                  })

df
df.pivot('지역', '요일')
df.pivot(index='지역', columns='요일', values='강수량')


# pivot과의 차이점은 중복데이터에 대한 처리가 가능
df = pd.DataFrame({
    '지역': ['서울', '서울', '서울', '경기', '경기', '부산', '서울', '서울', '부산', '경기', '경기', '경기'],
    '요일': ['월요일', '월요일', '수요일', '월요일', '화요일', '월요일', '목요일', '금요일', '화요일', '수요일', '목요일', '금요일'],
    '강수량': [100, 80, 1000, 200, 200, 100, 50, 100, 200, 100, 50, 100],
    '강수확률': [80, 70, 90, 10, 20, 30, 50, 90, 20, 80, 50, 10]
                  })
df
pd.pivot_table(df, index='지역', columns='요일', aggfunc=np.mean)
pd.pivot_table(df, index='지역', columns='요일', aggfunc=np.mean, values='강수량')



# stack과 unstack
df = pd.DataFrame({
    '지역': ['서울', '서울', '서울', '경기', '경기', '부산', '서울', '서울', '부산', '경기', '경기', '경기'],
    '요일': ['월요일', '화요일', '수요일', '월요일', '화요일', '월요일', '목요일', '금요일', '화요일', '수요일', '목요일', '금요일'],
    '강수량': [100, 80, 1000, 200, 200, 100, 50, 100, 200, 100, 50, 100],
    '강수확률': [80, 70, 90, 10, 20, 30, 50, 90, 20, 80, 50, 10]
                  })

new_df = df.set_index(['지역', '요일'])
new_df

    # unstack
new_df.unstack(level=0)
# new_df.unstack(level='지역')
new_df.unstack(level=1)
# new_df.unstack(level='요일')

# stack
unstack_df = new_df.unstack(level=0)
unstack_df

unstack_df.stack(level=0)
unstack_df.stack(level=1)

# melt
test_df = pd.DataFrame([{'a':10, 'b1':1, 'b2':2, 'b3':3, 'd':'abc'},
    {'a':20, 'b1':2, 'b2':4, 'b3':3, 'd':'fhg'},
     {'a':30, 'b1':6, 'b2':5, 'b3':4, 'd':'qwe'}])
test_df

test_df.melt(value_vars=['b1','b2','b3'], var_name='b_level', value_name='b_value')
test_df.melt(id_vars=['a','d'])

test_df.columns
test_df.columns.drop(['b1','b2','b3'])
test_df.melt(id_vars=test_df.columns.drop(['b1','b2','b3']), var_name='b_level', value_name='b_value')



# merge, join
customer = pd.DataFrame({'customer_id' : np.arange(6), 
                    'name' : ['철수'"", '영희', '길동', '영수', '수민', '동건'], 
                    '나이' : [40, 20, 21, 30, 31, 18]})


orders = pd.DataFrame({'customer_id' : [1, 1, 2, 2, 2, 3, 3, 1, 4, 9], 
                    'item' : ['치약', '칫솔', '이어폰', '헤드셋', '수건', '생수', '수건', '치약', '생수', '케이스'], 
                    'quantity' : [1, 2, 1, 1, 3, 2, 2, 3, 2, 1]})


customer
orders

pd.merge(left=customer, right=orders, on='customer_id')
pd.merge(left=customer, right=orders, on='customer_id', how='left')
pd.merge(left=customer, right=orders, on='customer_id', how='right')
pd.merge(left=customer, right=orders, on='customer_id', how='outer')




a = [1,2,3]
a.pop(3)









# --- Pandas실습 ----------------------------------------------------------------------------------------

s = pd.Series([1, 2, 3, 5, np.nan, 6, 8])
s

dates = pd.date_range('20130101', periods=6)
dates

df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=['A','B','C','D'])
df
df.head(3)
df.index
df.columns
df.values
df.info()
df.describe()
df.sort_values(by='B', ascending=False)
df['A']
df[0:3]
df['20130102':'20130104']

df.loc['20130102']
df.loc['2013-01-02']
df.loc[dates[1]]
df.loc[:,['A','B']]
df.loc['20130102':'20130104',['A','B']]
df.loc['2013-01-02',['A','B']]
df.loc[dates[0],['A']]
df.loc[dates[0],'A']

df.iloc[3]
df.iloc[3:5,0:2]
df.iloc[[1,2,4],[0,2]]

df[df['A']>0]
df[df>0]

df2 = df.copy()
df2

df2['E'] = ['One','One','Two','Three','Four','Three']
df2['E'].isin(['Two','Four'])
df2[df2['E'].isin(['Two','Four'])]

df
df.apply(np.cumsum)  # Column방향의 누적합
df.apply(np.mean)   # Column방향의 평균
df.apply(lambda x: x.max() - x.min())   # Column방향의 max값 - min값

df.apply(np.mean)


# --- Pandas실습 : concat ----------------------------------------------------------------------------------------
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'], 
                    'B': ['B0', 'B1', 'B2', 'B3'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']},
                   index=[0, 1, 2, 3])

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                    'B': ['B4', 'B5', 'B6', 'B7'],
                    'C': ['C4', 'C5', 'C6', 'C7'],
                    'D': ['D4', 'D5', 'D6', 'D7']},
                   index=[4, 5, 6, 7])

df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                    'B': ['B8', 'B9', 'B10', 'B11'],
                    'C': ['C8', 'C9', 'C10', 'C11'],
                    'D': ['D8', 'D9', 'D10', 'D11']},
                   index=[8, 9, 10, 11])

df4 = pd.DataFrame({'B': ['B2', 'B3', 'B6', 'B7'], 
                    'D': ['D2', 'D3', 'D6', 'D7'],
                    'F': ['F2', 'F3', 'F6', 'F7']},
                   index=[2, 3, 6, 7])



pd.concat([df1,df2,df3])
result = pd.concat([df1,df2,df3], keys=['x', 'y','z'])
result
result.index
result.index.get_level_values(0)
result.index.get_level_values(1)

# concat 함수는 index를 기준으로 합침
pd.concat([df1, df4], axis=1)
pd.concat([df1, df4], axis=1, join='inner')
pd.concat([df1, df4], ignore_index=True)    # ignore_index=True : index를 무시하고 열을 기준으로 합침


# --- Pandas실습 : merge ----------------------------------------------------------------------------------------
left = pd.DataFrame({'key': ['K0', 'K4', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})

right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})


pd.merge(left,right, on='key')
pd.merge(left,right, on='key', how='left')
pd.merge(left,right, on='key', how='right')
pd.merge(left,right, on='key', how='outer')
pd.merge(left,right, on='key', how='inner')\
    
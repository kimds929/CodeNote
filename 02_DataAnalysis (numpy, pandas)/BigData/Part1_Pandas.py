import numpy as np
import pandas as pd

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.float_format', "{:.10f}".format)

np.array(dir(pd))
help(pd.set_option)
dir(pd.set_option)
pd.__all__


# -----------------------------------------------------------------------------------------
df_coffee = pd.DataFrame({
    "메뉴":['아메리카노', '카페라떼', '카페모카', '카푸치노', '에스프레소', '밀크티', '녹차'],
    "가격":[4500, 5000, 5500, 5000, 4000, 5900, 5300],
    "칼로리":[10, 110, 250, 110, 20, 210, 0],
})
df_coffee['날짜'] = pd.date_range(start='2024-01-01', end='2024-01-07')

# Correlation(상관계수) : -1 ~ 1 사이의 값으로 표현되는 수치로, 두 변수 간의 선형 관계의 강도와 방향을 나타냅니다.

df_coffee[['가격','칼로리']].corr()     # correlation
df_coffee[['가격','칼로리']].cov()      # covariance
df_coffee.corr(numeric_only=True)


# -----------------------------------------------------------------------------------------
# 중복 값이 있는 데이터 생성
df_car = pd.DataFrame({
    "car":['Sedan','SUV','Sedan','SUV','SUV','SUV','Sedan','Sedan','Sedan','Sedan','Sedan'],
    "size":['S','M','S','S','M','M','L','S','S', 'M','S']
})

# 중복값제외하고 몇개씩 class가 있는지?
df_car.nunique()
df_car.drop_duplicates().value_counts()


# Describe
df_car.describe()

df_coffee.describe(include='all')        # 전체 datatype을 요약
df_coffee.describe(include='object')        # object 타입 컬럼만 골라서 요약하라
df_coffee.describe(include='number')        # number 타입 컬럼만 골라서 요약하라
df_coffee.describe(include='datetime')        # datetime 타입 컬럼만 골라서 요약하라

# -----------------------------------------------------------------------------------------
# 정렬
df_coffee.sort_values('가격', axis=0)

# 값 변경
df_coffee.replace('녹차', '그린티')

change = {'아메리카노':'룽고', '녹차':'그린티'}
df_coffee.replace(change)



# -----------------------------------------------------------------------------------------
df_study = pd.DataFrame({'A': ['데이터 분석', '기본 학습서', '퇴근 후 열공'],
                   'B': [10, 20, 30],
                   'C': ['ab cd', 'AB CD', 'ab cd ']
                   })

# replace
df_study['A'].replace('데이터 분석', ' 데이터 시각화')    # (단어 전체 매칭) A열에서 '데이터 분석'이라는 단어를 ' 데이터 시각화'로 대체
df_study['A'].str.replace('분석', '시각화')     # (단어 부분 매칭) A열에서 '분석'이라는 단어를 '시각화'로 대체

# contains / isin
df_study['A'].isin(['데이터 분석'])      # (단어 전체 포함) A열에 '데이터 분석'이라는 단어가 포함되어 있는지 여부를 True/False로 반환
df_study['A'].str.contains('데이터')      # (단어 부분 포함) A열에 '데이터'라는 단어가 포함되어 있는지 여부를 True/False로 반환


# strip
df_study['C'].str.strip()      # C열의 각 문자열에서 양쪽 공백을 제거
df_study['C'].str.lstrip()      # C열의 각 문자열에서 왼쪽 공백을 제거
df_study['C'].str.rstrip()      # C열의 각 문자열에서 오른쪽 공백을 제거


# sum
df_study.sum()
df_study.sum(numeric_only=True)


# nan-sum
df_coffee.loc[5, '가격'] = np.nan
df_coffee['가격'].sum()      # 가격 열의 합계를 계산하되, nan 값은 무시하고 계산
df_coffee['가격'].sum(skipna=False)      # 가격 열의 합계를 계산하되, nan 값이 포함된 경우 결과도 nan로 반환

df_coffee['가격'] = df_coffee['가격'].fillna(4500)
df_coffee['가격'].sum()


# mode
df_coffee['가격'].mode()      # 가격 열에서 가장 빈도가 높은 값을 반환
df_coffee['가격'].value_counts()      # 가격 열에서 각 값이 몇 번 나타나는지 계산하여 반환

df_car['car'].mode()      # car 열에서 가장 빈도가 높은 값을 반환

  
# nlargest / nsmallest
df_coffee['가격'].nlargest(3)      # 가격 열에서 가장 큰 3개의 값을 반환
df_coffee['가격'].nsmallest(3)      # 가격 열에서 가장 작은 3개의 값을 반환

# idxmax / idxmin
df_coffee['가격'].idxmax()  # 가격 열에서 가장 큰 값이 위치한 인덱스를 반환
df_coffee['가격'].idxmin()  # 가격 열에서 가장 작은 값이 위치한 인덱스를 반환



# -----------------------------------------------------------------------------------------
df_score = pd.DataFrame({
    "반": ["A반", "A반", "B반"],  # 반 정보 추가
    "이름": ["쿼카", "알파카", "시바견"],
    "수학": [90, 93, 85],
    "영어": [92, 84, 86],
    "국어": [91, 94, 83]
})

# melt : 데이터프레임을 길게 변환하는 함수로, 특정 열을 고정하고 나머지 열을 변수와 값으로 분리하여 새로운 데이터프레임을 생성
df_score.melt(id_vars=['이름'], var_name='과목', value_name='점수')     # 이름 열을 고정하고, 나머지 열을 과목과 점수로 분리하여 새로운 데이터프레임을 생성
df_score.melt(id_vars=['이름'], value_vars=['수학', '영어'], var_name='과목', value_name='점수')        #  이름 열을 고정하고, 수학과 영어 열을 과목과 점수로 분리하여 새로운 데이터프레임을 생성
df_score.melt(id_vars=['반', '이름'], var_name='과목', value_name='점수')  # id_vars로 고정할 열을 지정하고, var_name과 value_name으로 새로운 열의 이름을 지정하여 데이터프레임을 길게 변환


# set_index + stack : set_index로 특정 열을 인덱스로 설정한 후, stack으로 열을 행으로 변환하여 데이터프레임을 길게 변환
df_score.set_index(['반','이름']).stack().to_frame().reset_index().rename(columns={'level_2':'과목', 0:'점수'})


# -----------------------------------------------------------------------------------------

df_fruit = pd.DataFrame({
    '과일': ['딸기', '블루베리', '딸기', '블루베리', '딸기', '블루베리', '딸기', '블루베리'],
    '가격': [1000, None, 1500, None, 2000, 2500, None, 1800]  # 결측값 포함
})


# transform + fillna : transform으로 그룹별 평균 가격을 계산한 후, fillna로 결측값을 해당 그룹의 평균 가격으로 채우는 방법
price = df_fruit.groupby('과일')['가격'].transform('mean')
df_fruit['가격'] = df_fruit['가격'].fillna(price)



##########################################################################################
# 시계열 데이터



import pandas as pd
datetime_data = {
    'Date1': ['2024-02-17', '2024-02-18', '2024-02-19', '2024-02-20'],
    'Date2': ['2024:02:17', '2024:02:18', '2024:02:19', '2024:02:20'],
    'Date3': ['24/02/17', '24/02/18', '24/02/19', '24/02/20'],
    'Date4': ['02/17/2024', '02/18/2024', '02/19/2024', '02/20/2024'],
    'Date5': ['17-Feb-2024', '18-Feb-2024', '19-Feb-2024', '20-Feb-2024'],
    'DateTime1': ['24-02-17 13:45:30', '24-02-18 14:55:45', '24-02-19 15:30:15', '24-02-20 16:10:50'],
    'DateTime2': ['2024-02-17 13-45-30', '2024-02-18 14-55-45', '2024-02-19 15-30-15', '2024-02-20 16-10-50'],
    'DateTime3': ['02/17/2024 01:45:30 PM', '02/18/2024 02:55:45 PM', '02/19/2024 03:30:15 AM', '02/20/2024 04:10:50 AM'],
    'DateTime4': ['17 Feb 2024 13:45:30', '18 Feb 2024 14:55:45', '19 Feb 2024 15:30:15', '20 Feb 2024 16:10:50']
}

df_date = pd.DataFrame(datetime_data)
# df_date.to_csv("date_data.csv", index=False)
df_date.info()
for c in df_date.columns:
    format = None
    if c == 'Date2':
        format = '%Y:%m:%d'    
    elif c == 'Date3':
        format = '%y/%m/%d'
    elif c == 'DateTime1':
        format = '%y-%m-%d %H:%M:%S'
    elif c == 'DateTime2':
        format = '%Y-%m-%d %H-%M-%S'
    df_date[c] = pd.to_datetime(df_date[c], format=format, errors='coerce')  # errors='coerce'는 변환할 수 없는 값을 NaT로 처리하도록 지정하는 옵션입니다.


df_date['year'] = df_date['DateTime1'].dt.year      # DateTime1 열에서 연도 정보 추출
df_date['month'] = df_date['DateTime1'].dt.month    # DateTime1 열에서 월 정보 추출
df_date['day'] = df_date['DateTime1'].dt.day        # DateTime1 열에서 일 정보 추출
df_date['hour'] = df_date['DateTime1'].dt.hour      # DateTime1 열에서 시간 정보 추출
df_date['minute'] = df_date['DateTime1'].dt.minute  # DateTime1 열에서 분 정보 추출
df_date['second'] = df_date['DateTime1'].dt.second  # DateTime1 열에서 초 정보 추출

df_date['weekday'] = df_date['DateTime1'].dt.weekday  # DateTime1 열에서 요일 정보 추출

df_date.T

df_date['DateTime1'].dt.to_period('Y')      # 연 단위로 변환
df_date['DateTime1'].dt.to_period('Q')      # 분기 단위로 변환
df_date['DateTime1'].dt.to_period('M')      # 월 단위로 변환
df_date['DateTime1'].dt.to_period('D')      # 일 단위로 변환
df_date['DateTime1'].dt.to_period('h')      # 시간 단위로 변환
df_date['DateTime1'].dt.to_period('min')      # 분 단위로 변환
df_date['DateTime1'].dt.to_period('s')      # 초 단위로 변환


day_delta = pd.Timedelta(days=7)  # 7일을 나타내는 Timedelta 객체 생성
df_date['DateTime4'] + day_delta  # DateTime4 열의 각 날짜에 7일을 더하는 연산 수행


hour_delta = pd.Timedelta(hours=11, minutes=15)  # 11시간 30분을 나타내는 Timedelta 객체 생성
delta = df_date['DateTime4'] + hour_delta - df_date['DateTime4'].shift() 

delta.dt.round('D')     # 일단위로 반올림
delta.dt.round('h')    # 시간 단위로 반올림
delta.dt.total_seconds()     # Timedelta 객체의 총 초 단위로 변환





###################################################################################################################
df_test = pd.DataFrame({
    "메뉴": ['아메리카노', '카페라떼', '에스프레소', '카페모카', '바닐라라떼'],
    "가격": [4500, 5000, 4000, 5900, 5300],
    "칼로리": [10, 110, np.nan, 210, np.nan],
    "원두": ['과테말라', '브라질', '과테말라', np.nan, np.nan]
})

# Q1. '칼로리' 컬럼의 결측치를 칼로리 데이터 중 최솟값으로 채우시오.
df_test['칼로리'].fillna(df_test['칼로리'].min(), inplace=True)

# Q2. '원두' 컬럼의 결측치를 원산지 데이터 중 최빈값으로 채우시오.
df_test['원두'].fillna(df_test['원두'].mode()[0], inplace=True)


# Q3. 가격이 5,000 이상인 데이터의 수를 구하시오.
df_test.query('가격 >= 5000').shape[0]

# Q4. '이벤트가' 컬럼을 만들고 기존 가격에서 50% 할인된 가격을 채우시오.
df_test['이벤트가'] = (df_test['가격'] / 2).astype(int)

# Q5. '칼로리' 컬럼을 삭제하시오.
df_test = df_test.drop('칼로리', axis=1)

# Q6. 위에서부터 3개의 행만 선택하시오. (loc 사용)
df_test.loc[:2]

# Q7. 위에서부터 3개의 행만 선택하시오. (iloc 사용)
df_test.iloc[:3]

# Q8. 주어진 데이터(df)에서 아래 값을 loc를 활용해 데이터프레임으로 출력하시오.
# - 카페라떼 5000
# - 에스프레소 4000
df_test.loc[1, ['메뉴','가격']]
df_test.loc[2, ['메뉴','가격']]


# Q9. 주어진 데이터(df)에서 아래 값을 iloc를 활용해 데이터프레임으로 출력하시오.

# - 카페라떼 5000
# - 에스프레소 4000

# Q10. 메뉴 중 가격이 가장 비싼 순으로 정렬해 상위 3개의 값을 구하시오.
df_test['가격'].nlargest(3)
df_test.loc[df_test['가격'].nlargest(3).index]

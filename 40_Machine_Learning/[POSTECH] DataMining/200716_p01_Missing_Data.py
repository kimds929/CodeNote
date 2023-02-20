# Python_basic_posco.ipynb
# --- [ Main ] ----------------------------------------------------------------------------


import os
import pandas as pd
import numpy as np
import pickle
import xlrd

current_dir = os.getcwd()
# D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/dataset_city.csv
absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'


# --- [ pandas ] ----------------------------------------------------------------------------

df = pd.read_csv(absolute_path + 'dataset_city.csv')

df.head(5)

df['col2']
df.loc[0:2, 'col3':'col6']
df.loc[0:2]
df.loc[:,'col3':'col6']

df.dtypes
df.describe()
df['col3'].mean()


df['col2']
df.loc[0:2, 'col3':'col6']
df.loc[0:2]
df.loc[:,'col3':'col6']

idx1 = df['col5'] >= 1
idx1
df[idx1]

idx4 = df.dtypes == 'object'   #  data type이 'object'인 것만 가져오기
df.iloc[:, idx4.array]         #
df.iloc[:, (df.dtypes == 'object').array]


# ------------ 실습
idx_r1 = ((df['col3'] >=df['col3'].mean())& (df['col4'] >= df['col4'].mean())).array
idx_c1 = (df.dtypes == 'float64').array

df.iloc[idx_r1, idx_c1]


# 두 데이터 프레임 합치기
df1 = df[['col2', 'col4']]
df2 = df[['col1', 'col3']]

df3 = pd.concat([df1, df2], axis=1)


df.head()

df.isnull().sum()  # column별로 결측치 갯수를 나타내줌
df.isnull().sum(1) # row별로 결측치 갯수를 나타내줌
df[df.isnull()==False]


df.fillna(0)

values = {'col3':0, 'col4':1}
df.fillna(values)



# 연속형 변수의 NULL값을 평균으로 채우기
idx_contin = df.dtypes=='float64'
df_contin=df.loc[:,idx_contin]
print(df_contin)
df_contin_fill = df_contin.fillna(df_contin.mean())
print(df_contin_fill)

# categorical 변수를 mode(최빈값)으로 채우기
idx_cat = df.dtypes=='object'
idx_cat                                 
df_cat = df.loc[:,idx_cat]
print(df_cat)
df_cat


df['col1'].value_counts()

# mode() the value that appears most often. (최빈값)
# DataFrame.mode(self, axis=0, numeric_only=False, dropna=True) 
cat_mode = df_cat.mode().iloc[0]
df_cat.fillna(cat_mode)



# -- 실습2 
df2 = pd.read_excel(absolute_path + 'dataset_product.xlsx')

df2.head()
df2.dtypes
df2.dtypes.value_counts()

# object변수
df2.dtypes == 'object'
df2_obj = df2.iloc[:,(df2.dtypes == 'object').array]
df2_obj_mode = df2_obj.mode().iloc[0,:]
df2_obj_fill_mode = df2_obj.fillna(df2_obj_mode)


# numeric변수
df2_num = df2.iloc[:,((df2.dtypes == 'float64') | (df2.dtypes == 'int64')).array]
df2_num_mean = df2_num.mean().fillna(0)
df2_num_fill_mean = df2_num.fillna(df2_num_mean)

df2_fill = pd.concat([df2_obj_fill_mode, df2_num_fill_mean], axis=1)
df2_fill.isnull().sum()














# --- [ numpy ] ----------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import pickle
import xlrd


current_dir = os.getcwd()
# D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/dataset_city.csv
absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'

df = pd.read_csv(absolute_path + 'dataset_city.csv')
df2 = pd.read_excel(absolute_path + 'dataset_product.xlsx')




# dataframe → numpy
df_np = df.to_numpy() # 자동 형변환 int > float > object 으로 통일됨
    # 일반적으로 data_type을 동일한 set을 numpy로 바꾸어야 함

df_np.shape

# 숫자형 데이터만 가져오기
idx_contin = df.dtypes=='float64'
df_contin=df.loc[:,idx_contin]

df_contin_np = df_contin.values     # dataframe to numpy
print(df_contin_np)
df_contin_np.dtype
df_contin_np.shape
df_contin_np[0,0]
df_contin_np[0,:]   # 첫번째 row만 가져오기


# string 데이터만 가져오기
idx_cat = df.dtypes=='object'
idx_cat                                 
df_cat = df.loc[:,idx_cat]

df_cat_np = df_cat.values     # dataframe to numpy

    
    # 3.2 조건에 맞는 행이나 열 선택, 최대, 최소, 행렬 붙이기
df_contin_np

# 첫번째 열이 2 이상인 행 선택
idx7 = df_contin_np[:,1]>=2
idx7




df_contin.mean().mean()    #  데이터 전체 평균 (nan무시)
df_contin.mean(0)          # column별 평균 (nan무시)
    # df_contin.mean()
df_contin.mean(1)          # row별 평균 (nan무시)

# 연산 nan고려
print('전체 평균:',df_contin_np.mean()) # 전체 평균 출력
print('각 열별 평균:',df_contin_np.mean(0)) # (column단위) 각 column(열)별 평균 출력
print('각 행별 평균:',df_contin_np.mean(1)) # (row단위) 각 row(행)별 평균 출력


# nan 무시
print('전체 평균(nan무시):',np.nanmean(df_contin_np)) # 전체 평균 출력 (nan무시)
print('각 열별 평균(nan 무시):',np.nanmean(df_contin_np, 0)) # (column단위) 각 column(열)별 평균 출력 (nan무시)
print('각 행별 평균(nan 무시):',np.nanmean(df_contin_np, 1)) # (row단위) 각 row(행)별 평균 출력 (nan무시)




# 데이터 이어붙이기 ------------------------------------------------------------
df_np1 = df_contin_np[:3,:5]
df_np2 = df_contin_np[3:6,:5]

print(df_np1)
print(df_np1.shape)

print(df_np2)
print(df_np2.shape)

# 두 행렬을 세로로 이어 붙이기
df_np3 = np.concatenate((df_np1,df_np2))    # 위아래로 이어붙이기
print(df_np3)
print(df_np3.shape)

# 두 행렬을 가로로 이어 붙이기
df_np4=np.concatenate((df_np1,df_np2), axis=1)  # 옆으로 이어붙이기
print(df_np4)
print(df_np4.shape)


# 결측치 처리하기 ------------------------------------------------------------
np.isnan(df_contin_np)  # 결측치 확인

idx_missing = np.isnan(df_contin_np)
print(idx_missing)

print('전체 결측치의 개수:', np.isnan(df_contin_np).sum()) # 전체 결측치 갯수 확인
print('열별 결측치의 개수:', np.isnan(df_contin_np).sum(0)) # column(열)별 결측치 갯수 확인
print('행별 결측치의 개수:', np.isnan(df_contin_np).sum(1)) # row(행)별 결측치 갯수 확인


    # 실습 : 결측치가 200,000개 이상인 열들의 전체 평균과 결측치가 200,000개 미만인 열들의 전체 평균 비교하기

idx_col1 =np.isnan(df_contin_np).sum(0) >=200000
df_contin_np_20up = df_contin_np[:,idx_col1]

idx_col2 =np.isnan(df_contin_np).sum(0) <200000
df_contin_np_20down = df_contin_np[:,idx_col2]


print(f'결측칙 20만개 미만 평균 : {np.nanmean(df_contin_np_20down)}')
print(f'결측칙 20만개 이상 평균 : {np.nanmean(df_contin_np_20up)}')



# 결측치가 50% 이상인 열 제외
Num_obs = df_contin_np.shape[0]
Num_missing_by_column = idx_missing.sum(0)
idx_col_valid = Num_missing_by_column<(Num_obs * 0.5) # 열별 결측치의 수 < 전체 관측치의 수 
print(idx_col_valid)

df_contin_np2 = df_contin_np[:,idx_col_valid]
print(df_contin_np2)
















# --- [ sklearn ] ----------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import pickle
import xlrd

from sklearn.impute import KNNImputer
from sklearn.impute import *
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


current_dir = os.getcwd()
# D:/Python/★★Python_POSTECH_AI/Dataset_AI/preprocessing_test_data/dataset_city.csv
absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/preprocessing_test_data/'

df = pd.read_csv(absolute_path + 'dataset_city.csv')
df2 = pd.read_excel(absolute_path + 'dataset_product.xlsx')


# 숫자형 데이터만 가져오기
idx_contin = df.dtypes=='float64'
df_contin=df.loc[:,idx_contin]
df_contin_np = df_contin.values     # dataframe to numpy


# string 데이터만 가져오기
idx_cat = df.dtypes=='object'
df_cat = df.loc[:,idx_cat]
df_cat_np = df_cat.values     # dataframe to numpy


# 결측치가 50% 이상인 열 제외
idx_missing = np.isnan(df_contin_np)    # 결측치 확인
Num_obs = df_contin_np.shape[0]
Num_missing_by_column = idx_missing.sum(0)
idx_col_valid = Num_missing_by_column<(Num_obs * 0.5) # 열별 결측치의 수 < 전체 관측치의 수 
print(idx_col_valid)

df_contin_np2 = df_contin_np[:,idx_col_valid]
print(df_contin_np2)



# 평균으로 채워넣기
# ?SimpleImputer
    # SimpleImputer( *, missing_values=nan, strategy='mean', fill_value=None,
    #     verbose=0, copy=True, add_indicator=False)

imp_mean = SimpleImputer(strategy='mean', missing_values=np.nan)    # 1. 모형 선언,  결측치는 평균으로 채워넣고 결측치는 np.nan 값을 가지고
imp_mean.fit(df_contin_np2) # 2. 각 열별 평균을 EQ_contin_np2 행렬을 기준으로 계산
imp_mean.statistics_    # 계산 평균값들이 무엇인지 보여줌
print(imp_mean.transform(df_contin_np2))    ## 3. transform : statistics_에 저장된 값으로 결측치를 채워 넣겠다. 


# median으로 채워넣기
imp_median = SimpleImputer(strategy='median', missing_values=np.nan) # 선언, median으로 결측치를 채워 넣겠다
imp_median.fit(df_contin_np2) # median 값을 학습
1
imp_median.statistics_
print(imp_median.transform(df_contin_np2))



# KNNImputer
    # knn으로 채워넣기
imp_KNN = KNNImputer(n_neighbors=5, missing_values=np.nan)
imp_KNN.fit(df_contin_np2)
print(imp_KNN.transform(df_contin_np2))


    # IterativeImputer를 이용해서  채워넣기
imp_mice = IterativeImputer(missing_values=np.nan)
imp_mice.fit(df_contin_np2)
print(imp_mice.transform(df_contin_np2))





# 결측치 처리 참고
# https://datascienceschool.net/view-notebook/d96dcbf7f8ac4ee4bf3875b66b2da654/

# 실습 --------------------------------------------------------------------
# dataset_product.xlsx 데이터에서 다음에 따라 결측치를 처리하세요
    # object일 경우 mode로 처리
    # 연속형일 경우 KNN 으로 처리 (n_nieghbors로 2개 이상을 사용 하고 결과 비교)
    # 연속형일 경우 Simple imputer (most_frequent) 으로 처리

df2.dtypes.value_counts()
df2_obj = df2.iloc[:,(df2.dtypes == 'object').array]
df2_num = df2.iloc[:,((df2.dtypes == 'int64') | (df2.dtypes == 'float64')).array]


# object변수 결측치 처리
# df2_obj_np = df2_obj.values     # numpy변수로 변환
# np.isnan(df2_obj_np)    # (error) 결측치 확인
# pd.DataFrame(df2_obj_np).isna().values
df2_obj.isnull()

df2_obj_mode = df2_obj.mode().iloc[0]
df2_obj__mode_result = df2_obj.fillna(df2_obj_mode).values



# number변수 결측치 처리
df2_num_np = df2_num.values     # numpy변수로 변환

np.isnan(df2_num_np)    # 결측치 확인
# pd.DataFrame(df2_num_np).isna().values
# Numpy isnan ()은 float 배열에서만 적용가능 object배열 적용불가

    # Median
imp_median = SimpleImputer(strategy='median', missing_values=np.nan) # 선언, median으로 결측치를 채워 넣겠다
imp_median.fit(df2_num_np)
imp_median.statistics_
df2_num_median_result = imp_median.transform(df2_num_np)

    # KNN방법사용 : k = 2
imp_KNN_2 = KNNImputer(n_neighbors=2, missing_values=np.nan)
imp_KNN_2.fit(df2_num_np)
df2_num_KNN2_result = imp_KNN_2.fit_transform(df2_num_np)


# 결과
result_df2_1 = np.concatenate([df2_obj__mode_result, df2_num_KNN2_result], axis=1 )
df2_pd1 = pd.DataFrame(df2_obj__mode_result)
df2_pd2 = pd.DataFrame(df2_num_KNN2_result)
result_df2_2 = pd.concat([df2_pd1, df2_pd2], axis=1).to_numpy()





# [ Missingpy Library ]  -----------------------------------------------------------------------

from missingpy import MissForest        #  Randomforest 방식으로 결측치를 채우는 Library
# https://pypi.org/project/missingpy/
# imputer = MissForest()
# X_imputed = imputer.fit_transform(X)






# [ Pickle Library ]  -----------------------------------------------------------------------
    # 분석결과(변수 등..)를 파일로 저장하고 불러오는데 사용하는 library
# whos        # worksapce상에 있는 정의된 변수 list를 볼 수 있는 명령어

# 변수값을 파일로 저장
pickle.dump(df_contin_np2, open('df_contin_np2.pkl','wb'))      # df_contin_np2.pkl 파일 생성
    # pickle.dump(어떤변수를 저장?, open('어떤이름으로저장?.pkl','wb'))
pickle.dump(df,open('df_pd.pkl','wb'))
pickle.dump(imp_mean, open('imp_mean.pkl','wb'))

# 불어오기
df_contin_np2_load = pickle.load((open('df_contin_np2.pkl','rb')))
df_load = pickle.load((open('df_pd.pkl','rb')))
imp_mean_load = pickle.load((open('imp_mean.pkl','rb')))

# 불러온 데이터 확인
df_load
df_contin_np2_load
imp_mean_load






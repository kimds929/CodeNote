import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import *

from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer



# --- [ sklearn_train_test_split ] ----------------------------------------------------------------------------
current_dir = os.getcwd()
os.getcwd()
# D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/dataset_city.csv
absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'
# absolute_path = '/home/pirl/Postech_AI/3_DataMining/'

df = pd.read_csv(absolute_path + 'dataset_city.csv')
df2 = pd.read_excel(absolute_path + 'dataset_product.xlsx')



df.dtypes
df.isna()               # 데이터 개개별 결측치 확인
df.isna().sum(axis=0)        # column별 결측치 갯수 확인
df.isna().sum(axis=1)        # row별 결측치 갯수 확인


# 연속형 변수를 가지는 행 선택 and 결측치만 가지는 열들을 제외
idx_contin = (df.dtypes=='int64') | (df.dtypes=='float64')
    # idx_contin = np.logical_or(df.dtypes=='int64', df.dtypes=='float64')

df_contin_np = df.loc[:,idx_contin].to_numpy()
idx_missing = np.isnan(df_contin_np)
idx_missing2 = idx_missing.sum(0) < df_contin_np.shape[0]   # 특정Column이 전부 결측치인 Column이 있는지 확인
df_contin_np2 = df_contin_np[:,idx_missing2]
df_contin_np2

print('행렬 모양:', df_contin_np2.shape)
print('전체 결측치의 개수:', np.isnan(df_contin_np2).sum())
print('열별 결측치의 개수:', np.isnan(df_contin_np2).sum(0))


# ## Train test set 분리 ------------------------------------------------------------
df_contin_np2

np.isnan(df_contin_np2)
np.isnan(df_contin_np2) != True   # 결측치 값이 없는 항목들 → True
# ~np.isnan(df_contin_np2)

# ?np.argwhere  # True에 해당하는 값의 좌표값(r,c)을 반환 --------------
# get_ipython().run_line_magic('pinfo', 'np.argwhere')

# a = np.array([False, False, True])
# np.argwhere(a)      # 2

# a = np.array([[False, False, True],[True,True,False]])
# np.argwhere(a)  
# array([[0, 2], [1, 0], [1, 1]], dtype=int64)      # True의 위치 (0,2), (1,0), (1,1)
# ---------------------------------------------------------------------

non_missing_idx = np.argwhere(np.isnan(df_contin_np2)!=True)    # 값이 True인것(결측치 값이 없는 것)의 index들
# non_missing_idx = np.argwhere(~np.isnan(df_contin_np2))    # 값이 True인것(결측치 값이 없는 것)의 index들
    
print(non_missing_idx)          # [관측치의 row idx, 관측치의 column idx]
print(non_missing_idx.shape)    #전체 관측치의 갯수 = 87,402
df_contin_np2[0,2]

df_contin_np2[non_missing_idx[0,0], non_missing_idx[0,1]]


# idx = np.random.permutation(non_missing_idx.shape[0])
idx = np.arange(non_missing_idx.shape[0])
idx
idx.shape[0]


non_missing_idx     # 결측치가 없는 데이터의 좌표값 (r, c)

Num_obs = idx.shape[0]
Num_test = 1000000;             # 전체 중 1,000,000개로 모형 학습
Num_train = Num_obs - Num_test
idx_train_init, idx_test_init= train_test_split(idx,
                                                train_size=Num_train, test_size=Num_test, random_state=1)
idx_train = non_missing_idx[idx_train_init,:]
idx_test= non_missing_idx[idx_test_init,:]
idx_train_init      # 결측치가 없는 train_data_set의 좌표값 (r, c)
idx_test_init       # 결측치가 없는 test_data_set의 좌표값 (r, c)


print(idx_train.shape)
print(idx_train)

print(idx_test.shape)
print(idx_test)

df_contin_train = df_contin_np2.copy() 
df_contin_train[idx_test[:,0], idx_test[:,1]] = np.NaN      # Test_Data에 대해서는 결측치로 바꿔라

df_contin_np2
df_contin_train

print('원 행렬의 결측치의 개수:', np.isnan(df_contin_np2).sum())
print('test set의 수 (값을 지운 개수)', Num_test)
print('지운 행렬의 결측치의 개수:', np.isnan(df_contin_train).sum())


# 지워진 값들 저장
test_true = df_contin_np2[idx_test[:,0], idx_test[:,1]]
print(test_true.shape)
print(test_true)






# Mean(평균)  # SimpleImputer를 활용해서 평균값으로 채우기 -------------------------
imp_mean = SimpleImputer(strategy='mean', missing_values=np.nan)

imp_mean.fit(df_contin_train) # 중요: train만을 사용
df_contin_fill_mean = imp_mean.transform(df_contin_train)
df_contin_train
df_contin_fill_mean

test_hat_mean = df_contin_fill_mean[idx_test[:,0],idx_test[:,1]] 
# 그림에서 주황색 동그라미
print(test_hat_mean.shape)

mse_mean = mean_squared_error(test_true, test_hat_mean)
print('평균으로 결측치를 채울때 mse:', mse_mean)



# Median(중위수)  # SimpleImputer를 활용해서 중위수로 채우기 -------------------------
imp_median = SimpleImputer(strategy='median', missing_values=np.nan)

imp_median.fit(df_contin_train) # 중요: train만을 사용
df_contin_fill_median = imp_median.transform(df_contin_train)
df_contin_train
df_contin_fill_median

test_hat_median = df_contin_fill_median[idx_test[:,0],idx_test[:,1]] 
# 그림에서 주황색 동그라미
print(test_hat_median.shape)

mse_median = mean_squared_error(test_true, test_hat_median)
print('중위수로 결측치를 채울때 mse:', mse_median)



# KNN(K-Nearest-Neighbor)  # KNNImputer를 활용해서 K-Nearest-Neighbor값으로 채우기 -------------------------
imp_KNN = KNNImputer(n_neighbors=3, missing_values=np.nan, weights='uniform')   # weight='distance'
imp_KNN.fit(df_contin_train)
df_contin_fill_KNN = imp_KNN.transform(df_contin_train)
test_hat_KNN = df_contin_fill_KNN[idx_test[:,0],idx_test[:,1]]

mse_KNN = mean_squared_error(test_true, test_hat_KNN)
print('KNN(K=3)으로 결측치를 채울때 mse:', mse_KNN)




# IterativeImputer를 이용해서 채워넣기 # IterativeImputer를 활용해서 Regression값으로 채우기 -------------------------
imp_iter = IterativeImputer(missing_values=np.nan)
imp_iter.fit(df_contin_train)
df_contin_fill_iter = imp_iter.transform(df_contin_train)
df_contin_train
df_contin_fill_iter

test_hat_iter = df_contin_fill_iter[idx_test[:,0],idx_test[:,1]] 
# 그림에서 주황색 동그라미
print(test_hat_iter.shape)

mse_iter = mean_squared_error(test_true, test_hat_iter)
print('InteractiveImputer로 결측치를 채울때 mse:', mse_iter)




# ## 실습 -----------------------------------------------------------------------------
# ### dataset_product.xlsx에 대해서 적용하기
#  idx_train과 idx_test는 모든 방법론에 같이 적용
#  mean, median, most_frquent,
# #(+iterativeimputer) 결과 비교
# 
#  결측치가 10 ~ 25% 미만인 열만 선택
#  단계1. train과 test set 나누기 : Test set은 전체의 30%, test set은 나머지
#  단계2. train set을 이용해서 결측치 채워넣기
#  단계3. 채워넣은 값과 test set에서의 값을 비교하기

df2_contin_index = ((df2.dtypes == 'int64') | (df2.dtypes == 'float64'))
df2_contin = df2.loc[:, df2_contin_index]
df2_contin_np = df2_contin.to_numpy()

    #  결측치가 10 ~ 25% 미만인 열만 선택
df2_miss_up10 = df2_contin.isna().sum(0) >= df2_contin.shape[0]*0.10
df2_miss_down25 = df2_contin.isna().sum(0) < df2_contin.shape[0]*0.25
df2_filter_pd =  df2_contin.loc[:, df2_miss_up10 & df2_miss_down25]      # [pandas]
df2_filter_np = df2_filter_pd.to_numpy()
# df2_filter_np = df2_contin_np[:,df2_miss_up10 & df2_miss_down25]      # [numpy]

    
df2_non_missing_idx = np.argwhere(np.isnan(df2_filter_np) == False)
df2_idx = np.arange(df2_non_missing_idx.shape[0])

    #  단계1. train과 test set 나누기 : Test set은 전체의 30%, test set은 나머지
df2_train_idx, df2_test_idx = train_test_split(df2_idx, test_size=0.3, random_state=1)

non_missing_train = df2_non_missing_idx[df2_train_idx,:]
non_missing_test = df2_non_missing_idx[df2_test_idx,:]

df2_contin_train = df2_contin_np.copy()
df2_contin_train[non_missing_test[:,0], non_missing_test[:,1]] = np.nan

print('원 행렬의 결측치의 개수:', np.isnan(df2_filter_np).sum())
print('test set의 수 (값을 지운 개수)', len(df2_test_idx))
print('지운 행렬의 결측치의 개수:', np.isnan(df2_contin_train).sum())




    # 단계2. train_set을 이용해서 결측치 채워넣기
    # mean
df2_imp_mean = SimpleImputer(strategy='mean',missing_values=np.nan)
df2_imp_mean.fit(df2_train)






# Normaliztion (표준화) --------------------------------------------
from sklearn.preprocessing import StandardScaler

pd.DataFrame(df_contin_np2)     # 원래 데이터

scaler = StandardScaler()
scaler.fit(df_contin_np2)

df_contin_np2_scale = scaler.transform(df_contin_np2)
pd.DataFrame(df_contin_np2_scale)       # StandardScaler 적용결과

df_contin_recover = scaler.inverse_transform(df_contin_np2_scale)
pd.DataFrame(df_contin_recover)       # StandardScaler Inverse 되돌린것 결과

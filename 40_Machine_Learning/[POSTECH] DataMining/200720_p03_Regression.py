import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import statsmodels.api as sm
from statsmodels.formula.api import ols

import matplotlib.pyplot as plt

# https://scikit-learn.org/stable/modules/linear_model.html     # sklearn.linear_model



current_dir = os.getcwd()
os.getcwd()
# D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/dataset_city.csv
absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'


# Regression 실습 -----------------------------------------------------------------------------------------
# energy dataset ****
df_energy = pd.read_csv(absolute_path + 'energy_dataset.csv')
df_energy.head()


# 데이터 전처리 -------------------
df_energy
# df_energy.drop(columns=['forecast wind offshore eday ahead'], inplace=True)
df = df_energy.drop(columns=['forecast wind offshore eday ahead'])
df.drop(columns=['generation hydro pumped storage aggregated'], inplace=True)
df.dropna(inplace=True)

Input = df.loc[:,'generation biomass':'generation wind onshore']
Output =  df.loc[:,'price actual']

Input_np = Input.values
Output_np= Output.values

Input_np
Output_np


    # Sklearn LinearRegression 활용
# ?LinearRegression     # fit_intercept : 상수항 적용여부
# LinearRegression(
#     *,
#     fit_intercept=True,
#     normalize=False,
#     copy_X=True,
#     n_jobs=None,
# )
mdl = LinearRegression()
mdl.fit(Input_np, Output_np)

mdl.coef_
mdl.intercept_

Output_hat = mdl.predict(Input_np)
mse = mean_squared_error(Output_np,Output_hat)
print(f'mse : {mse}')



    # Statsmodels 활용
Input_np_aug = sm.add_constant(Input_np, prepend=False)
Input_np_aug[:3,:]

mod = sm.OLS(Output_np, Input_np_aug).fit()
print(mod.summary())

Output_hat2 = mod.predict(Input_np_aug)
mse2 = mean_squared_error(Output_np,Output_hat2)
print(mse2)


    # 시각화
plt.plot(Output, Output_hat, 'r.', markersize=12)
plt.xlabel("True")
plt.ylabel("Predicted")
plt.show()

plt.plot(Output_hat, mod.resid,'.')
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.show()

    # qqplot
probplot = sm.ProbPlot(mod.resid)
probplot.qqplot(line='45')
plt.show()







# Train-Test Split Data Regression -----------------------------------------------------------------------------------------
    # energy Data   ****
# df
# Input = df.loc[:,'generation biomass':'generation wind onshore']
# Output =  df.loc[:,'price actual']

# Input_np = Input.values
# Output_np= Output.values

Num_obs = Input_np.shape[0]
print(Num_obs)

Num_test = 15000
Num_train = Num_obs - Num_test
idx = np.arange(Num_obs)

idx
idx_train, idx_test = train_test_split(idx,
                                       train_size=Num_train,
                                       test_size=Num_test,
                                      random_state=6)

# random state에 idx_train과 idx_test가 달라짐
print(idx_train, idx_test)

Input_train= Input_np[idx_train,:]
Input_test= Input_np[idx_test,:]
Output_train = Output_np[idx_train]
Output_test = Output_np[idx_test]

print('Train input 정보:',Input_train.shape)
print('Test input 정보: ',Input_test.shape)


# train set만을 이용해서 모형 학습
mdl_train= LinearRegression(fit_intercept=True).fit(Input_train, Output_train)
# train set에서의 성능
Output_train_hat = mdl_train.predict(Input_train)
print(f'mse_energy_train : {mean_squared_error(Output_train, Output_train_hat)}')

# test set에서의 성능
Output_test_hat = mdl_train.predict(Input_test)
print(f'mse_energy_test : {mean_squared_error(Output_test,Output_test_hat)}')


# 실습: Intercept가 있는 모형과 없는 모형의 Train/Test set mse 비교하기
    # Intercept 없는 모델
mdl2_train= LinearRegression(fit_intercept=False).fit(Input_train, Output_train)
# train set에서의 성능
Output_train_hat2 = mdl2_train.predict(Input_train)
print(f'mse_energy_train(intercept x) : {mean_squared_error(Output_train, Output_train_hat2)}')

# test set에서의 성능
Output_test_hat2 = mdl2_train.predict(Input_test)
print(f'mse_energy_test(intercept x) : {mean_squared_error(Output_test,Output_test_hat2)}')



# ### 실습 - winearoma 데이터셋에 대해서 train과 test set 에러를 계산하기
# test set: 관측치 10개
Wine
Input_np_wine = Wine.iloc[:,:-1].to_numpy()
Output_np_wine = Wine.iloc[:,-1].to_numpy()

Num_obs_wine = Wine.shape[0]
Num_test_wine = 10
NUm_train_wine = Num_obs_wine - Num_test_wine

idx_wine = np.arange(Num_obs_wine)
idx_wine

idx_wine_train, idx_wine_test = train_test_split(idx_wine, 
                                                train_size=NUm_train_wine,
                                                test_size=Num_test_wine,
                                                random_state=1 )
idx_wine_train
idx_wine_test

Input_train_wine = Input_np_wine[idx_wine_train,:]
Input_test_wine = Input_np_wine[idx_wine_test,:]
Output_train_wine = Output_np_wine[idx_wine_train]
Output_test_wine = Output_np_wine[idx_wine_test]

print('Train input 정보:', Input_train_wine.shape)
print('Test input 정보: ', Input_test_wine.shape)

# train set만을 이용해서 모형 학습
mdl_wine_train = LinearRegression(fit_intercept=True)
mdl_wine_train.fit(Input_train_wine, Output_train_wine)

# train set에서의 성능
Output_wine_train_hat = mdl_wine_train.predict(Input_train_wine)
print(f'mse_wine_train : {mean_squared_error(Output_train_wine, Output_wine_train_hat)}')

# test set에서의 성능
Output_wine_test_hat = mdl_wine_train.predict(Input_test_wine)
print(f'mse_wine_test : {mean_squared_error(Output_test_wine, Output_wine_test_hat)}')



# ### 실습 - winearoma 데이터셋에 대해서 statsmodels를 이용해서 모형 학습하고 train set mse 계산하기 (
# ## Train_test_set_split & evaluation

# # Wine Aroma Data ****
#     # 데이터 불러오기
Wine = pd.read_excel(absolute_path + 'wine_aroma.xlsx')

Input_wine= Wine.iloc[:,:-1].values
Output_wine = Wine.iloc[:,-1].values

    # Train-Test 나누기
wine_train, wine_test = train_test_split(Wine, test_size=0.3)

wine_train_x = wine_train.iloc[:,:-1]
wine_train_y = wine_train.iloc[:,:1]

wine_test_x = wine_test.iloc[:,:-1]
wine_test_y = wine_test.iloc[:,-1]



    # Sklearn LinearRegression 활용
mdl_wine = LinearRegression()
mdl_wine.fit(wine_train_x, wine_train_y)

mdl_wine.coef_
mdl_wine.intercept_

Output_hat_wine = mdl_wine.predict(wine_test_x)
mse_wine = mean_squared_error(wine_test_y, Output_hat_wine)
print(f'mse_wine(sklearn) : {mse_wine}')


    # Statsmodels 활용
Input_wine_train_aug = sm.add_constant(wine_train_x.to_numpy(), prepend=False)
Output_wine_train = wine_train_y.to_numpy()

mod_wine2 = sm.OLS(Output_wine_train, Input_wine_train_aug).fit()
# print(mod_wine2.summary())

Input_wine_test_aug = sm.add_constant(wine_test_x.to_numpy(), prepend=False)
Output_wine_test = wine_test_y.to_numpy()

Output_hat2 = mod_wine2.predict(Input_wine_test_aug)
mse2 = mean_squared_error(Output_wine_test, Output_hat2)
print(f'mse_wine2(statsmodels) : {mse2}')







# Cross Validataion(K-Fold) ---------------------------------------------------------------------

# ## Hold out split 반복 **** --------------------------------

# 10회 반복, 매 반복시 15,000개를 test set으로 사용
    # energy data ****
# df
# Input = df.loc[:,'generation biomass':'generation wind onshore']
# Output =  df.loc[:,'price actual']

# Input_np = Input.values
# Output_np= Output.values

Num_rep = 10
Num_test = 15000
Num_train = Num_obs - Num_test
idx = np.arange(Num_obs)
idx

# index 저장 
# 정수 형태의 값이 0인 행렬 생성
idx_train_rep = np.zeros([Num_train, Num_rep]).astype(int) # 정수형 행렬
idx_test_rep = np.zeros([Num_test, Num_rep]).astype(int)
mse_rep = np.zeros([Num_rep, 2])


# index 저장 
for rep in np.arange(Num_rep):
    idx_train_rep[:,rep], idx_test_rep[:,rep] = train_test_split(
                                                idx, train_size = Num_train, test_size=Num_test)
    print('반복:',rep)
    
print(idx_train_rep)



for rep in np.arange(Num_rep):
    print('반복:',rep)
    # 각 반복 마다 train과 test set 생성
    idx_train,idx_test = idx_train_rep[:,rep], idx_test_rep[:,rep] 
    # 저장된 값을 불러오기
    Input_train= Input_np[idx_train,:]
    Input_test= Input_np[idx_test,:]
    Output_train = Output_np[idx_train]
    Output_test = Output_np[idx_test]

    mdl_rep= LinearRegression(fit_intercept=True).fit(Input_train,Output_train)
    # train set에서의 성능
    Output_train_hat = mdl_rep.predict(Input_train)
    mse_rep[rep,0] = mean_squared_error(Output_train, Output_train_hat)
    # test set에서의 성능
    Output_test_hat = mdl_rep.predict(Input_test)
    mse_rep[rep,1]=mean_squared_error(Output_test,Output_test_hat)


print(mse_rep)
print(mse_rep.mean(axis=0))
print(mse_rep.std(axis=0))


    # box-plot
plt.boxplot(mse_rep, labels=('train','test'))
plt.show()



# ### 실습 - winearoma 데이터셋에 대해서 10번 반복해서 train과 test set 에러를 계산하기
# ○ Intercept 있는 모형과 없는 모형 비교 / test set: 관측치 10개
    # Wine Data
Wine
Input_wine= Wine.iloc[:,:-1].values
Output_wine = Wine.iloc[:,-1].values

Num_obs_wine = Wine.shape[0]
Num_rep_wine = 10

Num_test_wine = 10
Num_train_wine = Num_obs_wine - Num_test_wine

idx_wine = np.arange(Num_obs_wine)
idx_wine

idx_train_rep_wine = np.zeros([Num_train_wine, Num_rep_wine]).astype(int) # 정수형 행렬
idx_test_rep_wine = np.zeros([Num_test_wine, Num_rep_wine]).astype(int)
mse_rep_wine = np.zeros([Num_rep_wine, 2])


# index 저장 
for rep in np.arange(Num_rep_wine):
    idx_train_rep_wine[:,rep], idx_test_rep_wine[:,rep] = train_test_split(idx_wine,
                                                                        train_size=Num_train_wine,
                                                                        test_size=Num_test_wine)
    print('반복:',rep)
print(idx_train_rep_wine)


for rep in np.arange(Num_rep_wine):
    print('반복:',rep)
    # 각 반복 마다 train과 test set 생성
    idx_train_wine, idx_test_wine = idx_train_rep_wine[:,rep], idx_test_rep_wine[:,rep] 

    # 저장된 값을 불러오기
    Input_train_wine = Input_wine[idx_train_wine,:]
    Input_test_wine = Input_wine[idx_test_wine,:]
    Output_train_wine = Output_wine[idx_train_wine]
    Output_test_wine = Output_wine[idx_test_wine]

    mdl_rep_wine3 = LinearRegression(fit_intercept=True)
    mdl_rep_wine3.fit(Input_train_wine, Output_train_wine)

    # train set에서의 성능
    Output_train_hat_wine3 = mdl_rep_wine3.predict(Input_train_wine)
    mse_rep_wine[rep,0] = mean_squared_error(Output_train_wine, Output_train_hat_wine3)

    # test set에서의 성능
    Output_test_hat_wine3 = mdl_rep_wine3.predict(Input_test_wine)
    mse_rep_wine[rep,1] = mean_squared_error(Output_test_wine, Output_test_hat_wine3)


print(mse_rep_wine)
print(mse_rep_wine.mean(axis=0))
print(mse_rep_wine.std(axis=0))


    # box-plot
plt.boxplot(mse_rep_wine, labels=('train','test'))
plt.show()








# ## Cross validation 이용 **** ---------------------------------

# cross validation 예제
idx = np.arange(10)
kf2 = KFold(n_splits=5, shuffle=False)
kf2

# fold : 몇번째 fold?
# idx_train : train_set Index
# idx_test : test_set Index
for fold,(idx_train, idx_test) in enumerate(kf2.split(idx)):
    print('fold:',fold)
    print("idx train:",idx_train, "idx test:",idx_test)

kf = KFold(n_splits=5, shuffle=True, random_state=1)    # shuffle = True와 shuffle=false 비교
CV_MSE = np.zeros((5,2)) # train과 test MSE 저장


    # energy data ****
# df
# Input = df.loc[:,'generation biomass':'generation wind onshore']
# Output =  df.loc[:,'price actual']

# Input_np = Input.values
# Output_np= Output.values


for fold, (idx_train, idx_test) in enumerate(kf.split(Input_np)):
    print("idx train:",idx_train, "idx test:",idx_test)
    Input_train= Input_np[idx_train,:]
    Input_test= Input_np[idx_test,:]
    Output_train = Output_np[idx_train]
    Output_test = Output_np[idx_test]
        
    mdl_cv=LinearRegression(fit_intercept=True).fit(Input_train,Output_train)
    # 빈칸: train과 test MSE 저장
    Output_train_hat = mdl_cv.predict(Input_train)
    Output_test_hat = mdl_cv.predict(Input_test)
    
    CV_MSE[fold,0] = mean_squared_error(Output_train,Output_train_hat)
    CV_MSE[fold,1] = mean_squared_error(Output_test,Output_test_hat)
    fold= fold+1


print(CV_MSE)
print(CV_MSE.mean(axis=0)) # column 별 평균 출력
print(CV_MSE.std(axis=0)) # column 별 편차 출력


# ### 실습 - winearoma 데이터셋에 대해서 cross validation을 이용했을때 train과 test error 계산하기
# ### 실습 - winearoma 데이터셋에 대해서 cross validation을 10번반복 했을때 test error 계산하기
Wine
wine_fold = 10
cv_wine = KFold(n_splits=wine_fold, shuffle=True, random_state=1)    # shuffle = True와 shuffle=false 비교

mse_wine_cv = np.zeros((wine_fold,2))

for fold, (idx_train, idx_test) in enumerate(cv_wine.split(Wine)):
        wine_train = Wine.iloc[idx_train]
        wine_test = Wine.iloc[idx_test]

        wine_train_y = wine_train.iloc[:, -1]
        wine_train_x = wine_train.iloc[:, :-1]
        wine_test_y = wine_test.iloc[:, -1]
        wine_test_x = wine_test.iloc[:, :-1]

        wine_train_x_add_const = sm.add_constant(wine_train_x, prepend=False)
        wine_test_x_add_const = sm.add_constant(wine_test_x, prepend=False)

        wine_cv_model = sm.OLS(wine_train_y, wine_train_x_add_const).fit()

        wine_cv_train_predict = wine_cv_model.predict(wine_train_x_add_const)
        mse_wine_cv_train = mean_squared_error(y_true=wine_train_y, y_pred=wine_cv_train_predict)

        wine_cv_test_predict = wine_cv_model.predict(wine_test_x_add_const)
        mse_wine_cv_test = mean_squared_error(y_true=wine_test_y, y_pred=wine_cv_test_predict)

        mse_wine_cv[fold, 0] = mse_wine_cv_train
        mse_wine_cv[fold, 1] = mse_wine_cv_test

mse_wine_cv_df = pd.DataFrame(mse_wine_cv, columns=['train', 'test'])
mse_wine_cv_df.index.name = 'fold'

print(mse_wine_cv_df)
print(mse_wine_cv_df.mean(axis=0))
print(mse_wine_cv_df.std(axis=0))

    # mse box-plot 시각화
plt.boxplot(mse_wine_cv, labels=('train','test'))
plt.show()



# # ## Cross validation 반복
# Num_fold=5
# cv_idx_rep = np.zeros([Num_obs,Num_rep]).astype(int) # 관측치 수 * 반복수: 어느 fold에 속하는지
# cv_mse_rep = np.zeros((Num_fold, Num_rep,2))



# for rep in np.arange(Num_rep):
#     print('반복:',rep)
#     kf = KFold(n_splits=Num_fold, shuffle=True)
    
#     for fold, (idx_train, idx_test) in enumerate(kf.split(Input_np)):
#         Input_train= Input_np[idx_train,:]
#         Input_test= Input_np[idx_test,:]
#         Output_train = Output_np[idx_train]
#         Output_test = Output_np[idx_test]
        
#         mdl_cv_rep=LinearRegression(fit_intercept=True).fit(Input_train,Output_train)
#         Output_train_hat = mdl_cv_rep.predict(Input_train)
#         Output_test_hat = mdl_cv_rep.predict(Input_test)
        
#         # 결과와 index 저장
#         cv_mse_rep[fold,rep,0] = mean_squared_error(Output_train,Output_train_hat)
#         cv_mse_rep[fold,rep,1] = mean_squared_error(Output_test,Output_test_hat)
#         cv_idx_rep[idx_test,rep]=fold


# print('Train set mse:',cv_mse_rep[:,:,0])
# print('Test set mse:',cv_mse_rep[:,:,1])


# # 각 rep별 결과 도출
# print(cv_mse_rep[:,:,0].mean(axis=0)) # train error
# print(cv_mse_rep[:,:,1].mean(axis=0)) # test error


# plt.boxplot([cv_mse_rep[:,:,0].mean(axis=0)
#              ,cv_mse_rep[:,:,1].mean(axis=0)], labels=('train','test'))
# plt.show()



# [ 실습 ] Energy Dataset
#  . 결측치 : 평균으로 채우기
#  . Train-Test set split을 10번 반복해서 train/test set error 평가하기

#  1) Train-Test set분할
#  2) Train set에서 평균으로 train set의 결측치 채우기
#  3) Train set에서 모형 학습
#  4) 2단계에서 계산한 값으로 test set의 결측치 채우기
#  5) Test set에서 모형 평가

df_energy2 = df_energy.copy()
df_energy2 = df_energy2.drop(columns=['time'])
df_energy2 = df_energy2.drop(columns=['generation hydro pumped storage aggregated'])
df_energy2 = df_energy2.drop(columns=['forecast wind offshore eday ahead'])
energy_train, energy_test = train_test_split(df_energy2, test_size=0.3)
energy_train.isna().sum()

energy_train.dtypes

# 결측치 평균으로 대체
from sklearn.impute import *
energy_train.dtypes
imp_mean = SimpleImputer(strategy='mean', missing_values=np.nan)
imp_mean.fit(energy_train)

enenrgy_train_impute = imp_mean.transform(energy_train)

enenrgy_train_impute_x = enenrgy_train_impute[:,:-1]
enenrgy_train_impute_y = enenrgy_train_impute[:,-1]
enenrgy_test_x = energy_test.iloc[:,:-1]
enenrgy_test_y = energy_test.iloc[:,-1]

enenrgy_train_impute_x_add_const = sm.add_constant(enenrgy_train_impute_x, prepend=False)
enenrgy_test_x_add_const = sm.add_constant(enenrgy_test_x, prepend=False)

energe_impute_model = sm.OLS(enenrgy_train_impute_y, enenrgy_train_impute_x_add_const).fit()
# energe_impute_model.summary()

energe_impute_model.predict(enenrgy_train_impute_x_add_const)









# categorical 변수에 대한 Dummy variable 이용하기 -----------------------------------------
df_product = pd.read_excel(absolute_path + 'dataset_product.xlsx')

df_product.dtypes
idx_cat = df_product.dtypes=='object'
df_product_cat = df_product.loc[:,idx_cat.array]
df_product_cat.head().T

df_product_cat_dummies = pd.get_dummies(df_product_cat)
df_product_cat_dummies.head().T

# 실습 : Aroma Dataset으로 train_test_split 10회와 10-fold regression 비교하기 -------------------
# load example dataset (wine aroma)
df_wine = pd.read_excel(absolute_path + 'wine_aroma.xlsx')

# Test_Train_Split 10 times ---------------
tts_mse = []
for i in range(10):
    train_y, test_y, train_x, test_x = train_test_split(df_wine.iloc[:,-1].to_frame(), df_wine.iloc[:,:-1],
                                                        test_size=0.3, random_state=i)
    train_x_const = sm.add_constant(train_x)

    model_wine_tts = sm.OLS(train_y, train_x_const).fit()
    # model_wine_tts.summary()

    wine_tts_predict = model_wine_tts.predict(sm.add_constant(test_x))
    wine_tts_mse = mean_squared_error(y_true=test_y, y_pred=wine_tts_predict)
    tts_mse.append(wine_tts_mse)
tts_mse_np = np.array(tts_mse).reshape((-1,1))

# 10-Fold Regression ---------------
cv10 = KFold(n_splits=10, shuffle=False)

cv_mse = []
for fold, (idx_train, idx_test) in enumerate(cv10.split(df_wine)):
        cv_train = df_wine.iloc[idx_train,:]
        cv_test = df_wine.iloc[idx_test,:]

        cv_train_y = cv_train.iloc[:, -1]
        cv_train_x = cv_train.iloc[:, :-1]
        cv_test_y = cv_test.iloc[:, -1]
        cv_test_x = cv_test.iloc[:, :-1]

        cv_train_x_const = sm.add_constant(cv_train_x)
        model_wine_cv = sm.OLS(cv_train_y, cv_train_x_const).fit()
        wine_cv_predict = model_wine_cv.predict(sm.add_constant(cv_test_x))
        wine_cv_mse = mean_squared_error(y_true=cv_test_y, y_pred=wine_cv_predict)
        cv_mse.append(wine_cv_mse)
cv_mse_np = np.array(cv_mse).reshape((-1,1))

tts_cv_mse = np.concatenate((tts_mse_np, cv_mse_np), axis=1)
tts_cv_mse.mean(axis=0)

    # mse box-plot 시각화
plt.boxplot(tts_cv_mse, labels=('tts','cv'))
plt.show()



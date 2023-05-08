import sys
sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import *
from DS_OLS import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm

from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'

# a = pd.read_clipboard()
# a.values.tolist()
# Lasso  ****
# get_ipython().run_line_magic('pinfo', 'Lasso')


# load example dataset (wine aroma)
df = pd.read_excel(absolute_path + 'wine_aroma.xlsx')
df_info = DS_DF_Summary(df)           # DS_Module
X = df.iloc[:, :-1] #.to_numpy()
y = df.iloc[:, -1]  #.to_numpy()

# y_column = ['Aroma']
# x_column = df.columns.drop(y_column)



# declare lasso object and train the model
# ?Lasso
# Lasso(
#     alpha=1.0,
#     *,
#     fit_intercept=True,
#     normalize=False,
#     precompute=False,
#     copy_X=True,
#     max_iter=1000,
#     tol=0.0001,
#     warm_start=False,
#     positive=False,
#     random_state=None,
#     selection='cyclic',
# )
model = Lasso(alpha=0.05, normalize=True)  # (default) alpha=1.0
model.fit(X, y)

dir(model)
# print regression coefficients
print('Regression coefficients')
# pd.Series(model.coef_, index=df.iloc[:, :-1].columns)
print(model.coef_)
print(model.intercept_)
pd.Series(model.coef_, index=X.columns)

# compute MSE
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print(f'MSE: {np.around(mse, decimals=3)}')
r2_score(y, y_pred)
# fun_evalueate_model(y_true=y, y_pred=y_pred, X=X, const=True, resid=True)      # DS_Module





# ridge ****
get_ipython().run_line_magic('pinfo', 'Ridge')
model_ridge = Ridge(alpha=0.01, normalize=True)
model_ridge.fit(X,y)

print(model_ridge.coef_)
# pd.Series(model_ridge.coef_, index=df.iloc[:, :-1].columns)
print(model_ridge.intercept_)

y_pred_ridge = model_ridge.predict(X)
mse_ridge = mean_squared_error(y, y_pred_ridge)
print(f'MSE: {np.around(mse_ridge, decimals=3)}')
r2_score(y, y_pred_ridge)




# # Compute test MSE (cross-validation) for two models above
n_splits = 5
kf = KFold(n_splits=n_splits)


# alpha = 0.05
mse_train = np.zeros(shape=(kf.n_splits,))
mse_test = np.zeros(shape=(kf.n_splits,))

for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index, :], X[test_index, :]
    y_train, y_test = y[train_index], y[test_index]
    
    model = Lasso(alpha=0.05, normalize=True)       # Lasso Model
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    mse_train[i] = mean_squared_error(y_train, y_train_pred)
    mse_test[i] = mean_squared_error(y_test, y_test_pred)
    

print('=' * 30)
print(f'Train MSE: {np.around(np.mean(mse_train), decimals=3)}')
print(f'Test MSE: {np.around(np.mean(mse_test), decimals=3)}')




# alpha = 0.01
mse_train = np.zeros(shape=(kf.n_splits,))
mse_test = np.zeros(shape=(kf.n_splits,))

for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index, :], X[test_index, :]
    y_train, y_test = y[train_index], y[test_index]
    
    model = Lasso(alpha=0.01, normalize=True)       # Lasso Model
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    mse_train[i] = mean_squared_error(y_train, y_train_pred)
    mse_test[i] = mean_squared_error(y_test, y_test_pred)
    
print('=' * 30)
print(f'Train MSE: {np.around(np.mean(mse_train), decimals=3)}')
print(f'Test MSE: {np.around(np.mean(mse_test), decimals=3)}')





# ## Make function to repeat test
def compute_mse(X, y, alpha, kf):   
    
    mse_train = np.zeros(shape=(kf.n_splits,))
    mse_test = np.zeros(shape=(kf.n_splits,))
    
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        model = Lasso(alpha=alpha, normalize=True)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        mse_train[i] = mean_squared_error(y_train, y_train_pred)
        mse_test[i] = mean_squared_error(y_test, y_test_pred)

    print('=' * 30)
    print(f'Alpha: {alpha}')
    print(f'Train MSE: {np.around(np.mean(mse_train), decimals=3)}')
    print(f'Test MSE: {np.around(np.mean(mse_test), decimals=3)}')
    
    return alpha, np.mean(mse_train), np.mean(mse_test)


# # Hyper-parameter search
n_splits=5
alpha_range = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001 ]


for alpha in alpha_range:
    a, train_mse, test_mse = compute_mse(X=X, y=y, alpha=alpha, kf=kf)


alpha_best=0.005
model_best = Lasso(alpha=alpha_best, normalize=True)
model_best.fit(X, y)

print('Coefficients')
print(np.around(model_best.coef_, decimals=3))
pd.Series(model_best.coef_, index=df.iloc[:,:-1].columns)


# Cross Validation을 통한 alpha값 선택      ************
# ## Cross validation to select alpha in Lasso ---------------
kf_hyp = KFold(n_splits=3, shuffle=True, random_state=1)
lasso_cv = LassoCV(normalize=True, cv=kf_hyp)
# ?LassoCV
# LassoCV(
#     *,
#     eps=0.001,
#     n_alphas=100,     # 몇개의 alpha들을 확인해볼 것인지?
#     alphas=None,
#     fit_intercept=True,
#     normalize=False,
#     precompute='auto',
#     max_iter=1000,
#     tol=0.0001,
#     copy_X=True,
#     cv=None,          # cross-validation 여부
#     verbose=False,
#     n_jobs=None,      # 병렬처리 여부
#     positive=False,
#     random_state=None,
#     selection='cyclic',
# )
lasso_cv.fit(X,y)

lasso_cv.alpha_
lasso_cv.alphas_
lasso_cv.mse_path_  # 각 fold별 mse
lasso_cv.coef_
# pd.Series(lasso_cv.coef_, index=df.iloc[:,:-1].columns)
lasso_cv.intercept_
dir(lasso_cv)



# min MSE from CV 
idx = np.argwhere(lasso_cv.alphas_==lasso_cv.alpha_)[0]
print(lasso_cv.mse_path_[idx])
print(lasso_cv.mse_path_[idx].mean())

lasso_cv.coef_
lasso_cv.mse_path_.mean(1)
y_pred_lasso_cv = lasso_cv.predict(X)


    # 시각화
plt.plot(lasso_cv.alphas_, lasso_cv.mse_path_.mean(1), 'k',
         label='Average across the folds', linewidth=2)
plt.xlabel(r'$\alpha$')
plt.ylabel('Mean square error')


# ## Cross validation to select alpha in Ridge ---------------
# ?RidgeCV
# RidgeCV(
#     alphas=(0.1, 1.0, 10.0),
#     *,
#     fit_intercept=True,
#     normalize=False,
#     scoring=None,
#     cv=None,
#     gcv_mode=None,
#     store_cv_values=False,
# )
ridge_cv = RidgeCV(normalize=True, alphas=[0.001, 1, 50. ], cv=kf_hyp)
ridge_cv = RidgeCV(normalize=True, alphas=np.logspace(start=-2, stop=2, num=17), cv=kf_hyp)
ridge_cv.fit(X,y)

ridge_cv.coef_
ridge_cv.alpha_
ridge_cv.best_score_
dir(ridge_cv)
np.logspace(start=-2, stop=2, num=17)






# ## 실습 
# energy datset에서 LassoCV를 train/test evaluation 10번 반복해서 평가하기
# (+ RidgeCV도 비교)

df = pd.read_csv(absolute_path + 'energy_dataset.csv')
df.drop(columns=['generation hydro pumped storage aggregated'], inplace=True)
df.drop(columns=['forecast wind offshore eday ahead'], inplace=True)
df.drop(columns=['time'],inplace=True)
df.dropna(inplace=True)

energy_info = DS_DF_Summary(df)
Input = df.loc[:,'generation biomass':'generation wind onshore']
Output =  df.loc[:,'price actual']
Input_np = Input.values
Output_np= Output.values


kf_hyp5 = KFold(n_splits=5, shuffle=True, random_state=1)
kf_hyp10 = KFold(n_splits=10, shuffle=True, random_state=1)

# for fold, (train_idx, test_idx) in enumerate(kf_hyp5.split(df)):
#     print(fold, '------')
#     print(len(train_idx), len(test_idx))

#     train_x = df.iloc[train_idx, :-1]
#     train_y = df.iloc[train_idx,-1]
#     test_x = df.iloc[test_idx, :-1]
#     test_y = df.iloc[test_idx, -1]



    # lasso
lasso_cv2 = LassoCV(normalize=True, cv=kf_hyp10, random_state=1)
lasso_cv2.fit(Input_np, Output_np)

lasso_cv2.alpha_
lasso_cv2.coef_
pd.Series(lasso_cv2.coef_, index=Input.columns )
lasso_cv2.intercept_

y_pred_lasso_cv2 = lasso_cv2.predict(Input_np)
mean_squared_error(Output_np, y_pred_lasso_cv2)



    # ridge
ridge_cv2 = RidgeCV(alphas=np.logspace(start=-5, stop=5, num=17), normalize=True, cv=kf_hyp10)
            # cv=None, store_cv_values=True ☞ leave one out (Test-set 1 ob, 나머지 Train-set)
ridge_cv2.fit(Input_np, Output_np)

ridge_cv2.alpha_
ridge_cv2.coef_
pd.Series(ridge_cv2.coef_, index=Input.columns )
ridge_cv2.intercept_
ridge_cv2.best_score_   # best r2_score

y_pred_ridge_cv2 = ridge_cv2.predict(Input_np)
mean_squared_error(Output_np, y_pred_ridge_cv2)
r2_score(Output_np, y_pred_ridge_cv2)





from mlxtend.feature_selection import SequentialFeatureSelector as SFS


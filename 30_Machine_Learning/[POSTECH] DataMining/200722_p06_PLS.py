import numpy as np
import pandas as pd
import statsmodels.api as sm

import sys
sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import *

from sklearn.cross_decomposition import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'

# load example dataset (wine aroma)
df = pd.read_excel(absolute_path + 'wine_aroma.xlsx')
wine_info = DS_DF_Summary(df)
X = df.iloc[:, :-1].to_numpy()
y = df.iloc[:, -1].to_numpy()

# PLS ****
# https://www.youtube.com/watch?v=OCprdWfgBkc
# ?PLSRegression
# PLSRegression(
#     n_components=2,       # Number of Latent variables (가장중요한 parameter)
#                           # 1이상 변수의 갯수 미만의 숫자를 넣어주어야 함
#     *,
#     scale=True,
#     max_iter=500,
#     tol=1e-06,
#     copy=True,
# )
# get_ipython().run_line_magic('pinfo', 'PLSRegression')

model = PLSRegression(n_components=3)
model.fit(X,y)

print(model.x_weights_)     # p
print(model.x_loadings_)
print(model.x_scores_)      # 각각의 x값별로 latent값

print(model.y_weights_)     # b
print(model.y_loadings_)
print(model.y_scores_)      # 각각의 y값별로 latent값

print(model.coef_)      # x 



def vip(model):     
    # 모든 계수들의 vip 평균 ==1, 1보다 작거나 같은 계수들은 중요하지 않다고 고려
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
        vips[i] = np.sqrt(p*(s.T @ weight)/total_s)
    return vips

    # vip score를 보고 변수선택을 진행
model_vip = vip(model)
print(model_vip)


idx_feature_valid = np.argwhere(model_vip>1)
print(idx_feature_valid)


model2 = PLSRegression(n_components=4)
model2.fit(X,y)

model2_vip = vip(model2)
print(model2_vip)

model2.predict(X)   # 전체 변수에 대해 PLS결과를 적용하여 예측된 y값

idx_feature_valid2 = np.argwhere(model2_vip>1)
print(idx_feature_valid2)       # 중요한 변수라고 선태된 변수
# 보통은 중요한 변수라고 선태된 변수들만 가지고 다시 PLS를 하거나 다른 회귀분석을 진행한다.





def compute_mse_pls(X, y, n_comp, kf):   
    mse_train = np.zeros(shape=(kf.n_splits,))
    mse_test = np.zeros(shape=(kf.n_splits,))
    
    for i, (train_index, test_index) in enumerate(kf.split(X)):        
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        model = PLSRegression(n_components=n_comp)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        mse_train[i] = mean_squared_error(y_train, y_train_pred)
        mse_test[i] = mean_squared_error(y_test, y_test_pred)

    print('=' * 30)
    print(f'N_comp: {n_comp}')
    print(f'Train MSE: {np.around(np.mean(mse_train), decimals=3)}')
    print(f'Test MSE: {np.around(np.mean(mse_test), decimals=3)}')
    
    return n_comp, mse_train, mse_test




n_splits=5
n_range= [1, 2, 4, 6]
kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)

for n in n_range:
    compute_mse_pls(X=X, y=y, n_comp=n, kf=kf)



# ## Cross validation to select n_comp in PLS
def selct_ncomp_pls(X, y, n_comp_list, kf):   # 최적의 n_component 찾는 함수
    mse_test = np.zeros(shape=(kf.n_splits,len(n_comp_list)))
    
    for i, (n_comp) in enumerate(n_comp_list):    
        for fold, (train_index, test_index) in enumerate(kf.split(X)):        
            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            model = PLSRegression(n_components=n_comp)
            model.fit(X_train, y_train)            
            y_test_pred = model.predict(X_test)            
            mse_test[fold,i] = mean_squared_error(y_test, y_test_pred)
    
    idx_best = np.argmin(mse_test.mean(0))
    n_comp_best = n_comp_list[idx_best]
    mse_test_best = np.min(mse_test.mean(0))
    return (n_comp_best, mse_test, mse_test_best)



n_comp_list = [1,2,3,4,5,6,7]
pls_hyp = selct_ncomp_pls(X, y, n_comp_list, kf)
pls_hyp

print('n_comp_best:', pls_hyp[0])
print('cv_mse_best:', pls_hyp[2])

pls_hyp[1].mean(0)
mse_test.mean(0)





# ## 실습
# energy datset에서 PLS를 train/test evaluation 10번 반복해서 평가하기
df = pd.read_csv(absolute_path + 'energy_dataset.csv')
df.drop(columns=['generation hydro pumped storage aggregated'], inplace=True)
df.drop(columns=['forecast wind offshore eday ahead'], inplace=True)
df.dropna(inplace=True)
Input = df.loc[:,'generation biomass':'generation wind onshore']
Output =  df.loc[:,'price actual']
Input_np = Input.values
Output_np= Output.values





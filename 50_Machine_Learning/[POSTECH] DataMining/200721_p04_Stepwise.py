import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import * 

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
    # StandardScaler(X): 평균이 0과 표준편차가 1이 되도록 변환.
    # RobustScaler(X): 중앙값(median)이 0, IQR(interquartile range)이 1이 되도록 변환.
    # MinMaxScaler(X): 최대값이 각각 1, 최소값이 0이 되도록 변환
    # MaxAbsScaler(X): 0을 기준으로 절대값이 가장 큰 수가 1또는 -1이 되도록 변환
# scaler = eval('StandardScaler' + '()')
# scaler_result = scaler.fit(a6)
# scaler_result.transform(a6)
# scaler_result = scaler.fit_transform(a6)
# scaler.inverse_transform(scaler_result)



current_dir = os.getcwd()
os.getcwd()
# D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/dataset_city.csv
absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'



# 전체 변수를 사용했을때
df_wine = pd.read_excel(absolute_path + 'wine_aroma.xlsx')
X = df_wine.iloc[:, :-1].to_numpy()
y = df_wine.iloc[:, -1].to_numpy()

model = LinearRegression(fit_intercept=True)
model.fit(X, y)

y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print(f'Overall | mse: {np.around(mse, decimals=3)}')



# Drop each variable and compute MSE
n_instances, n_features = X.shape


for col_drop in range(n_features):
    # (차집합) make sub-dataset without only one variable(column)
    sub_col = np.setdiff1d(np.arange(n_features), col_drop)
    # print(col_drop, sub_col)
    X_sub = X[:, sub_col]
    
    # train regression model with sub-dataset
    model_sub = LinearRegression()
    model_sub.fit(X_sub, y)
    
    # compute evaluation measure
    y_pred_sub = model_sub.predict(X_sub)
    mse_sub = mean_squared_error(y, y_pred_sub)
    print(f'Drop {col_drop}| mse: {np.around(mse_sub, decimals=3)}')


# SFS (Sequential-Feature-Selector) --------------------------
# ?SFS
    # SFS(
    #     estimator,        # 모델
    #     k_features = 1,     # 몇개의 독립변수 선택?
    #     forward = True,     # True : forward selection / False: backward selection
    #                           # Start지점에서 채우면서 시작하냐 / 빼면서 시작하는지?
    #     floating = False,   # (True) forward : 지우는작업 / backward : 채우는 작업
    #                         # stepwise search 기능?
    #     verbose=0,
    #     scoring=None,     # (default: r2 score) evaluation measure 선택 (sklearn.metrics.SCORERS.keys())
    #                       classifiers : {accuracy, f1, precision, recall, roc_auc} 
    #                       regressors : {'mean_absolute_error', 'mean_squared_error'/'neg_mean_squared_error',
    #                                   'median_absolute_error', 'r2'}
    #     cv=5,             # cv : None, False, 0 (no cross-validation)  / 그외 (cross-validation)
    #     n_jobs=1,         # 병렬처리
    #     pre_dispatch='2*n_jobs',
    #     clone_estimator=True,
    #     fixed_features=None,
    # )



# 특정 변수의 갯수를 정하고 할때
    # forward_selction ****
fs = SFS(LinearRegression(), 
            k_features=5, forward=True, floating=False, scoring='r2', cv=0)
fs.fit(X,y)

fs.subsets_
pd.DataFrame.from_dict(fs.get_metric_dict())        # DataFrame

print(fs.k_score_)              # 제일좋은 features들의 score
print(fs.k_feature_idx_)        # 제일좋은 features

    # 시각화
    # x축 : feature 갯수 / y축 : scores
fig = plot_sfs(fs.get_metric_dict())
plt.title('Forward Selection')
plt.grid()
plt.show()


    # backward_selction ****
bs = SFS(LinearRegression(),
           k_features=5, forward=False,floating=False, scoring='r2', cv = 0)
bs.fit(X,y)

    # 시각화
fig = plot_sfs(bs.get_metric_dict())
plt.title('Backward Selection')
plt.grid()
plt.show()


# cross validation을 사용해서 할때 -----------------------------
n_splits = 5
kf = KFold(n_splits=n_splits)

fs_cv = SFS(LinearRegression(),
           k_features=5, forward=True,floating=False,scoring = 'r2', cv = kf)
fs_cv.fit(X,y)

pd.DataFrame.from_dict(fs_cv.get_metric_dict()).T
print(fs_cv.k_score_)
print(fs_cv.k_feature_idx_)

    # 시각화
fig = plot_sfs(fs_cv.get_metric_dict(), kind='std_err')
plt.title('Forward Selection (w. StdErr)')
plt.grid()
plt.show()



# 독립변수의 갯수를 몇개부터 몇개까지중에 선택해달라고 할때 ----------------
fs_cv2 = SFS(LinearRegression(),
           k_features=(4,7), forward=True,floating=False,scoring = 'r2', cv = kf)
# k_features = (4,7)    # 4~7 사이에서 제일 좋은 걸 찾아줘라
fs_cv2.fit(X,y)

pd.DataFrame.from_dict(fs_cv2.get_metric_dict()).T

print('best combination (R2: %.3f): %s\n' % (fs_cv2.k_score_, fs_cv2.k_feature_idx_))

fig = plot_sfs(fs_cv2.get_metric_dict(), kind='std_err')
plt.title('Forward Selection (w. StdErr)')
plt.grid()
plt.show()



# 독립변수의 갯수를 지정하지 않을때 ----------------
fs_cv3= SFS(LinearRegression(),
           k_features='best', forward=True,floating=False,scoring = 'r2', cv = kf)
fs_cv3.fit(X,y)

pd.DataFrame.from_dict(fs_cv3.get_metric_dict()).T
print('best combination (R2: %.3f): %s\n' % (fs_cv3.k_score_, fs_cv3.k_feature_idx_))

fig = plot_sfs(fs_cv3.get_metric_dict(), kind='std_err')
plt.title('Forward Selection (w. StdErr)')
plt.grid()
plt.show()



# 독립변수의 갯수를 지정하지 않을때 ----------------
# parsimonious : best case에서 standard error범위내 중에 가장 변수의 갯수가 작은 case를 선택
fs_cv4= SFS(LinearRegression(),
           k_features='parsimonious', forward=True,floating=False,scoring = 'r2', cv = kf)
fs_cv4.fit(X,y)

pd.DataFrame.from_dict(fs_cv4.get_metric_dict()).T
print('best combination (R2: %.3f): %s\n' % (fs_cv4.k_score_, fs_cv4.k_feature_idx_))

fig = plot_sfs(fs_cv4.get_metric_dict(), kind='std_err')
plt.title('Forward Selection (w. StdErr)')
plt.grid()
plt.show()



# backward_selection    : 변수갯수 미지정
bs_cv = SFS(LinearRegression(),
           k_features='best', forward=False,floating=False,scoring = 'r2', cv = kf)
bs_cv.fit(X,y)

pd.DataFrame.from_dict(bs_cv.get_metric_dict()).T
print('best combination (R2: %.3f): %s\n' % (bs_cv.k_score_, bs_cv.k_feature_idx_))

fig = plot_sfs(bs_cv.get_metric_dict(), kind='std_err')
plt.title('Backward Selection (w. StdErr)')
plt.grid()
plt.show()





# (Stepwise) Forward Backward search
fbs_cv= SFS(LinearRegression(),
           k_features='best', forward=True,floating=True, scoring = 'r2', cv=kf)
fbs_cv.fit(X,y)

fbs_cv.subsets_
pd.DataFrame.from_dict(fbs_cv.get_metric_dict()).T
print(fbs_cv.k_score_)
print(fbs_cv.k_feature_idx_)

fig = plot_sfs(fbs_cv.get_metric_dict(), kind='std_err')
plt.title('Forward Backward Selection (w. StdErr)')
plt.grid()
plt.show()


# (Stepwise) Backward Forward search 
bfs_cv= SFS(LinearRegression(),
           k_features='best', forward=False,floating=True, scoring = 'r2', cv=kf)
bfs_cv.fit(X,y)

bfs_cv.subsets_
pd.DataFrame.from_dict(bfs_cv.get_metric_dict()).T

print(bfs_cv.k_score_)
print(bfs_cv.k_feature_idx_)


fig = plot_sfs(bfs_cv.get_metric_dict(), kind='std_err')
plt.title('Backward Forward Selection (w. StdErr)')
plt.grid()
plt.show()

bfs_cv.get_metric_dict()




# 실습 energy dataset
# 1. train_test_split 10번 반복비교
# 2. forward / backward / forward-backward / backward-forward
df_energy = pd.read_csv(absolute_path + 'energy_dataset.csv')
# df_energy.drop(columns=['forecast wind offshore eday ahead'], inplace=True)
df_enery_drop = df_energy.drop(columns=['forecast wind offshore eday ahead'])
df_enery_drop.drop(columns=['generation hydro pumped storage aggregated'], inplace=True)
df_enery_drop.dropna(inplace=True)


# DataFrame split to object, numeric variables
def fun_object_numeric_split(data):
    obj_data = data.loc[:, data.dtypes == 'object']
    numeric_data = data.loc[:, (data.dtypes == 'float64') | (data.dtypes == 'int64')]
    return obj_data, numeric_data

energy_obj, energy_num = fun_object_numeric_split(df_enery_drop)


train_y, test_y, train_x, test_x = train_test_split(energy_num.iloc[:,-1], energy_num.iloc[:,:-1],
                                                    test_size=0.3, random_state=1)
energy_fs= SFS(LinearRegression(),
            # k_features='best',
            k_features=(5,10),
            forward=True, floating=False, scoring = 'r2', cv=5)
energy_fs.fit(train_x, train_y)

energy_fs.subsets_
pd.DataFrame.from_dict(energy_fs.get_metric_dict()).T

print(energy_fs.k_score_)
print(energy_fs.k_feature_idx_)

fig = plot_sfs(energy_fs.get_metric_dict(), kind='std_err')
plt.title('Backward Forward Selection (w. StdErr)')
plt.grid()
plt.show()


train_selection = train_x[list(energy_fs.k_feature_names_)]
test_selection = test_x[list(energy_fs.k_feature_names_)]

    # sklearn
energy_fs_model = LinearRegression(fit_intercept=True)
energy_fs_model.fit(train_selection, train_y)

energy_fs_pred = energy_fs_model.predict(test_selection)
mean_squared_error(y_true=test_y, y_pred=energy_fs_pred)


    # stats-models
train_selection_add_const = sm.add_constant(train_selection)
test_selection_add_const = sm.add_constant(test_selection)

energy_ols = sm.OLS(train_y, train_selection_add_const).fit()
energy_ols.summary()

energy_ols_pred = energy_ols.predict(test_selection_add_const)
mean_squared_error(y_true=test_y, y_pred=energy_ols_pred)



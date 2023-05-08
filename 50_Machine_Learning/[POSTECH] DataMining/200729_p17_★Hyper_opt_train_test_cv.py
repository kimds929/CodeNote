#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
# 모형최적화 : 하이퍼파라미터 튜닝
# https://datascienceschool.net/view-notebook/ff4b5d491cc34f94aea04baca86fbef8/

import sys
sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import *
from DS_OLS import *

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'


# Bank
df = pd.read_csv(absolute_path + 'Bank.csv')
df.head()
df_info = DS_DF_Summary(df)

# 독립변수/종속변수 분리
X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# 전처리 y
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_new = le.fit_transform(y)
pd.Series(y_new).value_counts()

# X.dtypes  # X 변수 타입 확인
print(df_info)

# object 변수 개수 확인
ind_obj = (X.dtypes == 'object')
X.loc[:, ind_obj].nunique()


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop='first', sparse=False)
cat_var = ohe.fit_transform(X.loc[:, ind_obj])
cat_var

ohe.get_feature_names()
df_cat = pd.DataFrame(cat_var, 
                     columns=ohe.get_feature_names())
df_cat.head()

df_new = pd.concat((X.loc[:, ~ind_obj], df_cat), axis=1)
df_new
# DS_DF_Summary(df_new)
# df_new.nunique()




X_new = df_new
y_new = pd.Series(y_new)
n_instances = X_new.shape[0]
n_test = 2000
n_train = n_instances - n_test
idx = np.arange(n_instances)

train_index, test_index = train_test_split(idx,
                                           train_size=n_train,
                                           test_size=n_test, 
                                           random_state=0)

print(train_index.shape)
print(test_index.shape)


X_train, X_test = X_new.iloc[train_index, :], X_new.iloc[test_index, :]
y_train, y_test = y_new.iloc[train_index], y_new.iloc[test_index]
kf = KFold(n_splits=5, shuffle=True, random_state=1) 


# ## Grid search
# ### 1단계 param grid 정의하기
params = {'min_samples_split': [0.001, 0.005, 0.1],
          'max_depth': [5, 10, 20],
          'criterion': ['gini', 'entropy']}


# ## 2단계 최적 hyper parameter 찾기 ----------------------
# ?GridSearchCV
# get_ipython().run_line_magic('pinfo', 'GridSearchCV')
# GridSearchCV(
#     estimator,
#     param_grid,
#     *,
#     scoring=None,
#     n_jobs=None,
#     iid='deprecated',
#     refit=True,
#     cv=None,
#     verbose=0,
#     pre_dispatch='2*n_jobs',
#     error_score=nan,
#     return_train_score=False,
# )

# ## 주요 파라미터
# ### estimator
# ### param_grid
# ### scoring
# ### cv
# ### return_train_score
# ### refit : best모델 추후에 사용할 수 있도록 하게 해줌

dtree = DecisionTreeClassifier()
    # accuracy
grid_acc = GridSearchCV(dtree, 
                        params, 
                        cv=kf, 
                        scoring='accuracy', 
                        return_train_score=True)
grid_acc.fit(X_train.to_numpy(), y_train.to_numpy())

grid_acc.cv_results_        # 결과 정보
# pd.DataFrame(grid_acc.cv_results_).to_clipboard(index=False)
sorted(grid_acc.cv_results_.keys())
grid_acc.best_score_         # best_model score
grid_acc.best_params_        # best_model hyper-parameter
grid_acc.best_estimator_     # best_model instance

    # f1_score
grid_f1 = GridSearchCV(dtree, 
                       params, 
                       cv=kf, 
                       scoring='f1' )
grid_f1.fit(X_train, y_train)

grid_f1.best_score_
grid_f1.best_params_
grid_f1.best_estimator_     # best_model instance




# ### 3단계. 최적의 모형 test set에서 평가
model_best_acc = grid_acc.best_estimator_   # best_model
model_best_f1 = grid_f1.best_estimator_   # best_model

y_test_hat_grid_acc = model_best_acc.predict(X_test)
y_test_hat_grid_f1 = model_best_f1.predict(X_test)

con_mat_grid_acc = confusion_matrix(y_test, y_test_hat_grid_acc)
print('Grid search Accuracy 기준 선택')
print(con_mat_grid_acc)
print(accuracy_score(y_test, y_test_hat_grid_acc))
print(f1_score(y_test, y_test_hat_grid_acc))

con_mat_grid_f1 = confusion_matrix(y_test, y_test_hat_grid_f1)

print('Grid search F1 score 기준 선택')
print(con_mat_grid_f1)
print(accuracy_score(y_test, y_test_hat_grid_f1))
print(f1_score(y_test, y_test_hat_grid_f1))




# ## Random search ****
# ### 1단계 param 구간 정의하기
# 

from sklearn.utils.fixes import loguniform
import scipy.stats as stats

# params = {'min_samples_split':[0.001, 0.005, 0.1],
#             'max_depth':[5,10,20],
#             'criterion':['gini', 'entropy']}
params = {'min_samples_split':loguniform(1e-3,1e-1),
            'max_depth':stats.randint(5,20),
            'criterion':['gini', 'entropy']}

# ### 2단계 최적 hyper parameter 찾기
dtree = DecisionTreeClassifier()
n_iter_search=18
random_acc = RandomizedSearchCV(dtree, 
                                param_distributions=params, 
                                cv=kf, 
                                scoring='accuracy', 
                                n_iter=n_iter_search)
random_acc.fit(X_train, y_train)

random_acc.cv_results_
random_acc.cv_results_.keys()
random_acc.best_score_
random_acc.best_params_
random_acc.best_estimator_


random_f1 = RandomizedSearchCV(dtree, 
                               param_distributions=params,
                               cv=kf, 
                               scoring='f1', 
                               n_iter=n_iter_search)
random_f1.fit(X_train, y_train)
random_f1.best_score_
random_f1.best_params_
random_acc.best_estimator_





# ### 3단계. 최적의 모형 test set에서 평가 -----------------------------
model_best_acc_rd = random_acc.best_estimator_
model_best_f1_rd = random_f1.best_estimator_

y_test_hat_rd_acc = model_best_acc_rd.predict(X_test)
y_test_hat_rd_f1 = model_best_f1_rd.predict(X_test)

con_mat_rd_acc = confusion_matrix(y_test, y_test_hat_rd_acc)
print('Randomized search Accuracy 기준 선택')
print(con_mat_rd_acc)
print(accuracy_score(y_test, y_test_hat_rd_acc))
print(f1_score(y_test, y_test_hat_rd_acc))

print('Grid search Accuracy 기준 선택')
print(con_mat_grid_acc)
print(accuracy_score(y_test, y_test_hat_grid_acc))
print(f1_score(y_test, y_test_hat_grid_acc))

con_mat_rd_f1 = confusion_matrix(y_test, y_test_hat_rd_f1)
print('Randomized search f1 기준 선택')
print(con_mat_rd_f1)
print(accuracy_score(y_test, y_test_hat_rd_f1))
print(f1_score(y_test, y_test_hat_rd_f1))

print('Grid search F1 score 기준 선택')
print(con_mat_grid_f1)
print(accuracy_score(y_test, y_test_hat_grid_f1))
print(f1_score(y_test, y_test_hat_grid_f1))










# 실습 ----------------------------------------------------------------------------------
# 분류기 : 의사결정나무, 서포트벡터머신
# 0. 연속형 변수, 이산형 변수에 따른 처리
# 1. Train 85%, Test 15% 분리
# 2. Train GridSearchCV  또는 RandomizedSeachCV를 이용해
#   최적 파라미터와 최적 모형을 찾기 (scoring: f1_score)
# 3. 최적모형을 Train전체와 Test전체에 대해 Accuracy와 F1-Score계산


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.fixes import loguniform
import scipy.stats as stats

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


df_titanic = pd.read_csv(absolute_path + 'Titanic.csv')
df_titanic_info = DS_DF_Summary(df_titanic)
print(df_titanic_info)


# y_col = 'Survived'
# x_col = list(df_titanic.columns.drop(y_col))
# ind_obj = (df_titanic[x_col].dtypes == 'object')

# ohe = OneHotEncoder(drop='first', sparse=False)
# X_ohe = ohe.fit_transform(df_titanic[x_col].loc[:, ind_obj])
# X_ohe_df = pd.DataFrame(X_ohe, columns=ohe.get_feature_names())


# df_titanic.loc[:,ind_obj]



# Train-Test split
train_df, test_df = train_test_split(df_titanic, test_size=0.15, shuffle=True, random_state=1)
print(train_df.shape, test_df.shape)

# 최적파라미터 찾기
tree_mdl = DecisionTreeClassifier
# tree
tree_opt = {'criterion' : ['gini', 'entropy'],
            'min_samples_split': loguniform(1e-3,1e-1),
            'max_depth': stats.randint(5,20)}








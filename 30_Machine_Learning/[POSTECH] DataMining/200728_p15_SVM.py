#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
# https://www.youtube.com/embed/3liCbRZPrZA?rel=0       # Kernel

import sys
sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import *
from DS_OLS import *

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'


# ## Load dataset

# load dataset
df = pd.read_csv(absolute_path + 'Hepatitis.csv')
df.head()
df_info = DS_DF_Summary(df)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]


X.head()
y.head()


# ## Linear SVM 
# ?SVC
# get_ipython().run_line_magic('pinfo', 'SVC')
# SVC(
#     *,
#     C=1.0,
#     kernel='rbf',
#     degree=3,
#     gamma='scale',
#     coef0=0.0,
#     shrinking=True,
#     probability=False,
#     tol=0.001,
#     cache_size=200,
#     class_weight=None,
#     verbose=False,
#     max_iter=-1,
#     decision_function_shape='ovr',
#     break_ties=False,
#     random_state=None,
# )

# [Hyper Parameter]
# C : 슬랙변수(오분류 허용) 가중치 (Penalty term)
#       * C가 커진다면, 오분류에 대한 벌칙을 강하게 주는 것이고 (오분류 타이트하게 허용)
#           작아진다면, 벌칙을 약하게 주는 것이다   (오분류 너프하게 허용)
# kernel: {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}
#       * 2차 3차 등 곡선 Decision_Boundary형성시 쓰임
# degree (if poly)
# gamma (if poly, rbf, and sigmoid)     # gamma는 커널 자체에 대한 파라미터
#       * 가우시안 구(Gaussian sphere)같은, 훈련 샘플의 영향력을 증가시키는 등의 역할을 하는 파라미터 계수
# coef0 (if poly and sigmoid) Independent term in kernel function.
# decision_function_shape: {'ovo', 'ovr'}
#   * ovo : 3개 이상의 클래스에서 클래스 마다 1:1로 분류 
#           (n)(n-1)/2 가지 경우를 고려
#   * ovr : 3개 이상 클래스에서 1개의 클래스와 나머지 클래스를 분류 
#           n가지 경우를 고려


# [attribute]
# n_support_: 각 클래스의 서포트의 개수
# support_: 각 클래스의 서포트의 인덱스
# support_vectors_: 각 클래스의 서포트의 x 값.  x+ 와  x− 
# coef_:  w  벡터
# intercept_:  −w0 
# dual_coef_: 각 원소가  ai⋅yi 로 이루어진 벡터



# 모형 선언 후 학습
lin_svc = SVC(kernel='linear')
lin_svc.fit(X, y)

lin_svc.n_support_          # 각 Class별 서보트 벡터 갯수
lin_svc.support_            # 서보트 벡터에 해당하는 관측치 인덱스
# y[lin_svc.support_]
# y[lin_svc.support_].value_counts()

support_vectors = pd.DataFrame(lin_svc.support_vectors_, index=lin_svc.support_, columns=X.columns)  # 서포트 벡터들의 값을 가져옴
support_vectors
support_vectors.shape



# 모형 살펴보기: weights to the features
print(lin_svc.coef_)
print(lin_svc.intercept_)

y_pred = lin_svc.predict(X)
y_pred  # 예측된 label

con_mat = confusion_matrix(y, y_pred)
acc = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred)
print(con_mat)
print('accuracy:', acc, 'f1 score:', f1)


# ### Regularization parameter 바꿔보기
lin_svc2 = SVC(kernel='linear', C=5)
lin_svc2.fit(X, y)


y_pred2 = lin_svc2.predict(X)
con_mat_2 = confusion_matrix(y, y_pred2)
acc_2 = accuracy_score(y, y_pred2)
f1_2 = f1_score(y, y_pred2)
print(con_mat_2)
print('accuracy:', acc_2, 'f1 score:', f1_2)




# ### 여러 regularization parameter 에 대해서 모형 학습 해보기
from sklearn.model_selection import train_test_split
n_instances, n_features = X.shape
idx = np.arange(n_instances)
train_index, test_index = train_test_split(idx,
                                           train_size=0.8,
                                           test_size=0.2)
X_train, X_test = X.iloc[train_index,:], X.iloc[test_index, :]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]

print(X_train.shape)
print(X_test.shape)


c_range = [1, 5, 10, 100, 1000]

f1_train = np.zeros(shape=(5,))
acc_train = np.zeros(shape=(5,))
f1_test = np.zeros(shape=(5,))
acc_test = np.zeros(shape=(5,))


svc_list = []
for i, c in enumerate(c_range):
    svc = SVC(kernel='linear', C=c).fit(X_train, y_train)
    svc_list.append(svc)
    y_train_pred =svc.predict(X_train)
    y_test_pred = svc.predict(X_test)
    acc_train[i] = accuracy_score(y_train, y_train_pred)
    f1_train[i] = f1_score(y_train, y_train_pred)
    
    acc_test[i] = accuracy_score(y_test, y_test_pred)
    f1_test[i] = f1_score(y_test, y_test_pred)
    print('='*30)
    print('Regularization parameter:', c)
    print('train set acc:', acc_train[i])
    print('f1 score:', f1_train[i])
    print('test set acc:', acc_test[i])
    print('f1 score:',f1_test[i])
    


# ## Non-linear kernel
svc_rbf_C100 = SVC(kernel='rbf', max_iter=1000, C=100).fit(X_train,y_train)
svc_rbf_C1 = SVC(kernel='rbf', max_iter=1000, C=1).fit(X_train,y_train)
svc_rbf2 = SVC(kernel='rbf', max_iter=1000, C=100).fit(X_train,y_train)
svc_poly = SVC(kernel='poly', degree=2, max_iter=1000, C=1).fit(X_train,y_train)
svc_poly2 = SVC(kernel='poly', degree=2, max_iter=1000, C=10).fit(X_train,y_train)

svc_list = [svc_rbf_C1, svc_rbf2, svc_poly, svc_poly2]


for svc in svc_list:
    y_train_pred= svc.predict(X_train)
    y_test_pred = svc.predict(X_test)
    con_mat_train = confusion_matrix(y_train, y_train_pred)
    con_mat_test = confusion_matrix(y_test, y_test_pred)
    print('='*30)
    print(svc)
    print('train set confusion mat')
    print(con_mat_train)
    print('test set confusion mat')
    print(con_mat_test)



# ### Standard scaler
from sklearn.preprocessing import StandardScaler
X_scaler = StandardScaler()
X_scaler.fit(X_train)

X_train_std = X_scaler.transform(X_train)
X_test_std = X_scaler.transform(X_test)


svc_rbf_std = SVC(kernel='rbf', max_iter=1000,C=1).fit(X_train_std, y_train)
svc_rbf2_std = SVC(kernel='rbf', max_iter=1000, C=100).fit(X_train_std, y_train)
svc_poly_std = SVC(kernel='poly', degree=2,max_iter=1000, C=1).fit(X_train_std, y_train)
svc_poly2_std = SVC(kernel='poly', degree=2,max_iter=1000, C=10).fit(X_train_std, y_train)

svc_list_std= [svc_rbf_std, svc_rbf2_std, svc_poly_std, svc_poly2_std]


for svc in svc_list_std:
    y_train_pred = svc.predict(X_train_std)
    y_test_pred = svc.predict(X_test_std)
    con_mat_train = confusion_matrix(y_train, y_train_pred)
    con_mat_test = confusion_matrix(y_test, y_test_pred)
    print('='*30)
    print(svc)
    print('Standard scaler & train set confusion mat')
    print(con_mat_train)
    print('Standard scaler & test set confusion mat')
    print(con_mat_test)


# # 실습
# ## Dataset: Hepatitis
# ## Hyper-parameter
# ### - kernel
# ### - C
# ## 5-fold cross validation을 통해 각 모형 하이퍼 파라미터에 대한 평균 train, test 성능(acc, f1)을 계산하기.
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

kf_practice = KFold(n_splits=5, shuffle=True, random_state=3920)

svc_rbf_std_p = SVC(kernel='rbf', max_iter=1000,C=1)
svc_rbf2_std_p = SVC(kernel='rbf', max_iter=1000, C=100)
svc_poly_std_p = SVC(kernel='poly', degree=2,max_iter=1000, C=1)
svc_poly2_std_p = SVC(kernel='poly', degree=2,max_iter=1000, C=10)

def svc_acc(model, X_train, y_train, X_test, y_test):
    result = {}
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    result['train'] = train_acc
    result['test'] = test_acc
    return result

models = {'rbf': svc_rbf_std_p, 'rbf2' : svc_rbf2_std_p,
        'poly': svc_poly_std_p, 'poly2' : svc_poly2_std_p}

accuracy_obj = {}
for fold, (train_index, test_index) in enumerate(kf_practice.split(df)):
    print(fold, len(train_index), len(test_index))
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y[train_index], y[test_index]

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.fit_transform(X_test)

    accuracy_obj[str(fold)] = {}
    for model in models:
        accuracy_obj[str(fold)][model] = svc_acc(models[model], 
                                    X_train=X_train, X_test=X_test,
                                    y_train=y_train, y_test=y_test)
result = pd.DataFrame(accuracy_obj).T

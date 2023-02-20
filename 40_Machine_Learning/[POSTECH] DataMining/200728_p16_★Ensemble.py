#!/usr/bin/env python
# coding: utf-8
# https://statkclee.github.io/model/model-ensemble.html

import numpy as np
import pandas as pd
from sklearn.ensemble import (VotingClassifier, BaggingClassifier, 
                              RandomForestClassifier, AdaBoostClassifier, 
                             GradientBoostingClassifier)
from sklearn.ensemble import (VotingRegressor, BaggingRegressor, 
                              RandomForestRegressor, AdaBoostRegressor, 
                              GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                             f1_score, roc_auc_score)
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

import sys
sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import *
from DS_OLS import *

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'


# load dataset
df = pd.read_csv(absolute_path + 'breast_cancer.csv')
df.head()

df_info = DS_DF_Summary(df)

# 독립변수/종속변수 분리 및 확인
X = df.iloc[:, 2:]
y = df.iloc[:, 1]


# y class 종류 확인
print(np.unique(y))


# Bagging : dataset을 뽑을때 변수 갯수를 추출
# RandomForest : tree를 만들때 노드마다 random하게 변수 갯수를 추출










# # Bagging classifier ----------------------------------------------------------------------
# ?BaggingClassifier
# get_ipython().run_line_magic('pinfo', 'BaggingClassifier')

# BaggingClassifier(
#     base_estimator=None,
#           . The base estimator to fit on random subsets of the dataset. If None, then the base estimator is a decision tree.
#     n_estimators=10,          # 모형의 갯수
#     *,
#     max_samples=1.0,          # 데이터 샘플링 갯수 또는 비율
#           . If int, then draw max_samples samples.
#           . If float, then draw max_samples * X.shape[0] samples.
#     max_features=1.0,         # 변수 샘플링 갯수 또는 비율
#           . If int, then draw max_features features.
#           . If float, then draw max_features * X.shape[1] features.
#     bootstrap=True,           # 데이터 반복추출여부 : Whether samples are drawn with replacement.
#     bootstrap_features=False, # 변수 반복추출여부 : Whether features are drawn with replacement.
#     oob_score=False,
#     warm_start=False,
#     n_jobs=None,
#     random_state=None,
#     verbose=0,
# )

# ## 주요 파라미터
# ### base_estimator: 기본 모형
# ### n_estimators: 모형 개수
# ### bootstrap: 데이터 중복 사용 여부
# ### max_samples: 데이터 샘플 중 사용할 개수 또는 비율
# ### bootstrap_features: 변수 중복 사용 여부
# ### max_features: 변수 중 사용할 개수 또는 비율


lr = LogisticRegression(max_iter=100)
lr.fit(X, y)

y_pred = lr.predict(X)

con_mat_lr = confusion_matrix(y, y_pred)
print(con_mat_lr)


bagging = BaggingClassifier(base_estimator=LogisticRegression(max_iter=100),
                                n_estimators=10, 
                                max_features=0.99, 
                                max_samples=0.99)
bagging.fit(X, y)
bagging

y_pred = bagging.predict(X)
con_mat_bagging = confusion_matrix(y, y_pred)
print(con_mat_bagging)












# ## Random Forest ----------------------------------------------------------------------
# ?RandomForestClassifier
# get_ipython().run_line_magic('pinfo', 'RandomForestClassifier')

# RandomForestClassifier(
#     n_estimators=100,
#     *,
#     criterion='gini',
#     max_depth=None,
#     min_samples_split=2,
#     min_samples_leaf=1,
#     min_weight_fraction_leaf=0.0,
#     max_features='auto',
#     max_leaf_nodes=None,
#     min_impurity_decrease=0.0,
#     min_impurity_split=None,
#     bootstrap=True,
#     oob_score=False,
#     n_jobs=None,
#     random_state=None,
#     verbose=0,
#     warm_start=False,
#     class_weight=None,
#     ccp_alpha=0.0,
#     max_samples=None,
# )

# ## 주요 파라미터
# ### n_estimators: 의사결정트리의 수
# ### max_features: default=sqrt(n_features)
# ### -
# ### criterion: {'gini', 'entropy'}
# ### max_depth
# ### min_samples_split
# ### min_samples_leaf


# hyperparameters: n_estimators, oob_score max_depth, min_samples_leaf, min_samples_split, 
random_forest = RandomForestClassifier(max_depth=4, 
                                       min_samples_split=0.01, 
                                       oob_score=True)
random_forest.fit(X, y)


# Estimators
# list of trees in random forest
random_forest.estimators_
random_forest.estimators_[0]

feature_importance = random_forest.feature_importances_
pd.Series(feature_importance, index=X.columns)
feature_name = X.columns.to_numpy()


import matplotlib.pyplot as plt
sorted_idx = feature_importance.argsort()
y_ticks = np.arange(0, len(feature_name))
fig, ax = plt.subplots(figsize=(10,15))
ax.barh(y_ticks, feature_importance[sorted_idx])
ax.set_yticklabels(feature_name[sorted_idx])
ax.set_yticks(y_ticks)
ax.set_title("Random Forest Feature Importances based on impurity")
fig.tight_layout()
plt.show()


# out of bag (OOB) score
random_forest.oob_score_

y_pred = random_forest.predict(X)

con_mat_rf = confusion_matrix(y, y_pred)
acc = accuracy_score(y, y_pred)
# f1 = f1_score(y, y_pred)
print(con_mat_rf)










# ExtraTreesClassifier ------------------------------------------------------------
# ## AdaBoost ------------------------------------------------------------
# get_ipython().run_line_magic('pinfo', 'AdaBoostClassifier')
# ?AdaBoostClassifier

# AdaBoostClassifier(
#     base_estimator=None,
#     *,
#     n_estimators=50,
#     learning_rate=1.0,
#     algorithm='SAMME.R',
#     random_state=None,
# )

# ## 주요 파라미터
# ### base_estimator
# ### n_estimators

adaboost = AdaBoostClassifier(n_estimators=20)
adaboost.fit(X, y)

# Esimators information
print(adaboost.estimator_weights_)
print(adaboost.estimator_errors_)

# list of trees in adabosst
adaboost.estimators_
adaboost.estimators_[1]

# feature importance
feature_importance_ada = adaboost.feature_importances_
print(feature_importance_ada)


import matplotlib.pyplot as plt
sorted_idx = feature_importance_ada.argsort()
y_ticks = np.arange(0, len(feature_name))
fig, ax = plt.subplots(figsize=(10,15))
ax.barh(y_ticks, feature_importance_ada[sorted_idx])
ax.set_yticklabels(feature_name[sorted_idx])
ax.set_yticks(y_ticks)
ax.set_title("AdaBoost Feature Importances based on impurity")
fig.tight_layout()
# plt.tight_layout()
plt.show()


y_pred = adaboost.predict(X)

con_mat = confusion_matrix(y, y_pred)
acc = accuracy_score(y, y_pred)
# f1 = f1_score(y, y_pred)
print(con_mat)











# # GradientBoosting ------------------------------------------------------------
# ?GradientBoostingClassifier
# GradientBoostingClassifier(
#     *,
#     loss='deviance',
#     learning_rate=0.1,
#     n_estimators=100,
#     subsample=1.0,
#     criterion='friedman_mse',
#     min_samples_split=2,
#     min_samples_leaf=1,
#     min_weight_fraction_leaf=0.0,
#     max_depth=3,                  # Decision Tree의 Overfitting 방지를 위해 Max_depth 지정
#     min_impurity_decrease=0.0,
#     min_impurity_split=None,
#     init=None,
#     random_state=None,
#     max_features=None,
#     verbose=0,
#     max_leaf_nodes=None,
#     warm_start=False,
#     presort='deprecated',
#     validation_fraction=0.1,
#     n_iter_no_change=None,
#     tol=0.0001,
#     ccp_alpha=0.0,
# )
gb = GradientBoostingClassifier()
gb.fit(X, y)

y_pred = gb.predict(X)

con_mat = confusion_matrix(y, y_pred)
acc = accuracy_score(y, y_pred)
print(con_mat)










# # Majority Voting (VotingClassifier) ----------------------------------------------------------------------

# 세 개의 다른 모형을 합쳐 다수결로 분류
# 3개 기본 모델 생성
model1 = LogisticRegression(max_iter=10000)  # 수렴을 위해 max_iter 입력
model2 = QuadraticDiscriminantAnalysis()
model3 = DecisionTreeClassifier()

# 3개 모형의 조합을 이용한 앙상블 분류기 생성
estimator_list = [('lr', model1), 
                 ('qda', model2), 
                 ('dt', model3)]
ensemble = VotingClassifier(estimators=estimator_list, 
                           voting='soft')

ensemble_model = ensemble.fit(X, y)
dir(ensemble)
ensemble_pred_y = ensemble_model.predict(X)
ensemble.estimators_

# for문에서 이름과 함께 출력하기 위해 아래 dictionary 선언
model_dict = {'LogisticRegression': model1, 
             'QuadraticDiscriminantAnalysis': model2, 
             'DecisionTree': model3, 
             'Voting': ensemble}

model_dict.items()


# 각 모형에 대해 Accuracy, AUC 출력
for model_name, model in model_dict.items():
    model.fit(X, y)
    
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob[:, 1])
    
    print('='*40)
    print(model_name)
    print('Accuracy:', acc)
    print('AUC:', auc)




# 각 모형에 대해 5-fold CV의 Accuracy, AUC 출력
from sklearn.model_selection import KFold, cross_validate
seed = np.random.choice(1000)
print('Seed:', seed)

kf = KFold(n_splits=5, shuffle=True, random_state=seed)
for model_name, model in model_dict.items():
    scores = cross_validate(model, X, y, 
                            scoring='accuracy', 
                            return_train_score=True, 
                            cv=kf)
    print('='*30)
    print(model_name)
    print(scores['train_score'])
    print(scores['test_score'])


# ![image.png](attachment:image.png)
# model_dict.items()
# model
# cross_validate(model, X, y, scoring='accuracy', return_train_score=True, cv=kf)
# ?cross_validate


# 실습 ---------------------------------------------------------------------
# adaboost, random_forest 비교
# 5-fold cross validation (cross_validate 사용)
# X, y에 대해 5-fold train, test 평균 accuracy 사용

models2_dict = {'adaboost' : AdaBoostClassifier(n_estimators=20),
         'random_forest' : RandomForestClassifier(max_depth=4, min_samples_split=0.01, oob_score=True)}
kf2 = KFold(n_splits=5, shuffle=True, random_state=1)


for mdl_name, mdl in models2_dict.items():
    scores2 = cross_validate(mdl, X, y, 
                            scoring=['accuracy', 'recall', 'precision'],
                            return_train_score=True, 
                            cv=kf)
    print('='*30)
    print(mdl_name)
    print('train', np.mean(scores2['train_score']))
    print('test', np.mean(scores2['test_score']))

# Lable Encoder -----------------------------------------------
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_new = le.fit_transform(y)
le.classes_
le.inverse_transform(y_new)


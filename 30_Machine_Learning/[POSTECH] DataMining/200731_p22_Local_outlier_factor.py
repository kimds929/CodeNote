#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

import sys
sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import *
from DS_OLS import *

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'


# LOF 알고리즘 연습을 위한 예제 데이터 생성
from sklearn.datasets import make_blobs

# ?make_blobs     # 데이터를 만드는 함수
# get_ipython().run_line_magic('pinfo', 'make_blobs')
# make_blobs(
#     n_samples=100,                # object 갯수
#     n_features=2,                 # column 갯수
#     *,
#     centers=None,                 # center를 지정
#     cluster_std=1.0,              # center로부터의 편차값 지정
#     center_box=(-10.0, 10.0),     # 데이터 형성에 대한 제한범위 (limit)
#     shuffle=True,                 # 중복허용여부
#     random_state=None,            # random seed설정
#     return_centers=False,
# )

# np.random.seed(1)  # for reproducibility
x, _ = make_blobs(n_samples=200, n_features=2,
                  centers=1, 
                  cluster_std=0.3, 
                  center_box=(10, 10),
                  random_state=1)
x[:10] # data
_[:10] # center_class

# plot example data
plt.scatter(x[:, 0], x[:, 1])
plt.show()

np.cov(x, rowvar=False)   # 분산, 공분산 행렬   # rowvar : row가 feature로 간주
np.cov(x, rowvar=False).shape


# ## Local Outlier Factor(LOF)      # 이상점 찾기
# ![LOF](그림파일/LOF.png)
# ?LocalOutlierFactor
# get_ipython().run_line_magic('pinfo', 'LocalOutlierFactor')
# LocalOutlierFactor(
#     n_neighbors=20,       # 주변에 몇개의 값을 가지고 와서 이상점을 판단할 것인지?
#     *,
#     algorithm='auto',
#     leaf_size=30,
#     metric='minkowski',   # 어떤 거리함수를 활용할 것인지?
#           ['cityblock', 'cosine', 'euclidean', 'l1', 'l2','manhattan']
#           ['braycurtis', 'canberra', 'chebyshev', 'correlation', 
#           'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 
#           'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 
#           'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
#     p=2,                  # Parameter for the Minkowski metric from
#     metric_params=None,
#     contamination='auto',     # 사전확률, 값을 높이면 outlier라고 분류할 확률이 올라감
#     novelty=False,
#     n_jobs=None,
# )


# ### 유클리디안 거리와 마할라노비스 거리
# <img src="그림파일/mahalanobis_example.png" width="300" height="300">
# mahalanobis(마할라노비스) : 분산을 고려한 거리

# Declare local outlier factor and get the result
x.shape

lof = LocalOutlierFactor(n_neighbors=20, 
                        contamination=.03,      # 사전 확률 개념으로 예상 outlier 비율
                        metric='mahalanobis',
                        metric_params={'V': np.cov(x, rowvar=False)})       # 공분산값
lof_result = lof.fit_predict(x)
lof_result

lof.negative_outlier_factor_        
    # -1보다 클수록 주변 데이터와 밀집되어있음
    # -1보다 작을수록 주변 데이터와 많이 떨어져 있음
lof.negative_outlier_factor_[lof_result == -1]


# 군집화 결과 plotting
def lof_plot(x, model):
    predict = model.fit_predict(x)
    unique_labels = np.unique(predict)  # 군집 종류
    for i in unique_labels:  # 각 군집에 대해
        cluster_instance_mask = (predict == i)
        x_cluster_i = x[cluster_instance_mask, :]  # 해당 군집에 해당하는 인스턴스
        plt.scatter(x_cluster_i[:, 0], x_cluster_i[:, 1], label='cluster ' + str(i))  # 1번째, 2번째 변수를 이용해 plotting

    plt.title('example LOF result')
    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.legend()
    plt.show()
lof_plot(x, lof)


# contamination 값 조정 0.03 → 0.1
lof2 = LocalOutlierFactor(n_neighbors=20, 
                        contamination=.1,      # 사전 확률 개념으로 예상 outlier 비율
                        metric='mahalanobis',
                        metric_params={'V': np.cov(x, rowvar=False)})       # 공분산값
lof2.fit(x)
lof_plot(x, lof2)


# contamination 값 조정 'auto'
lof3 = LocalOutlierFactor(n_neighbors=20, 
                        contamination='auto',      # 사전 확률 개념으로 예상 outlier 비율
                        metric='mahalanobis',
                        metric_params={'V': np.cov(x, rowvar=False)})       # 공분산값
lof3.fit(x)
lof_plot(x, lof3)



# ## 실습 내용

# * 아래 생성된 data set을 통해 Local Outlier Factor를 측정하는 모형 만들기
# * 하나의 모형은 마할라노비스 거리로, 하나의 모형은 유클리디언 거리로 모형 생성할 것
# * contamination은 0.03과 0.1로 각각 변경해 생성해볼 것

# np.random.seed(1)  # for reproducibility
x2, _2 = make_blobs(n_samples=800, n_features=2,
                  centers=4, 
                  cluster_std= 1, 
                  center_box=(-20, 20),
                  random_state=1)

lof_maha1 = LocalOutlierFactor(metric='mahalanobis', contamination=0.03, metric_params={'V': np.cov(x2, rowvar=False)})
lof_maha2 = LocalOutlierFactor(metric='mahalanobis', contamination=0.1, metric_params={'V': np.cov(x2, rowvar=False)})
lof_mins1 = LocalOutlierFactor(metric='minkowski', contamination=0.03)
lof_mins2 = LocalOutlierFactor(metric='minkowski', contamination=0.1)

lof_maha1.fit(x2)
lof_maha2.fit(x2)
lof_mins1.fit(x2)
lof_mins2.fit(x2)

# lof_plot(x2, lof_maha1)
# lof_plot(x2, lof_maha2)
# lof_plot(x2, lof_mins1)
# lof_plot(x2, lof_mins2)

    # plot
fig, axe = plt.subplots(2,2, figsize=(10,10))
axe[0][0].set_title('mahalanobis / contamination=0.03')
axe[0][0].scatter(x2[:,0], x2[:,1], c=lof_maha1.fit_predict(x2))
axe[0][1].set_title('mahalanobis / contamination=0.1')
axe[0][1].scatter(x2[:,0], x2[:,1], c=lof_maha2.fit_predict(x2))
axe[1][0].set_title('minkowski / contamination=0.03')
axe[1][0].scatter(x2[:,0], x2[:,1], c=lof_mins1.fit_predict(x2))
axe[1][1].set_title('minkowski / contamination=0.1')
axe[1][1].scatter(x2[:,0], x2[:,1], c=lof_mins2.fit_predict(x2))
plt.show


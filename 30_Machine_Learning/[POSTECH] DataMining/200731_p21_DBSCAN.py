#!/usr/bin/env python
# coding: utf-8

# 라이브러리 임포트
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn import metrics

import sys
sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import *
from DS_OLS import *

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'


# 군집 알고리즘 연습을 위한 예제 데이터 생성
x = np.c_[[4, 20, 3, 19, 17, 8, 19, 18, 8, 3, 19, 18],
         [15, 13, 13, 4, 17, 11, 12, 6, 10, 15, 14, 5]]
x_df = pd.DataFrame(x, columns=['experience', 'violation'])
x_df

plt.scatter(x_df.iloc[:,0], x_df.iloc[:,1])

# ## DBSCAN
# <img src="그림파일/DBSCAN.png" width="300" height="300">

# ## DBSCAN 예제
# <img src="그림파일/DBSCAN_example.png" width="500" height="500">

# 파라미터 설정
eps = 4
min_samples = 3


# DBSCAN 군집화 알고리즘 실행
# ?DBSCAN
# DBSCAN(
#     eps=0.5,
#     *,
#     min_samples=5,
#     metric='euclidean',
#     metric_params=None,
#     algorithm='auto',
#     leaf_size=30,
#     p=None,
#     n_jobs=None,
# )
db = DBSCAN(eps=eps, min_samples=min_samples).fit(x)

db.core_sample_indices_   # core에 해당되는 관측치

# core sample들의 indicator
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
db.labels_  # 각 object들의 DBSCAN된 label



def plot_clustering(data, model):
    unique_labels = np.unique(model.labels_)  # 군집 종류
    for i in unique_labels:  # 각 군집에 대해
        cluster_instance_mask = (model.labels_ == i)
        x_cluster_i = data[cluster_instance_mask, :]  # 해당 군집에 해당하는 인스턴스
        plt.scatter(x_cluster_i[:, 0], x_cluster_i[:, 1], label='cluster ' + str(i))  # 1번째, 2번째 변수를 이용해 plotting
    plt.title('example DBSCAN result')
    plt.xlabel('experience')
    plt.ylabel('violation')
    plt.legend()
    plt.show()
    return

plot_clustering(x, db)


def plot_clustering_2(data, model_db):
    unique_labels = np.unique(model_db.labels_)
    core_samples_mask = np.zeros_like(model_db.labels_, dtype=bool)
    core_samples_mask[model_db.core_sample_indices_] = True
    
    for i in unique_labels:
        if i == -1:
            # 데이터 포인트가 noise로 판단될 경우, 데이터 포인트를 검정색으로 표시
            col = [0, 0, 0, 1]
        else:
            col = np.random.rand(1, 4)

        class_member_mask = (model_db.labels_ == i)

        xt = data[class_member_mask & core_samples_mask, :]
        plt.scatter(xt[:, 0], xt[:, 1], 
                    marker='o', 
                    color=col,
                    edgecolors='k', 
                    s=200)

        xt = data[class_member_mask & ~core_samples_mask, :]
        plt.scatter(xt[:, 0], xt[:, 1], 
                    marker='o', 
                    color=col, 
                    edgecolors='k', 
                    s=60)

    plt.title('example DBSCAN result')
    plt.xlabel('experience')
    plt.ylabel('violation')
    plt.show()
    return

plot_clustering_2(x, db)


# # DBSCAN 군집화가 필요한 경우
x_dbscan_df = pd.read_csv(absolute_path + 'dbscan_data.csv', header=None)
x_dbscan_df.head()
x_dbscan_df_info = DS_DF_Summary(x_dbscan_df)

x_dbscan = x_dbscan_df.iloc[:, 0:2].values


# 전체 데이터 plotting
plt.figure()
plt.scatter(x_dbscan[:, 0], x_dbscan[:, 1])
plt.title('Dataset for DBSCAN')
plt.xlabel('X0')
plt.ylabel('X1')
plt.show()


# # DBSCAN 데이터를 이용해 DBSCAN 군집 실행 및 결과 그림
# ### df = pd.read_csv('dbscan_data.csv', header=None)
# ##### * 첫 두 열만 이용할 것 (3rd column=class label)
# ##### * 군집 파라미터(eps, min_samples)를 정하기 위해 미리 데이터의 분포를 확인할 것 (ex. 2-d plot, statistics such as min, max, mean, std)

dbscan_eps = 1
dbscan_min_samples = x_dbscan.shape[1] + 1
db_model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
db_model.fit(x_dbscan)
plot_clustering_2(x_dbscan, db_model)



# eps, min_saples에 따른 dbscan결과
eps_list = np.linspace(1.5, 3.5, 7)
min_samples = range(3,8)
for i in eps_list:
    for j in min_samples:
        print('#'*10)
        print(f'eps:{i}, min_samples:{j}')
        db_model2 = DBSCAN(eps=i, min_samples=j).fit(x_dbscan)
        plot_clustering_2(x_dbscan, db_model2)



# k-distance method
def k_distances(x, k):
    dim0 = x.shape[0]
    dim1 = x.shape[1]
    p=-2*x.dot(x.T)+np.sum(x**2, axis=1).T+ np.repeat(np.sum(x**2, axis=1),dim0,axis=0).reshape(dim0,dim0)
    p = np.sqrt(p)
    p.sort(axis=1)
    p=p[:,:k]
    pm= p.flatten()
    pm= np.sort(pm)
    return p, pm

# k-distance plot
def k_distance_plot(x, k):
    m, m2= k_distances2(x, k)
    plt.plot(m2)
    plt.ylabel("k-distances")
    plt.grid(True)
    plt.show()

k_distance_plot(x_dbscan, 10)



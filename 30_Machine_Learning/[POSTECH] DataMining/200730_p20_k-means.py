#!/usr/bin/env python
# coding: utf-8

# 라이브러리 임포트
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.cluster import KMeans  # K-means 임포트
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import *
from DS_OLS import *

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'



# 군집 알고리즘 연습을 위한 예제 데이터 생성
x = np.c_[[4, 20, 3, 19, 17, 8, 19, 18, 12.5],
         [15, 13, 13, 4, 17, 11, 12, 6, 9]]
x_df = pd.DataFrame(x, columns=['experience', 'violation'])
x_df


# ## K-means 알고리즘
# ![kmeans](그림/kmeans.png)

# ?KMeans
# get_ipython().run_line_magic('pinfo', 'KMeans')
# KMeans(
#     n_clusters=8,
#     *,
#     init='k-means++',
#     n_init=10,
#     max_iter=300,
#     tol=0.0001,
#     precompute_distances='deprecated',
#     verbose=0,
#     random_state=None,
#     copy_x=True,
#     n_jobs='deprecated',
#     algorithm='auto',
# )

# 파라미터 설정
num_clusters = 3  # 군집 개수

# K-means 군집화 알고리즘 실행
kmeans = KMeans(n_clusters=num_clusters, init='k-means++')
# (init) k-means++ : 맨 처음 객체는 랜덤하게 선택, 그 관측치에서 가장 먼 관측치가 선택
#        random : 랜덤하게 시작점을 선택
kmeans.fit(x)
kmeans.labels_

# ![image.png](attachment:image.png)


# 군집화 결과 plotting
unique_labels = np.unique(kmeans.labels_)  # 군집 종류

centroids = []
for i in unique_labels:  # 각 군집에 대해
    cluster_instance_mask = (kmeans.labels_ == i)
    
    x_cluster_i = x[cluster_instance_mask, :]       # 해당 군집에 속하는 인스턴스
    centroids.append(np.mean(x_cluster_i, axis=0))  # 군집마다의 평균
    
    plt.scatter(x_cluster_i[:, 0], x_cluster_i[:, 1], label='cluster ' + str(i))  # 1번째, 2번째 변수를 이용해 plotting

plt.title('example K-means result')
plt.xlabel('experience')
plt.ylabel('violation')
plt.legend()
plt.show()


print('manual_centroid: ', centroids)
print('model_centroid: ', kmeans.cluster_centers_)


# # Practice 1: 아래 내용을 완성하세요 -----------------------------------------------------------------------
# 먼 객체들로 초기 객체 선정한 모형의 결과값(label) 저장
kmeans_pp = KMeans(n_clusters=3, init='k-means++')
kmeans_pp.fit(x)
kmeans_pp_labels = kmeans_pp.labels_
kmeans_pp_labels

plt.scatter(x[:,0], x[:,1], c=kmeans_pp.labels_)
plt.scatter(kmeans_pp.cluster_centers_[:,0], kmeans_pp.cluster_centers_[:,1], 
            marker='*', c='red', s=200, alpha=0.3)
plt.show()


# 랜덤 초기화 방법으로 K-means 군집화 알고리즘 실행
kmeans_random = KMeans(n_clusters=3, init='random', random_state=1)
kmeans_random.fit(x)
kmeans_random_labels = kmeans_random.labels_    # 랜덤 초기화 모형의 결과값(label) 저장
kmeans_random_labels


plt.scatter(x[:,0], x[:,1], c=kmeans_random.labels_)
plt.scatter(kmeans_random.cluster_centers_[:,0], kmeans_random.cluster_centers_[:,1], 
            marker='*', c='red', s=200, alpha=0.3)
plt.show()


# 랜덤 초기화 군집화 결과 plotting
unique_labels = np.unique(kmeans_random.labels_)  # 군집 종류

for i in unique_labels:  # 각 군집에 대해
    cluster_member_mask = (kmeans_random.labels_ == i)
    
    x_cluster_i = x[cluster_member_mask, :]  # 해당 군집에 해당하는 인스턴스
    plt.scatter(x_cluster_i[:, 0], x_cluster_i[:, 1], label='cluster ' + str(i))  # 1번째, 2번째 변수를 이용해 plotting

plt.scatter(kmeans_random.cluster_centers_[:, 0], kmeans_random.cluster_centers_[:, 1],
           s = 200, c = 'green', label = 'Centroids', marker = '*')
plt.title('example K-means result')
plt.xlabel('experience')
plt.ylabel('violation')
plt.legend()
plt.show()





# KMean, 초기 위치를 Manual로 지정 --------------------------------------------------------------------------------------------
# 데이터 x로부터 초기 중심객체로 사용할 1, 2, 3번째 인스턴스 추출
init_cents = x[:3, :]

# 선정된 초기 중심객체를 이용해 K-means 군집화 알고리즘 실행
kmeans_manual = KMeans(n_clusters=3, init=init_cents)
kmeans_manual.fit(x)

kmeans_manual_labels = kmeans_manual.labels_    # 결과값 저장

# 특정 객체 초기화 군집화 결과 plotting
unique_labels = np.unique(kmeans_manual.labels_)  # 군집 종류

for i in unique_labels:  # 각 군집에 대해
    cluster_member_mask = (kmeans_manual.labels_ == i)
    
    x_cluster_i = x[cluster_member_mask, :]  # 해당 군집에 해당하는 인스턴스

    plt.scatter(x_cluster_i[:, 0], x_cluster_i[:, 1], label='cluster ' + str(i))  # 1번째, 2번째 변수를 이용해 plotting

plt.scatter(kmeans_manual.cluster_centers_[:, 0], kmeans_manual.cluster_centers_[:, 1],
           s = 200, c = 'green', label = 'Centroids', marker = '*')
plt.title('example K-means result')
plt.xlabel('experience')
plt.ylabel('violation')
plt.legend()
plt.show()





# # Practice 2: 아래 내용을 완성하세요 ------------------------------------------------------------------------------
# ### 원본 데이터
# 군집 개수 2개일 경우 (kmeans++ & without standardization)
num_clusters2 = 2
kmeans2 = KMeans(n_clusters=num_clusters2, init='k-means++')
kmeans2.fit(x)
kmeans2_centroid = kmeans2.cluster_centers_

kmeans2_std = KMeans(n_clusters=num_clusters2, init='k-means++')
standard = StandardScaler()
kmeans2_std.fit(standard.fit_transform(x))
kmeans2_centroid_std = standard.inverse_transform(kmeans2_std.cluster_centers_)

    # plotting
fig2, axe2 = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
axe2[0].set_title('k-mean')
axe2[0].scatter(x[:,0], x[:,1], c=kmeans2.labels_)
axe2[0].scatter(kmeans2_centroid[:,0], kmeans2_centroid[:,1],
                marker='*', c='red', s=200, alpha=0.5)
axe2[1].set_title('k-mean_Standard_Scale')
axe2[1].scatter(x[:,0], x[:,1], c=kmeans2_std.labels_)
axe2[1].scatter(kmeans2_centroid_std[:,0], kmeans2_centroid_std[:,1],
                marker='*', c='red', s=200, alpha=0.5)
plt.show()



# ### 그냥 Clustering 했을 경우
unique_labels = np.unique(kmeans2.labels_)  # 군집 종류

for i in unique_labels:  # 각 군집에 대해
    cluster_member_mask = (kmeans2.labels_ == i)    
    x_cluster_i = x[cluster_member_mask, :]  # 해당 군집에 해당하는 인스턴스
    plt.scatter(x_cluster_i[:, 0], x_cluster_i[:, 1], label='cluster ' + str(i))  # 1번째, 2번째 변수를 이용해 plotting

    
plt.scatter(kmeans2.cluster_centers_[:, 0], kmeans2.cluster_centers_[:, 1],
           s = 200, c = 'green', label = 'Centroids', marker = '*')
plt.title('example K-means result')
plt.xlabel('experience')
plt.ylabel('violation')
plt.legend()
plt.show()


# ### 표준화(Standardization)된 데이터
unique_labels = np.unique(kmeans2_std.labels_)  # 군집 종류

for i in unique_labels:  # 각 군집에 대해
    cluster_member_mask = (kmeans2_std.labels_ == i)
    
    x_cluster_i = x[cluster_member_mask, :]  # 해당 군집에 해당하는 인스턴스

    plt.scatter(x_cluster_i[:, 0], x_cluster_i[:, 1], label='cluster ' + str(i))  # 1번째, 2번째 변수를 이용해 plotting

plt.scatter(kmeans2_centroid_std[:, 0], kmeans2_centroid_std[:, 1],
           s = 200, c = 'green', label = 'Centroids', marker = '*')
plt.title('example K-means result')
plt.xlabel('experience')
plt.ylabel('violation')
plt.legend()
plt.show()




# # Practice ----------------------------------------------------------------------------------------------------------------------
# ## Load Iris dataset - pd.read_csv('Iris.csv')
# ### Implement K-means clustering with 'n_clusters=3'
# ### and plot 2 figures of the result (1st figure x='SepalLengthCm', y='SepalWidthCm' and 2nd figure x='PetalLengthCm', y='PetalWidthCm')

df_iris = pd.read_csv(absolute_path + 'Iris.csv')

# x='SepalLengthCm', y='SepalWidthCm' 
# x='PetalLengthCm', y='PetalWidthCm'
case1 = ['SepalLengthCm', 'SepalWidthCm']
case2 = ['PetalLengthCm', 'PetalWidthCm']

case1_df = df_iris[case1]
case2_df = df_iris[case2]

num_clusters_iris = 3
standard_iris_case1 = StandardScaler()
standard_iris_case2 = StandardScaler()

    # case1
case1_kmean = KMeans(n_clusters=num_clusters_iris, init='k-means++', random_state=1)
case1_kmean.fit(standard_iris_case1.fit_transform(case1_df))
case1_kmean_label = case1_kmean.labels_
case1_kmean_centroid = standard_iris_case1.inverse_transform(case1_kmean.cluster_centers_)

    # case2
case2_kmean = KMeans(n_clusters=num_clusters_iris, init='k-means++', random_state=2)
case2_kmean.fit(standard_iris_case2.fit_transform(case2_df))
case2_kmean_label = case2_kmean.labels_
case2_kmean_centroid = standard_iris_case2.inverse_transform(case2_kmean.cluster_centers_)

    # plotting
fig_iris, axe_iris = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
axe_iris[0].set_title('case_1')
axe_iris[0].scatter(case1_df.iloc[:,0], case1_df.iloc[:,1], c=case1_kmean_label)
axe_iris[0].scatter(case1_kmean_centroid[:,0], case1_kmean_centroid[:,1],
                    marker='*', c='red', s=200, alpha=0.5)

axe_iris[1].set_title('case_2')
axe_iris[1].scatter(case2_df.iloc[:,0], case2_df.iloc[:,1], c=case2_kmean_label)
axe_iris[1].scatter(case2_kmean_centroid[:,0], case2_kmean_centroid[:,1],
                    marker='*', c='red', s=200, alpha=0.5)
plt.show()
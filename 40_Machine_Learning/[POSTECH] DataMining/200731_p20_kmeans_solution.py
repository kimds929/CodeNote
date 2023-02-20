#!/usr/bin/env python
# coding: utf-8

# [2]:


# 라이브러리 임포트
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.cluster import KMeans  # K-means 임포트
from sklearn.preprocessing import StandardScaler


# [3]:


# 군집 알고리즘 연습을 위한 예제 데이터 생성
x = np.c_[[4, 20, 3, 19, 17, 8, 19, 18, 12.5],
         [15, 13, 13, 4, 17, 11, 12, 6, 9]]
x_df = pd.DataFrame(x, columns=['experience', 'violation'])
x_df


# ## K-means 알고리즘

# ![kmeans](그림/kmeans.png)

# [4]:


# 파라미터 설정
num_clusters = 3  # 군집 개수


# [5]:


get_ipython().run_line_magic('pinfo', 'KMeans')


# [6]:


# K-means 군집화 알고리즘 실행
kmeans = KMeans(n_clusters=num_clusters, init='k-means++')
kmeans.fit(x)
kmeans.labels_


# ![image.png](attachment:image.png)

# [7]:


cluster_instance_mask


# [ ]:


# 군집화 결과 plotting
unique_labels = np.unique(kmeans.labels_)  # 군집 종류

centroids = []
for i in unique_labels:  # 각 군집에 대해
    cluster_instance_mask = (kmeans.labels_ == i)
    
    x_cluster_i = x[cluster_instance_mask, :]  # 해당 군집에 속하는 인스턴스
    centroids.append(np.mean(x_cluster_i, axis=0))
    
    plt.scatter(x_cluster_i[:, 0], x_cluster_i[:, 1], label='cluster ' + str(i))  # 1번째, 2번째 변수를 이용해 plotting

plt.title('example K-means result')
plt.xlabel('experience')
plt.ylabel('violation')
plt.legend()
plt.show()


# [ ]:


print('manual_centroid: ', centroids)
print('model_centroid: ', kmeans.cluster_centers_)


# # Practice 1: 아래 내용을 완성하세요

# [ ]:


# 먼 객체들로 초기 객체 선정한 모형의 결과값(label) 저장
kmeans_pp_labels = kmeans.labels_
kmeans_pp_labels


# [ ]:


# 랜덤 초기화 방법으로 K-means 군집화 알고리즘 실행
kmeans_random = KMeans(n_clusters=num_clusters, init='random')

# 랜덤 초기화 모형의 결과값(label) 저장
kmeans_random_labels = kmeans_random.fit(x).labels_


# [ ]:


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


# [ ]:


init_cents


# [ ]:


# 초기 중심객체로 사용할 1, 2, 3번째 인스턴스 추출
init_cents = x[:3, :]
# 선정된 초기 중심객체를 이용해 K-means 군집화 알고리즘 실행
kmeans_manual = KMeans(n_clusters=num_clusters, init= init_cents)

# 결과값 저장
kmeans_manual_labels = kmeans_manual.fit(x).labels_

# n_init 파라미터를 통해 객체를 10번 바꿔서 반복할 것이라 설정되어있음
# 하지만 초기 객체를 설정할 경우, 10번 반복하는 것이 의미가 없음(결과가 모두 동일)
# 따라서 설정한 초기객체로 1번만 실행된 결과를 출력했다는 에러메시지임.


# [ ]:


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


# # Practice 2: 아래 내용을 완성하세요

# ### 원본 데이터

# [ ]:


# 군집 개수 2개일 경우 (kmeans++ & without standardization)
num_clusters = 2

kmeans = KMeans(n_clusters=num_clusters, init= 'k-means++').fit(x)



# 특정 객체 초기화 군집화 결과 plotting
unique_labels = np.unique(kmeans_manual.labels_)  # 군집 종류


for i in unique_labels:  # 각 군집에 대해
    cluster_member_mask = (kmeans.labels_ == i)
    
    x_cluster_i = x[cluster_member_mask, :]  # 해당 군집에 해당하는 인스턴스

    plt.scatter(x_cluster_i[:, 0], x_cluster_i[:, 1], label='cluster ' + str(i))  # 1번째, 2번째 변수를 이용해 plotting

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           s = 200, c = 'green', label = 'Centroids', marker = '*')
plt.title('example K-means result')
plt.xlabel('experience')
plt.ylabel('violation')
plt.legend()
plt.show()


# ### 표준화된 데이터

# <img src="그림/stand.png" width="300" height="300">

# [ ]:


print(f'mean value: {x.mean(axis = 0)}, var value: {x.var(axis = 0)}')


# [8]:


# 군집 개수 2개일 경우 (kmeans++ & with standardization)
scaler = StandardScaler()
x_std = scaler.fit_transform(x)
num_clusters = 2

kmeans = KMeans(n_clusters=num_clusters, init= 'k-means++').fit(x_std)

# 특정 객체 초기화 군집화 결과 plotting
unique_labels = np.unique(kmeans.labels_)  # 군집 종류

for i in unique_labels:  # 각 군집에 대해
    cluster_member_mask = (kmeans.labels_ == i)    
    x_cluster_i = x_std[cluster_member_mask, :]  # 해당 군집에 해당하는 인스턴스
    plt.scatter(x_cluster_i[:, 0], x_cluster_i[:, 1], label='cluster ' + str(i))  # 1번째, 2번째 변수를 이용해 plotting

    
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           s = 200, c = 'green', label = 'Centroids', marker = '*')
plt.title('example K-means result')
plt.xlabel('experience')
plt.ylabel('violation')
plt.legend()
plt.show()


# # Practice
# ## Load Iris dataset - pd.read_csv('Iris.csv')
# ### Implement K-means clustering with 'n_clusters=3'
# ### and plot 2 figures of the result (1st figure x='SepalLengthCm', y='SepalWidthCm' and 2nd figure x='PetalLengthCm', y='PetalWidthCm')

# [9]:


dataset = pd.read_csv('Iris.csv')

x_df = dataset.iloc[:, 1:5]
x = x_df.to_numpy()
y_df = dataset.iloc[:, -1]
y = y_df.to_numpy()

kmeans = KMeans(n_clusters = 3).fit(x)



# 특정 객체 초기화 군집화 결과 plotting
unique_labels = np.unique(kmeans.labels_)  # 군집 종류

for i in unique_labels:  # 각 군집에 대해

    cluster_member_mask = (kmeans.labels_ == i)    
    x_cluster_i = x[cluster_member_mask, :]  # 해당 군집에 해당하는 인스턴스
    plt.scatter(x_cluster_i[:, 0], x_cluster_i[:, 1], label='cluster ' + str(i))  # 1번째, 2번째 변수를 이용해 plotting

    
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           s = 200, c = 'green', label = 'Centroids', marker = '*')
plt.title('Iris K-means result')
plt.legend()
plt.show()


# [10]:


# 특정 객체 초기화 군집화 결과 plotting
unique_labels = np.unique(kmeans.labels_)  # 군집 종류

for i in unique_labels:  # 각 군집에 대해

    cluster_member_mask = (kmeans.labels_ == i)    
    x_cluster_i = x[cluster_member_mask, :]  # 해당 군집에 해당하는 인스턴스
    plt.scatter(x_cluster_i[:, 2], x_cluster_i[:, 3], label='cluster ' + str(i))  # 1번째, 2번째 변수를 이용해 plotting

    
plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3],
           s = 200, c = 'green', label = 'Centroids', marker = '*')
plt.title('Iris K-means result')
plt.legend()
plt.show()


# [11]:


kmeans.cluster_centers_


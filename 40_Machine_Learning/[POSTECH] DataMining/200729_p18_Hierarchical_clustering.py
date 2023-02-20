# coding: utf-8

# 라이브러리 임포트
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage

from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

import sys
sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import *
from DS_OLS import *

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'


# Iris 데이터 (Iris.csv) 불러오기
x_df = pd.read_csv(absolute_path + 'Iris.csv')
x_df.head()
x_df_info = DS_DF_Summary(x_df)

# Iris 데이터 
x = x_df.iloc[:, 1:5].values

# 파라미터 설정
num_clusters = 2
n_instances, n_dim = x.shape


# 계층적 군집화 알고리즘 (Agglomerative - Ward) 실행 
# ?AgglomerativeClustering
# AgglomerativeClustering(
#     n_clusters=2,
#     *,
#     affinity='euclidean',
#     memory=None,
#     connectivity=None,
#     compute_full_tree='auto',
#     linkage='ward',
#     distance_threshold=None,
# )
ward = AgglomerativeClustering(n_clusters=num_clusters, 
                            affinity='euclidean', linkage='ward').fit(x)
# dir(ward)
ward.labels_        # Lable 결과



# 계층적 군집화 결과 plotting
unique_labels = np.unique(ward.labels_)
unique_labels

for i in unique_labels:
    cluster_member_mask = (ward.labels_ == i)
    x_cluster_i = x[cluster_member_mask, :]
    plt.scatter(x_cluster_i[:, 0], x_cluster_i[:, 1], label='cluster ' + str(i))

plt.title('example hierarchical clustering (ward) result')
plt.xlabel('SepalLengthCm')
plt.ylabel('SepalWidthCm')
plt.legend()
plt.show()


# 군집 중심 좌표 계산
C = np.zeros([num_clusters, n_dim])
for i in np.unique(ward.labels_):
    C[i, :] = np.mean(x[ward.labels_==i, :], axis=0)
C


# # 계층적 군집화에서 덴드로그램을 이용한 군집 수 결정
# 덴드로그램 작성을 위한 linkage matrix 계산
from scipy.cluster.hierarchy import linkage
Z = linkage(x, 'ward')
np.round(Z, decimals=0) 
# 0,1 column : 묶인 인 index, 
# 2: 묶인 index간 거리
# 3 : 묶인 군집의 갯수


# metric: euclidean, minkowski, cosine, jaccard, mahalanobis...
# (check metrics in scipy.spatial.distance.pdist)

# 덴드로그램 작성
def plot_dendrogram(link_mat, n_clusters, mode=None, truncate_p=100):
    plt.figure()
    plt.title('Hierarchical Clustering (Ward) Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        link_mat,
        color_threshold=link_mat[1-n_clusters, 2],
        truncate_mode=mode,
        p=truncate_p
    )
    plt.show()


# ![image.png](attachment:image.png)
# 덴드로그램 (last 100 aggregation step) 작성
plot_dendrogram(Z, num_clusters)

# 덴드로그램 (last 10 step) 작성
plot_dendrogram(Z, num_clusters, mode='lastp', truncate_p=10)

# 파라미터 설정
num_clusters = 3
n_instances, n_dim = x.shape

# 계층적 군집화 알고리즘 (Agglomerative - Ward) 실행 
ward = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward').fit(x)
# ward.labels_


# 계층적 군집화 결과 plotting
unique_labels = np.unique(ward.labels_)

for i in np.unique(ward.labels_):
    cluster_member_mask = (ward.labels_ == i)
    x_cluster_i = x[cluster_member_mask, :]
    plt.scatter(x_cluster_i[:, 0], x_cluster_i[:, 1], label='cluster ' + str(i))

plt.title('example hierarchical clustering (ward) result')
plt.xlabel('SepalLengthCm')
plt.ylabel('SepalWidthCm')
plt.legend()
plt.show()


# 군집 중심 좌표 계산
C = np.zeros([num_clusters, n_dim])
for i in np.unique(ward.labels_):
    C[i, :] = np.mean(x[ward.labels_==i, :], axis=0)
C


# 덴드로그램 (last 10 step) 작성
plot_dendrogram(Z, num_clusters, mode='lastp', truncate_p=10)


# # Practice
# ### 1. Load synthetic dataset - pd.read_excel('syn_data.xlsx')
# ### 2. Plot dendrogram with last 20 steps
# ### 3. Choose 2 most probable number of clusters with dendrogram
# ### 4. Plot the results of two cases

var_names = ['x1', 'x2']
x_df = pd.read_excel(absolute_path + 'syn_data.xlsx', header=None, names=var_names)
x_df_info = DS_DF_Summary(x_df)

n_class = 10
ward_test = AgglomerativeClustering(n_clusters=n_class, 
                affinity='euclidean', linkage='ward').fit(x_df)

pd.Series(ward_test.labels_).value_counts()


for i in range(n_class):
    test_mask = (ward_test.labels_ == i)
    test_cluster_i = x_df.iloc[test_mask,:]
    plt.scatter(test_cluster_i.iloc[:, 0], test_cluster_i.iloc[:, 1], label='cluster ' + str(i))
plt.title('example hierarchical clustering (ward) result')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()


Z_test = linkage(x_df, 'ward')
np.around(Z_test, decimals=0)

# 0,1 column : 묶인 인 index, 
# 2: 묶인 index간 거리
# 3 : 묶인 군집의 갯수

plot_dendrogram(Z_test, 2, mode='lastp', truncate_p=20)



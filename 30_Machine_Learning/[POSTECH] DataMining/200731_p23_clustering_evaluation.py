#!/usr/bin/env python
# coding: utf-8

# 라이브러리 임포트
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score, calinski_harabasz_score

import sys
sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import *
from DS_OLS import *

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'


# Iris 데이터 데이터 파일(Iris.csv) 불러오기
x_df = pd.read_csv(absolute_path + 'Iris.csv') 
x_df.head()


# 데이터를 numpy array 형태로 추출
x = x_df.iloc[:, 1:5].values


# 실제 label 도출
unique_species = np.unique(x_df['Species'])

labels_true = np.zeros(x.shape[0])
for i, species in enumerate(unique_species):
    labels_true[x_df['Species'] == species] = i
print(labels_true)


# 군집 개수 후보 리스트
num_clusters_set = np.arange(2, 11)


# ### Adjusted-RI (실제 결과가 있어야 평가가능) --------------------------------------------------------------------------------------
# * 값은 (Adjusted-RI) ≤ 1 에 위치함 
# * 클러스터링 결과가 label값과 일치할수록 1에 가깝고, 일치하지 않을수록 0과 가까움
# * 무작위로 군집화에도 Rand Index가 어느정도 나오기 때문에 이를 방지하기 위해 Adjusted-RI를 사용


labels_true
# ?adjusted_rand_score
# get_ipython().run_line_magic('pinfo', 'adjusted_rand_score')
# adjusted_rand_score(
#   labels_true, 
#   labels_pred)


for num_clusters in num_clusters_set:
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++')
    kmeans.fit(x)
    labels_pred = kmeans.labels_
    print('num_clusters: ', num_clusters, 
          '| adjusted RI: ', adjusted_rand_score(labels_true, labels_pred))


# ## 실루엣 (실제 결과가 없어도 평가가능)--------------------------------------------------------------------------------------
# * 실루엣계수는 값이 클수록 좋은 군집화라 판단할 수 있음.
# ?silhouette_score
# get_ipython().run_line_magic('pinfo', 'silhouette_score')
# silhouette_score(
#     X,                    # 데이터
#     labels,               # 예측한 결과 data
#     *,
#     metric='euclidean',   # 계산 방식
#     sample_size=None,     # 전체 데이터는 시간이 많이 걸리기 때문에, 몇개 sample을 뽑아서 평가
#     random_state=None,    # sample옵션 사용시, 데이터 뽑는 random seed
#     **kwds,
# )


for num_clusters in num_clusters_set:
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++')
    kmeans.fit(x)
    
    labels_pred = kmeans.labels_
    
    print('num_clusters: ', num_clusters, 
          '| silhouette: ', silhouette_score(x, labels_pred, metric='euclidean'))


# ## CH-index --------------------------------------------------------------------------------------
# * 클러스터 내의 분산 값과 클러스터 간의 분산 값의 비를 나타낸 것
# * 클러스터 내의 분산 값은 작고, 클러스터 간의 분산 값은 클수록 클러스터링의 결과가 좋다고 할 수 있음.
# * CH-index 값이 클수록 클러스터링 결과가 좋다고 할 수 있음.

# ?calinski_harabasz_score
# get_ipython().run_line_magic('pinfo', 'calinski_harabasz_score')
# calinski_harabasz_score(
#     X,        # 데이터
#     labels)   # 예측한 결과 data


for num_clusters in num_clusters_set:
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++')
    kmeans.fit(x)
    
    labels_pred = kmeans.labels_
    
    print('num_clusters: ', num_clusters, 
          '| CH-index: ', calinski_harabasz_score(x, labels_pred))


# # 실습
# * syn_unbalanced.xlsx 파일을 불러온다.
# 2. k-means나 gaussian 군집 방법을 선택한다.
# 3. 군집수 3~10까지에 대해 silhouette value와 CH-index를 이용해 적합한 군집수 후보를 3개 찾는다.
# 4. 선택한 3개 군집수에 대한 분석 결과 plot을 그리기

from sklearn.mixture import GaussianMixture
sys_data = pd.read_excel(absolute_path + 'syn_unbalance.xlsx')
sys_data_info = DS_DF_Summary(sys_data)

result = {}
result['kmean_silhouette'] = []
result['kmean_ch_index'] = []
result['gaussian_silhouette'] = []
result['gaussian_ch_index'] = []

for i in range(3,11):
        # kmean
    kmeans_sys = KMeans(n_clusters=i, init='k-means++').fit(sys_data)
    kmean_sys_labels = kmeans_sys.labels_

    kmean_silhouette = silhouette_score(X=sys_data, labels=kmean_sys_labels, metric='euclidean')
    result['kmean_silhouette'].append(kmean_silhouette)

    kmean_ch_index = calinski_harabasz_score(X=sys_data, labels=kmean_sys_labels)
    result['kmean_ch_index'].append(kmean_ch_index)


        # gaussian
    gmm_sys = GaussianMixture(n_components=i, 
                covariance_type='full', random_state=1).fit(sys_data)
    gmm_sys_labels = gmm_sys.predict(sys_data)

    gaussian_silhouette = silhouette_score(X=sys_data, labels=gmm_sys_labels, metric='euclidean')
    result['gaussian_silhouette'].append(gaussian_silhouette)

    gaussian_ch_index = calinski_harabasz_score(X=sys_data, labels=gmm_sys_labels)
    result['gaussian_ch_index'].append(gaussian_ch_index)


result
pd.DataFrame(result)




def plot_clustering(data, model, method='k-means'):
    if type(data) == np.ndarray:
        data_np = data.copy()
    else:
        data_np = data.to_numpy()

    if method == 'k-means':
        model_label = model.labels_
    elif method == 'gmm':
        model_label = model.predict(data_np)
    unique_labels = np.unique(model_label)  # 군집 종류

    for i in unique_labels:  # 각 군집에 대해
        cluster_instance_mask = (model_label == i)
        x_cluster_i = data_np[cluster_instance_mask, :]  # 해당 군집에 해당하는 인스턴스
        plt.scatter(x_cluster_i[:, 0], x_cluster_i[:, 1], label='cluster ' + str(i))  # 1번째, 2번째 변수를 이용해 plotting

    plt.title('Clustering result')
    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.legend()
    plt.show()
    return None


for i in [5,6,7,8,9]:
    print(i, '-'*30)
    kmeans_sys = KMeans(n_clusters=i, init='k-means++').fit(sys_data)
    gmm_sys = GaussianMixture(n_components=i, 
                covariance_type='full', random_state=1).fit(sys_data)

    print('k-mean ***')
    plot_clustering(sys_data, kmeans_sys, method='k-means')

    print('gaussian ***')
    plot_clustering(sys_data, gmm_sys, method='gmm')

# ## Reference
# - https://datascienceschool.net/view-notebook/54ee87f1caf84311a0efcbe73fa9e1ea/
# - http://scikit-learn.org/stable/modules/clustering.html






# Plotting Function ****
def clustering_plot(X, labels=None, figsize='auto', alpha=1):
    import itertools

    combination_set = list(itertools.combinations(X.columns,2))
    len_comb = len(combination_set)

    if len_comb == 1:
        dim = 1;        nrows = 1;        ncols = 1;
    elif len_comb == 2:
        dim = 1;        nrows = 1;        ncols = 2;
    elif len_comb == 3:
        dim = 1;        nrows = 1;        ncols = 3;
    elif len_comb % 3 == 0:
        dim = 2;        nrows = len_comb // 3;        ncols = 3;
    elif len_comb % 2 == 0:
        dim = 2;        nrows = len_comb // 2;        ncols = 2;

    # print(dim, nrows, ncols)
    if list(labels):
        unique_label = np.unique(labels)

    if figsize == 'auto':
        figsize = (ncols*5, nrows*4)
    fig, axe = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for ci, (x1, x2) in enumerate(combination_set):
        # print(ci, x1, x2)
        if dim == 1:
            if list(labels):
                for l in unique_label:
                    cluster_mask = (labels == l)
                    X_mask = X.iloc[cluster_mask, :]
                    axe[ci].scatter(X_mask[x1], X_mask[x2], label=str(l), edgecolors='grey', alpha=alpha)
                axe[ci].legend()
            else:
                axe[ci].scatter(X[x1], X[x2], c='skyblue', edgecolors='grey', alpha=alpha)
            axe[ci].set_xlabel(x1)
            axe[ci].set_ylabel(x2)
        else: # dim == 2:
            if list(labels):
                for l in unique_label:
                    cluster_mask = (labels == l)
                    X_mask = X.iloc[cluster_mask, :]
                    axe[ci//ncols-1][ci%ncols].scatter(X_mask[x1], X_mask[x2], label=str(l), edgecolors='grey', alpha=alpha)
                axe[ci//ncols-1][ci%ncols].legend()
            else:
                axe[ci//ncols][ci%ncols].scatter(X[x1], X[x2], c='skyblue', edgecolors='grey', alpha=alpha)
            axe[ci//ncols][ci%ncols].set_xlabel(x1)
            axe[ci//ncols][ci%ncols].set_ylabel(x2)
    plt.show()
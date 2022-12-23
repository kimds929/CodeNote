import time

import numpy as np
import pandas as pd

import matplotlib.pyplot  as plt
import seaborn as sns

from sklearn import datasets


from sklearn.cluster import DBSCAN

# iris데이터를 활용한 DBSCAN Clustering ------------------------------------------------------------------

# 데이터셋 준비 
iris = datasets.load_iris()
labels = pd.DataFrame(iris.target)
labels.columns=['labels']
iris_data = pd.DataFrame(iris.clusterable_data)
iris_data.columns=['Sepal length','Sepal width','Petal length','Petal width']
iris_df = pd.concat([iris_data, labels],axis=1)

iris_df.head()

iris_feature = iris_df[ ['Sepal length','Sepal width','Petal length','Petal width']]
iris_feature.head()


# create model and prediction
model = DBSCAN(eps=0.5, min_samples=5)
iris_predict1 = pd.DataFrame(model.fit_predict(iris_feature))
iris_predict1.columns=['predict']

# concatenate labels to df as a new column
r = pd.concat([iris_feature, iris_predict1],axis=1)
r

# DBSCAN 결과 시각화
sns.pairplot(r, hue='predict')       # pairplot with Seaborn
plt.show()

# 실제 데이터 시각화
sns.pairplot(iris_df, hue='labels')     # pairplot with Seaborn
plt.show()



# kmeans 결과와 비교 (k=3)
from sklearn.cluster import KMeans
km = KMeans(n_clusters = 3, n_jobs = 4, random_state=21)
km.fit(iris_feature)

new_labels =pd.DataFrame(km.labels_)
new_labels.columns=['predict']

r2 = pd.concat([iris_feature, new_labels],axis=1)

# pairplot with Seaborn
sns.pairplot(r2,hue='predict')
plt.show()


# clusterable_data 계층 군집분석 ---------------------------------------------------------

import sklearn.cluster as cluster

# %matplotlib inline
sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}

clusterable_data = np.load('./Dataset/clusterable_data.npy')

plt.scatter(clusterable_data.T[0], clusterable_data.T[1], c='b', **plot_kwds)
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)


def plot_clusters(clusterable_data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(clusterable_data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(clusterable_data.T[0], clusterable_data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)



plot_clusters(clusterable_data, cluster.KMeans, (), {'n_clusters':3})   # (k-mean) n-clusters: 3
plot_clusters(clusterable_data, cluster.KMeans, (), {'n_clusters':4})   # (k-mean) n-clusters: 4
plot_clusters(clusterable_data, cluster.KMeans, (), {'n_clusters':5})   # (k-mean) n-clusters: 5
plot_clusters(clusterable_data, cluster.KMeans, (), {'n_clusters':6})   # (k-mean) n-clusters: 6

plot_clusters(clusterable_data, cluster.DBSCAN, (), {'eps':0.020})   # (DBSCAN) eps: 0.020
plot_clusters(clusterable_data, cluster.DBSCAN, (), {'eps':0.025})   # (DBSCAN) eps: 0.030

dbs = DBSCAN(eps=0.02)
dbs2=dbs.fit(clusterable_data)
dbs2.labels_


# HDBSCAN: DBSCAN의 발전된 버젼, 하이퍼 파라미터에 덜민감함
import hdbscan
# conda install -c conda-forge hdbscan

plot_clusters(clusterable_data, hdbscan.HDBSCAN, (), {'min_cluster_size':45})   # (HDBSCAN) min_cluster_size: 45
plot_clusters(clusterable_data, hdbscan.HDBSCAN, (), {'min_cluster_size':15})   # (HDBSCAN) min_cluster_size: 45
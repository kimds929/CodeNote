import os
import time

import numpy as np
import pandas as pd

from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt



from sklearn.cluster import KMeans

#  Iris 데이터를 활용 Kmeans clustering------------------------------------------------
iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

# 기존 Data분류 (y : kinds of flowers)
plt.scatter(X[:,0], X[:,1], c=y, cmap='gist_rainbow')
plt.xlabel('Spea1 Length', fontsize=18)
plt.ylabel('Sepal Width', fontsize=18)
plt.show()

KMeans()
km = KMeans(n_clusters = 3, n_jobs = 4, random_state=21)
# KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300,
#        tol=0.0001, precompute_distances='auto', verbose=0,
#        random_state=None, copy_x=True, n_jobs=None, algorithm='auto')
km.fit(X)

centers = km.cluster_centers_
print(centers)



new_labels = km.labels_
# Plot the identified clusters and compare with the answers
fig, axes = plt.subplots(1, 2, figsize=(16,8))
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='gist_rainbow',
edgecolor='k', s=150)
axes[1].scatter(X[:, 0], X[:, 1], c=new_labels, cmap='jet',
edgecolor='k', s=150)
axes[0].set_xlabel('Sepal length', fontsize=18)
axes[0].set_ylabel('Sepal width', fontsize=18)
axes[1].set_xlabel('Sepal length', fontsize=18)
axes[1].set_ylabel('Sepal width', fontsize=18)
axes[0].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[1].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[0].set_title('Actual', fontsize=18)
axes[1].set_title('Predicted', fontsize=18)




# 2차원의 가상 데이터에 Kmeans clustering ------------------------------------------------
from sklearn.datasets import make_blobs
# create dataset
X, y = make_blobs(
   n_samples=150, n_features=2,
   centers=3, cluster_std=0.5,
   shuffle=True, random_state=0
)

# plot
plt.scatter(
   X[:, 0], X[:, 1],
   c='white', marker='o',
   edgecolor='black', s=50
)
plt.show()

# K-means predict
km2 = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km2 = km2.fit_predict(X)
y_km2


# K-means Clusting Result Plot (K:3) 
plt.scatter(
    X[y_km2 == 0, 0], X[y_km2 == 0, 1],
    s=50, c='lightgreen', marker='s', edgecolor='black', label='cluster 1'
    )
plt.scatter(
    X[y_km2 == 1, 0], X[y_km2 == 1, 1],
    s=50, c='orange', marker='o', edgecolor='black', label='cluster 2'
    )
plt.scatter(
    X[y_km2 == 2, 0], X[y_km2 == 2, 1],
    s=50, c='lightblue', marker='v', edgecolor='black', label='cluster 3'
    )

    # plot the centroids
plt.scatter(
    km2.cluster_centers_[:, 0], km2.cluster_centers_[:, 1],
    s=250, marker='*',c='red', edgecolor='black', label='centroids'
    )
plt.legend(scatterpoints=1)
plt.grid()
plt.show()



# k 를 4로 할경우 ---------------------------------------------------------------------
km3 = KMeans(
    n_clusters=4, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km3 = km3.fit_predict(X)

# K-means Clusting Result Plot (K: 4)
plt.scatter(
    X[y_km3 == 0, 0], X[y_km3 == 0, 1],
    s=50, c='lightgreen', marker='s', edgecolor='black', label='cluster 1'
    )
plt.scatter(
    X[y_km3 == 1, 0], X[y_km3 == 1, 1],
    s=50, c='orange', marker='o', edgecolor='black', label='cluster 2'
    )
plt.scatter(
    X[y_km3 == 2, 0], X[y_km3 == 2, 1],
    s=50, c='lightblue', marker='v', edgecolor='black', label='cluster 3'
    )
plt.scatter(
    X[y_km3 == 3, 0], X[y_km3 == 3, 1],
    s=50, c='purple', marker='d', edgecolor='black', label='cluster 4'
    )
# plot the centroids
plt.scatter(
    km3.cluster_centers_[:, 0], km3.cluster_centers_[:, 1],
    s=250, marker='*',c='red', edgecolor='black', label='centroids'
    )
plt.legend(scatterpoints=1)
plt.grid()
plt.show()


# Elbow Method -----------------------------------------------------------
distortions = []
for i in range(1, 11):
    km_dist = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km_dist.fit(X)
    distortions.append(km_dist.inertia_)

# plot
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()




# K-medoid

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

documents = ["This little kitty came to play when I was eating at a restaurant.","hello kitty is my favorite character",
             "Merley has the best squooshy kitten belly.","Is Google translator so good?","google google"
             "google Translate app is incredible.","My dog s name is Kong","dog dog dog","cat cat"
             "If you open 100 tab in google you get a smiley face.","Kong is a very cute and lovely dog",
             "Best cat photo I've ever taken.","This is a cat house"
             "Climbing ninja cat kitty.","What's your dog's name?","Cat s paws look like jelly",
             "Impressed with google map feedback.","I want to join google","You have to wear a collar when you walk the dog",
             "Key promoter extension for google Chrome.","Google is the best company","Google researcher"]

# documnet 정보를 벡터로 치환
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

X       # row: 19, column: 55 인 Matrix 생성
# <19x55 sparse matrix of type '<class 'numpy.float64'>'
# 	with 74 stored elements in Compressed Sparse Row format>


true_k = 3
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

model.labels_

[x for x, y in zip(documents, model.labels_) if  y == 0]    # 0 으로 예측한 문장
[x for x, y in zip(documents, model.labels_) if  y == 1]    # 1 로 예측한 문장
[x for x, y in zip(documents, model.labels_) if  y == 2]    # 2 로 예측한 문장


Y = vectorizer.transform(["chrome browser to open."])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["I want to have a dog"])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["My cat is hungry."])
prediction = model.predict(Y)
print(prediction)



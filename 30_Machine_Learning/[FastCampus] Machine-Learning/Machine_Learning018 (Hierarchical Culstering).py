import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as shc

# 간단한 데이터를 활용한 Hierarchical Culstering(계층적 군집화) --------------------------------------------
# 데이터 생성
X = np.array([[5,3],
    [10,15],
    [15,12],
    [24,10],
    [30,30],
    [85,70],
    [71,80],
    [60,78],
    [70,55],
    [80,91],])


# scatter-plot 시각화
labels = range(1, 11)
plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(X[:,0],X[:,1], label='True Position')
for label, x, y in zip(labels, X[:, 0], X[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-3, 3),
        textcoords='offset points', ha='right', va='bottom')
plt.show()


linked = linkage(X, 'single')

# 계층 시각화
labelList = range(1, 11)
plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()



# 예제 데이터를 이용한 실습 ----------------------------------------------------------------------------------------
# 데이터 로드
os.getcwd()     # 현재경로 확인
customer_data = pd.read_csv('./Dataset/shopping-data.csv')   # 데이터 불러오기
customer_data.head()

shopping_data = customer_data.iloc[:, 3:5].values

# scatter-plot 시각화
plt.scatter(x=data[:,0], y=data[:,1])

dend_model = shc.linkage(data, method='ward')
dir(dend_model)

# 계층정보 시각화
plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms")
dend = shc.dendrogram(dend_model)



from sklearn.cluster import AgglomerativeClustering

# 갯수 지정하여 군집화
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(shopping_data)

# 군집화 결과 Scatter-Plot
plt.figure(figsize=(10, 7))
plt.scatter(shopping_data[:,0], shopping_data[:,1], c=cluster.labels_, cmap='rainbow')


# boston house 집값 데이터를 활용한 군집화 --------------------------------------------------------------------------
boston_data = pd.read_csv("./Dataset/Boston_house.csv") 

'''
< y값: 타겟 데이터 >
1978 보스턴 주택 가격
506개 타운의 주택 가격 중앙값 (단위 1,000 달러)

< X값: 특징 데이터 >
CRIM: 범죄율
INDUS: 비소매상업지역 면적 비율
NOX: 일산화질소 농도
RM: 주택당 방 수
LSTAT: 인구 중 하위 계층 비율
B: 인구 중 흑인 비율
PTRATIO: 학생/교사 비율
ZN: 25,000 평방피트를 초과 거주지역 비율
CHAS: 찰스강의 경계에 위치한 경우는 1, 아니면 0
AGE: 1940년 이전에 건축된 주택의 비율
RAD: 방사형 고속도로까지의 거리
DIS: 직업센터의 거리
TAX: 재산세율'''

boston_data.shape
boston_data.head() # 데이터 확인

boston_y = boston_data['Target']
boston_X = boston_data.drop(['Target'], axis = 1) 


# Dendiagram
plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms")
dend = shc.dendrogram(shc.linkage(boston_X, method='ward'))

# 2개의 군집으로 지정하여 군집화
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster.fit_predict(boston_X)

np.mean([x for x, y in zip(boston_y, cluster.fit_predict(boston_X)) if  y == 0])    # 0번 클러스터의 평균 y값
np.mean([x for x, y in zip(boston_y, cluster.fit_predict(boston_X)) if  y == 1])    # 1번 클러스터의 평균 y값

boston_X.iloc[cluster.fit_predict(boston_X)==0,:].describe()
boston_X.iloc[cluster.fit_predict(boston_X)==1,:].describe()
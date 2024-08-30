import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

import sys
sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import DS_DF_Summary, DS_OneHotEncoder, DS_LabelEncoder
from DS_OLS import *

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Postech_AI 4) Aritificial_Intelligent/교재_실습_자료/'
# absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'




# Clustering ---------------------------------------------------------------------
# data generation
G0 = np.random.multivariate_normal([1, 1], np.eye(2), 100)
G1 = np.random.multivariate_normal([3, 5], np.eye(2), 100)
G2 = np.random.multivariate_normal([9, 9], np.eye(2), 100)

X = np.vstack([G0, G1, G2])
X = np.asmatrix(X)
print(X.shape)

plt.figure(figsize = (10, 8))
plt.plot(X[:,0], X[:,1], 'b.')
plt.axis('equal')
plt.show()



# The number of clusters and data
k = 3
m = X.shape[0]

# ramdomly initialize mean points
mu = X[np.random.randint(0, m, k), :]
pre_mu = mu.copy()
print(mu)

plt.figure(figsize = (10, 8))
plt.plot(X[:,0], X[:,1], 'b.')
plt.plot(mu[:,0], mu[:,1], 'r*', markersize=20)
plt.axis('equal')
plt.show()



# K-means
y = np.empty([m,1])


# Run K-means
for n_iter in range(500):
    for i in range(m):
        d0 = np.linalg.norm(X[i,:] - mu[0,:], 2)
        d1 = np.linalg.norm(X[i,:] - mu[1,:], 2)
        d2 = np.linalg.norm(X[i,:] - mu[2,:], 2)

        y[i] = np.argmin([d0, d1, d2])
    
    err = 0
    for i in range(k):
        mu[i,:] = np.mean(X[np.where(y == i)[0]], axis = 0)
        err += np.linalg.norm(pre_mu[i,:] - mu[i,:], 2)
    
    pre_mu = mu.copy()
    
    if err < 1e-10:
        print("Iteration:", n_iter)
        break    


X0 = X[np.where(y==0)[0]]
X1 = X[np.where(y==1)[0]]
X2 = X[np.where(y==2)[0]]

plt.figure(figsize = (10, 8))
plt.plot(X0[:,0], X0[:,1], 'b.', label = 'C0')
plt.plot(X1[:,0], X1[:,1], 'g.', label = 'C1')
plt.plot(X2[:,0], X2[:,1], 'r.', label = 'C2')
plt.axis('equal')
plt.legend(fontsize = 12)
plt.show()




# use kmeans from the scikit-learn module ----------------------------------------------
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, random_state = 0)
kmeans.fit(X)
np.unique(kmeans.labels_)

plt.figure(figsize = (10,8))
plt.plot(X[kmeans.labels_ == 0,0],X[kmeans.labels_ == 0,1], 'b.', label = 'C0')
plt.plot(X[kmeans.labels_ == 1,0],X[kmeans.labels_ == 1,1], 'g.', label = 'C1')
plt.plot(X[kmeans.labels_ == 2,0],X[kmeans.labels_ == 2,1], 'r.', label = 'C2')
plt.axis('equal')
plt.legend(fontsize = 12)
plt.show()





# 4.2. Choosing the Number of Clusters -------------------------------------------------

# data generation
G0 = np.random.multivariate_normal([1, 1], np.eye(2), 100)
G1 = np.random.multivariate_normal([3, 5], np.eye(2), 100)
G2 = np.random.multivariate_normal([9, 9], np.eye(2), 100)

X = np.vstack([G0, G1, G2])
X = np.asmatrix(X)


cost = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 0).fit(X)
    cost.append(abs(kmeans.score(X)))

plt.figure(figsize = (10,8))
plt.stem(range(1,11), cost)
plt.xticks(np.arange(11))
plt.xlim([0.5, 10.5])
plt.grid(alpha = 0.3)
plt.show()





# Clustering Example ------------------------------------------------------------------

# 1. Setting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.cluster import KMeans
# %matplotlib inline
absolute_path = 'D:/Python/★★Python_POSTECH_AI/Postech_AI 4) Aritificial_Intelligent/교재_실습_자료/'


# 2. Data
# data load
iris = pd.read_csv(absolute_path + "Iris.csv")
iris.shape
iris.head()

iris_id_drop = iris.drop(['Id'],axis=1)


# data preprocessing
X = iris.iloc[:, 1:5].values
X.shape
X[:10]

# 2.4 Data_Labeling
temp = iris['Species'].values
temp[:10]

y = pd.Categorical(iris['Species']).codes
y
Species = y.reshape(-1,1)
Species.shape


# 2.5 Data_Combined
data = np.hstack([X, Species])
data[:10]

df = pd.DataFrame(data, columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm","Species"])


# 3. Select Variables
# 3.1 Select Variables_Correlation
iris_id_drop.corr()
sns.pairplot(iris_id_drop, hue='Species')

correlation_matrix = df.corr()
correlation_matrix



# 3.1.1 Correlation_Visualization
c_matrix = np.array(correlation_matrix)
column_name = list(correlation_matrix.columns)
plt.figure(figsize=(15,15))
plt.title('Correlation between data variables', fontsize = 20)
plt.imshow(correlation_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.xticks(np.arange(len(column_name)), column_name, rotation=45)
plt.yticks(np.arange(len(column_name)), column_name)
for y in range(c_matrix.shape[0]):
    for x in range(c_matrix.shape[1]):
        plt.text(x, y, '%.4f' % c_matrix[y, x],
                 horizontalalignment='center',
                 verticalalignment='center',fontsize = 20
                 )
plt.grid(False)
plt.show()



import seaborn as sns
plt.figure(figsize=(15,15))
plt.title('Correlation between data variables', fontsize = 20)
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt = '.2f', linewidths=.5, cmap='Blues')
heatmap.set_ylim(0,5)



# 3.2 Select_Variables_Combined
val_SepalWidthCm = df['SepalWidthCm'].values
val_SepalLengthCm = df['SepalLengthCm'].values
val_PetalWidthCm = df['PetalWidthCm'].values

val_SepalWidthCm = val_SepalWidthCm.reshape(-1,1)
val_SepalLengthCm = val_SepalLengthCm.reshape(-1,1)
val_PetalWidthCm = val_PetalWidthCm.reshape(-1,1)

data_values = np.hstack([val_SepalWidthCm, val_SepalLengthCm, val_PetalWidthCm])

data_values[:10]




# 4. Clustering_Algorithm -----------------------------------------------------------
from sklearn.cluster import KMeans

cost = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(data_values)
    cost.append(abs(kmeans.score(data_values)))

plt.figure(figsize=(15,10))
plt.plot(range(1,11), cost)
plt.title('The K-means Clustering cost', fontsize = 20)
plt.xlabel('number of clusters', fontsize = 20)
plt.ylabel('cost', fontsize = 20)
plt.show()



# 4.1.2 K-means_Clustering
kmeansmodel = KMeans(n_clusters= 3, init='k-means++', random_state=0)
y_kmeans= kmeansmodel.fit_predict(data_values)

real_values = df["Species"].astype('int')

print("Real Values : ")
real_unique_class , real_unique_class_counts = np.unique(real_values, return_counts=True)
for x , y in zip(real_unique_class, real_unique_class_counts):
    print("The number of observations assigned class",x,"is",y)


print("Result of kmeans : ")
kmeans_unique_class , kmeans_unique_class_counts = np.unique(y_kmeans, return_counts=True)
for x , y in zip(kmeans_unique_class, kmeans_unique_class_counts):
    print("The number of observations assigned cluster",x,"is",y)


# 4.2 K-means_Visualization
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib inline

# 4.2.1 Visualization_Real
    # Real-Data Visulization
plt.style.use('default')
fig=plt.figure(figsize=(15,10))
ax=Axes3D(fig)
ax.scatter(data_values[real_values == 0, 0], data_values[real_values == 0, 1], data_values[real_values == 0, 2], s = 30, c = 'green', label = 'Setosa')
ax.scatter(data_values[real_values == 1, 0], data_values[real_values == 1, 1], data_values[real_values == 1, 2], s = 30, c = 'blue', label = 'Versicolor')
ax.scatter(data_values[real_values == 2, 0], data_values[real_values == 2, 1], data_values[real_values == 2, 2], s = 30, c = 'red', label = 'Virginica')
ax.set_title('Real Value', size = 25)
ax.set_xlabel('Sepal_Width  (cm)', size = 13)
ax.set_ylabel('Sepal_Length (cm)', size = 13)
ax.set_zlabel('Petal_Width  (cm)', size = 13)
plt.legend(fontsize=15)
plt.show()



# 4.2.2 Visualization_K-means
    # Prediction-Data Visulization
plt.style.use('default')
fig=plt.figure(figsize=(15,10))
ax=Axes3D(fig)
ax.scatter(data_values[y_kmeans == 0, 0], data_values[y_kmeans == 0, 1], data_values[y_kmeans == 0, 2], s = 30, c = 'cyan', label = 'Cluster 0')
ax.scatter(data_values[y_kmeans == 1, 0], data_values[y_kmeans == 1, 1], data_values[y_kmeans == 1, 2], s = 30, c = 'magenta', label = 'Cluster 1')
ax.scatter(data_values[y_kmeans == 2, 0], data_values[y_kmeans == 2, 1], data_values[y_kmeans == 2, 2], s = 30, c = 'black', label = 'Cluster 2')
ax.set_title('K-means Clustering', size = 25)
ax.set_xlabel('Sepal_Width  (cm)', size = 13)
ax.set_ylabel('Sepal_Length (cm)', size = 13)
ax.set_zlabel('Petal_Width  (cm)', size = 13)
plt.legend(fontsize=15)
plt.show()


# 4.2.3 Visualization_K-means with centroid
kmeansmodel.cluster_centers_

plt.style.use('default')
fig=plt.figure(figsize=(15,10))
ax=Axes3D(fig)
ax.scatter(data_values[y_kmeans == 0, 0], data_values[y_kmeans == 0, 1], data_values[y_kmeans == 0, 2], s = 30, c = 'cyan', label = 'Cluster 0')
ax.scatter(data_values[y_kmeans == 1, 0], data_values[y_kmeans == 1, 1], data_values[y_kmeans == 1, 2], s = 30, c = 'magenta', label = 'Cluster 1')
ax.scatter(data_values[y_kmeans == 2, 0], data_values[y_kmeans == 2, 1], data_values[y_kmeans == 2, 2], s = 30, c = 'black', label = 'Cluster 2')
ax.scatter(kmeansmodel.cluster_centers_[0:3, 0], kmeans.cluster_centers_[0:3, 1], kmeans.cluster_centers_[0:3, 2], s = 100, c = 'red', label = 'Centroid')
ax.set_title('K-means Clustering with Centroid', size = 25)
ax.set_xlabel('Sepal_Width  (cm)', size = 13)
ax.set_ylabel('Sepal_Length (cm)', size = 13)
ax.set_zlabel('Petal_Width  (cm)', size = 13)
plt.legend(fontsize=15)
plt.show()




# 4.3 K-means_Accuracy
temp = iris['Species'].values
real_species = temp.reshape(-1,1)

kmeans_species = []
for sp in y_kmeans:
    if sp == 0:
        sp = 'Iris-setosa'
        kmeans_species.append(sp)
    elif sp == 2:
        sp = 'Iris-versicolor'
        kmeans_species.append(sp)
    else:
        sp = 'Iris-virginica'
        kmeans_species.append(sp)

kmeans_species
real_species.shape

kmeans_species = np.array(kmeans_species)
kmeans_species = kmeans_species.reshape(-1,1)
kmeans_species.shape


i = 0
j = 0
for real, predict in zip(real_species,kmeans_species):
    print(real, predict)
    if real == predict:
        i += 1
    j += 1


accuracy = np.round((i/j)*100,1)
print('K-means Total accuracy :{}'.format(accuracy),'%')



# 4.3.3 Accuracy_Case by Case
i = 0
setosa_num = 0
j = 0
versicolor_num = 0
k = 0
virginica_num = 0

for real, predict in zip(real_species,kmeans_species):
    if real == 'Iris-setosa':
        setosa_num += 1
        if real == predict:
            i += 1
    elif real == 'Iris-versicolor':
        versicolor_num += 1
        if real == predict:
            j += 1
    else:
        virginica_num += 1
        if real == predict:
            k += 1


ac_setosa = np.round((i/setosa_num)*100,1)
print('K-means setosa accuracy :{}'.format(ac_setosa),'%')

ac_virginica = np.round((k/virginica_num)*100,1)
print('K-means virginica accuracy :{}'.format(ac_virginica),'%')

ac_versicolor = np.round((j/versicolor_num)*100,1)
print('K-means versicolor accuracy :{}'.format(ac_versicolor),'%')

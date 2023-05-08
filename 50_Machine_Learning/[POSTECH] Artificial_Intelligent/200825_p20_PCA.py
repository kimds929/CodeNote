# Principal Component Analysis (PCA)


import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Postech_AI 4) Aritificial_Intelligent/교재_실습_자료/'

# data generation
m = 5000
mu = np.array([0, 0])
sigma = np.array([[3, 1.5], 
                  [1.5, 1]])

X = np.random.multivariate_normal(mu, sigma, m) # 다차원에서 normalize 분포를 그려주는 함수
X = np.asmatrix(X)

fig = plt.figure(figsize = (10, 8))
plt.plot(X[:,0], X[:,1], 'k.', alpha = 0.3)
plt.axis('equal')
plt.grid(alpha = 0.3)
plt.show()

# PCA Linalgbra
S = 1/(m-1) * X.T * X
 
D, U = np.linalg.eig(S)

idx = np.argsort(-D)
D = D[idx]
U = U[:,idx]

print(D, '\n')
print(U)


# PCA결과
h = U[1,0]/U[0,0]
xp = np.arange(-6, 6, 0.1)
yp = h*xp

fig = plt.figure(figsize = (10, 8))
plt.plot(X[:,0], X[:,1], 'k.', alpha = 0.3)
plt.plot(xp, yp, 'r', linewidth = 3)
plt.axis('equal')
plt.grid(alpha = 0.3)
plt.show()




# Histogram
plt.figure(figsize = (10, 8))
plt.hist(X*U[:,0], 51, alpha=0.5, label='PC1')
plt.hist(X*U[:,1], 51, alpha=0.5, label='PC2')
plt.legend()
plt.show()




# sklearn
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
pca.fit(X)
pca.transform(X)
dir(pca)
pca.explained_variance_         # 분산
pca.explained_variance_ratio_   # 분산비



u = pca.transform(X)

plt.figure(figsize = (10, 8))
plt.hist(u, 51)
plt.show()










from six.moves import cPickle
X = cPickle.load(open(absolute_path + '/pca_spring.pkl','rb'))
X = np.asmatrix(X.T)
X.shape

print(X.shape)
m = X.shape[0]

# plotting
plt.figure(figsize = (12, 6))
plt.subplot(1,3,1)
plt.plot(X[:, 0], -X[:, 1], 'r')
plt.axis('equal')
plt.title('Camera 1')

plt.subplot(1,3,2)
plt.plot(X[:, 2], -X[:, 3], 'b')
plt.axis('equal')
plt.title('Camera 2')

plt.subplot(1,3,3)
plt.plot(X[:, 4], -X[:, 5], 'k')
plt.axis('equal')
plt.title('Camera 3')

plt.show()



# PCA Linalgbra
X = X - np.mean(X, axis = 0)

S = 1/(m-1)*X.T*X
 
D, U = np.linalg.eig(S)

idx = np.argsort(-D)
D = D[idx]
U = U[:,idx]

print(D, '\n')
print(U)


plt.figure(figsize = (10,8))
plt.stem(np.sqrt(D))
plt.grid(alpha = 0.3)
plt.show()




# relative magnitutes of the principal components

Z = X*U
m = len(Z)
xp = np.arange(0, m)/24    # 24 frame rate

plt.figure(figsize = (10, 8))
plt.plot(xp, Z)
plt.yticks([])
plt.show()


## projected onto the first principal component
# 6 dim -> 1 dim (dim reduction)
# relative magnitute of the first principal component

Z = X*U[:,0]

plt.figure(figsize = (10, 8))
plt.plot(Z)
plt.yticks([])
plt.show()











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



# KNN Regressor --------------------------------------------------------------------------
from sklearn import neighbors

# Dataset
N = 100
w1 = 0.5
w0 = 2
x = np.random.normal(0, 15, N).reshape(-1,1)
y = w1*x + w0 + 5*np.random.normal(0, 1, N).reshape(-1,1)

plt.figure(figsize = (10, 8))
plt.title('Data Set', fontsize = 15)
plt.plot(x, y, '.', label = 'Data')
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.legend(fontsize = 15)
plt.axis('equal')
plt.axis([-40, 40, -30, 30])
plt.grid(alpha = 0.3)
plt.show()



# single neighbors # neighbors = 1
# model learning
reg = neighbors.KNeighborsRegressor(n_neighbors = 1)
reg.fit(x, y)

# prediction
x_new = np.array([[5]])
pred = reg.predict(x_new)[0,0] 
print(pred)

# plotting
xp = np.linspace(-30, 30, 100).reshape(-1, 1)
yp = reg.predict(xp)

plt.figure(figsize = (10, 8))
plt.title('k-Nearest Neighbor Regression', fontsize = 15)

plt.plot(x, y, '.', label = 'Original Data')
plt.plot(xp, yp, label = 'kNN')
plt.plot(x_new, pred, 'o', label = 'Prediction')

plt.plot([x_new[0,0], x_new[0,0]], [-30, pred], 'k--', alpha = 0.5)
plt.plot([-40, x_new[0,0]], [pred, pred], 'k--', alpha = 0.5)

plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.legend(fontsize = 12)
plt.axis('equal')
plt.axis([-40, 40, -30, 30])
plt.grid(alpha = 0.3)
plt.show()




# big-size neighbors
n = 15

# model learning
reg = neighbors.KNeighborsRegressor(n_neighbors = n)
reg.fit(x, y)

# plotting
xp = np.linspace(-30, 30, 100).reshape(-1, 1)
yp = reg.predict(xp)

plt.figure(figsize = (10, 8))
plt.title('k-Nearest Neighbor Regression', fontsize=15)

plt.plot(x, y, '.', label='Original Data')
plt.plot(xp, yp, label = 'Regression Result')
plt.plot(x_new, pred, 'o', label='Prediction')

plt.plot([x_new[0,0], x_new[0,0]], [-30, pred], 'k--', alpha=0.5)
plt.plot([-40, x_new[0,0]], [pred, pred], 'k--', alpha=0.5)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.legend(fontsize = 12)
plt.axis('equal')
plt.axis([-40, 40, -30, 30])
plt.grid(alpha = 0.3)
plt.show()











# KNN Classification ---------------------------------------------------------------------------------

from sklearn import neighbors

# Dataset
m = 1000
X = -1.5 + 3*np.random.uniform(size = (m,2))

y = np.zeros([m,1])
for i in range(m):
    if np.linalg.norm(X[i,:], 2) <= 1:
        y[i] = 1      

C1 = np.where(y == 1)[0]
C0 = np.where(y == 0)[0]

theta = np.linspace(0, 2*np.pi, 100)

plt.figure(figsize = (8,8))
plt.plot(X[C1,0], X[C1,1], 'o', label = 'C1', markerfacecolor = "k", markeredgecolor = 'k', markersize = 4)
plt.plot(X[C0,0], X[C0,1], 'o', label = 'C0', markerfacecolor = "None", alpha = 0.3, markeredgecolor = 'k', markersize = 4)
plt.plot(np.cos(theta), np.sin(theta), '--', color = 'orange')
plt.axis([-1.5, 1.5, -1.5, 1.5])
plt.axis('equal')
plt.axis('off')
plt.show()



# single neighbors # neighbors = 1  ****
# model learning
clf = neighbors.KNeighborsClassifier(n_neighbors = 1)
clf.fit(X, np.ravel(y))

# predict
X_new = np.array([1, 1]).reshape(1,-1)
result = clf.predict(X_new)[0]
print(result)


# plotting
res = 0.01
[X1gr, X2gr] = np.meshgrid(np.arange(-1.5,1.5,res), np.arange(-1.5,1.5,res))

Xp = np.hstack([X1gr.reshape(-1,1), X2gr.reshape(-1,1)])
Xp = np.asmatrix(Xp)

inC1 = clf.predict(Xp).reshape(-1,1)
inCircle = np.where(inC1 == 1)[0]

plt.figure(figsize = (8, 8))
plt.plot(X[C1,0], X[C1,1], 'o', label = 'C1', markerfacecolor = "k", alpha = 0.5, markeredgecolor = 'k', markersize = 4)
plt.plot(X[C0,0], X[C0,1], 'o', label = 'C0', markerfacecolor = "None", alpha = 0.3, markeredgecolor='k', markersize = 4)
plt.plot(np.cos(theta), np.sin(theta), '--', color = 'orange')
plt.plot(Xp[inCircle][:,0], Xp[inCircle][:,1], 's', alpha = 0.5, color = 'r', markersize = 1)
plt.axis([-1.5, 1.5, -1.5, 1.5])
plt.axis('equal')
plt.axis('off')
plt.show()   






# When outliers exist  ---------------------------------------------
m = 1000
X = -1.5 + 3*np.random.uniform(size = (m,2))

y = np.zeros([m,1])
for i in range(m):
    if np.linalg.norm(X[i,:], 2) <= 1:
        if np.random.uniform() < 0.05:
            y[i] = 0
        else:
            y[i] = 1    
    else:
        if np.random.uniform() < 0.05:
            y[i] = 1
        else:
            y[i] = 0

C1 = np.where(y == 1)[0]
C0 = np.where(y == 0)[0]

theta = np.linspace(0, 2*np.pi, 100)

plt.figure(figsize = (8,8))
plt.plot(X[C1,0], X[C1,1], 'o', label = 'C1', markerfacecolor = "k", markeredgecolor = 'k', markersize = 4)
plt.plot(X[C0,0], X[C0,1], 'o', label = 'C0', markerfacecolor = "None", alpha = 0.3, markeredgecolor = 'k', markersize = 4)
plt.plot(np.cos(theta), np.sin(theta), '--', color = 'orange')
plt.axis([-1.5, 1.5, -1.5, 1.5])
plt.axis('equal')
plt.axis('off')
plt.show()



# single neighbor :  k = 1   ****
clf = neighbors.KNeighborsClassifier(n_neighbors = 1)
clf.fit(X, np.ravel(y))

res = 0.01
[X1gr, X2gr] = np.meshgrid(np.arange(-1.5,1.5,res), np.arange(-1.5,1.5,res))

Xp = np.hstack([X1gr.reshape(-1,1), X2gr.reshape(-1,1)])
Xp = np.asmatrix(Xp)

inC1 = clf.predict(Xp).reshape(-1,1)
inCircle = np.where(inC1 == 1)[0]

plt.figure(figsize = (8, 8))
plt.plot(X[C1,0], X[C1,1], 'o', label = 'C1', markerfacecolor = "k", alpha = 0.5, markeredgecolor = 'k', markersize = 4)
plt.plot(X[C0,0], X[C0,1], 'o', label = 'C0', markerfacecolor = "None", alpha = 0.3, markeredgecolor='k', markersize = 4)
plt.plot(np.cos(theta), np.sin(theta), '--', color = 'orange')
plt.plot(Xp[inCircle][:,0], Xp[inCircle][:,1], 's', alpha = 0.5, color = 'r', markersize = 1)
plt.axis([-1.5, 1.5, -1.5, 1.5])
plt.axis('equal')
plt.axis('off')
plt.show()   




# big-size neighbor :  k > 1   ****
n = 100
clf = neighbors.KNeighborsClassifier(n_neighbors = n)
clf.fit(X, np.ravel(y))

res = 0.01
[X1gr, X2gr] = np.meshgrid(np.arange(-1.5,1.5,res), np.arange(-1.5,1.5,res))

Xp = np.hstack([X1gr.reshape(-1,1), X2gr.reshape(-1,1)])
Xp = np.asmatrix(Xp)

inC1 = clf.predict(Xp).reshape(-1,1)
inCircle = np.where(inC1 == 1)[0]

plt.figure(figsize = (8, 8))
plt.plot(Xp[inCircle][:,0], Xp[inCircle][:,1], 's', alpha = 0.5, color = 'r', markersize = 1)
plt.plot(X[C1,0], X[C1,1], 'o', label = 'C1', markerfacecolor = "k", alpha = 0.5, markeredgecolor = 'k', markersize = 4)
plt.plot(X[C0,0], X[C0,1], 'o', label = 'C0', markerfacecolor = "None", alpha = 0.3, markeredgecolor='k', markersize = 4)
plt.plot(np.cos(theta), np.sin(theta), '--', color = 'orange')
# plt.legend(fontsize = 12)
plt.axis([-1.5, 1.5, -1.5, 1.5])
plt.axis('equal')
plt.axis('off')
plt.show()   





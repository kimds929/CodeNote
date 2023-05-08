import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

import sys
sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import *
from DS_OLS import *

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Postech_AI 4) Aritificial_Intelligent/교재_실습_자료/'


# 초기 Data생성
m = 100
x1 = 8 * np.random.rand(m,1)    # 0~ 1 사이의 m × 1 행렬 생성
x2 = 7 * np.random.rand(m,1) - 4

g = 0.8 *x1 + x2 - 3        # classification line을 생성(초기에 안다고 가정)

C1 = np.where(g>=1)     # [만족하는경우 index, 만족하지 않는경우 index]
C0 = np.where(g<-1)


C1 = C1[0]
C0 = C0[0]


plt.figure(figsize=(10,8))
plt.plot(x1[C1], x2[C1], 'ro', label='C1')
plt.plot(x1[C0], x2[C0], 'bo', label='C0')

plt.title('Linearly Separable Classes', fontsize = 15)
plt.legend(loc = 1, fontsize = 15)
plt.xlabel(r'$x_1$', fontsize = 15)
plt.ylabel(r'$x_2$', fontsize = 15)
plt.show()


# 
X1 = np.hstack([np.ones([C1.shape[0],1]), x1[C1], x2[C1]])
X0 = np.hstack([np.ones([C0.shape[0],1]), x1[C0], x2[C0]])

print(X1.shape, X0.shape)

X = np.vstack([X1, X0])
y = np.vstack([np.ones([C1.shape[0],1]), -np.ones([C0.shape[0], 1])])
print(X.shape, y.shape)

X = np.asmatrix(X)
y = np.asmatrix(y)


# a = np.array([5,2,3]).reshape(-1,1)
# np.where(a >2)



w = np.ones([3,1])
w = np.asmatrix(w)

n_iter = 20

# w값 Update
for k in range(n_iter):
    for i in range(y.shape[0]):
        if y[i,0] != np.sign(X[i,:] @ w)[0, 0]:
            w += y[i,0] * X[i,:].T
print(w)


x1p = np.linspace(0, 8, 100).reshape(-1,1)
x2p = -w[1,0]/w[2,0] * x1p - w[0,0]/w[2,0]

plt.figure(figsize=(10,8))

plt.plot(x1[C1], x2[C1], 'ro', label='C1')
plt.plot(x1[C0], x2[C0], 'bo', label='C0')
plt.plot(x1p, x2p, c='k', label='perceptron')

plt.xlim([0, 8])
plt.xlabel('$x_1$', fontsize = 15)
plt.ylabel('$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 15)
plt.show()








from sklearn import linear_model

# intercept False
X1 = np.hstack([np.ones([C1.shape[0],1]), x1[C1], x2[C1]])
X0 = np.hstack([np.ones([C0.shape[0],1]), x1[C0], x2[C0]])
X = np.vstack([X1, X0])

# Perceptron(
#     *,
#     penalty=None,
#     alpha=0.0001,
#     fit_intercept=True,
#     max_iter=1000,
#     tol=0.001,
#     shuffle=True,
#     verbose=0,
#     eta0=1.0,
#     n_jobs=None,
#     random_state=0,
#     early_stopping=False,
#     validation_fraction=0.1,
#     n_iter_no_change=5,
#     class_weight=None,
#     warm_start=False,
# )

clf = linear_model.Perceptron(tol=1e-3, fit_intercept=False)
clf.fit(X,y)
clf.coef_
clf.intercept_

w0 = clf.coef_[0,0]
w1 = clf.coef_[0,1]
w2 = clf.coef_[0,2]

x1p = np.linspace(0,8,100).reshape(-1,1)
x2p = -w1/w2 * x1p - w0/w2

plt.figure(figsize=(10,8))
plt.plot(x1[C1], x2[C1], 'ro', label='C1')
plt.plot(x1[C0], x2[C0], 'bo', label='C0')
plt.plot(x1p, x2p, c='k', label='perceptron')

plt.xlim([0, 8])
plt.xlabel('$x_1$', fontsize = 15)
plt.ylabel('$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 15)




# intercept auto

X1 = np.hstack([x1[C1], x2[C1]])
X0 = np.hstack([x1[C0], x2[C0]])
X = np.vstack([X1, X0])
clf = linear_model.Perceptron(tol=1e-3, fit_intercept=True)
clf.fit(X, y)

w0 = clf.intercept_
w1 = clf.coef_[0,0]
w2 = clf.coef_[0,1]

x1p = np.linspace(0,8,100).reshape(-1,1)
x2p = -w1/w2 * x1p - w0/w2


plt.figure(figsize=(10,8))
plt.plot(x1[C1], x2[C1], 'ro', label='C1')
plt.plot(x1[C0], x2[C0], 'bo', label='C0')
plt.plot(x1p, x2p, c='k', label='perceptron')

plt.xlim([0, 8])
plt.xlabel('$x_1$', fontsize = 15)
plt.ylabel('$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 15)














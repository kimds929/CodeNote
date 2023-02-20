import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

import sys
sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import *
from DS_OLS import *

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Postech_AI 4) Aritificial_Intelligent/교재_실습_자료/'


# 3. Illustrative Example ----------------------------------------------------------------

# training data gerneration --------------------------------------
x1 = 8*np.random.rand(100,1)
x2 = 7*np.random.rand(100,1) - 4

g = 0.8 * x1 + x2 -3
g1 = g -1
g0 = g + 1

C1 = np.where(g1>=0)[0]
C0 = np.where(g0<0)[0]

xp = np.linspace(-1, 9, 100).reshape(-1,1)
ypt = -0.8 * xp + 3


plt.figure(figsize=(10,8))
plt.plot(x1[C1], x2[C1], 'ro', label='C1')
plt.plot(x1[C0], x2[C0], 'bo', label='C0')
plt.plot(xp, ypt, 'k', label='True')

plt.title('Linearly and Strictly Separable Classes', fontsize = 15)
plt.xlabel(r'$x_1$', fontsize = 15)
plt.ylabel(r'$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 12)
plt.axis('equal')
plt.xlim([0, 8])
plt.ylim([-4, 3])


# support vector display
plt.figure(figsize=(10, 8))
plt.plot(x1[C1], x2[C1], 'ro', alpha = 0.4, label = 'C1')
plt.plot(x1[C0], x2[C0], 'bo', alpha = 0.4, label = 'C0')
plt.plot(xp, ypt, 'k', linewidth = 3, label = 'True')
plt.plot(xp, ypt-1, '--k')
plt.plot(xp, ypt+1, '--k')
plt.title('Linearly and Strictly Separable Classes', fontsize = 15)
plt.xlabel(r'$x_1$', fontsize = 15)
plt.ylabel(r'$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 12)
plt.axis('equal')
plt.xlim([0, 8])
plt.ylim([-4, 3])
plt.show()




# 3.1. The First Attempt -----------------------------------------------------
# CVXPY using simple classification --------------------------------------
import cvxpy as cvx

X1 = np.hstack([x1[C1], x2[C1]])
X0 = np.hstack([x1[C0], x2[C0]])


X1 = np.asmatrix(X1)
X0 = np.asmatrix(X0)

N = X1.shape[0]
M = X0.shape[0]


# support margin이 1인 경우에서의 decision boundary
# constraint:  ω0+X1ω≥1, ω0+X0ω≤−1

w0 = cvx.Variable([1,1])
w = cvx.Variable([2,1])


obj = cvx.Minimize(1)
const = [w0 + X1 @ w >= 1, w0 + X0 @ w <= -1]
prob = cvx.Problem(obj, const).solve()

w0 = w0.value
w = w.value

xp = np.linspace(-1,9,100).reshape(-1,1)
yp = - w[0,0]/w[1,0]*xp - w0/w[1,0]

plt.figure(figsize=(10, 8))
plt.plot(x1[C1], x2[C1], 'ro', alpha = 0.4, label = 'C1')
plt.plot(x1[C0], x2[C0], 'bo', alpha = 0.4, label = 'C0')
plt.plot(xp, ypt, 'k', alpha = 0.3, label = 'True')
plt.plot(xp, ypt-1, '--k', alpha = 0.3)
plt.plot(xp, ypt+1, '--k', alpha = 0.3)
plt.plot(xp, yp, 'g', linewidth = 3, label = 'Attempt 1')
plt.title('Linearly and Strictly Separable Classes', fontsize = 15)
plt.xlabel(r'$x_1$', fontsize = 15)
plt.ylabel(r'$x_2$', fontsize = 15)
plt.legend(loc = 1)
plt.axis('equal')
plt.xlim([0, 8])
plt.ylim([-4, 3])
plt.show()





# constraint:  X1ω≥1,   X0ω≤−1
w = cvx.Variable([3,1])

obj = cvx.Minimize(1)
const = [X1*w >= 1, X0*w <= -1]
prob = cvx.Problem(obj, const).solve()

w = w.value

xp = np.linspace(-1,9,100).reshape(-1,1)
yp = - w[1,0]/w[2,0]*xp - w[0,0]/w[2,0]

plt.figure(figsize=(10, 8))
plt.plot(x1[C1], x2[C1], 'ro', alpha = 0.4, label = 'C1')
plt.plot(x1[C0], x2[C0], 'bo', alpha = 0.4, label = 'C0')
plt.plot(xp, ypt, 'k', alpha = 0.3, label = 'True')
plt.plot(xp, ypt-1, '--k', alpha = 0.3)
plt.plot(xp, ypt+1, '--k', alpha = 0.3)
plt.plot(xp, yp, 'g', linewidth = 3, label = 'Attempt 1')
plt.title('Linearly and Strictly Separable Classes', fontsize = 15)
plt.xlabel(r'$x_1$', fontsize = 15)
plt.ylabel(r'$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 12)
plt.axis('equal')
plt.xlim([0, 8])
plt.ylim([-4, 3])
plt.show()



# 3.2. Outlier -----------------------------------------------------------
# Note that in the real world, you may have noise, errors, or outliers that do not accurately represent the actual phenomena
# Non-separable case
# No solutions (hyperplane) exist
# We will allow some training examples to be misclassified !
# But we want their number to be minimized


# X1 = np.hstack([np.ones([N,1]), x1[C1], x2[C1]])
# X0 = np.hstack([np.ones([M,1]), x1[C0], x2[C0]])
X1 = np.hstack([np.ones([x1[C1].shape[0],1]), x1[C1], x2[C1]])
X0 = np.hstack([np.ones([x1[C0].shape[0],1]), x1[C0], x2[C0]])

outlier = np.array([1, 2, 2]).reshape(1,-1)
X0 = np.vstack([X0, outlier])

X1 = np.asmatrix(X1)
X0 = np.asmatrix(X0)

plt.figure(figsize=(10,8))
plt.plot(X1[:,1], X1[:,2], 'ro', alpha = 0.4, label = 'C1')
plt.plot(X0[:,1], X0[:,2], 'bo', alpha = 0.4, label = 'C0')
plt.title('When Outliers Exist', fontsize = 15)
plt.xlabel(r'$x_1$', fontsize = 15)
plt.ylabel(r'$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 12)
plt.axis('equal')
plt.xlim([0, 8])
plt.ylim([-4, 3])
plt.show()



w = cvx.Variable([3,1])

obj = cvx.Minimize(1)
const = [X1*w >= 1, X0*w <= -1]
prob = cvx.Problem(obj, const).solve()

print(w.value)



# 3.3. The Second Attempt ---------------------------------------------------


# N = x1[C1].shape[0]
# M = x1[C0].shape[0]
# X1 = np.hstack([np.ones([N,1]), x1[C1], x2[C1]])
# X0 = np.hstack([np.ones([M,1]), x1[C0], x2[C0]])

X1 = np.hstack([np.ones([x1[C1].shape[0],1]), x1[C1], x2[C1]])
X0 = np.hstack([np.ones([x1[C0].shape[0],1]), x1[C0], x2[C0]])

outlier = np.array([1, 2, 2]).reshape(1,-1)
X0 = np.vstack([X0, outlier])

X1 = np.asmatrix(X1)
X0 = np.asmatrix(X0)

N = X1.shape[0]
M = X0.shape[0]


w = cvx.Variable([3,1])
u = cvx.Variable([N,1])
v = cvx.Variable([M,1])

obj = cvx.Minimize(np.ones((1,N))*u + np.ones((1,M))*v)
const = [X1*w >= 1-u, X0*w <= -(1-v), u >= 0, v >= 0 ]
prob = cvx.Problem(obj, const).solve()

w = w.value



xp = np.linspace(-1,9,100).reshape(-1,1)
yp = - w[1,0]/w[2,0]*xp - w[0,0]/w[2,0]

plt.figure(figsize=(10, 8))
plt.plot(X1[:,1], X1[:,2], 'ro', alpha = 0.4, label = 'C1')
plt.plot(X0[:,1], X0[:,2], 'bo', alpha = 0.4, label = 'C0')
plt.plot(xp, ypt, 'k', alpha = 0.1, label = 'True')
plt.plot(xp, ypt-1, '--k', alpha = 0.1)
plt.plot(xp, ypt+1, '--k', alpha = 0.1)
plt.plot(xp, yp, 'g', linewidth = 3, label = 'Attempt 2')
plt.plot(xp, yp-1/w[2,0], '--g')
plt.plot(xp, yp+1/w[2,0], '--g')
plt.title('When Outliers Exist', fontsize = 15)
plt.xlabel(r'$x_1$', fontsize = 15)
plt.ylabel(r'$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 12)
plt.axis('equal')
plt.xlim([0, 8])
plt.ylim([-4, 3])
plt.show()






# 3.4. Maximize Margin (Finally, it is Support Vector Machine) -------------------------
import time
from IPython.display import clear_output
# clear_output(wait=True)

time.sleep(2)


g = 2
w = cvx.Variable([3,1])
u = cvx.Variable([N,1])
v = cvx.Variable([M,1])

obj = cvx.Minimize(cvx.norm(w,2) + g*(np.ones((1,N))*u + np.ones((1,M))*v))
const = [X1*w >= 1-u, X0*w <= -(1-v), u >= 0, v >= 0 ]
prob = cvx.Problem(obj, const).solve()

w = w.value



xp = np.linspace(-1,9,100).reshape(-1,1)
yp = - w[1,0]/w[2,0]*xp - w[0,0]/w[2,0]

plt.figure(figsize=(10, 8))
plt.plot(X1[:,1], X1[:,2], 'ro', alpha = 0.4, label = 'C1')
plt.plot(X0[:,1], X0[:,2], 'bo', alpha = 0.4, label = 'C0')
plt.plot(xp, ypt, 'k', alpha = 0.1, label = 'True')
plt.plot(xp, ypt-1, '--k', alpha = 0.1)
plt.plot(xp, ypt+1, '--k', alpha = 0.1)
plt.plot(xp, yp, 'g', linewidth = 3, label = 'SVM')
plt.plot(xp, yp-1/w[2,0], '--g')
plt.plot(xp, yp+1/w[2,0], '--g')

plt.title('When Outliers Exist (g :' + str(g) + ')', fontsize = 15)
plt.xlabel(r'$x_1$', fontsize = 15)
plt.ylabel(r'$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 12)
plt.axis('equal')
plt.xlim([0, 8])
plt.ylim([-4, 3])
plt.show()



# g값에 따른 margin
for g in [0.01, 0.05, 0.1, 0.5, 1, 2, 3, 5, 10]:
    w = cvx.Variable([3,1])
    u = cvx.Variable([N,1])
    v = cvx.Variable([M,1])

    obj = cvx.Minimize(cvx.norm(w,2) + g*(np.ones((1,N))*u + np.ones((1,M))*v))
    const = [X1*w >= 1-u, X0*w <= -(1-v), u >= 0, v >= 0 ]
    prob = cvx.Problem(obj, const).solve()

    w = w.value



    xp = np.linspace(-1,9,100).reshape(-1,1)
    yp = - w[1,0]/w[2,0]*xp - w[0,0]/w[2,0]

    plt.figure(figsize=(10, 8))
    plt.plot(X1[:,1], X1[:,2], 'ro', alpha = 0.4, label = 'C1')
    plt.plot(X0[:,1], X0[:,2], 'bo', alpha = 0.4, label = 'C0')
    plt.plot(xp, ypt, 'k', alpha = 0.1, label = 'True')
    plt.plot(xp, ypt-1, '--k', alpha = 0.1)
    plt.plot(xp, ypt+1, '--k', alpha = 0.1)
    plt.plot(xp, yp, 'g', linewidth = 3, label = 'SVM')
    plt.plot(xp, yp-1/w[2,0], '--g')
    plt.plot(xp, yp+1/w[2,0], '--g')

    plt.title('When Outliers Exist (g :' + str(g) + ')', fontsize = 15)
    plt.xlabel(r'$x_1$', fontsize = 15)
    plt.ylabel(r'$x_2$', fontsize = 15)
    plt.legend(loc = 1, fontsize = 12)
    plt.axis('equal')
    plt.xlim([0, 8])
    plt.ylim([-4, 3])
    plt.show()

    time.sleep(1)
    clear_output(wait=True)










# 5. Nonlinear Support Vector Machine ---------------------------------------
# https://www.youtube.com/embed/3liCbRZPrZA?rel=0
X1 = np.array([[-1.1,0],[-0.3,0.1],[-0.9,1],[0.8,0.4],[0.4,0.9],[0.3,-0.6],
               [-0.5,0.3],[-0.8,0.6],[-0.5,-0.5]])
     
X0 = np.array([[-1,-1.3], [-1.6,2.2],[0.9,-0.7],[1.6,0.5],[1.8,-1.1],[1.6,1.6],
               [-1.6,-1.7],[-1.4,1.8],[1.6,-0.9],[0,-1.6],[0.3,1.7],[-1.6,0],[-2.1,0.2]])

X1 = np.asmatrix(X1)
X0 = np.asmatrix(X0)

plt.figure(figsize=(10, 8))
plt.plot(X1[:,0], X1[:,1], 'ro', label = 'C1')
plt.plot(X0[:,0], X0[:,1], 'bo', label = 'C0')
plt.title('SVM for Nonlinear Data', fontsize = 15)
plt.xlabel(r'$x_1$', fontsize = 15)
plt.ylabel(r'$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 12)
plt.axis('equal')
plt.show()





# Kernel Display -------------------------------------------------------
N = X1.shape[0]
M = X0.shape[0]

X = np.vstack([X1, X0])
y = np.vstack([np.ones([N,1]), -np.ones([M,1])])

X = np.asmatrix(X)
y = np.asmatrix(y)

m = N + M
Z = np.hstack([np.ones([m,1]), np.square(X[:,0]), np.sqrt(2)*np.multiply(X[:,0],X[:,1]), np.square(X[:,1])])

g = 10

w = cvx.Variable([4, 1])
d = cvx.Variable([m, 1])

obj = cvx.Minimize(cvx.norm(w, 2) + g*np.ones([1,m])*d)
const = [cvx.multiply(y, Z*w) >= 1-d, d>=0]
prob = cvx.Problem(obj, const).solve()

w = w.value
print(w)

# to plot
[X1gr, X2gr] = np.meshgrid(np.arange(-3,3,0.1), np.arange(-3,3,0.1))

Xp = np.hstack([X1gr.reshape(-1,1), X2gr.reshape(-1,1)])
Xp = np.asmatrix(Xp)

m = Xp.shape[0]
Zp = np.hstack([np.ones([m,1]), np.square(Xp[:,0]), np.sqrt(2)*np.multiply(Xp[:,0],Xp[:,1]), np.square(Xp[:,1])])
q = Zp*w

B = []
for i in range(m):
    if q[i,0] > 0:
        B.append(Xp[i,:])       

B = np.vstack(B)

plt.figure(figsize=(10, 8))
plt.plot(X1[:,0], X1[:,1], 'ro', label = 'C1')
plt.plot(X0[:,0], X0[:,1], 'bo', label = 'C0')
plt.plot(B[:,0], B[:,1], 'gs', markersize = 10, alpha = 0.1, label = 'SVM')

plt.title('SVM with Kernel', fontsize = 15)
plt.xlabel(r'$x_1$', fontsize = 15)
plt.ylabel(r'$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 12)
plt.axis('equal')
plt.show()

q.shape





# g값에 따른 Kernel Margin
# g = 10


for g in [ 0.005, 0.01, 0.07, 0.1, 0.2, 0.3, 0.5, 1, 5, 10]:
    N = X1.shape[0]
    M = X0.shape[0]
    m = N + M

    w = cvx.Variable([4, 1])
    d = cvx.Variable([m, 1])

    obj = cvx.Minimize(cvx.norm(w, 2) + g*np.ones([1,m])*d)
    const = [cvx.multiply(y, Z*w) >= 1-d, d>=0]
    prob = cvx.Problem(obj, const).solve()

    w = w.value

    # to plot
    [X1gr, X2gr] = np.meshgrid(np.arange(-3,3,0.1), np.arange(-3,3,0.1))

    Xp = np.hstack([X1gr.reshape(-1,1), X2gr.reshape(-1,1)])
    Xp = np.asmatrix(Xp)

    m = Xp.shape[0]
    Zp = np.hstack([np.ones([m,1]), np.square(Xp[:,0]), np.sqrt(2)*np.multiply(Xp[:,0],Xp[:,1]), np.square(Xp[:,1])])
    q = Zp*w

    B = []
    for i in range(m):
        if q[i,0] > 0:
            B.append(Xp[i,:])       

    B = np.vstack(B)

    plt.figure(figsize=(10, 8))
    plt.plot(X1[:,0], X1[:,1], 'ro', label = 'C1')
    plt.plot(X0[:,0], X0[:,1], 'bo', label = 'C0')
    plt.plot(B[:,0], B[:,1], 'gs', markersize = 10, alpha = 0.1, label = 'SVM')

    plt.title('SVM with Kernel (g : ' + str(g) + ')', fontsize = 15)
    plt.xlabel(r'$x_1$', fontsize = 15)
    plt.ylabel(r'$x_2$', fontsize = 15)
    plt.legend(loc = 1, fontsize = 12)
    plt.axis('equal')
    plt.show()

    time.sleep(1.5)
    clear_output(wait=True)
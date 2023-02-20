import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cvxpy as cvx



# data points in column vector [input, output]
x = np.array([0.1, 0.4, 0.7, 1.2, 1.3, 1.7, 2.2, 2.8, 3.0, 4.0, 4.3, 4.4, 4.9]).reshape(-1, 1)
y = np.array([0.5, 0.9, 1.1, 1.5, 1.5, 2.0, 2.2, 2.8, 2.7, 3.0, 3.5, 3.7, 3.9]).reshape(-1, 1)

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'bo')
plt.title('Data', fontsize = 15)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.axis('equal')
plt.grid(alpha = 0.3)
plt.xlim([0, 5])
plt.show()


A = np.hstack([x**0, x])
# A = np.hstack([np.ones((x.shape[0],1)), x])

#  θ = np.linalg.inv(A.T @ A) @ A.T @ y
A = np.asmatrix(A)
theta = (A.T @ A).I @ A.T @ y
theta


# to plot
plt.figure(figsize = (10, 8))
plt.title('Regression', fontsize = 15)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.plot(x, y, 'ko', label = "data")


# to plot a straight line (fitted line)
xp = np.arange(0, 5, 0.01).reshape(-1,1)
# yp = np.hstack([xp**0,xp]) @ theta
yp = theta[0, 0] + theta[1, 0] * xp


plt.plot(xp, yp, 'r', linewidth = 2, label = "regression")
plt.legend(fontsize = 15)
plt.axis('equal')
plt.grid(alpha = 0.3)
plt.xlim([0, 5])
plt.show()



# Use Gradient Descent -----------------------------------------------------
alpha = 0.001
theta_grad = np.random.randn(2,1)    # 초기값 설정

for _ in range(1000):
    df = 2 * (A.T @ A @ theta_grad - A.T @ y)
    theta_grad = theta_grad - alpha * df
theta_grad       # result

# to plot
plt.figure(figsize = (10, 8))
plt.title('Regression', fontsize = 15)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.plot(x, y, 'ko', label = "data")

# to plot a straight line (fitted line)
xp = np.arange(0, 5, 0.01).reshape(-1,1)
# yp = np.hstack([xp**0,xp]) @ theta_grad
yp = theta_grad[0, 0] + theta_grad[1, 0] * xp

plt.plot(xp, yp, 'r', linewidth = 2, label = "regression")
plt.legend(fontsize = 15)
plt.axis('equal')
plt.grid(alpha = 0.3)
plt.xlim([0, 5])
plt.show()




# cvxpy 활용 ------------------------------------------------------------------------
# L2 Norm
thetaL2 = cvx.Variable(shape=(2,1))
objL2 = cvx.Minimize(cvx.norm(A @ thetaL2 - y, 2))
probL2 = cvx.Problem(objL2, [])
# probL2 = cvx.Problem(objL2)
resultL2 = probL2.solve()

print(f'optimized_L2: {thetaL2.value}')
print(f'optimized_L2: {resultL2}')


# L1 Norm
thetaL1 = cvx.Variable(shape=(2,1))
objL1 = cvx.Minimize(cvx.norm(A @ thetaL1 - y, 1))
probL1 = cvx.Problem(objL1, [])
# probL1 = cvx.Problem(objL1)
resultL1 = probL1.solve()

print(f'optimized_L1: {thetaL1.value}')
print(f'optimized_L1: {resultL1}')




# to plot
plt.figure(figsize = (10, 8))
plt.title('Regression', fontsize = 15)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.plot(x, y, 'ko', label = "data")

# to plot a straight line (fitted line)
xp = np.arange(0, 5, 0.01).reshape(-1,1)
# yp_L1 = np.hstack([xp**0,xp]) @ thetaL1.value
yp_L1 = thetaL1.value[0, 0] + thetaL1.value[1, 0] * xp

# yp_L2 = np.hstack([xp**0,xp]) @ thetaL2.value
yp_L2 = thetaL2.value[0, 0] + thetaL2.value[1, 0] * xp

plt.plot(xp, yp_L1, 'b', linewidth = 2, label = "cvx_L1")
plt.plot(xp, yp_L2, 'r', linewidth = 2, label = "cvx_L2")
plt.legend(fontsize = 15)
plt.axis('equal')
plt.grid(alpha = 0.3)
plt.xlim([0, 5])
plt.show()








# add outliers -------------------------------------------------------------------
x_out = np.vstack([x, np.array([0.5, 3.8]).reshape(-1, 1)])
y_out = np.vstack([y, np.array([3.9, 0.3]).reshape(-1, 1)])

A_out = np.hstack([x_out**0, x_out])
print(f'{A.shape} → {A_out.shape})')

plt.figure(figsize = (10, 8))
plt.plot(x_out, y_out, 'ko', label = 'data')
plt.axis('equal')
plt.xlim([0, 5])
plt.grid(alpha = 0.3)
plt.show()


theta_out_L1 = cvx.Variable(shape=(2,1))
obj_out_L1 = cvx.Minimize(cvx.norm( A_out @ theta_out_L1 - y_out, 1))
prob_out_L1 = cvx.Problem(obj_out_L1)
result_out_L1 = prob_out_L1.solve()

theta_out_L2 = cvx.Variable(shape=(2,1))
obj_out_L2 = cvx.Minimize(cvx.norm( A_out @ theta_out_L2 - y_out, 2))
prob_out_L2 = cvx.Problem(obj_out_L2)
result_out_L2 = prob_out_L2.solve()



# to plot straight lines (fitted lines)
plt.figure(figsize = (10, 8))
plt.plot(x_out, y_out, 'ko', label = 'data')

xp_out = np.arange(0, 5, 0.01).reshape(-1,1)
yp_out_L1 = theta_out_L1.value[0, 0] + theta_out_L1.value[1, 0] * xp_out
yp_out_L2 = theta_out_L2.value[0, 0] + theta_out_L2.value[1, 0] * xp_out

plt.plot(xp_out, yp_out_L1, 'b', linewidth = 2, label = '$L_1$')
plt.plot(xp_out, yp_out_L2, 'r', linewidth = 2, label = '$L_2$')
plt.axis('equal')
plt.xlim([0, 5])
plt.legend(fontsize = 15, loc = 5)
plt.grid(alpha = 0.3)
plt.show()  







# ----- Scikit-Learn ----------------------------------------------------------------------

from sklearn import linear_model
# x
# y
reg = linear_model.LinearRegression()
reg.fit(x,y)
reg.coef_[0,0], reg.intercept_[0]


xp_reg = np.arange(0, 5, 0.01).reshape(-1,1) 

# to plot
plt.figure(figsize = (10, 8))
plt.title('Regression', fontsize = 15)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.plot(x, y, 'ko', label = "data")

# to plot a straight line (fitted line)
plt.plot(xp_reg, reg.predict(xp), 'r', linewidth = 2, label = "regression")
plt.legend(fontsize = 15)
plt.axis('equal')
plt.grid(alpha = 0.3)
plt.xlim([0, 5])
plt.show()










# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# 2. Multivariate Linear Regression (다항 회귀분석)

from mpl_toolkits.mplot3d import Axes3D     # for 3D plot


# y = theta0 + theta1*x1 + theta2*x2 + noise

n = 200
x1 = np.random.randn(n, 1)
x2 = np.random.randn(n, 1)
noise = 0.5*np.random.randn(n, 1);

y = 2 + 1*x1 + 3*x2 + noise
# y = 2 + 1*x1 + 3*x2

fig = plt.figure(figsize = (10, 8))
ax = fig.add_subplot(1, 1, 1, projection = '3d')
ax.set_title('Generated Data', fontsize = 15)
ax.set_xlabel('$X_1$', fontsize = 15)
ax.set_ylabel('$X_2$', fontsize = 15)
ax.set_zlabel('Y', fontsize = 15)
ax.scatter(x1, x2, y, marker = 'o', label = 'Data')
#ax.view_init(30,30)
plt.legend(fontsize = 15)
plt.show()




# % matplotlib qt5
A_3d = np.hstack([x1**0, x1, x2])
A_3d = np.asmatrix(A_3d)
theta = (A_3d.T @ A_3d).I @ A_3d.T @ y

X1, X2 = np.meshgrid(np.arange(np.min(x1), np.max(x1), 0.5), 
                     np.arange(np.min(x2), np.max(x2), 0.5))
YP = theta[0,0] + theta[1,0]*X1 + theta[2,0]*X2

fig = plt.figure(figsize = (10, 8))
ax = fig.add_subplot(1, 1, 1, projection = '3d')
ax.set_title('Regression', fontsize = 15)
ax.set_xlabel('$X_1$', fontsize = 15)
ax.set_ylabel('$X_2$', fontsize = 15)
ax.set_zlabel('Y', fontsize = 15)
ax.scatter(x1, x2, y, marker = '.', label = 'Data')
ax.plot_wireframe(X1, X2, YP, color = 'k', alpha = 0.3, label = 'Regression Plane')
# ax.view_init(30,30)
plt.legend(fontsize = 15)
plt.show()










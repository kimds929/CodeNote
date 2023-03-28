
# 3. Nonlinear Regression ----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

n = 100            
x = -5 + 15*np.random.rand(n, 1)
noise = 10*np.random.randn(n, 1)

y = 10 + 1*x + 2*x**2 + noise

plt.figure(figsize = (10, 8))
plt.title('True x and y', fontsize = 15)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.plot(x, y, 'o', markersize = 4, label = 'actual')
plt.xlim([np.min(x), np.max(x)])
plt.grid(alpha = 0.3)
plt.legend(fontsize = 15)
plt.show()



A = np.hstack([x**0, x, x**2])
A = np.asmatrix(A)

theta = (A.T @ A).I @ A.T @ y
print('theta:\n', theta)



xp = np.linspace(np.min(x), np.max(x))
yp = theta[0,0] + theta[1,0] * xp + theta[2,0] * xp **2

plt.figure(figsize = (10, 8))
plt.plot(x, y, 'o', markersize = 4, label = 'actual')
plt.plot(xp, yp, 'r', linewidth = 2, label = 'estimated')

plt.title('Nonlinear regression', fontsize = 15)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.xlim([np.min(x), np.max(x)])
plt.grid(alpha = 0.3)
plt.legend(fontsize = 15)
plt.show()




# Linear Basis Function Models -------------------------------------------------
xp = np.arange(-1, 1, 0.01).reshape(-1,1)

# A = np.hstack([x**0, x, x**2])
poly_degree = 6
polybasis = np.hstack([xp ** i for i in range(poly_degree)])
polybasis.shape


plt.figure(figsize = (10, 8))

for i in range(poly_degree):
    plt.plot(xp, polybasis[:,i], label = '$x^{}$'.format(i))
    
plt.title('Polynomial Basis', fontsize = 15)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.axis([-1, 1, -1.1, 1.1])
plt.grid(alpha = 0.3)
plt.legend(fontsize = 15)
plt.show()







# RBF Function ----------------------------------------------------------------------
# bi(x)=exp(− (∥x−μi∥**2 / (2*σ2) ))
d = 9                       # RBF 갯수
u = np.linspace(-1, 1, d)   # RBF 범위
sigma = 0.2                 # RBF Sigma

rbfbasis = np.hstack([np.exp(-(xp - u[i]) ** 2 / (2 * sigma **2 )) for i in range(d)])


plt.figure(figsize = (10, 8))

for i in range(d):
    plt.plot(xp, rbfbasis[:,i], label='$\mu = {}$'.format(u[i]))
    
plt.title('RBF basis', fontsize = 15)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.axis([-1, 1, -0.1, 1.1])
plt.legend(loc = 'lower right', fontsize = 15)
plt.grid(alpha = 0.3)
plt.show()







#   Nonlinear Regression with Polynomial Functions
xp = np.arange(np.min(x), np.max(x), 0.01).reshape(-1, 1)

d = 3
polybasis = np.hstack([xp**i for i in range(d)])
A =  np.hstack([x**i for i in range(d)])
A = np.asmatrix(A)

theta = (A.T @ A).I @ A.T @ y
yp = polybasis * theta

plt.figure(figsize = (10, 8))
plt.plot(x, y, 'o', label = 'Data')
plt.plot(xp, yp, label = 'Polynomial')
plt.title('Regression with Polynomial basis', fontsize = 15)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.grid(alpha = 0.3)
plt.legend(fontsize = 15)
plt.show()



# 4.2. Nonlinear Regression with RBF Functions ****
xp = np.arange(np.min(x), np.max(x), 0.01).reshape(-1, 1)

d = 6
u = np.linspace(np.min(x), np.max(x), d)
sigma = 4

rbfbasis = np.hstack([np.exp(-(xp - u[i]) ** 2 / (2 * sigma **2 )) for i in range(d)])
A = np.hstack([np.exp(-(x - u[i]) ** 2 / (2 * sigma **2 )) for i in range(d)])
A = np.asmatrix(A)
print(f'x_shape: {x.shape} / rbf_shape: {A.shape}') #  각각의 rbf 마다 x값을 매칭시켜 여러개의 rbf를 생성

theta = (A.T @ A).I @ A.T @ y
yp = rbfbasis * theta


plt.figure(figsize = (10, 8))
plt.plot(x, y, 'o', label = 'Data')
plt.plot(xp, yp, label = 'RBF')
plt.title('Regression with RBF basis', fontsize = 15)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.grid(alpha = 0.3)
plt.legend(fontsize = 15)
plt.show()



# over-fitting ----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# 10 data points
n = 10
x = np.linspace(-4.5, 4.5, 10).reshape(-1, 1)
y = np.array([0.9819, 0.7973, 1.9737, 0.1838, 1.3180, -0.8361, -0.6591, -2.4701, -2.8122, -6.2512]).reshape(-1, 1)

plt.figure(figsize = (10, 8))
plt.plot(x, y, 'o', label = 'Data')
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.grid(alpha = 0.3)
plt.show()


A = np.hstack([x**0, x])
A = np.asmatrix(A)
theta = (A.T @ A).I @ A.T @ y
print(theta)


# to plot
xp = np.arange(-4.5, 4.5, 0.01).reshape(-1, 1)
yp = theta[0,0] + theta[1,0] * xp

plt.figure(figsize = (10, 8))
plt.plot(x, y, 'o', label = 'Data')
plt.plot(xp[:,0], yp[:,0], linewidth = 2, label = 'Linear')
plt.title('Linear Regression', fontsize = 15)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.legend(fontsize = 15)
plt.grid(alpha = 0.3)
plt.show()



# 2차식 근사 ----------------------------------------------------------

A = np.hstack([x**0, x, x**2])
A = np.asmatrix(A)
theta = (A.T @ A).I @ A.T @ y
print(theta)


# to plot
xp = np.arange(-4.5, 4.5, 0.01).reshape(-1, 1)
yp = theta[0,0] + theta[1,0] * xp + theta[2,0] * xp ** 2

plt.figure(figsize = (10, 8))
plt.plot(x, y, 'o', label = 'Data')
plt.plot(xp[:,0], yp[:,0], linewidth = 2, label = '2nd degree')
plt.title('Nonlinear Regression with Polynomial Functions', fontsize = 15)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.legend(fontsize = 15)
plt.grid(alpha = 0.3)
plt.show()  




# n차식 근사 ----------------------------------------------------------
d = 10
A = np.hstack([x**i for i in range(d)])
A = np.asmatrix(A)

theta = (A.T @ A).I @ A.T @ y
print(theta)

# to plot
xp = np.arange(-4.5, 4.5, 0.01).reshape(-1, 1)
polybasis = np.hstack([xp**i for i in range(d)])
yp = polybasis @ theta

plt.figure(figsize = (10, 8))
plt.plot(x, y, 'o', label = 'Data')
plt.plot(xp[:,0], yp[:,0], linewidth = 2, label = '9th degree')
plt.title('Nonlinear Regression with Polynomial Functions', fontsize = 15)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.legend(fontsize = 15)
plt.grid(alpha = 0.3)
plt.show()



# 1, 3, 5, 9 차식 근사 ----------------------------------------------------------
d = [1, 3, 5, 9]
RSS = []

plt.figure(figsize = (12, 10))
plt.suptitle('Nonlinear Regression', fontsize = 15)

for k in range(4):
    A = np.hstack([x**i for i in range(d[k]+1)])
    polybasis = np.hstack([xp**i for i in range(d[k]+1)])
    
    A = np.asmatrix(A)
    polybasis = np.asmatrix(polybasis) 
    
    theta = (A.T @ A).I @ A.T @ y
    yp = polybasis @ theta
    
    RSS.append( np.linalg.norm(A @ theta - y) ** 2 )
    
    plt.subplot(2, 2, k+1)
    plt.plot(x, y, 'o')
    plt.plot(xp, yp)
    plt.axis([-5, 5, -12, 6])
    plt.title('degree = {}'.format(d[k]))
    plt.grid(alpha=0.3)
    
plt.show()
print(RSS)

plt.figure(figsize = (10, 8))
plt.stem(d, RSS, label = 'RSS')
plt.title('Residual Sum of Squares', fontsize = 15)
plt.xlabel('degree', fontsize = 15)
plt.ylabel('RSS', fontsize = 15)
plt.legend(fontsize = 15)
plt.grid(alpha = 0.3)
plt.show()







# 2. Overfitting with RBF Functions ----------------------------------------------------------
# xp = np.arange(-4.5, 4.5, 0.01).reshape(-1, 1)

# d = 10
# u = np.linspace(-4.5, 4.5, d)
# sigma = 0.2

# A = 
# rbfbasis = 

# A = np.asmatrix(A)
# rbfbasis = np.asmatrix(rbfbasis)

# theta = 
# yp = 

# plt.figure(figsize = (10, 8))
# plt.plot(x, y, 'o', label = 'Data')
# plt.plot(xp, yp, label = 'Overfitted')
# plt.title('(Overfitted) Regression with RBF basis', fontsize = 15)
# plt.xlabel('X', fontsize = 15)
# plt.ylabel('Y', fontsize = 15)
# plt.grid(alpha = 0.3)
# plt.legend(fontsize = 15)
# plt.show()








# d = [2, 4, 6, 10]
# sigma = 1

# plt.figure(figsize = (12, 10))

# for k in range(4):
#     u = np.linspace(-4.5, 4.5, d[k])
    
#     A = 
#     rbfbasis = 
    
#     A = np.asmatrix(A)
#     rbfbasis = np.asmatrix(rbfbasis)
    
#     theta = 
#     yp = 
    
#     plt.subplot(2, 2, k+1)
#     plt.plot(x, y, 'o')
#     plt.plot(xp, yp)
#     plt.axis([-5, 5, -12, 6])
#     plt.title('num RBFs = {}'.format(d[k]), fontsize = 10)
#     plt.grid(alpha = 0.3)

# plt.suptitle('Nonlinear Regression with RBF Functions', fontsize = 15)
# plt.show()








# 3. Regularization (Shrinkage Methods) ----------------------------------------------------------
# CVXPY code
import cvxpy as cvx


# Regression -------------------------------------
d = 10
u = np.linspace(-4.5, 4.5, d)

sigma = 1

A = np.hstack([np.exp(-(x-u[i])**2/(2*sigma**2)) for i in range(d)])
rbfbasis = np.hstack([np.exp(-(xp-u[i])**2/(2*sigma**2)) for i in range(d)])

A = np.asmatrix(A)
rbfbasis = np.asmatrix(rbfbasis)
    
theta = cvx.Variable([d, 1])
obj = cvx.Minimize(cvx.sum_squares(A @ theta - y))
prob = cvx.Problem(obj)
prob.solve()

yp = rbfbasis @ theta.value

plt.figure(figsize = (10, 8))
plt.plot(x, y, 'o', label = 'Data')
plt.plot(xp, yp, label = 'Overfitted')
plt.title('(Overfitted) Regression', fontsize = 15)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.axis([-5, 5, -12, 6])
plt.legend(fontsize = 15)
plt.grid(alpha = 0.3)
plt.show()

# Variable Coefficient
plt.figure(figsize = (10, 8))
plt.title(r'LinearRegression: magnitude of $\theta$', fontsize = 15)
plt.xlabel(r'$\theta$', fontsize = 15)
plt.ylabel('magnitude', fontsize = 15)
plt.stem(np.linspace(1, 10, 10).reshape(-1, 1), theta.value)
plt.xlim([0.5, 10.5])
# plt.ylim([-10, 10])
plt.grid(alpha = 0.3)
plt.show()




# Ridge Regression -------------------------------------
lamb = 0.1
theta = cvx.Variable([d, 1])
obj = cvx.Minimize(cvx.sum_squares(A @ theta - y) + lamb * cvx.sum_squares(theta))
prob = cvx.Problem(obj)
prob.solve()

yp = rbfbasis @ theta.value

plt.figure(figsize = (10, 8))
plt.plot(x, y, 'o', label = 'Data')
plt.plot(xp, yp, label = 'Ridge')
plt.title('Ridge Regularization (L2)', fontsize = 15)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.axis([-5, 5, -12, 6])
plt.legend(fontsize = 15)
plt.grid(alpha = 0.3)
plt.show()

# Regulization (= ridge nonlinear regression) encourages small weights, but not exactly 0
plt.figure(figsize = (10, 8))
plt.title(r'Ridge: magnitude of $\theta$', fontsize = 15)
plt.xlabel(r'$\theta$', fontsize = 15)
plt.ylabel('magnitude', fontsize = 15)
plt.stem(np.linspace(1, 10, 10).reshape(-1, 1), theta.value)
plt.xlim([0.5, 10.5])
plt.ylim([-5, 5])
plt.grid(alpha = 0.3)
plt.show()


# (Ridge) lambda 값에 따른 변수 축소
lamb = np.arange(0,3,0.01)

theta_record = []
for k in lamb:
    theta = cvx.Variable([d, 1])
    obj = cvx.Minimize(cvx.sum_squares(A*theta - y) + k * cvx.sum_squares(theta))
    prob = cvx.Problem(obj).solve()
    theta_record.append(np.ravel(theta.value))

plt.figure(figsize = (10, 8))
plt.plot(lamb, theta_record, linewidth = 1)
plt.title('Ridge coefficients as a function of regularization', fontsize = 15)
plt.xlabel('$\lambda$', fontsize = 15)
plt.ylabel(r'weight $\theta$', fontsize = 15)
plt.show()






# Lasso Regression -------------------------------------
lamb = 2
theta = cvx.Variable([d, 1])
obj = cvx.Minimize(cvx.sum_squares(A @ theta - y) + lamb * cvx.norm(theta,1))
prob = cvx.Problem(obj)
prob.solve()

yp = rbfbasis @ theta.value

# LASSO regression 
plt.figure(figsize = (10, 8))
plt.title('LASSO Regularization (L1)', fontsize = 15)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.plot(x, y, 'o', label = 'Data')
plt.plot(xp, yp, label = 'LASSO')
plt.axis([-5, 5, -12, 6])
plt.legend(fontsize = 15)
plt.grid(alpha = 0.3)
plt.show()


# Regulization (= Lasso nonlinear regression) encourages zero weights
plt.figure(figsize = (10, 8))
plt.title(r'LASSO: magnitude of $\theta$', fontsize = 15)
plt.xlabel(r'$\theta$', fontsize = 15)
plt.ylabel('magnitude', fontsize = 15)
plt.stem(np.arange(1,11), theta.value)
plt.xlim([0.5, 10.5])
plt.ylim([-5,1])
plt.grid(alpha = 0.3)
plt.show()


# (Lasso) lambda 값에 따른 변수 축소
lamb = np.arange(0,3,0.01)

theta_record = []
for k in lamb:
    theta = cvx.Variable([d, 1])
    obj = cvx.Minimize(cvx.sum_squares(A @ theta - y) + k*cvx.norm(theta, 1))
    prob = cvx.Problem(obj).solve()
    theta_record.append(np.ravel(theta.value))

plt.figure(figsize = (10, 8))
plt.plot(lamb, theta_record, linewidth = 1)
plt.title('LASSO coefficients as a function of regularization', fontsize = 15)
plt.xlabel('$\lambda$', fontsize = 15)
plt.ylabel(r'weight $\theta$', fontsize = 15)
plt.show()













# RBF Model ----------------------------------------------------------------------
# reduced order model 
# we will use only theta 2, 3, 8, 10 

d = 4
u = np.array([-3.5, -2.5, 2.5, 4.5])

sigma = 1

A = np.hstack([np.exp(-(x-u[i])**2/(2*sigma**2)) for i in range(d)])
rbfbasis = np.hstack([np.exp(-(xp-u[i])**2/(2*sigma**2)) for i in range(d)])

A = np.asmatrix(A)
rbfbasis = np.asmatrix(rbfbasis)
    
theta = cvx.Variable([d, 1])
obj = cvx.Minimize(cvx.norm(A @ theta - y, 2))
prob = cvx.Problem(obj).solve()

yp = rbfbasis * theta.value

plt.figure(figsize = (10, 8))
plt.plot(x, y, 'o', label = 'Data')
plt.plot(xp, yp, label = 'Overfitted')
plt.title('(Overfitted) Regression', fontsize = 15)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.axis([-5, 5, -12, 6])
plt.legend(fontsize = 15)
plt.grid(alpha = 0.3)
plt.show()



# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

# 5. L2 and L1 Regularizers     -------------------------------------------------------------------

# 5.1. L2 Penality (Ridge)
x = np.arange(-4,4,0.1)
k = 4
y = (x-1)**2 + 1/6*(x-1)**3 + k*x**2
x_star = x[np.argmin(y)]
print(x_star)

plt.plot(x, y, 'g', linewidth = 2.5)
plt.axvline(x = x_star, color = 'k', linewidth = 1, linestyle = '--')
plt.ylim([0,10])
plt.show()



for k in [0,1,2,4]:
    y = (x-1)**2 + 1/6*(x-1)**3 + k*x**2
    x_star = x[np.argmin(y)]
    
    plt.plot(x,y, 'g', linewidth = 2.5)
    plt.axvline(x = x_star, color = 'k', linewidth = 1, linestyle = '--')
    plt.ylim([0,10])
    plt.title('Ridge: k = {}'.format(k))
    plt.show()





# 5.2. L1 Penalty (Lasso)   -------------------------------------------------------------------
x = np.arange(-4,4,0.1)
k = 2
y = (x-1)**2 + 1/6*(x-1)**3 + k*abs(x)

x_star = x[np.argmin(y)]
print(x_star)

plt.plot(x, y, 'g', linewidth = 2.5)
plt.axvline(x = x_star, color = 'k', linewidth = 1, linestyle = '--')
plt.ylim([0,10])
plt.show()


for k in [0,1,2]:
    y = (x-1)**2 + 1/6*(x-1)**3 + k*abs(x)
    x_star = x[np.argmin(y)]
    
    plt.plot(x,y, 'g', linewidth = 2.5)
    plt.axvline(x = x_star, color = 'k', linewidth = 1, linestyle = '--')
    plt.ylim([0,10])
    plt.title('LASSO: k = {}'.format(k))
    plt.show()
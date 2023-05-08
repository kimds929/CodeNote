import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

########################################################################
# Gaussian Process Concept_1
def func(x):
    return 10*x + 5

def variance(x):
    return x**2

def variance_func(x):
    return x**2*np.random.randn(len(x))

def gpr(x):
    return func(x) + variance_func(x)


xs = np.arange(-5, 5.2, 0.001)
plt.scatter(xs, gpr(xs), alpha=0.1, s=5)
plt.plot(xs, func(xs)+variance(xs), color='coral', alpha=0.5)
plt.plot(xs, func(xs)-variance(xs), color='coral', alpha=0.5)
plt.plot(xs, func(xs), color='orange')



# Gaussian Process Concept_2

# xs = (-5:0.2:5); ns = size(xs,1); keps = 1e-9;
# m = inline('0.25*x.^2');
# K = inline('exp(-0.5*(repmat(p'',size(q))-repmat(q,size(p''))).^2)');
# fs = m(xs) + chol(K(xs,xs)+keps*eye(ns))*randn(ns,1);
# plot(xs,fs,'.')

import numpy as np
import matplotlib.pyplot as plt

xs = np.arange(-5, 5.2, 0.5)
# xs = np.arange(-5, 5.0001, 0.)


def m(xi):
    return 0.25 * xi**2

def K(xi, xj):
    return np.exp(-1/2 * (xi.reshape(-1,1) - xj.reshape(-1,1).T)**2) 

def gaussian_random_function(xi, keps=1e-9, random_state=None):
    rng = np.random.RandomState(random_state)
    ns = len(xs)
    cov = K(xi, xi)

    # eigen decomposition
    # val, vec = np.linalg.eig(cov + keps * np.eye(ns))
    # fs = m(xi).ravel() + vec @ np.sqrt(np.diag(val)) @ rng.randn(ns)

    # cholesky decomposition
    L = np.linalg.cholesky(cov + keps * np.eye(ns))
    fs = m(xi).ravel() + L @ rng.randn(ns)
    
    return fs

kernel = K(xs, xs) 
variance = np.sqrt(np.diag(kernel))


plt.figure(figsize=(10,2))
for i in range(100):
    plt.plot(xs, gaussian_random_function(xs, random_state=i), alpha=0.03, color='steelblue')
    # plt.plot(xs, np.random.multivariate_normal(mean=m(xs), cov=K(xs, xs)), alpha=0.03, color='steelblue')
plt.plot(xs, m(xs) + 1.96*variance, color='coral',alpha=0.2)
plt.plot(xs, m(xs) - 1.96*variance, color='coral',alpha=0.2)
plt.plot(xs, m(xs), color='orange')
plt.show()





########################################################################s


# np.set_printoptions(suppress=True)
# np.set_printoptions(suppress=False)

def RBF_kernel(X1, X2, l=None, sigma_f=None):
    """ 
    :param X1: numpy array, (n_samples_1, n_features) 
    :param X2: numpy array, (n_samples_2, n_features) 
    :param l: float, length parameter 
    :param sigma_f: float, scaling parameter 
    :return: numpy array, (n_samples_1, n_samples_2) 
    """ 
    l = 1 if l is None else l
    sigma_f = (2*np.pi)**(-1/4) if sigma_f is None else sigma_f     # (default) 0.63161877

    n_samples_1, n_features = X1.shape 
    n_samples_2, n_features = X2.shape 
    
    K = np.zeros((n_samples_1, n_samples_2)) 
    
    for i in range(n_samples_1): 
        for j in range(n_samples_2): 
            diff = X1[i] - X2[j] 
            K[i, j] = sigma_f ** 2 * np.exp(-1 / (2 * l**2) * np.dot(diff.T, diff))
            # K(x1, x2) = σf² · exp( 1/(2σl²) ·(x1 - x2).T@(x1 - x2) )
    return K 

def polynomial_kernel(X1, X2, order=1, sigma_f=None):
    sigma_f = 1 if sigma_f is None else sigma_f     # (default) 1

    polynomial_X1 = np.ones((len(X1),order+1))
    polynomial_X2 = np.ones((len(X2),order+1))
    for i in range(order):
        polynomial_X1[:,[-(i+2)]] = X1**(i+1)
        polynomial_X2[:,[-(i+2)]] = X2**(i+1)
    # return polynomial_X1, polynomial_X2
    return sigma_f * (polynomial_X1 @ polynomial_X2.T)



# def polynomial_kernel(X1, X2, order=1, sigma_f=None):
#     sigma_f = 1 if sigma_f is None else sigma_f     # (default) 1
#     return sigma_f * (X1 @ X2.T)





########################################################################
# import scipy.stats as stats
# norm = stats.norm(0,1)
import numpy as np
# dataset
X_truth = np.linspace(-4, 4, 101).reshape(-1,1)
y_truth = np.sin(X_truth)
idx = np.arange(101)

rng.choice(idx, )

rng = np.random.RandomState(10)
X_all = X_truth.copy()
y_all = y_truth.copy()

# random noise 추가
noise = True
if noise is True:
    noise_amp = 0.3
    # noise = rng.randn(len(y_all)).reshape(-1,1) * noise_amp
    noise = rng.normal(0, noise_amp, size=len(y_all)).reshape(-1,1)
    y_all = y_all + noise


n_sample = 5
train_idx = rng.choice(idx, n_sample)

X = X_all[train_idx]
y = y_all[train_idx]

X_train = X.reshape(-1,1)
y_train = y.reshape(-1,1)

print(X_train.shape, y_train.shape)
########################################################################
# https://greeksharifa.github.io/bayesian_statistics/2020/07/12/Gaussian-Process/
# https://aistory4u.tistory.com/entry/%EA%B0%80%EC%9A%B0%EC%8B%9C%EC%95%88-%ED%94%84%EB%A1%9C%EC%84%B8%EC%8A%A4-%ED%9A%8C%EA%B7%80
# https://pasus.tistory.com/209
# https://october25kim.github.io/paper/kernel%20method/2020/10/31/gpr-paper/


X_data = X_all.copy()


# Linear Kernel ***
x2_order = 1
x2_sigma_f = 1
x2_kernel = polynomial_kernel(X_train, X_train, order=x2_order, sigma_f=x2_sigma_f)
w2 = np.linalg.pinv(x2_kernel)@y_train

x2_train_cov = x2_kernel.copy()
x2_traintest_cov = polynomial_kernel(X_data, X_train, order=x2_order, sigma_f=x2_sigma_f)
x2_test_cov = polynomial_kernel(X_data, X_data, order=x2_order, sigma_f=x2_sigma_f)
x2_cov_mat = x2_test_cov - (x2_traintest_cov @ np.linalg.pinv(x2_train_cov) @ x2_traintest_cov.T) + 1e-10
x2_std = np.sqrt(np.diag(x2_cov_mat).reshape(-1,1))


# Quadratic Kernel ***
x3_order = 2
x3_sigma_f = 1
x3_kernel = polynomial_kernel(X_train, X_train, order=x3_order, sigma_f=x3_sigma_f)
w3 = np.linalg.pinv(x3_kernel)@y_train

x3_train_cov = x3_kernel.copy()
x3_traintest_cov = polynomial_kernel(X_data, X_train, order=x3_order, sigma_f=x3_sigma_f)
x3_test_cov = polynomial_kernel(X_data, X_data, order=x3_order, sigma_f=x3_sigma_f)
x3_cov_mat = x3_test_cov - (x3_traintest_cov @ np.linalg.pinv(x3_train_cov) @ x3_traintest_cov.T) + 1e-10
x3_std = np.sqrt(np.diag(x3_cov_mat).reshape(-1,1))




# RBF Kernel ***
l=None
sigma_f = None
x4_kernel = RBF_kernel(X_train, X_train, l=l, sigma_f=sigma_f)

# w4 = np.linalg.pinv(x4_kernel.T@x4_kernel)@x4_kernel.T@y_train
w4 = np.linalg.pinv(x4_kernel)@y_train
# Gaussian process에서 training data 끼리의 RBF (Radial Basis Function) kernel function은 positive definite한 함수이므로, 역행렬이 존재합니다. 
# 하지만 "singular matrix" 에러가 발생한다면, 이는 역행렬을 계산할 때 반올림 오차(round-off error)로 인해 발생할 수 있는 문제입니다.
train_cov = x4_kernel.copy()
traintest_cov = RBF_kernel(X_data, X_train, l=l, sigma_f=sigma_f)
test_cov = RBF_kernel(X_data, X_data, l=l, sigma_f=sigma_f)
cov_mat = test_cov - (traintest_cov @ np.linalg.pinv(train_cov) @ traintest_cov.T) + 1e-10
standard_deviation = np.sqrt(np.diag(cov_mat).reshape(-1,1))


# (sklearn gaussian process) ------------------------------------------------------------------------
# from sklearn.gaussian_process import GaussianProcessRegressor
# model_gpr = GaussianProcessRegressor()
# # from sklearn.gaussian_process.kernels import RBF
# # kernel_RBF = 1 * RBF(length_scale=10, length_scale_bounds=(1e-2, 1e2))
# # model_gpr= GaussianProcessRegressor(kernel=kernel_RBF)

# model_gpr.fit(X_train.reshape(-1,1), y_train)
# preds, stds = model_gpr.predict(X_data, return_std=True)
# ---------------------------------------------------------------------------------------------------




y_data = y_all.copy()



f = plt.figure(figsize=(8,4))


# Observation
plt.scatter(X_train, y_train, color='red')

# Linear_Kernel
plt.plot(X_data, x2_traintest_cov@w2, alpha=0.5, color='steelblue', label='linear_kernel')

# Quadratic_Kernel
plt.plot(X_data, x3_traintest_cov@w3, alpha=0.5, color='orange', label='quadratic_kernel')

# RBF_Kernel
plt.plot(X_data, traintest_cov@w4, color='purple', alpha=0.5, label='rbf_kernel')
plt.fill_between(X_data.ravel()
                ,( traintest_cov@w4 - 1.96*standard_deviation ).ravel()
                ,( traintest_cov@w4 + 1.96*standard_deviation ).ravel()
                ,alpha=0.05, color='purple')
for _ in range(100):
    sample_function = np.random.multivariate_normal(traintest_cov@w4.ravel(), cov_mat)
    plt.plot(X_data, sample_function, color='purple', alpha=0.02)
# plt.plot(X_data, preds, color='mediumseagreen', alpha=0.5, label='gaussian')
# plt.fill_between(X_data.ravel()
#     ,preds.ravel()-1.96*stds
#     ,preds.ravel()+1.96*stds
#     , color='mediumseagreen', alpha=0.05)

# Ground Truth
plt.scatter(X_truth, y_truth, edgecolor='black', s=10, facecolor='None', alpha=0.3, label='ground_truth')
plt.legend(bbox_to_anchor=(1,1))
plt.show()



















































# (Python) Gaussian Process Regressor 230424
# Gaussian Process Regressor는 supervised learning에서 kernel regression 모델을 통해
# target function을 추론하는 Bayesian 방법론입니다. 

# 다변량 정규분포가 평균 벡터와 공분산 행렬로 표현되는 것처럼, GP 또한 평균 함수와 공분산 함수를 통해 다음과 같이 정의된다.
# P(x,y) ~ N ([μx, μy], [[Σx, Σxy], [Σxy.T, Σy]])       # 어떤 평균과 공분산을 가진 distribution에서 sampling
# P(X) ~ GP(m(t), k(x1, x2))                            # 어떤 평균과 Kernel을 가진 distributional function에서 sampling


# 상세 구현을 위해 우선적으로 kernel 함수를 정의합니다. 
# kernel 함수는 데이터 샘플 간의 유사도를 측정하는데 사용됩니다. 
# 이번 경우는 Radial Basis Function(RBF) kernel 함수를 사용합니다.

# python 

def RBF_kernel(X1, X2, l=1, sigma_f=1):
    """ 
    :param X1: numpy array, (n_samples_1, n_features) 
    :param X2: numpy array, (n_samples_2, n_features) 
    :param l: float, length parameter 
    :param sigma_f: float, scaling parameter 
    :return: numpy array, (n_samples_1, n_samples_2) 
    """ 
    n_samples_1, n_features = X1.shape 
    n_samples_2, n_features = X2.shape 
    
    K = np.zeros((n_samples_1, n_samples_2)) 
    
    for i in range(n_samples_1): 
        for j in range(n_samples_2): 
            diff = X1[i] - X2[j] 
            K[i, j] = sigma_f ** 2 * np.exp(-1 / (2 * l**2) * np.dot(diff.T, diff))
    return K 

# 다음으로 Gaussian Process Regressor 모델을 구현합니다. 
# python 

class GPR: 
    """ Gaussian Process Regressor Implementation """ 
    def __init__(self, x_train, y_train, l=1.0, sigma_f=1.0): 
        """ 
        :param x_train: numpy array, (n_samples, n_features) 
        :param y_train: numpy array, (n_samples, ) 
        :param l: float, length parameter 
        :param sigma_f: float, scaling parameter 
        """ 
        self.x_train = x_train 
        self.y_train = y_train 
        
        self.l = l 
        self.sigma_f = sigma_f 
        self.K = RBF_kernel(self.x_train, self.x_train, self.l, self.sigma_f) 
        
    def predict(self, x_test): 
        """ 
        :param x_test: numpy array, (n_samples, n_features) 
        :return: numpy array, (n_samples, ) 
        """ 
        K_test = RBF_kernel(self.x_train, x_test, self.l, self.sigma_f) 
        K_inv = np.linalg.inv(self.K) 
        mu = K_test.T.dot(K_inv).dot(self.y_train) 
        cov = RBF_kernel(x_test, x_test, self.l, self.sigma_f) - K_test.T.dot(K_inv).dot(K_test) 
        return mu, cov 
    
# 이제 Gaussian Process Regressor를 사용하여 predictor를 생성할 수 있습니다. 
# python 
import numpy as np 

# Training Data 
X_train = np.array([[-4], [-3], [-2], [-1], [0]]) 
Y_train = np.array([4, 1, 0, 1, 4]) 

# Test Data 
X_test = np.arange(-5, 5, 0.2).reshape(-1, 1) 

# Gaussian Process Regressor 
gp = GPR(X_train, Y_train, l=1.0, sigma_f=1.0) 
Y_pred_mean, Y_pred_cov = gp.predict(X_test) 

# 위의 코드를 실행하면, X_test 범위 내의 결과값의 평균과 공분산을 얻을 수 있습니다. 
# 이를 바탕으로 관측치의 값과의 차이를 분석하고, 
# 모델을 조정해 보다 정확한 Predictor를 구축하는데 활용할 수 있습니다.




gp = GPR(X_train.reshape(-1,1), y_train, l=1.0, sigma_f=1.0) 
Y_pred_mean, Y_pred_cov = gp.predict(np.expand_dims(X_truth, axis=1)) 
Y_pred_mean

preds = Y_pred_mean
stds = np.diag(Y_pred_cov)

np.diag(Y_pred_cov)

# model_gpr = GaussianProcessRegressor()
# from sklearn.gaussian_process.kernels import RBF
kernel_RBF = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
model_gpr= GaussianProcessRegressor(kernel=kernel_RBF)
# model_gpr= GaussianProcessRegressor()

model_gpr.fit(X_train.reshape(-1,1), y_train)
preds, stds = model_gpr.predict(np.expand_dims(X_truth, axis=1), return_std=True)





















### Gausssian Process #################################################################################
import numpy as np
# dataset
X_truth = np.linspace(-4, 4, 101)
y_truth = np.sin(X_truth)
idx = np.arange(101)

rng = np.random.RandomState(2)
n_sample = 5
train_idx = rng.choice(idx, n_sample)

X_train = X_truth[train_idx]
y_train = np.sin(X_train) + 0.5*rng.rand(X_train.shape[0])

# plot
plt.scatter(X_train, y_train, label='obeservation')
plt.plot(X_truth, y_truth, alpha=0.1, color='red', label='real')
plt.legend()
plt.show()

# sklearn gaussian process
from sklearn.gaussian_process import GaussianProcessRegressor
# ?GaussianProcessRegressor


model_gpr = GaussianProcessRegressor()
# from sklearn.gaussian_process.kernels import RBF
# kernel_RBF = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
# model_gpr= GaussianProcessRegressor(kernel=kernel_RBF)

model_gpr.fit(X_train.reshape(-1,1), y_train)
preds, stds = model_gpr.predict(np.expand_dims(X_truth, axis=1), return_std=True)

import matplotlib.pyplot as plt
plt.figure(figsize=(6, 8))
plt.plot(X_train, y_train, linestyle='none', marker='x')
plt.plot(X_truth, y_truth, color='red')
plt.plot(X_truth, preds, color='blue')
plt.fill_between(X_truth, preds - 1.96 * stds, preds + 1.96 * stds, alpha=0.1, color='blue')
plt.legend(['sample', 'real', 'preds', 'uncertainty'],)
plt.show()
####################################################################################



# https://pasus.tistory.com/209




import scipy

# https://sonsnotation.blogspot.com/2020/11/11-2-gaussian-progress-regression.html
# https://greeksharifa.github.io/bayesian_statistics/2020/07/12/Gaussian-Process/

# Define the exponentiated quadratic 
# rbf_kernel function for two points x1, x2 computes the similarity or how close they are each other.
def rbf_kernel(x1, x2, sigma_f=1.0, sigma_l=1.0):
    """rbf_kernel : Exponentiated quadratic  with σ=1
                    K(x1, x2) = σf² · exp( 1/(2σl²) ·||x1 - x2||² )
                      . hyperparameter
                         ㄴ σf : similarity의 크기 조정 (σf는 클수록 전체적인 similarity의 절대값을 높게 측정하기 때문에, uncertainty 절대적인 양이 커진다)
                         ㄴ σl :  분포의 폭을 조절 (작을수록 similarity를 높게 측정하여 uncertainty가 요동친다, 크면 smooth해진다)
                      . ||x1 - x2||²  Euclidean(L2 norm) distance
                      * similarities have values between 0(min) and 1(max)
                      * RBF Kernel is popular because of its similarity to K-Nearest Neighborhood Algorithm.
        
        # HyperParameter
  
    """
    # L2 distance (Squared Euclidian)
    sq_dist = scipy.spatial.distance.cdist(x1, x2, 'sqeuclidean')
    sq_norm = -1/(2*(sigma_l**2)) * sq_dist
    # X = np.arange(3).reshape(-1,1)  # [1,2,3]
    # scipy.spatial.distance.cdist(X, X, 'euclidean')       # (L1)
    # scipy.spatial.distance.cdist(X, X, 'sqeuclidean')     # (L2)
    # (L1) 1 2 3      (L2) 1 2 3
    #   1  0 1 2        1  0 1 4
    #   2  1 0 1        2  1 0 1
    #   3  2 1 0        3  4 1 0
    return sigma**2 * np.exp(sq_norm)






# rbf_kernel_function -----------------------------------------------------
a1 = np.linspace(-5,5, 50).reshape(-1,1)
a2 = np.zeros(a1.shape)

result1 = rbf_kernel(a1, a1, sigma=2)
result2 = rbf_kernel(a1, a2, sigma=2)
result2[:,0].std()
result2[:,0]

plt.contourf(a1.ravel(), a1.ravel() ,result1, cmap='Blues')
plt.colorbar()
plt.show()

plt.plot(a1.ravel(), result2[:,0])
plt.show()


# Gaussian Process --------------------------------------------------------

# Sample from the Gaussian process distribution
nb_of_samples = 3  # Number of points in each function
number_of_functions = 5  # Number of functions to sample

# Independent variable samples
X = np.linspace(-4, 4, nb_of_samples).reshape(-1,1)
cov = rbf_kernel(X, X)  # Kernel of data points

# Draw samples from the prior at our data points.
# Assume a mean of 0 for simplicity
ys = np.random.multivariate_normal(
    mean=np.zeros(nb_of_samples), cov=cov, 
    size=number_of_functions)
















############################################################################
# Gaussian Process
import numpy as np
import matplotlib.pyplot as plt




# Define the true function and generate some data
f = lambda x: np.array(np.sin(0.9*x)).reshape(-1,1)
X = np.sort(np.random.uniform(-5, 5, size=(20, 1)), axis=0)  # (20,1)
# y = f(X) + np.zeros((1,30))
y = f(X) + 1 * np.random.randn(1,30) # (20,3)


plt.scatter(X + np.zeros((1,30)), y, alpha=0.1)
plt.plot(X, f(X))

# X.shape
# y.shape

# Define the GaussianProcess object with the RBF kernel
gp = GaussianProcess(kernel=rbf_kernel, noise_var=1e-5)

# Fit the model to the data
gp.fit(X, y)

# Generate some test data
X_test = np.linspace(-5, 5, 100).reshape(-1, 1)

X_test.shape

# Make predictions with the model
y_pred, y_std = gp.predict(X_test)
y_pred.shape

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(X_test, f(X_test), 'b-', label='True Function')
plt.plot(X, y, 'kx', label='Training Data')
plt.plot(X_test, y_pred, 'r-', label='Predicted Mean')
# plt.fill_between(X_test.flatten(), y_pred.flatten() - 2*y_std, y_pred.flatten() + 2*y_std, alpha=0.2)
plt.xlabel('x')
plt.ylabel('f(x)')
# plt.legend(loc='upper left')
plt.show()






# Gaussian Process
import numpy as np

class GaussianProcess:
    def __init__(self, kernel, noise_var=1e-10):
        self.kernel = kernel
        self.noise_var = noise_var
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        K = self.kernel(X, X) + self.noise_var * np.eye(len(X))
        self.L = np.linalg.cholesky(K)

    def predict(self, X_test):
        K_s = self.kernel(self.X, X_test)
        L_inv_y = np.linalg.solve(self.L, self.y)
        alpha = np.linalg.solve(self.L.T, L_inv_y)
        mu = np.dot(K_s.T, alpha)
        v = np.linalg.solve(self.L, K_s)
        var = self.kernel(X_test, X_test) - np.dot(v.T, v)
        return mu, var

# This code defines a GaussianProcess class that takes a kernel function and a noise variance as input. The fit method takes in training inputs X and outputs y, computes the covariance matrix K, adds the noise variance to the diagonal, and performs a Cholesky decomposition to obtain the lower triangular matrix L. The predict method takes in test inputs X_test, computes the covariance matrix between the training inputs X and the test inputs X_test, and uses the Cholesky decomposition to compute the mean and variance of the predictive distribution.
# The kernel function takes two inputs, X1 and X2, and returns the 



import numpy as np
import matplotlib.pyplot as plt

# Define the kernel function
def rbf_kernel(x1, x2, l=1.0, sigma_f=1.0):
    distance = np.sum((x1 - x2) ** 2)
    return sigma_f ** 2 * np.exp(-0.5 * distance / l ** 2)

# Define the Gaussian Process class
class GaussianProcess:
    def __init__(self, kernel, noise_variance=1e-10):
        self.kernel = kernel
        self.noise_variance = noise_variance

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

        # Calculate the kernel matrix for the training data
        K = np.zeros((len(self.X_train), len(self.X_train)))
        for i in range(len(self.X_train)):
            for j in range(len(self.X_train)):
                K[i, j] = self.kernel(self.X_train[i], self.X_train[j])

        # Add the noise variance to the diagonal of the kernel matrix
        K += self.noise_variance * np.eye(len(self.X_train))

        # Calculate the Cholesky decomposition of the kernel matrix
        L = np.linalg.cholesky(K)

        # Define the negative log marginal likelihood function
        def nll(params):
            self.noise_variance = np.exp(params[0])
            self.kernel_length_scale = np.exp(params[1])
            self.kernel_amplitude = np.exp(params[2])

            K = np.zeros((len(self.X_train), len(self.X_train)))
            for i in range(len(self.X_train)):
                for j in range(len(self.X_train)):
                    K[i, j] = self.kernel(self.X_train[i], self.X_train[j], self.kernel_length_scale, self.kernel_amplitude)

            K += self.noise_variance * np.eye(len(self.X_train))

            L = np.linalg.cholesky(K)

            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))

            return 0.5 * np.dot(self.y_train, alpha) + np.sum(np.log(np.diag(L))) + 0.5 * len(self.X_train) * np.log(2 * np.pi)

        # Minimize the negative log marginal likelihood function to obtain the hyperparameters
        initial_params = np.log([self.noise_variance, 1.0, 1.0])
        res = minimize(nll, initial_params, method='L-BFGS-B')

        self.noise_variance = np.exp(res.x[0])
        self.kernel_length_scale = np.exp(res.x[1])
        self.kernel_amplitude = np.exp(res.x[2])

        self.L = np.linalg.cholesky(K + self.noise_variance * np.eye(len(self.X_train)))

    def predict(self, X):
        K_s = np.zeros((len(X), len(self.X_train)))
        for i in range(len(X)):
            for j in range(len(self.X_train)):
                K_s[i, j] = self.kernel(X[i], self.X_train[j], self.kernel_length_scale, self.kernel_amplitude)

        alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y_train))
        f_mean = np.dot(K_s, alpha)
        v = np.linalg.solve(self.L, K_s.T)
        f_var = self.kernel(X, X) - np.dot(v.T, v)

        return f_mean, np.diag(f_var)


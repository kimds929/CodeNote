import numpy as np
import matplotlib.pyplot as plt
rng = np.random.RandomState()

######################################################################
# True Parameters
feature_dim = 2
true_b = rng.rand(1).reshape(-1,1)
true_theta = rng.rand(feature_dim).reshape(-1,1)
print(true_theta, true_b)

# Data
n_samples = 100
X = rng.normal(0, 1, size=(n_samples, feature_dim))   # (n, feature_dim)
noise = rng.normal(0, 0.2, size=n_samples).reshape(-1,1)  # noise

Y_obs = X @ true_theta + true_b + noise

# -------------------------------------------------------------------
plt.figure()
plt.scatter(X, Y_obs)
plt.show()

# -------------------------------------------------------------------
# Methods 1 : Derivative
X_bar = X - X.mean(axis=0)
Y_bar = Y_obs - Y_obs.mean(axis=0)

b1 = np.linalg.inv( X_bar.T @ X_bar ) @ X_bar.T @ Y_bar
b0 = Y_obs.mean(axis=0) - b1.T  @ X.mean(axis=0)
print(b1, b0)

# -------------------------------------------------------------------
# Methods 2 : Linear Algebra
X_b = np.concatenate([X, np.ones( (len(X),1) )], axis=1)

theta_hat = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ Y_obs
print(theta_hat[:-1], theta_hat[-1])


# -------------------------------------------------------------------
# Methods 3 : Library (sklearn)
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X, Y_obs)
print(LR.coef_.reshape(-1,1), LR.intercept_)



###########################################################################
# Methods 4 : Gradient Descent

# initialize
lr = 1e-3       # learning_rate
n_iter = 1000

b0_hat = rng.rand(1).reshape(-1,1)
b1_hat = rng.rand(feature_dim).reshape(-1,1)

b0_learning = [b0_hat]
b1_learning = [b1_hat]

for _ in range(n_iter):
    # calculate y_hat
    Y_hat = X @ b1_hat + b0_hat
    

    # calculate gradient
    b0_grad = np.zeros((1,1))
    b1_grad = np.zeros((feature_dim,1))
    for x, y_obs, y_hat in zip(X, Y_obs, Y_hat):
        b0_grad += (-2 * (y_obs - y_hat)).reshape(-1,1)
        b1_grad += (-2 * x * (y_obs - y_hat)).reshape(-1,1)
        
    # b0_grad = (-2 * (Y_obs - Y_hat)).sum(axis=0, keepdims=True)
    # b1_grad = (-2 * X.T @ (Y_obs - Y_hat))

    # update
    b0_hat = b0_hat - lr * b0_grad
    b1_hat = b1_hat - lr * b1_grad

    b0_learning.append(b0_hat)
    b1_learning.append(b1_hat)

plt.plot(np.stack(b0_learning).reshape(-1,1))
plt.plot(np.stack(b1_learning).reshape(-1, feature_dim))


print(b1_hat, b0_hat)

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import matplotlib.pylab as plt
# from itertools import product

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


num_data_train = X_train.shape[0]
num_data_test = X_test.shape[0]

dim_data_train = X_train.shape[0]
dim_data_test = X_test.shape[0]


# 1개의 column만 골라서 분석 시행
dim_target = 5

X_train_1D = X_train[:, dim_target][..., tf.newaxis]
X_test_1D = X_test[:, dim_target][..., tf.newaxis]

X_train_1D = (X_train_1D - X_train_1D.mean(0)) / X_train_1D.std(0)      # Normalization
X_test_1D = (X_test_1D - X_test_1D.mean(0)) / X_train_1D.std(0)         # Normalization

print(X_train_1D.shape, X_test_1D.shape)


# Plot 그려주는 함수 구현
def plot_graph(X, y, X_hat=None, y_hat=None, str_title=None):
    num_X = X.shape[0]
    
    fig = plt.figure(figsize=(8,6))

    if str_title is not None:
        plt.title(str_title, fontsize=20, pad=20)

    plt.plot(X, y, ls='none', marker='o', markeredgecolor='white')
    
    if X_hat is not None and y_hat is not None:
        plt.plot(X_hat, y_hat)

    plt.tick_params(axis='both', labelsize=14)

np.linalg.inv(X_train_1D.T @ X_train_1D) @ X_train_1D.T @ y_train

# Plot 보기
# plot_graph(X_train_1D, y_train, str_title=f'Distribution of Train Data: dim {dim_target+1}')


# theta = inv(X.T @ X) @ X.T @ y
Xmat_T_Xmat = tf.transpose(X_train_1D) @ tf.constant(X_train_1D)
theta = tf.linalg.inv(Xmat_T_Xmat) @ tf.transpose(X_train_1D) @ y_train[..., tf.newaxis]
print(theta)

loss = tf.math.sqrt(tf.reduce_mean((y_test - X_test_1D @ theta)**2))
print(loss)

X_train_lp = np.linspace(np.min(X_train_1D), np.max(X_train_1D), 100)[..., np.newaxis]
X_test_lp = np.linspace(np.min(X_test_1D), np.max(X_test_1D), 100)[..., np.newaxis]


# LinearRegression Plot
# Train_set
plot_graph(X=X_train_1D, y=y_train,
        X_hat=X_train_lp, y_hat=X_train_lp @ tf.constant(theta),
        str_title=f'Linear Regression for training Data: dim {dim_target+1}')

# Test_set
plot_graph(X=X_test_1D, y=y_test,
        X_hat=X_test_lp, y_hat=X_test_lp @ tf.constant(theta),
        str_title=f'Linear Regression for test Data: dim {dim_target+1}')



# featrue function (kernel) -----------------------------------------
dim_target = 12

X_train_1D = X_train[:, dim_target][..., tf.newaxis]
X_test_1D = X_test[:, dim_target][..., tf.newaxis]
X_train_lp = np.linspace(np.min(X_train_1D), np.max(X_train_1D), 100)[..., np.newaxis]
X_test_lp = np.linspace(np.min(X_test_1D), np.max(X_test_1D), 100)[..., np.newaxis]


tf.concat([X_train_1D, X_train_1D**0], axis=1).shape

def featrue_f(X, degree=1):
    num_X = tf.shape(X)[0]
    phi = tf.ones([num_X, 1], dtype=tf.float64)

    for i in range(0, degree):
        phi = phi * X
        if i == degree-1:
            phi = tf.concat([featrue_f(X, degree-1), phi], axis=1)
    return phi

featrue_f(X=X_train_1D, degree=2)[:10,:]


num_degree = 2
phi_train = featrue_f(X_train_1D, num_degree)
phi_test = featrue_f(X_test_1D, num_degree)

phi_train_lp = featrue_f(X_train_lp, num_degree)
phi_test_lp = featrue_f(X_test_lp, num_degree)

# print(phi_train[:10])

phi_theta = tf.linalg.inv(tf.transpose(phi_train) @ phi_train) @ tf.transpose(phi_train) @ y_train.reshape(-1,1)
phi_loss = tf.math.sqrt(tf.reduce_mean((y_test - phi_test @ phi_theta)**2))
phi_loss


# LinearRegression Plot (phi)
# Train_set
plot_graph(X=X_train_1D, y=y_train,
        X_hat=X_train_lp, y_hat=phi_train_lp @ tf.constant(phi_theta),
        str_title=f'Linear Regression for training Data: dim {dim_target+1}')

# Test_set
plot_graph(X=X_test_1D, y=y_test,
        X_hat=X_test_lp, y_hat=phi_test_lp @ tf.constant(phi_theta),
        str_title=f'Linear Regression for test Data: dim {dim_target+1}')







# Gradient Descent 방법으로 최적의 Theta값 추정
def RMSE(y_pred, y_true):
    return tf.math.sqrt(tf.reduce_mean(tf.square(tf.squeeze(y_pred) - y_true)))

def run_optimization():
    with tf.GradientTape() as Tape:
        pred = phi_train @ theta_gd
        loss = RMSE(pred, y_train)
    
    gradients = Tape.gradient(loss, theta_gd)           # loss를 theta에 대해 미분한다.
    optimizer.apply_gradients([(gradients, theta_gd)])  # gradient를 바탕으로 theta를 업데이트 해줘
                            # list of (gradient, variable(parameter))
    # tf.optimizers.SGD.apply_gradients(self, grads_and_vars, name=None)
    #       grads_and_vars: List of (gradient, variable) pairs.
    #       name: Optional name for the returned operation.  Default to the name
    #           passed to the `Optimizer` constructor.
# RMSE(phi_train[:3] @ theta_gd, y_train[:3])

# learning_rate = 0.1
learning_rate = 0.001
optimizer = tf.optimizers.SGD(learning_rate)

# theta_gd = tf.Variable(np.random.randn(*phi_theta.shape),dtype=tf.float64)
theta_gd = tf.Variable(np.random.randn(*phi_theta.shape), dtype=tf.float64)

training_steps = 1000
display_step = 100

for step in range(1,training_steps+1):
    run_optimization()

    if step % display_step == 0:
        pred = phi_train @ theta_gd
        loss = RMSE(pred, y_train)
        print('step: ', step, 'loss: ', loss.numpy(), ' theta: ', theta_gd.numpy())

print(f'phi_theta: \n{phi_theta} \ntheta_gd: \n{theta_gd}')

# LinearRegression Plot (Gradient_Descent)
# Train_set
plot_graph(X=X_train_1D, y=y_train,
        X_hat=X_train_lp, y_hat=phi_train_lp @ theta_gd,
        str_title=f'Linear Regression for training Data: dim {dim_target+1}')

# Test_set
plot_graph(X=X_test_1D, y=y_test,
        X_hat=X_test_lp, y_hat=phi_test_lp @ theta_gd,
        str_title=f'Linear Regression for test Data: dim {dim_target+1}')




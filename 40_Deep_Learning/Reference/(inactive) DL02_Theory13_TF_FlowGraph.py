import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
import tensorflow as tf

import time
from IPython.display import clear_output
# clear_output(wait=True)



# --------------------------------------------------------------------------------------------------------
# Tensorflow Regressor Problem ---------------------------------------------------------------------


# Dataset Define ----
test_dict = {'y': [10, 8, 20, 17, 15],
        'x1': [3, 2, 5, 4, 6],
        'x2': [10, 8, 5, 12, 7]}

df = pd.DataFrame(test_dict)

# Dataset 정의 ----
y_cols = ['y']
# X_cols = df.columns.drop(y_cols)
X_cols = ['x1']

y = df[y_cols]
X = df[X_cols]
y
X


# numpy data ----
X_np0 = X.copy()
X_np0['const'] = 1
X_np0

X_np1 = X_np0[['const'] + list(X_cols)]

X_np = X_np1.to_numpy()
y_np = y.to_numpy()
X_np
y_np




# w_init
w_init = np.array([15, -2]).reshape(-1,1)  # w initialize (Random)
print(f'X.shape: {X_np.shape} / w_init.shape : {w_init.shape}  /  y.shape : {y_np.shape}')


# Linear_Regression (sklearn)
LR = LinearRegression()
LR.fit(X_np, y_np)




# plotting
def plotting(w=False):
    if type(w) != bool:
        line_x = np.linspace(-1, 10, 50).reshape(-1,1)
        line_x = np.hstack([line_x ** 0, line_x])
        line_y = line_x @ w
        # print(w)

    plt.figure()
    plt.hlines(y=0, xmin=-5, xmax=10, alpha=0.5)
    plt.vlines(x=0, ymin=-5, ymax=30, alpha=0.5)
    plt.scatter(data=df, x='x1', y='y', s=50)

    if type(w) != bool:
        plt.plot(line_x[:,1], line_y, 'r--', alpha=0.3)
    plt.plot(line_x[:,1], LR.predict(line_x), 'orange', alpha=0.7) 
    plt.xlim([-1, 10])
    plt.ylim([-1, 30])
    
    plt.grid(alpha=0.5)
    plt.show()

def result_plot(w_history, mse_history, grad_history):
    w_history_reshape = np.array(w_history).reshape(-1, 2)
    grad_history_reshape = np.array(grad_history).reshape(-1,2)

    plt.figure(figsize=(15, 4.5))

    # mse plot
    plt.subplot(1, 3, 1)
    plt.plot(mse_history, label='mse')
    plt.xscale('log')
    plt.legend()

    # weight hitory plot
    plt.subplot(1, 3, 2)
    plt.title(w_history_reshape[-1])
    plt.plot(w_history_reshape[:,0], label='const')
    plt.plot(w_history_reshape[:,1], label='x1')
    plt.xscale('log')
    plt.legend()

    # gradient history plot
    plt.subplot(1, 3, 3)
    plt.title(grad_history_reshape[-1])
    plt.plot(grad_history_reshape[:,0], label='const')
    plt.plot(grad_history_reshape[:,1], label='x1')
    plt.xscale('log')
    plt.legend()
    plt.show()


# plotting
# w_init = np.array([15, -2]).reshape(-1,1)  # w initialize (Random)
plotting(w=w_init)




# tensorflow optimizer minimize 활용------------------------------------------------

# Deicision 영역 그리는 함수
def plot_decision(W1, W2, b1, b2):
    x_plot, y_plot = np.meshgrid(np.linspace(-0.5, 1.5), np.linspace(-0.5, 1.5))

    X_plot = np.hstack([x_plot.reshape(-1, 1), y_plot.reshape(-1, 1)])
    z1 = tf.sigmoid(X_plot @ W1 + b1)
    z2 = tf.sigmoid(z1 @ W2 + b2)

    plt.contour(x_plot.reshape(50, 50), y_plot.reshape(50, 50), z2.numpy().reshape(50, 50))
    plt.scatter(X[:, 0], X[:, 1], s=200, c=y, marker='*')
    plt.show()

leaning_rate = 0.001
epoch = 5000

n_features = len(X.columns)
n_hiddens = 3
n_output = len(y.columns)


w_tf1 = tf.Variable(tf.random.normal(shape=[n_features, n_hiddens]), dtype=tf.float32)
b_tf1 = tf.Variable(tf.random.normal(shape=[n_hiddens]), dtype=tf.float32)
w_tf2 = tf.Variable(tf.random.normal(shape=[n_hiddens, n_output]), dtype=tf.float32)
b_tf2 = tf.Variable(tf.random.normal(shape=[n_output]), dtype=tf.float32)

# forward propagation
def forward(X):
    hidden = X @ w_tf1 + b_tf1
    hidden_logit = tf.sigmoid(hidden)
    output = hidden_logit @ w_tf2 + b_tf2
    return output

# loss function
def tf_mse(y_pred, y_true):
    return tf.reduce_mean((y_pred-y_true)**2)

# loss = tf.reduce_mean(tf.losses.mean_squared_error)
# loss(forward(X.to_numpy()), y_np)

# weight history
w_tf_history_optMin2 = []

# optimizer
optimizer = tf.optimizers.SGD(learning_rate=leaning_rate)   # optimizer 활용 **

for _ in range(epoch):
    loss_func = lambda: tf_mse(forward(X.to_numpy()), y_np)
    optimizer.minimize(loss_func, var_list=[w_tf1, b_tf1, w_tf2, b_tf2])

    opt_result = [w_tf1.numpy(), b_tf1.numpy(), w_tf2.numpy(), b_tf2.numpy()]
    # history 저장
    w_tf_history_optMin2.append(opt_result)

    if _ <10 or (_ <100 and (_+1) % 20 == 0) or (_ < 1000 and (_+1) % 100 == 0) or ( (_+1) % 1000 == 0):
        print(f'{_+1}번째 epoch')
        # plot_decision()
        clear_output(wait=True)
        time.sleep(0.2)



# weight history
w_optMin2_history_reshape = np.array(w_tf_history_optMin2).reshape(-1, 2)
plt.figure()
plt.title(w_optMin2_history_reshape[-1])
plt.plot(w_optMin2_history_reshape[:,0], label='const')
plt.plot(w_optMin2_history_reshape[:,1], label='x1')
plt.xscale('log')
plt.legend()
plt.show()

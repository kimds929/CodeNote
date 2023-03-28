import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import torch

import os
import time
from IPython.display import clear_output
# clear_output(wait=True)


# --------------------------------------------------------------------------------------------------------
# Tensorflow Classifier Problem ---------------------------------------------------------------------

# Dataset Define ----
x_data = {'x1':[0,0,1,1], 'x2':[0,1,0,1]}
# y_data = [0,0,1,1]
# y_data = [0,0,0,1]
y_data = [1,0,0,0]

x_df = pd.DataFrame(x_data)
y_df = pd.Series(y_data, name='y').to_frame()

df = pd.concat([x_df, y_df], axis=1)
df


# Dataset 정의 ----
y_cols = ['y']
X_cols = df.columns.drop(y_cols)

y = df[y_cols]
X = df[X_cols]


# numpy Data ----
X_np0 = X.copy()
X_np0['const'] = 1
X_np0

X_np1 = X_np0[['const'] + list(X_cols)]

X_np = X_np1.to_numpy()
y_np = y.to_numpy()


# w_init
w_init = np.array([6, -1, -2]).reshape(-1,1)  # w initialize (Random)
print(f'X.shape: {X_np.shape} / w_init.shape : {w_init.shape}  /  y.shape : {y_np.shape}')


# plotting
def plotting(w=False, mesh=False):
    if type(w) != bool:
        line_x = np.linspace(-1, 3, 30).reshape(-1,1)
        line_x = np.hstack([line_x ** 0, line_x])
        coef_w = [-w[0]/w[2], -w[1]/w[2]]
        line_y = line_x @ coef_w
        # print(coef_w)

    plt.figure()
    plt.hlines(y=0, xmin=-5, xmax=5, alpha=0.5)
    plt.vlines(x=0, ymin=-5, ymax=5, alpha=0.5)
    for i, v in df.groupby('y'):
        plt.scatter(data=v, x='x1', y='x2', label=i, s=100)

    if type(w) != bool:
        plt.plot(line_x[:,1], line_y, 'r--', alpha=0.3)    
    plt.legend()
    plt.xlim([-1, 3])
    plt.ylim([-1, 3])

    if mesh:
        xp1 = np.linspace(-1,3, 100)
        xp2 = np.linspace(-1,3, 100)
        x1_mesh, x2_mesh = np.meshgrid(xp1, xp2)
        mesh_np = np.hstack([x1_mesh.reshape(-1,1) ** 0, x1_mesh.reshape(-1,1), x2_mesh.reshape(-1,1)])
        cond = np.sign((mesh_np @ w).ravel()) == 1
        plt.plot(mesh_np[cond][:,1], mesh_np[cond][:,2], color='orange', marker='s', alpha=0.05)

    plt.grid(alpha=0.5)
    plt.show()

# result summary plotting
def result_plot(w_history, loss_history, grad_history):
    w_history_reshape = np.array(w_history).reshape(-1, 2)
    grad_history_reshape = np.array(grad_history).reshape(-1,2)

    plt.figure(figsize=(15, 4.5))

    # loss_function plot
    plt.subplot(1, 3, 1)
    plt.title('loss_function')
    plt.plot(loss_history, label='mse')
    plt.xscale('log')
    plt.legend()

    # weight hitory plot
    plt.subplot(1, 3, 2)
    plt.title('weight\n' + str(w_history_reshape[-1]))
    plt.plot(w_history_reshape[:,0], label='const')
    plt.plot(w_history_reshape[:,1], label='x1')
    plt.xscale('log')
    plt.legend()

    # gradient history plot
    plt.subplot(1, 3, 3)
    plt.title('weight_gradient\n' + str(grad_history_reshape[-1]))
    plt.plot(grad_history_reshape[:,0], label='const')
    plt.plot(grad_history_reshape[:,1], label='x1')
    plt.xscale('log')
    plt.legend()
    plt.show()


# w_init = np.array([6, -1, -2]).reshape(-1,1)  # w initialize (Random)
plotting(w_init, mesh=True)



# numpy ---------------------------------
learning_rate = 0.01
epoch = 5000

# w_init = np.array([6, -1, -2]).reshape(-1,1)  # w initialize (Random)
w_np = w_init.copy()


def cross_entropy_np(y_pred, y_true):
    return np.mean(- y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

w_np_history = []
loss_np_history = []
grad_np_history = []


for _ in range(epoch):
    # gradient 구하기
        # small delta 를 활용한 미분 ★★★
    loss_np = cross_entropy_np(y_pred=sigmoid(X_np @ w_np), y_true=y_np)

    loss_grad_list = []
    delta = 1e-6
    for i in range(len(w_np)):
        delta_list = [0, 0, 0]
        delta_list[i] = delta
        delta_np = np.array(delta_list).reshape(-1,1)

        loss_X = cross_entropy_np(y_pred=sigmoid(X_np @ w_np), y_true=y_np) 
        loss_deltaX = cross_entropy_np(y_pred=sigmoid(X_np @ (w_np - delta_np)), y_true=y_np)
        loss_grad = (loss_X - loss_deltaX)/delta
        loss_grad_list.append(loss_grad)

    loss_grad_w_np = np.array(loss_grad_list).reshape(-1,1)

    # history 저장
    w_np_history.append(w_np)
    loss_np_history.append(loss_np)
    grad_np_history.append(loss_grad_w_np)

    # w값 업데이트
    w_np = w_np - learning_rate * loss_grad_w_np
    
    if _ <5 or (_ <100 and (_+1) % 20 == 0) or (_ < 1000 and (_+1) % 100 == 0) or ( (_+1) % 1000 == 0):
        print(f'{_+1}번째 epoch')
        print(f'w : {w_np.ravel()}')
        plotting(w_np)
        time.sleep(0.1)
        clear_output(wait=True)


plotting(w_np, mesh=True)
result_plot(w_history=w_np_history, loss_history=loss_np_history, grad_history=grad_np_history)








# tensorflow basic ------------------------------------------------
learning_rate = 0.01
epoch = 5000

# w_init = np.array([6, -1, -2]).reshape(-1,1)  # w initialize (Random)
w_tf = tf.Variable(w_init, shape=[3,1], dtype=tf.float32)

def cross_entropy_tf(y_pred, y_true):
    return tf.reduce_mean(- y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred))

w_tf_history = []
loss_tf_history = []
grad_tf_history = []

for _ in range(epoch):
    # tensorflow GradientTape ★★★
    with tf.GradientTape() as tape:     # Gradient를 구해주는 함수
        y_pred = tf.sigmoid(X_np @ w_tf)
        loss_tf = cross_entropy_tf(y_pred=y_pred, y_true=y_np)
    loss_grad_tf = tape.gradient(loss_tf, w_tf)

    w_tf_history.append(w_tf.numpy())
    loss_tf_history.append(loss_tf.numpy())
    grad_tf_history.append(loss_grad_tf.numpy())

    w_tf.assign(w_tf - learning_rate * loss_grad_tf)    # update ★★★

    if _ <5 or (_ <100 and (_+1) % 20 == 0) or (_ < 1000 and (_+1) % 100 == 0) or ( (_+1) % 1000 == 0):
        print(f'{_+1}번째 epoch')
        print(f'w : {w_tf.numpy().ravel()}')
        plotting(w_tf.numpy())
        time.sleep(0.1)
        clear_output(wait=True)

plotting(w_tf.numpy(), mesh=True)
result_plot(w_history=w_tf_history, loss_history=loss_tf_history, grad_history=grad_tf_history)









# tensorflow optimizer활용 ------------------------------------------------
learning_rate = 0.01
epoch = 5000

# w_init = np.array([6, -1, -2]).reshape(-1,1)  # w initialize (Random)
w_tf_opt = tf.Variable(w_init, shape=[3,1], dtype=tf.float32)

def cross_entropy_tf_opt(y_pred, y_true):
    return tf.reduce_mean(- y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred))

optimizer = tf.optimizers.SGD(learning_rate=learning_rate)  # optimizer ★★★

w_tf_opt_history = []
loss_tf_opt_history = []
grad_tf_opt_history = []

for _ in range(epoch):
    with tf.GradientTape() as tape:     # Gradient를 구해주는 함수
        y_pred = tf.sigmoid(X_np @ w_tf_opt)
        loss_tf_opt = cross_entropy_tf_opt(y_pred=y_pred, y_true=y_np)
        loss_grad_tf_opt = tape.gradient(loss_tf_opt, w_tf_opt)

    w_tf_opt_history.append(w_tf_opt.numpy())
    loss_tf_opt_history.append(loss_tf_opt.numpy())
    grad_tf_opt_history.append(loss_grad_tf_opt.numpy())

    optimizer.apply_gradients(grads_and_vars=[(loss_grad_tf_opt, w_tf_opt)])    # update ★★★

    if _ <5 or (_ <100 and (_+1) % 20 == 0) or (_ < 1000 and (_+1) % 100 == 0) or ( (_+1) % 1000 == 0):
        print(f'{_+1}번째 epoch')
        print(f'w : {w_tf_opt.numpy().ravel()}')
        plotting(w_tf_opt.numpy())
        time.sleep(0.1)
        clear_output(wait=True)

plotting(w_tf_opt.numpy(), mesh=True)
result_plot(w_history=w_tf_opt_history, loss_history=loss_tf_opt_history, grad_history=grad_tf_opt_history)









# tensorflow optimizer minimize 활용------------------------------------------------
learning_rate = 0.01
epoch = 5000
w_tf_optMin = tf.Variable(w_init, shape=[3,1], dtype=tf.float32)

def cross_entropy_tf_optMin(y_pred, y_true):
    return tf.reduce_mean(- y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred))

optimizer = tf.optimizers.SGD(learning_rate=learning_rate)

w_tf_optMin_history = []
# loss_tf_optMin_history = []
# grad_tf_optMin_history = []

for _ in range(epoch):
    loss_func = lambda: cross_entropy_tf_optMin(y_pred=tf.sigmoid(X_np @ w_tf_optMin), y_true=y_np)
    
    # optimizer.minimize ★★★
    optimizer.minimize(loss_func, [w_tf_optMin])

    w_tf_optMin_history.append(w_tf_optMin.numpy())
    # loss_tf_optMin_history.append(loss_tf_optMin.numpy())
    # grad_tf_optMin_history.append(loss_grad_tf_optMin.numpy())


    if _ <5 or (_ <100 and (_+1) % 20 == 0) or (_ < 1000 and (_+1) % 100 == 0) or ( (_+1) % 1000 == 0):
        print(f'{_+1}번째 epoch')
        print(f'w : {w_tf_optMin.numpy().ravel()}')
        plotting(w_tf_optMin.numpy())
        time.sleep(0.1)
        clear_output(wait=True)

plotting(w=w_tf_optMin.numpy(), mesh=True)








# Keras Basic ---------------------------------------------------------------
learning_rate = 0.01
epoch = 5000

X
y
n_input = len(X.columns)
n_output = len(y.columns)

# Sequential modeling
keras_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(n_output, activation='sigmoid', input_shape=[n_input,])
    ])

# compile
optimizer = tf.optimizers.SGD(learning_rate=learning_rate)
keras_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# fit
keras_result = keras_model.fit(X, y, batch_size=len(X), epochs=epoch, verbose=0)

# result
keras_weight = np.array([[keras_model.weights[1].numpy()[0]], *keras_model.weights[0].numpy()])



# result plotting
plotting(w=keras_weight, mesh=True)

plt.plot(keras_result.history['loss'])
plt.show()

plt.plot(keras_result.history['accuracy'])
plt.show()



# Keras Expert ---------------------------------------------------------------
X_np
y_np


# class model
class Tensorflow_Logistic(tf.keras.Model):
    def __init__(self):
        super(Tensorflow_Logistic, self).__init__()
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid', use_bias=False)
    
    def call(self, x):
        x = self.dense(x)
        return x
keras_model_class = Tensorflow_Logistic()

learning_rate = 0.01
epoch = 5000
# epoch = 5
optimizer = tf.keras.optimizers.SGD(learning_rate)

w_keras_history = []
mse_keras_history = []
grad_keras_history = []

# tf.graph
@tf.function
def trainning():
    with tf.GradientTape() as Tape:
        pred = keras_model_class(X_np)
        loss = tf.keras.losses.binary_crossentropy(y_pred=pred, y_true=y_np)
    gradients = Tape.gradient(loss, keras_model_class.trainable_variables)
    optimizer.apply_gradients(zip(gradients, keras_model_class.trainable_variables))
    return loss, gradients

for _ in range(epoch):
    # with tf.GradientTape() as Tape:
    #     pred = keras_model_class(X_np)
    #     loss = tf.keras.losses.binary_crossentropy(y_pred=pred, y_true=y_np)
    # gradients = Tape.gradient(loss, keras_model_class.trainable_variables)
    # optimizer.apply_gradients(zip(gradients, keras_model_class.trainable_variables))

    # tf.function (graph)
    loss, gradients = trainning()

    # history 저장
    w_keras_history.append(keras_model_class.weights[0].numpy())
    mse_keras_history.append(loss.numpy())
    grad_keras_history.append(gradients[0].numpy())
    
    if _ <10 or (_ <100 and (_+1) % 20 == 0) or (_ < 1000 and (_+1) % 100 == 0) or ( (_+1) % 1000 == 0):
        print(f'{_+1}번째 epoch')
        print(f'w : {keras_model_class.weights[0].numpy().ravel()}')
        plotting(keras_model_class.weights[0].numpy())
        clear_output(wait=True)
        time.sleep(0.2)

plotting(w=keras_model_class.weights[0].numpy(), mesh=True)











# Pytorch Autograd---------------------------------------------------------------
y_torch = torch.as_tensor(y_np, dtype=torch.float)
X_torch = torch.as_tensor(X_np, dtype=torch.float)

learning_rate = 0.01
epoch = 5000

w_init = np.array([6, -1, -2]).reshape(-1,1)  # w initialize (Random)
w_torch = torch.as_tensor(w_init, dtype=torch.float)
w_torch.requires_grad_(True)
# w_torch = torch.tensor(w_init, dtype=torch.float, requires_grad=True)

# X_torch.shape, w_torch.shape

w_torch_history = []
mse_torch_history = []
grad_torch_history = []
for _ in range(1, epoch+1):
    pred = torch.sigmoid(X_torch @ w_torch)
    loss = torch.nn.functional.binary_cross_entropy(pred, y_torch)

    loss.backward()

    with torch.no_grad():
        # history 저장
        w_torch_history.append(w_torch.numpy())
        mse_torch_history.append(loss.numpy())
        grad_torch_history.append(w_torch.grad.numpy())

        w_torch -= learning_rate * w_torch.grad
        w_torch.grad.zero_()

    if _ <10 or (_ <100 and (_+1) % 20 == 0) or (_ < 1000 and (_+1) % 100 == 0) or ( (_+1) % 1000 == 0):
        print(f'{_+1}번째 epoch')
        print(f'w : {w_torch.detach().numpy().ravel()}')
        plotting(w_torch.detach().numpy())
        clear_output(wait=True)
        time.sleep(0.2)

plotting(w=w_torch.detach().numpy(), mesh=True)


# Pytorch Model---------------------------------------------------------------
y_torch = torch.as_tensor(y_np, dtype=torch.float)
X_torch = torch.as_tensor(X_np, dtype=torch.float)

# class model
class Torch_Logistic(torch.nn.Module):
    def __init__(self):
        super(Torch_Logistic, self).__init__()
        self.linear = torch.nn.Linear(3, 1, bias=False)
    
    def forward(self, x):
        x = torch.nn.functional.sigmoid(self.linear(x))
        return x

torch_model = Torch_Logistic()

learning_rate = 0.01
epoch = 5000

optimizer = torch.optim.SGD(torch_model.parameters(), lr=learning_rate)

w_torch_mdl_history = []
mse_torch_mdl_history = []
grad_torch_mdl_history = []
for _ in range(epoch):
    optimizer.zero_grad()                   # gradients를 clear해서 새로운 최적화 값을 찾기 위해 준비

    pred = torch_model(X_torch)                                     # forward
    loss = torch.nn.functional.binary_cross_entropy(pred, y_torch)  # loss
    loss.backward()                                                 # backward

    optimizer.step()                        # weight update

    with torch.no_grad():
        # history 저장
        w_torch_mdl_history.append(torch_model.linear.weight.numpy())
        mse_torch_mdl_history.append(loss.numpy())
        grad_torch_mdl_history.append(torch_model.linear.weight.grad.numpy())

    if _ <10 or (_ <100 and (_+1) % 20 == 0) or (_ < 1000 and (_+1) % 100 == 0) or ( (_+1) % 1000 == 0):
        print(f'{_+1}번째 epoch')
        print(f'w : {torch_model.linear.weight.grad.numpy().ravel()}')
        plotting(torch_model.linear.weight.detach().numpy().reshape(-1,1))
        clear_output(wait=True)
        time.sleep(0.2)

plotting(w=torch_model.linear.weight.detach().numpy().reshape(-1,1), mesh=True)








import random

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import matplotlib.pylab as plt

from IPython.display import clear_output
import time
# from itertools import product


random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)


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
    plt.show()


x_0 = 3 + np.random.randn(20)
y_0 = np.zeros(20)

x_1 = 5.5 + np.random.randn(20)
y_1 = np.ones(20)

x = np.concatenate((x_0, x_1))
y = np.concatenate((y_0, y_1))

print(x)
print(y)

plot_graph(x, y, str_title='dataset')


# Weight, bias Initialize
W = tf.Variable(np.random.randn())
b = tf.Variable(np.random.randn())

def cross_entropy(y_pred, y_true):
    # [Cross_Entropy] y · log(y_hat) + (1 - y) · log(1 - y_hat)
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.0)
    return -tf.reduce_sum(y_true * tf.math.log(y_pred) + 
                        (1 - y_true) * tf.math.log(1 - y_pred))

lr = 0.001
optimizer = tf.optimizers.SGD(lr)

def run_optimization():
    with tf.GradientTape() as tape:
        y_pred = 1 / (1 + tf.exp(-1*(W * x + b)))
        loss = cross_entropy(y_pred=y_pred, y_true=y)
    
    gradients = tape.gradient(loss, [W,b])
    optimizer.apply_gradients(zip(gradients, [W,b]))

    return y_pred, loss

x_fl = np.linspace(np.min(x), np.max(x), num=100)
plot_graph(x, y, X_hat=x_fl, y_hat=1 / (1 + tf.exp(-1*(W * x_fl + b))),
        str_title='Logistic Regression')


steps = 0
display_step = 1000
training_interval = 10000

for step in range(steps, steps+training_interval+1):
    pred, loss = run_optimization()

    if step % display_step == 0:
        plot_graph(x, y, X_hat=x_fl, y_hat=1 / (1 + tf.exp(-1*(W * x_fl + b))),
            str_title=f'Logistic Regression\nstep: {step} / loss: {loss}')
        clear_output(wait=True)

steps = step


plot_graph(x, y, X_hat=x_fl, y_hat=1 / (1 + tf.exp(-1*(W * x_fl + b))),
            str_title=f'Logistic Regression\nstep: {step} / loss: {loss}')



# Mesh-plot
def sigmoid(w, x, b):
    return 1 / (1 + tf.exp(-1*(x * x + b)))

x_mesh = np.linspace(0,7, 100)
y_mesh = np.linspace(0,1, 100)

xs, ys = np.meshgrid(x_mesh, y_mesh)
k = np.hstack([xs.reshape(-1,1), ys.reshape(-1,1)])


cond = np.array(sigmoid(w=W, x=k[:,0], b=b)) < k[:,1]

fig = plt.figure(figsize=(8,6))
plt.title('Logistic_Regression', fontsize=20, pad=20)
plt.plot(x, y, ls='none', marker='o', markeredgecolor='white')
plt.plot(x_fl, sigmoid(w=W, x=x_fl, b=b))
plt.plot(k[cond][:,0], k[cond][:,1], 'rs', alpha=0.1, markeredgecolor='white')
plt.show()







# Multi-Class ---------------------------------------------------------------

# MNIST Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

num_classes = np.max(y_train) + 1
num_features = x_train.shape[1] * x_train.shape[2]
print(num_classes)
print(num_features)

x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
x_train, x_test = x_train/255, x_test/255


# 학습변수 정의
W = tf.Variable(tf.ones([num_features, num_classes]))
b = tf.Variable(tf.zeros([num_classes]))


def logistic_multiclass(x, W, b):
    return tf.nn.softmax(x @ W + b)


def cross_entropy(y_pred, y_true):
    y_true = tf.one_hot(y_true, depth=num_classes)
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1)
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), 1))
# tf.one_hot(y_train[:5], depth=10)

def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def logistic_run_optimization(x, y):
    with tf.GradientTape() as tape:
        y_pred = logistic_multiclass(x=x, W=W, b=b)
        loss = cross_entropy(y_pred=y_pred, y_true=y)
    
    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))
    accuracy_score = accuracy(y_pred=y_pred, y_true=y)

    return y_pred, loss, accuracy_score




batch_size = 200    # batch_size
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.shuffle(60000).batch(batch_size)  #.prefetch(1)        
    # prefetch : 미리 다음의 배치를 준비하고 있어라

# list_data = list(train_data.as_numpy_iterator())
# n_data = len(list_data)
# print(len(list_data))
# print(type(list_data[0]))


# img, label = list_data[0]
# print(img.shape)
# print(label)

n_epoch = 4
display_step = 50

for epoch in range(1, n_epoch+1):
    for step, (batch_x, batch_y) in enumerate(train_data, start=1):
        logistic_run_optimization(batch_x, batch_y)

        if step % display_step == 0:
            pred = logistic_multiclass(batch_x, W, b)
            loss = cross_entropy(pred, batch_y)
            acc = accuracy(pred, batch_y)

            print(' .epoch:', epoch, ' .step:', step, ' .loss:', loss.numpy(), ' .acurracy:', acc.numpy())



pred_test = logistic_multiclass(x_test, W, b)
print('Test_Accuracy: ', accuracy(pred_test, y_test).numpy())






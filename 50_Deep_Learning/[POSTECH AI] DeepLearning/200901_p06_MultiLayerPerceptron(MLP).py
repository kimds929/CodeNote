import random

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import matplotlib.pylab as plt

from IPython.display import clear_output
import time


# Backpropagation ==============================================================
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# y = relu(m·x + b)

# activation function
def relu(x):
    if x > 0.0:
        return x
    else:
        return 0.0

def square_error(m, b, data):
    totalError = 0
    for i in range(0, len(data)):
        x = data[i, 0]
        y = data[i, 1]
        totalError += (y - relu(x * m + b))**2
    return totalError / float(len(data))

def step_gradient(m_current, b_current, data, learning_rate):
    b_gradient = 0
    m_gradient = 0

    N = float(len(data))

    for i in range(len(data)):
        x = data[i, 0]
        y = data[i, 1]
        dL_df1 = 1.0
        dL_df2 = dL_df1 * 2.0 * (y - relu(m_current * x + b_current))
        dL_df3 = dL_df2 * 1.0
        dL_df4 = dL_df3 * (-1.0)

        if (m_current * x) + b_current > 0.0:
            dL_df5 = dL_df4 * 1.0
        else:
            dL_df5 = dL_df4 * 0.0
        dL_df6 = dL_df5 * 1.0

        b_gradient += (1/N) * (dL_df5 * 1.0)
        m_gradient += (1/N) * (dL_df6 * x)
    
    new_b = b_current - learning_rate * b_gradient
    new_m = m_current - learning_rate * m_gradient

    return new_m, new_b

def gradient_descent_runner(data, starting_m, starting_b, learning_rate, num_iterations):
    m = starting_m
    b = starting_b

    for i in range(num_iterations):
        m, b = step_gradient(m, b, np.array(data), learning_rate)

    return m, b


data = 50 + 15 * np.random.randn(100,2)
print(data)

learning_rate = 0.0001
initial_b = np.random.randn()
initial_m = np.random.randn()
num_iterations = 50000

print(initial_b, initial_m)
step_gradient(initial_m, initial_b, np.array(data), learning_rate)

print(f'Start Error: {square_error(initial_m, initial_b, data)}')
m, b = gradient_descent_runner(data, initial_m, initial_b, learning_rate, num_iterations)
print(f'Start Error: {square_error(m, b, data)}')








# Multilayer Perceptron =====================================================================
import tensorflow as tf
from tensorflow.keras import layers, Model


# Dataset Load, Preprocessing
tf.random.set_seed(1)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)

num_classes = np.max(y_train)+1
num_features = x_train.shape[1] * x_train.shape[2]

x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
x_train, x_test = x_train/255, x_test/255


# random_normal = tf.initializers.RandomNormal()
# # tf.initializers.RandomNormal()([3,2])

# weight = {
#     'h1': tf.Variable(random_normal([num_features, 128])),
#     'h2': tf.Variable(random_normal([128, 256])),
#     'out': tf.Variable(random_normal([256, num_classes]))
# }

# bias = {
#     'b1': tf.Variable(tf.zeros([128])),
#     'b2': tf.Variable(tf.zeros([256])),
#     'out': tf.Variable(tf.zeros([num_classes]))
# }

# def neural_net(x):
#     Layer_1 = tf.add(x @ weight['h1'], bias['b1'])
#     Layer_1 = tf.nn.relu(Layer_1)

#     Layer_2 = tf.add(x @ weight['h2'], bias['b2'])
#     Layer_2 = tf.nn.relu(Layer_2)

#     out_Layer = tf.add(x @ weight['out'], bias['out'])
#     return out_Layer
#     # return tf.nn.softmax(out_Layer)


class NeuralNet(Model):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = layers.Dense(128, activation=tf.nn.relu)
        self.fc2 = layers.Dense(256, activation=tf.nn.relu)
        self.out = layers.Dense(num_classes)
    
    def call(self, x, is_training=False):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)

        if is_training == False:
            x = tf.nn.softmax(x)
        return x

def cross_entropy_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return tf.reduce_mean(loss)

def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


lr = 0.0003
optimizer = tf.optimizers.SGD(lr)

neural_net = NeuralNet()

# cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy()
# tf.nn.sparse_softmax_cross_entropy_with_logits()


# @tf.function
def run_optimization(x, y):
    with tf.GradientTape() as tape:
        pred = neural_net(x, is_training=True)
        loss = cross_entropy_loss(pred, y)

    train_variables = neural_net.trainable_variables
    gradients = tape.gradient(loss, train_variables)
    optimizer.apply_gradients(zip(gradients, train_variables))


batch_size = 200
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(batch_size)

# n_epoch = 76
n_epoch = 8
display_epoch = 4

for epoch in range(1, n_epoch+1):
    for step, (batch_x, batch_y) in enumerate(train_data, start=1):
        run_optimization(batch_x, batch_y)

    if epoch % display_epoch == 0:
        pred = neural_net(batch_x, is_training=True)
        loss = cross_entropy_loss(pred, batch_y)
        print(f'epoch: {epoch}, loss:{loss}')




pred = neural_net(x_test, is_training=False)
print(f'Test Accuracy: {accuracy(pred, y_test)}')


n_images = 5
test_images = x_test[:n_images]
predictions = neural_net(test_images)

plt.figure(figsize=(10,6))
for i in range(n_images):
    plt.subplot(2,3,i+1)
    plt.title(f'Model prediction : {np.argmax(predictions.numpy()[i])}')
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray_r')
plt.show()



pred.shape
batch_y.shape

tf.cast(batch_y, tf.int64)
batch_y
tf.keras.losses.SparseCategoricalCrossentropy(pred, tf.cast(batch_y, tf.int64))
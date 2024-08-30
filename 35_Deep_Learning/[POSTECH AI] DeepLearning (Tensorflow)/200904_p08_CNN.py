import random

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

import tensorflow as tf
from tensorflow.keras import Model, layers
# import torch

import sys
import copy
from IPython.display import clear_output
import time


random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Data Preprocessing ------------------------------------
(x_train_val, y_train_val), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

num_classes = 10
num_features = 784

x_train_val = x_train_val.astype('float32')/255
x_test = x_test.astype('float32')/255

y_train_val = y_train_val.astype('float32')
y_test = y_test.astype('float32')

# train_validation_split ------------------------------------
indices = np.random.permutation(x_train_val.shape[0])
train_indices = indices[:-10000]
valid_indices = indices[-10000:]

x_train = x_train_val[train_indices]
x_valid = x_train_val[valid_indices]
y_train = y_train_val[train_indices]
y_valid = y_train_val[valid_indices]

# print(indices[:10])     # indice Sample
print(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape)


# Dataset ----------------------------------------------------------------------------
batch_size = 200
train_data_01 = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(batch_size).prefetch(1)
valid_data_01 = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).shuffle(60000).batch(batch_size).prefetch(1)
test_data_01 = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(60000).batch(batch_size).prefetch(1)


# Essential Function ------------------------------------------------------------------
# Loss_Function
def cross_entropy_loss(x, y):
    y = tf.cast(y, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    return tf.reduce_mean(loss)

# Metrics
def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

# Batch Accuracy
def accuracy_batch(model, test_data):
    acc = 0
    for step, (batch_x, batch_y) in enumerate(test_data, 1):
        pred = model(batch_x, is_training=False)
        acc += accuracy(pred, batch_y)
    acc = acc / step * 100
    return acc


# Training Model Overall Run
def train_model(train_data, model, lr, epochs, 
                print_loss=True, plot_graph=True):
    step_l = []
    loss_l = []

    optimizer = tf.optimizers.SGD(lr, momentum=0.9)
    n_batch = len(list(train_data))
    # Training Run
    # @tf.function
    # def train_running(X, y, model, loss, optimizer):
    #     with tf. GradientTape() as Tape:
    #         y_pred = model(X, is_training=True)
    #         model_loss = loss(y_pred, y)

    #     train_weights = model.trainable_variables
    #     gradients = Tape.gradient(model_loss, train_weights)
    #     optimizer.apply_gradients(zip(gradients, train_weights))
    #     return model_loss

    for epoch in range(1, epochs+1):

        running_loss = 0.0
        for step, (batch_x, batch_y) in enumerate(train_data, 1):

            # model_loss = train_running(batch_x, batch_y, model, cross_entropy_loss, optimizer)
            with tf. GradientTape() as Tape:
                y_pred = model(batch_x, is_training=True)
                model_loss = cross_entropy_loss(y_pred, batch_y)

            train_weights = model.trainable_variables
            gradients = Tape.gradient(model_loss, train_weights)
            optimizer.apply_gradients(zip(gradients, train_weights))

            running_loss += model_loss.numpy()

            if plot_graph:
                if step % 10 == 0:
                    step_l.append(epoch * n_batch + step)
                    loss_l.append(running_loss/10)
                    running_loss = 0.0
        
        if print_loss:
            print(f'epoch: {epoch},  loss: {model_loss.numpy()}')

    if plot_graph:
        plt.plot(step_l, loss_l)
        plt.show()
    
    return model


# Early_Stoping
def early_stopping(train_data, valid_data, model, lr):
    el = []     # epoch_list
    vll = []    # error_list

    p = 4
    i = 0   # While Loop n_iter (epoch)
    j = 0
    v = sys.float_info.max      # 시스템 최대값
    i_s = i
    # model_s = copy.deepcopy(model)
    weigt_s = model.weights

    while j < p:
        train_model(train_data, model, lr, 1, print_loss=False, plot_graph=False)
        acc = 0
        for step, (batch_x, batch_y) in enumerate(valid_data, 1):
            pred = model(batch_x, is_training=False)
            acc += accuracy(pred, batch_y)

        acc = 100. * acc/step
        error = 100. - acc

        i = i+1
        temp_v = error.numpy()

        el.append(i)
        vll.append(error)

        if temp_v < v:
            j = 0
            model_s = copy.deepcopy(model)
            # weigt_s = copy.deepcopy(model.weights)
            i_s = i
            v = temp_v
        else:
            j = j+1
        print(f'epoch reached: {i}, val_error={error.numpy()}, smallest_error: {v}')

    plt.plot(el, vll)
    plt.show()
    print(f'best_epoch: {i_s}')

    return model_s, i_s
    # return weigt_s, i_s







# CNN (Convolution Neural Network) ===================================================================

# CNN Modeling
class CNN(Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1,1), 
                                        padding='same', activation=tf.nn.relu)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), 
                                        padding='same', activation=tf.nn.relu)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)

        self.flatten3 = tf.keras.layers.Flatten()

        self.dense4 = tf.keras.layers.Dense(units=128)
        self.out = tf.keras.layers.Dense(units=num_classes)
    
    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 28, 28, 1])
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = self.flatten3(x)

        x = self.dense4(x)
        x = self.out(x)
        if not is_training:
            x = tf.nn.softmax(x)
        return x

# ?tf.keras.layers.Conv2D
# ?tf.keras.layers.MaxPool2D
# ?tf.keras.layers.Dense
# tf.keras.layers.ZeroPadding1D?

cnn = CNN()
cnn, cnn_best_epoch = early_stopping(train_data_01, valid_data_01, cnn, 0.001)

cnn_acc = accuracy_batch(cnn, test_data_01).numpy()
print(f'Test_Accuracy_of_CNN: {cnn_acc}')



# ※ padding option ****
input_shape = (4,28,28,1)
x = tf.random.normal(input_shape)
y1 = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=2,
                    padding='same', activation='relu')(x)
y2 = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=3,
                    padding='same', activation='relu')(x)
y3 = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=4,
                    padding='same', activation='relu')(x)
y4 = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=5,
                    padding='same', activation='relu')(x)
print(y1.shape)
print(y2.shape)
print(y3.shape)
print(y4.shape)

y5 = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=1,
                    padding=[[0,0], [1,1], [1,1], [0,0]], activation='relu')(x)
                        # batch, width(left,right) , height(top, bottom), channel
y6 = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=1,
                    padding=[[0,0], [2,2], [2,2], [0,0]], activation='relu')(x)
                        # batch, width(left,right) , height(top, bottom), channel
print(y5.shape)
print(y6.shape)





# ResNet ===================================================================
# Layer가 깊어질수록 Optimization이 잘 안됨

# ResNet Inner Unit
class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, strides=strides,
                    kernel_size=3, padding='same', kernel_initializer=tf.keras.initializers.he_normal())
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.ac1 = tf.keras.layers.Activation('relu')

        self.conv2 = tf.keras.layers.Conv2D(filters=filters, strides=1,
                    kernel_size=3, padding='same', kernel_initializer=tf.keras.initializers.he_normal())
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.ac2 = tf.keras.layers.Activation('relu')

        if strides == 1:
            self.downsample = lambda x: x
        else:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filters, strides=strides, kernel_size=1, padding='same'))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        
    def call(self, x, is_training=False):
        residual =self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x, training=is_training)
        x = self.ac1(x)
        x = self.conv2(x)
        x = self.bn2(x, training=is_training)
        x = self.ac2(tf.keras.layers.add([residual, x]))
        return x

# ResNet Block
def make_basic_block_layer(filters, blocks, strides=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filters=filters, strides=strides))

    for i in range(1, blocks):
        res_block.add(BasicBlock(filters=filters, strides=1))
    
    return res_block

# ResNet Modeling
class ResNet18(Model):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same',
                                        kernel_initializer=tf.keras.initializers.he_normal())
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.ac1 = tf.keras.layers.Activation(tf.nn.relu)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')

        self.layer2 = make_basic_block_layer(filters=64, blocks=2)
        self.layer3 = make_basic_block_layer(filters=128, blocks=2, strides=2)
        self.layer4 = make_basic_block_layer(filters=256, blocks=2, strides=2)
        self.layer5 = make_basic_block_layer(filters=512, blocks=2, strides=2)

        self.pool6 = tf.keras.layers.GlobalAveragePooling2D()   # 모든데이터를 1개짜리로 pooling

        self.out = tf.keras.layers.Dense(num_classes)

    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 28, 28, 1])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.ac1(x)
        x = self.pool1(x, training=is_training)

        x = self.layer2(x, training=is_training)
        x = self.layer3(x, training=is_training)
        x = self.layer4(x, training=is_training)
        x = self.layer5(x, training=is_training)

        x = self.pool6(x)

        x = self.out(x)
        if not is_training:
            x = tf.nn.softmax(x)
        return x


resnet18 = ResNet18()
resnet18 = train_model(train_data_01, resnet18, 0.0001, 5)

resnet18_acc = accuracy_batch(resnet18, test_data_01).numpy()
print(f'Test_Accuracy_of_CNN: {resnet18_acc}')





















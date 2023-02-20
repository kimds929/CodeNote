import random

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

import tensorflow as tf
# from tensorflow.keras import Model, layers
# import torch

import scipy.io
import skimage
import skimage.transform
# pip install scikit-image==0.16.2
# conda install scikit-image
# pip install scipy==1.4.1
# pip install --upgrade scikit-image

import os
import sys
import copy
from IPython.display import clear_output
import time
from ipywidgets import interact

# Data Download
# !wget -O attention_train.mat https://vo.la/P5KIl
# !wget -O attention_test.mat https://vo.la/pLjvp

# os.getcwd()
# sorted(os.listdir())
mat_data_train = scipy.io.loadmat('attention_train.mat')
mat_data_test = scipy.io.loadmat('attention_test.mat')

train_x = mat_data_train['X_train']
train_y = mat_data_train['Y_train']

test_x = mat_data_test['X_test']
test_y = mat_data_test['Y_test']

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)


img = train_x[134,:,:]
plt.imshow(img, cmap='gray')
plt.show()

train_x = train_x[..., np.newaxis]
test_x = test_x[..., np.newaxis]
image_shape = train_x.shape[1:]


# LSTM_function
def LSTM_cell(c_prev, h_prev, x_input, Wi, Wf, Wo, Wg, bi, bf, bo, bg):
    h_x_concat = tf.concat([h_prev, x_input], axis=1)
    forget_gate = tf.sigmoid(h_x_concat @ Wf + bf)
    input_gate = tf.sigmoid(h_x_concat @ Wi + bi)
    out_gate = tf.sigmoid(h_x_concat @ Wo + bo)
    gate = tf.tanh(h_x_concat @ Wg + bg)

    cellState_t = forget_gate * c_prev + input_gate * gate
    hiddenState_t = out_gate * tf.tanh(cellState_t)

    return cellState_t, hiddenState_t

def soft_attention(h_prev, feature_map, Wa, Wh):
    m_list = []

    # feature_map :  a list of size 196 which of element has shape (batch, 512_feature)
    for feature_vector in feature_map:
        m_list.append(feature_vector @ Wa + h_prev @ Wh)
    m_concat = tf.concat(m_list, axis=1)        # (batch_size, 
    alpha = tf.nn.softmax(m_concat) #
    z_list = []

    for i in range(len(feature_map)):
        z_list.append(feature_map[i] * alpha[:, i, tf.newaxis])
    z_stack = tf.stack(z_list, axis=2)  # (batch, 512, 196)
    z = tf.reduce_sum(z_stack, axis=2)  # 196개의 위치를 모두 더해준다

    return alpha, z

# tf.concat vs tf.stack
# t1 = [[1,2,3], [4,5,6]]
# t2 = [[7,8,9], [10,11,12]]

# tf.reshape(t1, (1,6))
# tf.concat(t1, axis=0)
# tf.constant(t1)
# tf.stack(t1, axis=0)

# Model Build
class LSTM_attention_model(tf.keras.Model):
    def __init__(self, h_dim, num_lstm, num_label):
        super(LSTM_attention_model, self).__init__()
        self.h_dim = h_dim
        self.num_lstm = num_lstm        # LSTM 몇번 반복?

        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2,2), padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(2,2), padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(2,2), padding='same', activation='relu')

        # LSTM variables initialize
        self.Wi = tf.Variable(tf.initializers.GlorotUniform()(shape=(h_dim + 512, h_dim)))
        self.Wf = tf.Variable(tf.initializers.GlorotUniform()(shape=(h_dim + 512, h_dim)))
        self.Wo = tf.Variable(tf.initializers.GlorotUniform()(shape=(h_dim + 512, h_dim)))
        self.Wg = tf.Variable(tf.initializers.GlorotUniform()(shape=(h_dim + 512, h_dim)))
        
        self.bi = tf.Variable(tf.constant_initializer(1.)(shape=(h_dim, )), dtype=tf.float32)
        self.bf = tf.Variable(tf.constant_initializer(1.)(shape=(h_dim, )), dtype=tf.float32)
        self.bo = tf.Variable(tf.constant_initializer(1.)(shape=(h_dim, )), dtype=tf.float32)
        self.bg = tf.Variable(tf.constant_initializer(1.)(shape=(h_dim, )), dtype=tf.float32)

        # Attention Variables
        self.Wa = tf.Variable(tf.initializers.GlorotUniform()(shape=(512, 1)))
        self.Wh = tf.Variable(tf.initializers.GlorotUniform()(shape=(h_dim, 1)))

        # Dense layer
        self.fc = tf.keras.layers.Dense(num_label)

        # alpha tracking
        self.alpha = None

    def call(self, x_batch):
        # x_batch: (batch, 112, 112, 1)
        batch_size = x_batch.shape[0]

        conv1 = self.conv1(x_batch) # (batch, 112/2, 112/2, 64)
        conv2 = self.conv2(conv1)   # (batch, 112/4, 112/4, 256)
        conv3 = self.conv3(conv2)   # (batch, 112/8, 112/8, 512)

        conv_size = conv3.shape[1]  
        conv_flat = tf.reshape(conv3, [-1, conv_size * conv_size, 512]) # (batch, 14*14=196, 512)
        conv_unstack = tf.unstack(conv_flat, axis=1)                    # list(size: 196), element: (batch, 512)

        c = tf.constant(0, dtype=tf.float32, shape=(batch_size, self.h_dim)) # (batch, h_dim)
        h = tf.constant(0, dtype=tf.float32, shape=(batch_size, self.h_dim)) # (batch, h_dim)
        
        for i in range(self.num_lstm):
            alpha, z = soft_attention(h_prev=h, feature_map=conv_unstack, 
                    Wa=self.Wa, Wh=self.Wh)
            c, h = LSTM_cell(c_prev=c, h_prev=h, x_input=z,
                    Wi=self.Wi, Wf=self.Wf, Wo=self.Wo, Wg=self.Wg, 
                    bi=self.bi, bf=self.bf, bo=self.bo, bg=self.bg)
        self.alpha = alpha

        output = self.fc(h)
        return output


# soft_attention(h_prev, feature_map, Wa, Wh)
# LSTM_cell(c_prev, h_prev, x_input, Wi, Wf, Wo, Wg, bi, bf, bo, bg)


model = LSTM_attention_model(h_dim=30, num_lstm=4, num_label=10)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=100, epochs=3, verbose=1)
# model.fit(train_x[:10000], train_y[:10000], batch_size=100, epochs=3, verbose=1)
# model(train_x[:10,:,:,:])


# evaluate model
model.evaluate(test_x, test_y, batch_size=100)


def plot(id):
    pred = model(test_x[id: id+1])
    alpha = model.alpha
    alpha_size = int(np.sqrt(alpha.shape[1]))
    alpha_reshape = np.reshape(alpha, (alpha_size, alpha_size))
    alpha_resize = skimage.transform.pyramid_expand(alpha_reshape.copy(), upscale=16, sigma=20)

    f1, ax = plt.subplots(1,2)
    ax[0].imshow(alpha_resize, cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('Attention Heatmap')
    ax[1].imshow(test_x[id].squeeze(), cmap='gray')
    ax[1].axis('off')
    ax[1].set_title(f'Prediction: {str(np.argmax(pred))} / Label: {str(np.argmax(test_y[id]))}')

plot(330)


interact(plot, id=(1,100))




# -------------------------------------------------------------------
# pip install ipywidgets
# !jupyter nbextension enable --py widgetsnbextension
from ipywidgets import interact

def f(x):
    return x*x
interact(f, x=(-30, 30, 1))

def f2(x, y):
    return x*x + y*y
    
interact(f, x=(-30, 30, 1))
interact(f2, x=(-30, 30, 1), y=(-30,30,1))


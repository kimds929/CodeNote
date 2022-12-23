import numpy as np
import pandas as pd
import matplotlib.pylab as plt

import tensorflow as tf
from scipy import io

import math
import time
from six.moves import cPickle

from images import *

import skimage
import skimage.transform

images, labels = load_all()

print(images.shape, labels.shape)
# np.min(images), np.max(images)

class Attention(tf.keras.Model):
    def __init__(self):
        super(Attention, self).__init__()
        
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 3, 2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(32, 3, 2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, 3, 2, padding='same', activation='relu'),
        ])  # 13 * 25 * 64

        # self.w_k = self.add_weight(shape)
        self.w_q = self.add_weight(shape=(64, 1), initializer='random_normal', trainable=True)
        self.dense = tf.keras.layers.Dense(10, activation='sigmoid')
    
    def call(self, x, training=False):
        x = self.conv(x)
        x = tf.reshape(x, (-1, 13*25, 64))

        score = tf.nn.softmax(tf.matmul(x, self.w_q), axis=1)
        self.score = score

        x = tf.reduce_sum(x * score, axis=1)
        x = self.dense(x)
        return x

attention01 = Attention()
# pred = attention01(images[:10,:,:,:])
# (pred >= 0.5).numpy().astype('int')
attention01.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
result01 = attention01.fit(images, labels, epochs=10, batch_size=64)


attention01(images[25][np.newaxis,...])
score_map = attention01.score.numpy().reshape(13,25)

plt.imshow(images[25])
plt.imshow(score_map)








class MultiAttention(tf.keras.Model):
    def __init__(self):
        super(MultiAttention, self).__init__()
        
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 3, 2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(32, 3, 2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, 3, 2, padding='same', activation='relu'),
        ])  # 13 * 25 * 64

        # self.w_k = self.add_weight(shape)
        self.w_q = self.add_weight(shape=(64, 10), initializer='random_normal', trainable=True)
        # self.dense = tf.keras.layers.Dense(10, activation='sigmoid')
        
        self.dense1 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.dense3 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.dense4 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.dense5 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.dense6 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.dense7 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.dense8 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.dense9 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.dense10 = tf.keras.layers.Dense(1, activation='sigmoid')

    
    def call(self, x, training=False):
        x = self.conv(x)
        x = tf.reshape(x, (-1, 13*25, 64))

        score = tf.nn.softmax(tf.matmul(x, self.w_q), axis=1)[..., tf.newaxis]
        self.score = score
        
        x = tf.expand_dims(x, axis=-2)
        x = tf.reduce_sum(x * score, axis=1)        # 10, 64
        # x = tf.keras.layers.Flatten()(x)
        # x = self.dense(x)
        x1 = self.dense1(x[:,0,:])
        x2 = self.dense2(x[:,1,:])
        x3 = self.dense3(x[:,2,:])
        x4 = self.dense4(x[:,3,:])
        x5 = self.dense5(x[:,4,:])
        x6 = self.dense6(x[:,5,:])
        x7 = self.dense7(x[:,6,:])
        x8 = self.dense8(x[:,7,:])
        x9 = self.dense9(x[:,8,:])
        x10 = self.dense10(x[:,9,:])

        x = tf.concat([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], axis=-1)
        return x
        

attention_multi = MultiAttention()
attention_multi.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.05), metrics=['accuracy'])
result_multi = attention_multi.fit(images, labels, epochs=10, batch_size=64)


attention_multi(images[10][np.newaxis,...])
score_maps = attention_multi.score.numpy()

plt.imshow(images[10])
plt.show()


for i in range(10):
    plt.title(i)
    plt.imshow(score_maps[0,:,i,0].reshape(13, 25))
    plt.show()









class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 3, 2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(32, 3, 2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, 3, 2, padding='same', activation='relu'),
        ])  # 13 * 25 * 64

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='sigmoid')
        
    def call(self, x, training=False):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

cnn = CNN()
cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
result_cnn = cnn.fit(images, labels, epochs=10, batch_size=64)

(cnn(images[10][np.newaxis,...]).numpy() > 0.5).astype(int)

plt.imshow(images[10])
plt.show()
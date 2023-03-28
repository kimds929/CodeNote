import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
# import tensorflow_hub as tf_hub

import cv2
import tqdm

(m_x_train, m_y_train), (m_x_test, m_y_test) = tf.keras.datasets.mnist.load_data()
(c_x_train, c_y_train), (c_x_test, c_y_test) = tf.keras.datasets.cifar10.load_data()
print('mnist_shape: ', m_x_train.shape, m_y_train.shape, m_x_test.shape, m_y_test.shape)
print('cifar10_shape: ', c_x_train.shape, c_y_train.shape, c_x_test.shape, c_y_test.shape)

m_y_train = tf.one_hot(m_y_train, 10)
m_y_test = tf.one_hot(m_y_test, 10)
c_y_train = tf.one_hot(tf.squeeze(c_y_train), 10)
c_y_test = tf.one_hot(tf.squeeze(c_y_test), 10)

print('mnist_y_shape: ', m_y_train.shape, m_y_test.shape)
print('cifar10_y_shape: ',c_y_train.shape, c_y_test.shape)

m_x_train_up = []
for img in tqdm.tqdm_notebook(m_x_train):
    m_x_train_up.append(
        # cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_LINEAR)
        cv2.resize(img, dsize=(96, 96), interpolation=cv2.INTER_LINEAR)
        # INTER_LINEAR for upsampling, INTER_AREA for downsampling
    )

m_x_test_up = []
for img in tqdm.tqdm_notebook(m_x_test):
    m_x_test_up.append(
        # cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_LINEAR)
        cv2.resize(img, dsize=(96, 96), interpolation=cv2.INTER_LINEAR)
        # INTER_LINEAR for upsampling, INTER_AREA for downsampling
    )

m_x_train = np.array(m_x_train_up).astype('float32') / 255.0
m_x_train = np.stack([m_x_train, m_x_train, m_x_train], axis=-1)

m_x_test = np.array(m_x_test_up).astype('float32') / 255.0
m_x_test = np.stack([m_x_test, m_x_test, m_x_test], axis=-1)

c_x_train = c_x_train.astype('float32') / 255.0
c_x_test = c_x_test.astype('float32') / 255.0
print('mnist_shape: ', m_x_train.shape, m_y_train.shape, m_x_test.shape, m_y_test.shape)
print('cifar10_shape: ', c_x_train.shape, c_y_train.shape, c_x_test.shape, c_y_test.shape)

# plt.imshow(m_x_train[0], cmap='gray')

class TransferModel(tf.keras.Model):
    def __init__(self):
        super(TransferModel, self).__init__()
        self.weight_trainable = True

        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2,2), padding='same', 
                        use_bias=False, input_shape=(32, 32, 3))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2,2), padding='same', 
                        use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(2,2), padding='same', 
                        use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.dense1 = tf.keras.layers.Dense(50)
        self.dense2 = tf.keras.layers.Dense(10)
    
    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.keras.layers.ReLU()(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.keras.layers.ReLU()(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Flatten()(x)

        x = self.dense1(x)
        x = self.dense2(x)
        return x

    def freeze_except_final(self):
        self.weight_trainable = not self.weight_trainable

        self.conv1.trainable = self.weight_trainable    # 학습 X
        self.bn1.trainable = self.weight_trainable      # 학습 X
        self.conv2.trainable = self.weight_trainable    # 학습 X
        self.bn2.trainable = self.weight_trainable      # 학습 X
        self.conv3.trainable = self.weight_trainable    # 학습 X
        self.bn3.trainable = self.weight_trainable      # 학습 X
        self.dense1.trainable = self.weight_trainable      # 학습 X

        print(f'trainableL {self.weight_trainable}')


model_m = TransferModel()
model_m_small = TransferModel()
model_m_c = TransferModel()

# model_m.trainable
# model_m.conv1.trainable
# model_m.freeze_except_final()

# compile
optimizer = tf.keras.optimizers.Adam(1e-3)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

model_m.compile(optimizer, loss, metrics=['accuracy'])
model_m_small.compile(optimizer, loss, metrics=['accuracy'])
model_m_c.compile(optimizer, loss, metrics=['accuracy'])

# fit
model_m.fit(m_x_train, m_y_train, batch_size=30, epochs=4)
model_m.evaluate(m_x_test, m_y_test)

model_m_small.fit(m_x_train[:50], m_y_train[:50], batch_size=30, epochs=8)
model_m_small.evaluate(m_x_test, m_y_test)

model_m_c.fit(c_x_train, c_y_train, batch_size=30, epochs=10, validation_split=0.2)
model_m_c.evaluate(c_x_test, c_y_test)


# 마지막 cifar10으로 학습시킨 모델로 mnist에 적용
model_m_c.summary()

model_m_c.freeze_except_final()
model_m_c.weight_trainable
model_m_c.summary()

model_m_c.fit(c_x_train[:50], c_y_train[:50], batch_size=30, epochs=10, validation_split=0.2)
model_m_c.evaluate(c_x_test, c_y_test)






# 미리 학습된 모델 불러오기
# trained_model_url = r'D:\Python\★★Python_POSTECH_AI\Postech_AI 5) Machine_Learning & Deep Learning\dataset\imagenet_mobilenet_v4'
# trained_model = tf.keras.models.load_model(filepath=trained_model_url)
# trained_model.summary()


pretrained_model = tf.keras.Sequential([
    # hub.KerasLayer('https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/4', input_shape=(96,96,3), trainable=False),
    # trained_model,
    tf.keras.layers.Dense(10)
])
pretrained_model.summary()
# pretrained_model.build([None, 96, 96, 3])     # Input_Shape 지정하는것과 효과가 동일


# compile
optimizer_pt = tf.keras.optimizers.Adam(1e-3)
loss_pt = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
pretrained_model.compile(optimizer_pt, loss_pt, metrics=['accuracy'])
pretrained_model.fit(m_x_train, m_y_train, batch_size=30, epochs=4)

pretrained_model.evaluate(m_x_test, m_y_test)


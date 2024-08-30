import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import torch

from sklearn.preprocessing import OneHotEncoder

import sys
sys.path.append(r'C:\Users\Admin\Desktop\DataScience\Code0) DS_Module')    # 모듈 경로 추가
from DS_DataFrame import DS_DF_Summary, DS_OneHotEncoder, DS_LabelEncoder
from DS_Image import Image_load, LoadImage_from_folder, EvaluateClassifier, show_image
# absolute_path = r'D:\Python\강의) [FastCampus] 딥러닝 올인원 패키지\dataset\'
# absolute_path = r'D:\Python\강의) [FastCampus] 딥러닝 올인원 패키지\source\'


# [ CNN ] =========================================
#  . transition = invariant
#  . scale = variant
#  . rotation = variant     ( steerable CNNs )
#  . 해상도 = variant (해상도에 따라 네트워크 구조를 다르게 해야함)
# ==================================================


# Tensorflow 2.0 basic --------------------------------------------------------------------
# 데이터불러오기

# MIST set불러오기
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

image = x_train[0]
plt.imshow(image, 'gray_r')
plt.show()


# 차원늘리기
exp_x_train = np.expand_dims(x_train, -1)
# x_train[..., np.newaxis]
exp_x_test = np.expand_dims(x_test, -1)

print(exp_x_train.shape, exp_x_test.shape)

# plt.imshow(exp_x_train[0], 'gray_r')      # error
plt.imshow(np.squeeze(exp_x_train[0]), 'gray_r')
plt.show()


# Label Dataset 확인하기
y_train.shape
y_train[0]

# image
plt.title(y_train[0], fontsize=20)
plt.imshow(np.squeeze(exp_x_train[0]), 'gray_r')
plt.show()


# OneHot Encoding
print(tf.keras.utils.to_categorical(0, num_classes=10))
print(tf.keras.utils.to_categorical(1,  num_classes=10))

label_ohe0 = tf.keras.utils.to_categorical(y_train[0], num_classes=10)
# y_df = pd.DataFrame(y_train.astype(str), columns=['y'])
# ohe = OneHotEncoder(sparse=False)
# ohe.fit_transform(y_df)
# DS_ohe = DS_OneHotEncoder(sparse=False)
# DS_ohe.fit_transform(y_df).to_numpy()

# image
plt.title(label_ohe0, fontsize=20)
plt.imshow(np.squeeze(exp_x_train[0]), 'gray_r')
plt.show()


# Convolution Layer ------------------------------------------------------------------
# filters: layer에서 나갈 때 몇 개의 filter를 만들 것인지 (a.k.a weights, filters, channels)  
# kernel_size: filter(Weight)의 사이즈  
# strides: 몇 개의 pixel을 skip 하면서 훑어지나갈 것인지 (사이즈에도 영향을 줌)  
# padding: zero padding을 만들 것인지. VALID는 Padding이 없고, SAME은 Padding이 있음 (사이즈에도 영향을 줌)  
# activation: Activation Function을 만들것인지. 당장 설정 안해도 Layer층을 따로 만들 수 있음

# Convolution Layer
tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')
tf.keras.layers.Conv2D(3, 3, 1, 'same')

# ?tf.keras.layers.Conv2D
# tf.keras.layers.Conv2D(
#     filters,
#     kernel_size,
#     strides=(1, 1),
#     padding='valid',
#     data_format=None,
#     dilation_rate=(1, 1),
#     activation=None,
#     use_bias=True,
#     kernel_initializer='glorot_uniform',
#     bias_initializer='zeros',
#     kernel_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     bias_constraint=None,
#     **kwargs,
# )


# Visualization  ------------------------------------------------------------------
image = x_train[0][tf.newaxis, ..., tf.newaxis]
image.dtype     # dtype('uint8')
image.shape     # (1, 28, 28, 1)
image = tf.cast(image, tf.float32)
plt.imshow(tf.squeeze(image), 'gray_r')


# Convolution_Transform1 ***
conv_layer1 = tf.keras.layers.Conv2D(5, 3, 1, 'same')   # filters=5, kernel_size=(3,3)
conv_output1 = conv_layer1(image)
print(conv_output1.shape)     # TensorShape([1, 28, 28, 5])

# 통과 후 각 filter node의 image
plt.figure(figsize=(15,7))
plt.subplot(511)
plt.imshow(conv_output1[0,:,:,0], 'gray_r')
plt.colorbar()
plt.subplot(512)
plt.imshow(conv_output1[0,:,:,1], 'gray_r')
plt.colorbar()
plt.subplot(513)
plt.imshow(conv_output1[0,:,:,2], 'gray_r')
plt.colorbar()
plt.subplot(514)
plt.imshow(conv_output1[0,:,:,3], 'gray_r')
plt.colorbar()
plt.subplot(515)
plt.imshow(conv_output1[0,:,:,4], 'gray_r')
plt.colorbar()
plt.show()

# 통과 전 후 픽셀값 변화
print(f'min: {np.min(image)} ~ max: {np.max(image)}')
print(f'min: {np.min(conv_output1)} ~ max: {np.max(conv_output1)}')




# Convolution_Transform2 ***
conv_layer2 = tf.keras.layers.Conv2D(3, 3, 1, 'same')   # filters=3, kernel_size=(3,3)
conv_output2 = conv_layer2(image)
print(conv_output2.shape)     # TensorShape([1, 28, 28, 3])

# 통과 후 각 filter node의 image
plt.figure(figsize=(12,7))
plt.subplot(311)
plt.imshow(conv_output2[0,:,:,0], 'gray_r')
plt.colorbar()
plt.subplot(312)
plt.imshow(conv_output2[0,:,:,1], 'gray_r')
plt.colorbar()
plt.subplot(313)
plt.imshow(conv_output2[0,:,:,2], 'gray_r')
plt.colorbar()
plt.show()

# 통과 전 후 픽셀값 변화
print(f'min: {np.min(image)} ~ max: {np.max(image)}')
print(f'min: {np.min(conv_output2)} ~ max: {np.max(conv_output2)}')

# 통과 전 후 픽셀값 변화 histogram
plt.figure(figsize=(10,3))
plt.subplot(121)
plt.hist(image.numpy().ravel())
# plt.yscale('log')
plt.subplot(122)
plt.hist(conv_output2.numpy().ravel())
# plt.yscale('log')
plt.show()


# weigt 확인
weight2 = conv_layer2.get_weights()
len(weight2)    # [weight, bias]
print(weight2[0].shape, weight2[1].shape)
print(weight2)


# filter(weight) image 확인
plt.figure(figsize=(12,7))
plt.subplot(311)
plt.imshow(weight2[0][:,:,0,0], 'gray_r')
plt.colorbar()
plt.subplot(312)
plt.imshow(weight2[0][:,:,0,1], 'gray_r')
plt.colorbar()
plt.subplot(313)
plt.imshow(weight2[0][:,:,0,2], 'gray_r')
plt.colorbar()
plt.show()


# convolution layer 통과과정 확인
def convolution():
    plt.figure(figsize=(13,9))
    plt.subplot(331)
    plt.title('Input')
    plt.imshow(image[0,:,:,0], 'gray_r')
    plt.colorbar()

    plt.subplot(332)
    plt.title('Filter')
    plt.imshow(weight2[0][:,:,0,0], 'gray_r')
    plt.colorbar()
    plt.subplot(335)
    plt.imshow(weight2[0][:,:,0,1], 'gray_r')
    plt.colorbar()
    plt.subplot(338)
    plt.imshow(weight2[0][:,:,0,2], 'gray_r')
    plt.colorbar()

    plt.subplot(333)
    plt.title('Conv')
    plt.imshow(conv_output2[0,:,:,0], 'gray_r')
    plt.colorbar()
    plt.subplot(336)
    plt.imshow(conv_output2[0,:,:,1], 'gray_r')
    plt.colorbar()
    plt.subplot(339)
    plt.imshow(conv_output2[0,:,:,2], 'gray_r')
    plt.colorbar()
    plt.show()

convolution()


# Activation function ---------------------------------------------------------------------
# Input Data
print(conv_output2.shape)
plt.figure(figsize=(10,7))
plt.subplot(311)
plt.imshow(conv_output2[0,:,:,0], 'RdBu')
plt.clim(-255,255)
plt.colorbar()
plt.subplot(312)
plt.imshow(conv_output2[0,:,:,1], 'RdBu')
plt.clim(-255,255)
plt.colorbar()
plt.subplot(313)
plt.imshow(conv_output2[0,:,:,2], 'RdBu')
plt.clim(-255,255)
plt.colorbar()
plt.show()

# Activation_Function ***
act_layer1 = tf.keras.layers.ReLU()
act_output1 = act_layer1(conv_output2)
print(act_output1.shape)

# Activation Function 통과 전 후 픽셀값 변화
print(f'min: {np.min(conv_output2)} ~ max: {np.max(conv_output2)}')
print(f'min: {np.min(act_output1)} ~ max: {np.max(act_output1)}')


# Activation_function 통과 전 후 픽셀값 변화 histogram
plt.figure(figsize=(10,3))
plt.subplot(121)
plt.hist(conv_output2.numpy().ravel())
plt.xlim(-150,150)
plt.subplot(122)
plt.hist(act_output1.numpy().ravel())
plt.xlim(-150,150)
plt.show()


# Input ~ Activation Function(ReLU) 과정 확인
def activation_relu():
    plt.figure(figsize=(12,12))
    plt.subplot(341)
    plt.title('Input')
    plt.imshow(image[0,:,:,0], 'RdBu')
    plt.clim(-255,255)
    plt.colorbar()

    plt.subplot(342)
    plt.title('Filter')
    plt.imshow(weight2[0][:,:,0,0], 'RdBu')
    plt.colorbar()
    plt.subplot(346)
    plt.imshow(weight2[0][:,:,0,1], 'RdBu')
    plt.colorbar()
    plt.subplot(3,4,10)
    plt.imshow(weight2[0][:,:,0,2], 'RdBu')
    plt.colorbar()

    plt.subplot(343)
    plt.title('Conv')
    plt.imshow(conv_output2[0,:,:,0], 'RdBu')
    plt.clim(-255,255)
    plt.colorbar()
    plt.subplot(347)
    plt.imshow(conv_output2[0,:,:,1], 'RdBu')
    plt.clim(-255,255)
    plt.colorbar()
    plt.subplot(3,4,11)
    plt.imshow(conv_output2[0,:,:,2], 'RdBu')
    plt.clim(-255,255)
    plt.colorbar()

    plt.subplot(344)
    plt.title('Relu')
    plt.imshow(act_output1[0,:,:,0], 'RdBu')
    plt.clim(-255,255)
    plt.colorbar()
    plt.subplot(348)
    plt.imshow(act_output1[0,:,:,1], 'RdBu')
    plt.clim(-255,255)
    plt.colorbar()
    plt.subplot(3,4,12)
    plt.imshow(act_output1[0,:,:,2], 'RdBu')
    plt.clim(-255,255)
    plt.colorbar()

    plt.show()

activation_relu()



# Pooling (Max Pooling) ------------------------------------------------------------------------------------
# Input Data
print(act_output1.shape)
plt.figure(figsize=(10,7))
plt.subplot(311)
plt.imshow(act_output1[0,:,:,0], 'RdBu')
plt.clim(-255,255)
plt.colorbar()
plt.subplot(312)
plt.imshow(act_output1[0,:,:,1], 'RdBu')
plt.clim(-255,255)
plt.colorbar()
plt.subplot(313)
plt.imshow(act_output1[0,:,:,2], 'RdBu')
plt.clim(-255,255)
plt.colorbar()
plt.show()


pool_layer1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')
pool_output1 = pool_layer1(act_output1)
# ?tf.keras.layers.MaxPool2D
# tf.keras.layers.MaxPool2D(
#     pool_size=(2, 2),
#     strides=None,
#     padding='valid',
#     data_format=None,
#     **kwargs,
# )

print(pool_output1.shape)

print(f'min: {np.min(act_output1)} ~ max: {np.max(act_output1)}')
print(f'min: {np.min(pool_output1)} ~ max: {np.max(pool_output1)}')


# MaxPooling 통과 전 후 픽셀값 변화 histogram
plt.figure(figsize=(10,3))
plt.subplot(121)
plt.hist(act_output1.numpy().ravel())
plt.xlim(-150,150)
plt.subplot(122)
plt.hist(pool_output1.numpy().ravel())
plt.xlim(-150,150)
plt.show()



# Input ~ MaxPooling 과정 확인
def maxpooling():
    plt.figure(figsize=(12,15))
    plt.subplot(351)
    plt.title('Input')
    plt.imshow(image[0,:,:,0], 'RdBu')
    plt.clim(-255,255)
    plt.colorbar()

    plt.subplot(352)
    plt.title('Filter')
    plt.imshow(weight2[0][:,:,0,0], 'RdBu')
    plt.colorbar()
    plt.subplot(357)
    plt.imshow(weight2[0][:,:,0,1], 'RdBu')
    plt.colorbar()
    plt.subplot(3,5,12)
    plt.imshow(weight2[0][:,:,0,2], 'RdBu')
    plt.colorbar()

    plt.subplot(353)
    plt.title('Conv')
    plt.imshow(conv_output2[0,:,:,0], 'RdBu')
    plt.clim(-255,255)
    # plt.colorbar()
    plt.subplot(358)
    plt.imshow(conv_output2[0,:,:,1], 'RdBu')
    plt.clim(-255,255)
    # plt.colorbar()
    plt.subplot(3,5,13)
    plt.imshow(conv_output2[0,:,:,2], 'RdBu')
    plt.clim(-255,255)
    # plt.colorbar()

    plt.subplot(354)
    plt.title('Relu')
    plt.imshow(act_output1[0,:,:,0], 'RdBu')
    plt.clim(-255,255)
    # plt.colorbar()
    plt.subplot(359)
    plt.imshow(act_output1[0,:,:,1], 'RdBu')
    plt.clim(-255,255)
    # plt.colorbar()
    plt.subplot(3,5,14)
    plt.imshow(act_output1[0,:,:,2], 'RdBu')
    plt.clim(-255,255)
    # plt.colorbar()

    plt.subplot(355)
    plt.title('MaxPooling')
    plt.imshow(pool_output1[0,:,:,0], 'RdBu')
    plt.clim(-255,255)
    plt.colorbar()
    plt.subplot(3,5,10)
    plt.imshow(pool_output1[0,:,:,1], 'RdBu')
    plt.clim(-255,255)
    plt.colorbar()
    plt.subplot(3,5,15)
    plt.imshow(pool_output1[0,:,:,2], 'RdBu')
    plt.clim(-255,255)
    plt.colorbar()

    plt.show()

maxpooling()





# Flatten ------------------------------------------------------------------------------------
# Input Data
print(pool_output1.shape)
plt.figure(figsize=(10,7))
plt.subplot(311)
plt.imshow(pool_output1[0,:,:,0], 'RdBu')
plt.clim(-255,255)
plt.colorbar()
plt.subplot(312)
plt.imshow(pool_output1[0,:,:,1], 'RdBu')
plt.clim(-255,255)
plt.colorbar()
plt.subplot(313)
plt.imshow(pool_output1[0,:,:,2], 'RdBu')
plt.clim(-255,255)
plt.colorbar()
plt.show()


# Flatten ***
flatten_layer1 = tf.keras.layers.Flatten()
flatten_output1 = flatten_layer1(pool_output1)
# ?tf.keras.layers.Flatten
# tf.keras.layers.Flatten(data_format=None, **kwargs)

print(flatten_output1.shape)

print(f'min: {np.min(pool_output1)} ~ max: {np.max(pool_output1)}')
print(f'min: {np.min(flatten_output1)} ~ max: {np.max(flatten_output1)}')

# Input ~ Flatten 과정 확인
plt.figure()
# plt.imshow(flatten_output1)
plt.imshow(np.repeat(flatten_output1.numpy(), 30, axis=0), 'RdBu')
plt.colorbar()
plt.clim(-255,255)
plt.show()


def flatten():
    plt.figure(figsize=(12,18))
    plt.subplot(361)
    plt.title('Input')
    plt.imshow(image[0,:,:,0], 'RdBu')
    plt.clim(-255,255)
    plt.colorbar()

    plt.subplot(362)
    plt.title('Filter')
    plt.imshow(weight2[0][:,:,0,0], 'RdBu')
    plt.colorbar()
    plt.subplot(368)
    plt.imshow(weight2[0][:,:,0,1], 'RdBu')
    plt.colorbar()
    plt.subplot(3,6,14)
    plt.imshow(weight2[0][:,:,0,2], 'RdBu')
    plt.colorbar()

    plt.subplot(363)
    plt.title('Conv')
    plt.imshow(conv_output2[0,:,:,0], 'RdBu')
    plt.clim(-255,255)
    # plt.colorbar()
    plt.subplot(369)
    plt.imshow(conv_output2[0,:,:,1], 'RdBu')
    plt.clim(-255,255)
    # plt.colorbar()
    plt.subplot(3,6,15)
    plt.imshow(conv_output2[0,:,:,2], 'RdBu')
    plt.clim(-255,255)
    # plt.colorbar()

    plt.subplot(364)
    plt.title('Relu')
    plt.imshow(act_output1[0,:,:,0], 'RdBu')
    plt.clim(-255,255)
    # plt.colorbar()
    plt.subplot(3,6,10)
    plt.imshow(act_output1[0,:,:,1], 'RdBu')
    plt.clim(-255,255)
    # plt.colorbar()
    plt.subplot(3,6,16)
    plt.imshow(act_output1[0,:,:,2], 'RdBu')
    plt.clim(-255,255)
    # plt.colorbar()

    plt.subplot(365)
    plt.title('MaxPooling')
    plt.imshow(pool_output1[0,:,:,0], 'RdBu')
    plt.clim(-255,255)
    plt.colorbar()
    plt.subplot(3,6,11)
    plt.imshow(pool_output1[0,:,:,1], 'RdBu')
    plt.clim(-255,255)
    plt.colorbar()
    plt.subplot(3,6,17)
    plt.imshow(pool_output1[0,:,:,2], 'RdBu')
    plt.clim(-255,255)
    plt.colorbar()

    plt.subplot(366)
    plt.title('Flatten')
    plt.imshow(np.repeat(flatten_output1.numpy(), 50, axis=0), 'RdBu')
    plt.colorbar()
    plt.clim(-255,255)

    plt.show()

flatten()









# Artificial Neural Networks =========================================================================

# Dense Layer  ------------------------------------------------------------------------------------
# Input Data
print(flatten_output1.shape)
plt.title('Flatten')
plt.imshow(np.repeat(flatten_output1.numpy(), 50, axis=0), 'RdBu')
plt.colorbar()
plt.clim(-255,255)
plt.show()


dense_layer1 = tf.keras.layers.Dense(units=32, activation='relu')
dense_output1 = dense_layer1(flatten_output1)
# ?tf.keras.layers.Dense
# tf.keras.layers.Dense(
#     units,
#     activation=None,
#     use_bias=True,
#     kernel_initializer='glorot_uniform',
#     bias_initializer='zeros',
#     kernel_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     bias_constraint=None,
#     **kwargs,
# )

dense_output1
print(dense_output1.shape)

plt.title('Dense Output')
plt.imshow(dense_output1, 'RdBu')
plt.colorbar()
plt.clim(-255,255)
plt.show()


# Dropout  ------------------------------------------------------------------------------------
# Input Data
print(dense_output1.shape)


dropout = tf.keras.layers.Dropout(0.7)
dropout_output1 = dropout(dense_output1)
# tf.keras.layers.Dropout?
#  tf.keras.layers.Dropout(rate, noise_shape=None, seed=None, **kwargs)
#       - rate: Float between 0 and 1. Fraction of the input units to drop.
dense_output1
print(dropout_output1.shape)

















# 【 Summary : 전체 과정 한꺼번에 보기 】 ===================================================================================


# (Modeling) -----------------------------------------------------------
input_shape = image.shape[1:]
output_n_class = 10

# [ CNN ]
input_layer = tf.keras.layers.Input(shape=input_shape)
# tf.keras.layers.Input?
# tf.keras.layers.Input(
#     shape=None,
#     batch_size=None,
#     name=None,
#     dtype=None,
#     sparse=False,
#     tensor=None,
#     ragged=False,
#     **kwargs,
# )

# Convolution Part : Feature Extraction
net = tf.keras.layers.Conv2D(32, 3, padding='same')(input_layer)
    # net = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same)(input_layer)
net = tf.keras.layers.Activation('relu')(net)
net = tf.keras.layers.MaxPool2D(2)(net)
    # net = tf.keras.layers.MaxPool2D(pool_size=(2,2))(net)
net = tf.keras.layers.Dropout(0.25)(net)

net = tf.keras.layers.Conv2D(32, 3, padding='same')(net)
net = tf.keras.layers.Activation('relu')(net)
net = tf.keras.layers.Conv2D(32, 3, padding='same')(net)
net = tf.keras.layers.Activation('relu')(net)
net = tf.keras.layers.MaxPool2D(2)(net)
net = tf.keras.layers.Dropout(0.25)(net)

net = tf.keras.layers.Flatten()(net)

# ANN Part
net = tf.keras.layers.Dense(512)(net)
net = tf.keras.layers.Activation('relu')(net)
net = tf.keras.layers.Dropout(0.25)(net)
net = tf.keras.layers.Dense(10)(net)
net = tf.keras.layers.Activation('softmax')(net)

model = tf.keras.Model(inputs=input_layer, outputs=net, name='Basic_CNN')

# tf.keras.Model(*args, **kwargs)
    # 1 - With the "functional API", where you start from `Input`,
    # you chain layer calls to specify the model's forward pass,
    # and finally you create your model from inputs and outputs:

    # inputs = tf.keras.Input(shape=(3,))
    # x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
    # outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
    # model = tf.keras.Model(inputs=inputs, outputs=outputs)



    # 2 - By subclassing the `Model` class: in that case, you should define your
    # layers in `__init__` and you should implement the model's forward pass in `call`.
    # If you subclass `Model`, you can optionally have
    # a `training` argument (boolean) in `call`, which you can use to specify
    # a different behavior in training and inference:

    # class MyModel(tf.keras.Model):

    #   def __init__(self):
    #     super(MyModel, self).__init__()
    #     self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    #     self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

    # def call(self, inputs, training=False):
    #     x = self.dense1(inputs)
    #     if training:
    #       x = self.dropout(x, training=training)
    #     return self.dense2(x)

    # model = MyModel()


model.summary()


# (Compiling) -----------------------------------------------------------
    # 모델 학습 전에 설정
# Loss_Function: 최소화 시키고자 하는 함수
# optimization: 최적화방법
# Metrics: 평가지표

# Loss_Function ----------------------------------
tf.keras.losses.binary_crossentropy                 # 분류가 2개
tf.keras.losses.categorical_crossentropy            # 분류가 3개 이상  (Onehot Encoding 을 하지 않았을때)
tf.keras.losses.sparse_categorical_crossentropy     # 분류가 3개 이상 (Onehot Encoding 을 했을때)


# metrics -----------------------------------------
tf.keras.metrics.Accuracy()
tf.keras.metrics.Precision()
tf.keras.metrics.Recall()
tf.keras.metrics.AUC()


# optimizer ---------------------------------------
tf.keras.optimizers.SGD()
tf.keras.optimizers.RMSprop()
tf.keras.optimizers.Adam()
# ?tf.optimizers.Adam
# tf.optimizers.Adam(
#     learning_rate=0.001,
#     beta_1=0.9,
#     beta_2=0.999,
#     epsilon=1e-07,
#     amsgrad=False,
#     name='Adam',
#     **kwargs,
# )


# compile ------------
model.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy'])

# model.compile?
# model.compile(
#     optimizer='rmsprop',
#     loss=None,
#     metrics=None,
#     loss_weights=None,
#     sample_weight_mode=None,
#     weighted_metrics=None,
#     target_tensors=None,
#     distribute=None,
#     **kwargs,
# )


# (Model_Fit) -----------------------------------------------------------

# Data Dimension Transform ------------------------------------
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# x값은 4차원 데이터로 들어가야함 (데이터갯수, width, height, channel)

x_train_4d = x_train[..., tf.newaxis]
# x_train_4d = x_train[..., np.newaxis]
# x_train_4d = np.expand_dims(x_train, axis=-1)

x_test_4d = x_test[..., tf.newaxis]
# x_test_4d = x_test[..., np.newaxis]
# x_test_4d = np.expand_dims(x_test, axis=-1)

print(x_train_4d.shape, y_train.shape, x_test_4d.shape, y_test.shape)


# Data Normalize------------------------------------
print('normalize 전')
print(np.min(x_train_4d), np.max(x_train_4d))
print(np.min(x_test_4d), np.max(x_test_4d))

x_train_4d_norm = x_train_4d/255.0
x_test_4d_norm = x_test_4d/255.0

print('normalize 후')
print(np.min(x_train_4d_norm), np.max(x_train_4d_norm))
print(np.min(x_test_4d_norm), np.max(x_test_4d_norm))



# Model_Fit -------------------------------------
    # epoch
    # batch_size

model.fit(x_train_4d_norm, y_train, epochs=3, batch_size=50, shuffle=True)
# ?model.fit
# model.fit(
#     x=None,
#     y=None,
#     batch_size=None,
#     epochs=1,
#     verbose=1,
#     callbacks=None,
#     validation_split=0.0,
#     validation_data=None,
#     shuffle=True,
#     class_weight=None,
#     sample_weight=None,
#     initial_epoch=0,
#     steps_per_epoch=None,
#     validation_steps=None,
#     validation_freq=1,
#     max_queue_size=10,
#     workers=1,
#     use_multiprocessing=False,
#     **kwargs,
# )






# (Model Evaluation) -------------------------------------
result = model.fit(x_train_4d_norm, y_train, epochs=3, batch_size=50, shuffle=True)


# (Evaluate)
model.evaluate(x_test_4d_norm, y_test, batch_size=50)


# (결과 확인)
    # 1개의 결과만 확인
test_image = x_test_4d_norm[0,:,:,0]
pred_proba = model.predict(test_image.reshpae(1,28,28,1))
pred = np.argmax(pred_proba)

plt.title(f'y_true: {y_test[0]} / y_pred : {pred}')
plt.imshow(test_image, 'gray_r')
plt.show()


    # n개의 결과 확인
test_n_batch = 32
test_batch = x_test_4d_norm[0:32]
test_batch.shape

pred_proba_batch = model.predict(test_batch)
pred_batch = np.argmax(pred_proba_batch, axis=-1)
# pred_batch = np.argmax(pred_proba_batch, axis=1)

    # 전체 결과 확인
pred_proba_overall = model.predcit(x_test_4d_norm)
pred_overall = np.argmax(pred_proba_overall, axis=1)


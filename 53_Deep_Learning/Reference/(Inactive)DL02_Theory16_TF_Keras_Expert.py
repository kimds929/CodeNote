import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf


# Tensorflow 공식 홈페이지에서 설명하는 Expert 버전 =======================================
# ## 네트워크 구조 정의 ------------------------------------------------------------------
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        # tf.keras.layers.Activation(activation, **kwargs)
        # tf.keras.layers.BatchNormalization(),     # batch normalization
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(128, activation='relu')
        self.dense4 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout4 = tf.keras.layers.Dropout(0.5)                # drop_out
        # tf.keras.layers.Dropout(rate, noise_shape=None, seed=None, **kwargs)
        self.dense5 = tf.keras.layers.Dense(10, activation='softmax')       
    
    def call(self, x, training=None, mask=None):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dropout4(x)
        return self.dense5(x)



# tf.data --------------------------------------
    # train_set
train_ds = tf.data.Dataset.from_tensor_slices((x_train_4d_norm, y_train))
# tf.data.Dataset.from_tensor_slices?
# tf.data.Dataset.from_tensor_slices(tensors)

train_ds = train_ds.shuffle(1000)      # 1,000 정도가 적당
# train_ds.shuffle?
# train_ds.shuffle(buffer_size, seed=None, reshuffle_each_iteration=None)
    # This dataset fills a buffer with `buffer_size` elements, then randomly
    # samples elements from this buffer, replacing the selected elements with new
    # elements. For perfect shuffling, a buffer size greater than or equal to the
    # full size of the dataset is required.

train_ds = train_ds.batch(32)
# train_ds.batch?
# train_ds.batch(batch_size, drop_remainder=False)


    # test_set
test_ds = tf.data.Dataset.from_tensor_slices((x_test_4d_norm, y_test))
test_ds = test_ds.batch(32)



# Visualization Data ---------------------
# train_ds.take(2)

for image, label in train_ds.take(2):
    print(image.shape, label.shape)
    plt.title(label[0].numpy())
    plt.imshow(image[0,:,:,0], 'gray_r')
    plt.show()


image, label = next(iter(train_ds))
plt.title(label[0].numpy())
plt.imshow(image[0,:,:,0], 'gray_r')
plt.show()



# Compile_model -----------------------
    # model.compile(optimizer='adam', 
    #             loss='sparse_categorical_crossentropy', 
    #             metrics=['accuracy'])

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')





# 학습 - 테스트 함수 및 학습 루프 구현  ---------------------------------------------
# @tf.function - 기존 session 열었던 것처럼 바로 작동안하고, 그래프를 만들고 학습이 시작되면 돌아가도록함

# ## 학습 함수 구현
@tf.function
def train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images)  # BatchSize x 10(Classes)      # forward propagation
        loss = loss_object(labels, predictions)                     # loss_function
    gradients = tape.gradient(loss, model.trainable_variables)      # loss_function을 model내 모든 weight에 대하여 미분
    # model.trainable_variables : model에 대해 모든 weight에 대하여
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))    # 미분값에 대해 optimizer를 이용하여 1 step 업데이트
    
    train_loss(loss)
    train_accuracy(labels, predictions)


# ## 테스트 함수 구현
@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):
    predictions = model(images)  # BatchSize x 10(Classes)
    loss = loss_object(labels, predictions)
    
    test_loss(loss)
    test_accuracy(labels, predictions)




# ## 모델 생성
model = MyModel()


# ## 손실함수 및 최적화 알고리즘 정의

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()


# ## 성능 지표 정의
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')




# ## 학습 루프 구현
EPOCHS = 10
for epoch in range(EPOCHS):
    for images, labels in train_ds:
        # images # x값 image의 matrix
        # labels # y값 image의 정답
        train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy)
        
    for images, labels in test_ds:
        test_step(model, images, labels, loss_object, test_loss, test_accuracy)
        
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch +1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))
    
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()




# --------------------------------------------------------------------------------------
# # model compile
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])


# # ## Early Stopping Callback
# earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)


# # ## 모델 학습
# history = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=[earlystopper])
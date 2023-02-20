import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

import tqdm


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

x_train = (x_train.astype('float32')/255.)[..., np.newaxis]
x_test = (x_test.astype('float32')/255.)[..., np.newaxis]

print(x_train.shape, x_test.shape)


def make_AE_model(latent_dim=20):
    model = tf.keras.Sequential([
        # Encoding
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, 
            padding='same', activation='relu', input_shape=(28,28,1)),  # (batch, 14, 14, 32)
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2,
            padding='same', activation='relu'),                         # (batch, 7, 7, 64)
        tf.keras.layers.Flatten(),                                      # (batch, 7*7*64)
        tf.keras.layers.Dense(units=latent_dim),                        # (batch, latent_dim)

        # Decoding
        tf.keras.layers.Dense(units=7 * 7 * 32, activation='relu'),     # (batch, 7*7*32)
        tf.keras.layers.Reshape(target_shape=(7, 7, 32)),               # (batch, 7, 7, 32)
        tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, 
            padding='same', activation='relu'),                         # (batch, 14, 14, 64)
        tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, 
            padding='same', activation='relu'),                         # (batch, 28, 28, 32)
        tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')    # (batch, 28, 28, 1)
    ])
    return model

model = make_AE_model(latent_dim=20)
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, x_train, batch_size=100, epochs=10)
model.evaluate(x_test, x_test)

# Image 비교
plt.figure(figsize=(10,4))
plt.subplot(121)
plt.title('original_image')
plt.imshow(x_test[96,:,:,0], cmap='gray')

plt.subplot(122)
plt.title('reconstructed_image')
plt.imshow(model.predict(x_test[96][np.newaxis, ...])[0,:,:,0], cmap='gray')
plt.show()


# Noise Data
noise = np.random.random(size=(1,28,28,1))

# Noise_Data
plt.figure(figsize=(10,4))
plt.subplot(121)
plt.title('original_image')
plt.imshow(noise[0,:,:,0], cmap='gray')

plt.subplot(122)
plt.title('reconstructed_image')
plt.imshow(model.predict(noise)[0,:,:,0], cmap='gray')
plt.show()



# MSE 비교 -----------------------------------------------
mse_loss = tf.keras.losses.MeanSquaredError()

# 정상데이터
mse_loss(y_true=x_test[96], y_pred=model.predict(x_test[96][np.newaxis, ...])[0] )

# Noise
mse_loss(y_true=noise, y_pred=model.predict(noise))


# 정상 Data MSE Loss 분포
mse_losses = []
for test_image in tqdm.tqdm_notebook(x_test):
    image_true = test_image[np.newaxis, ...]
    image_pred = model(image_true)
    mse_image = mse_loss(y_true=image_true, y_pred=image_pred)
    mse_losses.append(mse_image)
    
mse_losses_np = np.array(mse_losses)
mse_losses_np.mean()
mse_losses_np.std()

plt.hist(mse_losses_np, bins=40, color='skyblue', edgecolor='gray')
plt.show()














# 시계열 데이터 ----------------------------------------------------------
# !wget -O power_data.txt https://vo.la/H8uJu

data = []
with open('power_data.txt') as f:
    for line in f:
        data.append(int(line))
data = np.array(data)
# len(data) / 365         # 96
# len(data) / 365 * 7     # 672

# 전체 데이터 plot
plt.figure(figsize=(10,3), dpi=120)     # dpi = 선명하게 나옴
plt.plot(data)
plt.xlim(0, 36000)
plt.ylim(500, 2500)
plt.show()


# 일주일치 데이터 plot
plt.figure(figsize=(10,3), dpi=120)     # dpi = 선명하게 나옴
plt.plot(data)
plt.xlim(672*10, 672*11)
plt.ylim(500, 2500)
plt.show()



# 다른경향성이 보이는 일주일치 Data Plot
plt.figure(figsize=(10,3), dpi=120)     # dpi = 선명하게 나옴
plt.plot(data)
plt.xlim(87*96, 87*96+672)
plt.ylim(500, 2500)
plt.show()




# 3시간 간격으로 끊어줌
data_avg = np.mean(data.reshape(-1, 12), axis=-1)
print(data_avg.shape)
print(data_avg.shape[0] / 365)


# train_test_split
data_avg_train = data_avg[:2000]
data_avg_test = data_avg[2000:]

# Normalize Data
train_mean = np.mean(data_avg_train)    # 평균
train_std = np.std(data_avg_train)      # 표준편차

data_avg_train_norm = (data_avg_train - train_mean) / train_std
data_avg_test_norm = (data_avg_test - train_mean) / train_std


plt.figure(figsize=(10,3), dpi=120)     # dpi = 선명하게 나옴
plt.plot(data_avg_train_norm)
plt.xlim(0, 2000)
plt.ylim(-3, 3)
plt.show()


# Data 일주일단위로 끊어줌
x_seq_train = []
for i in range(len(data_avg_train) - 56):       # 마지막 일주일을 빼줌
    x_seq_train.append(data_avg_train_norm[i:i+56])
x_seq_train = np.array(x_seq_train)[..., np.newaxis]
print(x_seq_train.shape)


x_seq_test = []
for i in range(len(data_avg_test) - 56):       # 마지막 일주일을 빼줌
    x_seq_test.append(data_avg_test_norm[i:i+56])
x_seq_test = np.array(x_seq_test)[..., np.newaxis]
print(x_seq_test.shape)


def make_LSTM_model(X):
    model = tf.keras.Sequential()

    # encoding
    model.add( tf.keras.layers.LSTM(units=64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False) )
    # shape of encoding: (batch, 64)

    # decoding
    model.add(tf.keras.layers.RepeatVector(n=X.shape[1]))                           # (56, 64) 단순 Copy
    model.add(tf.keras.layers.LSTM(units=64, return_sequences=True))                # (56, 64)
    model.add(tf.keras.layers.TimeDistributed( layer=tf.keras.layers.Dense(1)) )    # (56, 1) 64 features → 1 feature (weight 공유)
    return model

lstm_model = make_LSTM_model(X=x_seq_train)
lstm_model.summary()


def scheduler(epoch, lr):
    if epoch % 20 == 0:
        return lr * 0.5
    return lr

lstm_model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(0.01))

lstm_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
lstm_model.fit(x_seq_train, x_seq_train, epochs=150, batch_size=100, callbacks=[lstm_callback], verbose=2)

lstm_model.evaluate(x_seq_test, x_seq_test)


# train_predict
lstm_mae = tf.keras.losses.MeanAbsoluteError()
x_seq_train_pred = lstm_model.predict(x_seq_train)
x_seq_test_pred = lstm_model.predict(x_seq_test)

lstm_mae(y_true=x_seq_train, y_pred=x_seq_train_pred)
lstm_mae(y_true=x_seq_test, y_pred=x_seq_test_pred)


seq_index = np.random.randint(0, 1944, size=2)
seq_min = np.min(seq_index)
seq_max = np.max(seq_index)

seq_min = 10
seq_max = 15
plt.figure(figsize=(20,5))
plt.title(f'seq: {seq_min} ~ {seq_max}')
plt.plot(x_seq_train[seq_min:seq_max,:].flatten(), alpha=0.7, label='origin')
plt.plot(lstm_model(x_seq_train)[seq_min:seq_max,:].numpy().flatten(), alpha=0.7, label='predict')
plt.show()




from matplotlib.ticker import AutoMinorLocator

# Train MAE Loss
fig = plt.figure(figsize=(20,5), dpi=120)
ax = fig.add_subplot(1,1,1)
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
plt.grid(which='both')
plt.plot(x_seq_train_pred[::56, 0])
plt.show()


# Test MAE Loss
fig = plt.figure(figsize=(20,5), dpi=120)
ax = fig.add_subplot(1,1,1)
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
plt.grid(which='both')
plt.plot(x_seq_test_pred[::56, 0])
plt.show()



















# class AE_Model(tf.keras.Model):
#     def __init__(self, latent_dim=20):
#         super(AE_Model, self).__init__()
#         # Encoding
#         self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, 
#             padding='same', activation='relu')  # (batch, 14, 14, 32)
#         self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2,
#             padding='same', activation='relu')                         # (batch, 7, 7, 64)
#         self.flatten = tf.keras.layers.Flatten()                                      # (batch, 7*7*64)
#         self.dense_e = tf.keras.layers.Dense(units=latent_dim)                       # (batch, latent_dim)

#         self.dense_d = tf.keras.layers.Dense(units=7 * 7 * 32, activation='relu')     # (batch, 7*7*32)
#         self.reshape = tf.keras.layers.Reshape(target_shape=(7, 7, 32))               # (batch, 7, 7, 32)
#         self.convT1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, 
#             padding='same', activation='relu')                         # (batch, 14, 14, 64)
#         self.convT2 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, 
#             padding='same', activation='relu')                         # (batch, 28, 28, 32)
#         self.convT3 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')    # (batch, 28, 28, 1)

#         self.eoncoder = tf.keras.Sequential([])

#     def call(self, x, training=False):
        


# def make_AE_model(latent_dim=20):
#     model = tf.keras.Sequential([
#         # Encoding
#         tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, 
#             padding='same', activation='relu', input_shape=(28,28,1)),  # (batch, 14, 14, 32)
#         tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2,
#             padding='same', activation='relu'),                         # (batch, 7, 7, 64)
#         tf.keras.layers.Flatten(),                                      # (batch, 7*7*64)
#         tf.keras.layers.Dense(units=latent_dim),                        # (batch, latent_dim)

#         # Decoding
#         tf.keras.layers.Dense(units=7 * 7 * 32, activation='relu'),     # (batch, 7*7*32)
#         tf.keras.layers.Reshape(target_shape=(7, 7, 32)),               # (batch, 7, 7, 32)
#         tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, 
#             padding='same', activation='relu'),                         # (batch, 14, 14, 64)
#         tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, 
#             padding='same', activation='relu'),                         # (batch, 28, 28, 32)
#         tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')    # (batch, 28, 28, 1)
#     ])
#     return model
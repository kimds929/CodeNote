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
top_words = 10000


# 영화 리뷰 데이터 (x: 리뷰, y: 긍정/부정)
(x_trainval, y_trainval), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=top_words)

print(x_trainval.shape, y_trainval.shape, x_test.shape, y_test.shape)
print(x_trainval[0][:10])   # text_data에 대한 숫자형식 list
print(y_trainval[0])        # 긍정 부정여부
wi = tf.keras.datasets.imdb.get_word_index()
# wi['fawn']
iw = {v: k for k, v in wi.items()}
np.array([iw[w] for w in x_test[0]])


unique_class = np.unique(y_trainval)
num_classes = len(unique_class)


# x_train review에서 n개(maxlen)의 sequence만 잘라서 표현, n개보다 짧은애들은 0 padding으로 맞춰줌
x_trainval80 = tf.keras.preprocessing.sequence.pad_sequences(x_trainval, maxlen=80)
x_trainval160 = tf.keras.preprocessing.sequence.pad_sequences(x_trainval, maxlen=160)
print(x_trainval80)
print(x_trainval80.shape,  x_trainval160.shape)

x_test40 = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=40)
x_test80 = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=80)
x_test160 = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=160)


# Type 변경
x_trainval80 = x_trainval80.astype('float32')
x_trainval160 = x_trainval160.astype('float32')
y_trainval = y_trainval.astype('float32')

x_test40 = x_test40.astype('float32')
x_test80 = x_test80.astype('float32')
x_test160 = x_test160.astype('float32')
y_test = y_test.astype('float32')
print(x_trainval80)


# train_valid_split
indices = np.random.permutation(x_trainval80.shape[0])
train_indices = indices[:-5000]
val_indices = indices[-5000:]

x_train80 = x_trainval80[train_indices]
x_train160 = x_trainval160[train_indices]
y_train = y_trainval[train_indices]

x_valid80 = x_trainval80[val_indices]
x_valid160 = x_trainval160[val_indices]
y_valid = y_trainval[val_indices]


# Dataset ----------------------------------------------------------------------------
batch_size = 200
train_data80 = tf.data.Dataset.from_tensor_slices((x_train80, y_train)).shuffle(x_train80.shape[0]).batch(batch_size).prefetch(1)
train_data160 = tf.data.Dataset.from_tensor_slices((x_train160, y_train)).shuffle(x_train160.shape[0]).batch(batch_size).prefetch(1)

valid_data80 = tf.data.Dataset.from_tensor_slices((x_valid80, y_valid)).shuffle(x_valid80.shape[0]).batch(batch_size).prefetch(1)
valid_data160 = tf.data.Dataset.from_tensor_slices((x_valid160, y_valid)).shuffle(x_valid160.shape[0]).batch(batch_size).prefetch(1)

test_data40 = tf.data.Dataset.from_tensor_slices((x_test40, y_test)).shuffle(x_test40.shape[0]).batch(batch_size).prefetch(1)
test_data80 = tf.data.Dataset.from_tensor_slices((x_test80, y_test)).shuffle(x_test80.shape[0]).batch(batch_size).prefetch(1)
test_data160 = tf.data.Dataset.from_tensor_slices((x_test160, y_test)).shuffle(x_test160.shape[0]).batch(batch_size).prefetch(1)



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
def train_model(train_data, model, optimizer, epochs, 
                print_loss=True, plot_graph=True):
    step_l = []
    loss_l = []

    # optimizer = tf.optimizers.SGD(lr, momentum=0.9)

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
def early_stopping(train_data, valid_data, model, optimizer):
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
        train_model(train_data, model, optimizer, 1, print_loss=False, plot_graph=False)
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


# ?tf.keras.layers.SimpleRNN
# tf.keras.layers.SimpleRNN(
#     units,
#     activation='tanh',
#     use_bias=True,
#     kernel_initializer='glorot_uniform',
#     recurrent_initializer='orthogonal',
#     bias_initializer='zeros',
#     kernel_regularizer=None,
#     recurrent_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     recurrent_constraint=None,
#     bias_constraint=None,
#     dropout=0.0,
#     recurrent_dropout=0.0,
#     return_sequences=False,
#     return_state=False,
#     go_backwards=False,
#     stateful=False,
#     unroll=False,
#     **kwargs,
# )
x_train_test = x_train80[:2][..., np.newaxis]
x_train_test.shape

RNN_layer = tf.keras.layers.SimpleRNN(2, return_sequences=True, return_state=True)
output, hidden = RNN_layer(x_train_test)

LSTM_layer = tf.keras.layers.LSTM(2, return_sequences=True, return_state=True)
LSTM_layer(x_train_test)
# tf.keras.layers.LSTM?
# tf.keras.layers.LSTM(
#     units,
#     activation='tanh',
#     recurrent_activation='sigmoid',
#     use_bias=True,
#     kernel_initializer='glorot_uniform',
#     recurrent_initializer='orthogonal',
#     bias_initializer='zeros',
#     unit_forget_bias=True,
#     kernel_regularizer=None,
#     recurrent_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     recurrent_constraint=None,
#     bias_constraint=None,
#     dropout=0.0,
#     recurrent_dropout=0.0,
#     implementation=2,
#     return_sequences=False,
#     return_state=False,
#     go_backwards=False,
#     stateful=False,
#     time_major=False,
#     unroll=False,
#     **kwargs,
# )

# RNN (Recurrent Neural Network) ===================================================================
# RNN
# LSTM
# Neural word Embedding: word2Vec  # 단어간의 연관성을 숫자 vector로 표현 (Dense Representation)




# top_words = 10000        # 가장빈도 높은 top_words 갯수만 가져오겠다.

# RNN Model ---------------------------------------------------------------------------------
class RNN(tf.keras.Model):
    def __init__(self):
        super(RNN, self).__init__()
        self.embedding = tf.keras.layers.Embedding(top_words, 100)  # 10000개의 Data를 100차원 Data로 바꿔달라는 의미
                                    # [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

        self.rnn = tf.keras.layers.SimpleRNN(64)
        self.out = layers.Dense(num_classes)
    
    def call(self, x, is_training=False):
        # (rnn 접근)
        # rnn input [batch, timestep, feature]
        # x = tf.expand_dims(x, axis=2)
        # x.shape = [20000, 80] → [20000, 80, 1] ? → NO
        x = self.embedding(x)
        x = self.rnn(x)
        x = self.out(x)
        if not is_training:
            x = tf.nn.softmax(x)
        return x

batch_train80_x = next(iter(x_train80))
print(f'train: {batch_train80_x.shape}')

embedding = tf.keras.layers.Embedding(top_words, 100)
embed_result = embedding(batch_train80_x)
print(f'embedding: {embed_result.shape}')

rnn0 = tf.keras.layers.SimpleRNN(64)
rnn1 = tf.keras.layers.SimpleRNN(64, return_sequences=True)
rnn2 = tf.keras.layers.SimpleRNN(64, return_state=True)

rnn0_result = rnn0(embed_result)
rnn1_result = rnn1(embed_result)
rnn2_result = rnn2(embed_result)
print(f'rnn_0: {rnn0_result.shape}')
print(f'rnn_1: {rnn1_result.shape}')
# print(f'rnn_2: {rnn2_result.shape}')


# RNN_80 ****
rnn80 = RNN()
rnn80_optimizer = tf.optimizers.Adam(0.0001)

print('--- RNN_80 -----------')
# rnn80, rnn80_best_epoch = early_stopping(train_data80, valid_data80, rnn80, rnn80_optimizer)
rnn80_result = train_model(train_data80, rnn80, rnn80_optimizer, epochs=5)

rnn80_acc40 = accuracy_batch(rnn80_result, test_data40)
rnn80_acc80 = accuracy_batch(rnn80_result, test_data80)
rnn80_acc160 = accuracy_batch(rnn80_result, test_data160)

print(f'(rnn80_accuracy) test_40: {rnn80_acc40}, test_80: {rnn80_acc80}, test_160: {rnn80_acc160}')


# RNN_160 ****
rnn160 = RNN()
rnn160_optimizer = tf.optimizers.Adam(0.0001)

print('--- RNN_160 -----------')
# rnn160, rnn160_best_epoch = early_stopping(train_data80, valid_data80, rnn160, rnn160_optimizer)
rnn160_result = train_model(train_data160, rnn160, rnn160_optimizer, epochs=5)

rnn160_acc40 = accuracy_batch(rnn160_result, test_data40)
rnn160_acc80 = accuracy_batch(rnn160_result, test_data80)
rnn160_acc160 = accuracy_batch(rnn160_result, test_data160)

print(f'(rnn160_accuracy) test_40: {rnn160_acc40}, test_80: {rnn160_acc80}, test_160: {rnn160_acc160}')






# LSTM Model ---------------------------------------------------------------------------------
class LSTM(tf.keras.Model):
    def __init__(self):
        super(LSTM, self).__init__()
        self.embedding = tf.keras.layers.Embedding(top_words, 100)  # 10000개의 Data를 100차원 Data로 바꿔달라는 의미
                                    # [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
        self.lstm = tf.keras.layers.LSTM(64)
        self.out = layers.Dense(num_classes)
    
    def call(self, x, is_training=False):
        # (rnn 접근)
        # rnn input [batch, timestep, feature]
        # x = tf.expand_dims(x, axis=2)
        # x.shape = [20000, 80] → [20000, 80, 1] ? → NO
        x = self.embedding(x)
        x = self.lstm(x)
        x = self.out(x)
        if not is_training:
            x = tf.nn.softmax(x)
        return x

# LSTM_80 ****
lstm80 = LSTM()
lstm80_optimizer = tf.optimizers.Adam(0.0001)

print('--- LSTM_80 -----------')
# lstm, lstm_best_epoch = early_stopping(train_data80, valid_data80, lstm80, lstm80_optimizer)
lstm80_result = train_model(train_data80, lstm80, lstm80_optimizer, epochs=5)

lstm80_acc40 = accuracy_batch(lstm80_result, test_data40)
lstm80_acc80 = accuracy_batch(lstm80_result, test_data80)
lstm80_acc160 = accuracy_batch(lstm80_result, test_data160)

print(f'(lstm80_accuracy) test_40: {lstm80_acc40}, test_80: {lstm80_acc80}, test_160: {lstm80_acc160}')
print()


# LSTM_160 ****
lstm160 = LSTM()
lstm160_optimizer = tf.optimizers.Adam(0.0001)

print('--- LSTM_160 -----------')
# lstm160, lstm160_best_epoch = early_stopping(train_data80, valid_data80, lstm160, lstm160_optimizer)
lstm160_result = train_model(train_data160, lstm160, lstm160_optimizer, epochs=5)

lstm160_acc40 = accuracy_batch(lstm160_result, test_data40)
lstm160_acc80 = accuracy_batch(lstm160_result, test_data80)
lstm160_acc160 = accuracy_batch(lstm160_result, test_data160)

print(f'(lstm160_accuracy) test_40: {lstm160_acc40}, test_80: {lstm160_acc80}, test_160: {lstm160_acc160}')
print()







# MultiLayer LSTM Model ----------------------------------------------------------------------------
class Multi_LSTM(tf.keras.Model):
    def __init__(self):
        super(Multi_LSTM, self).__init__()
        self.embedding = tf.keras.layers.Embedding(top_words, 100)  # 10000개의 Data를 100차원 Data로 바꿔달라는 의미
                                    # [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
        self.lstm1 = tf.keras.layers.LSTM(64, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(64)
        self.out = layers.Dense(num_classes)
    
    def call(self, x, is_training=False):
        # print(f'input_shape: {x.shape}')
        x = self.embedding(x)
        # print(f'embedding_shape: {x.shape}')
        x = self.lstm1(x)
        # print(f'lstm1_shape: {x.shape}')
        x = self.lstm2(x)
        # print(f'lstm2_shape: {x.shape}')
        x = self.out(x)
        # print(f'out_shape: {x.shape}')
        if not is_training:
            x = tf.nn.softmax(x)
        return x

# mutl_lstm = Multi_LSTM()
# mutl_lstm(x_train80[0][np.newaxis, ...])
    # input_shape: (1, 80)
    # embedding_shape:(1, 80, 100)
    # lstm1_shape:(1, 64)
    # out_shape:(1, 2)

# mutl_lstm(x_train80[:10])
    # input_shape: (10, 80)
    # embedding_shape: (10, 80, 100)
    # lstm1_shape: (10, 64)
    # out_shape: (10, 2)

# mutl_lstm(x_train80[:10])
    # input_shape: (10, 80)
    # embedding_shape: (10, 80, 100)
    # lstm1_shape: (10, 80, 64)     # return_sequences=True
    # lstm2_shape: (10, 64)
    # out_shape: (10, 2)




# Multi_LSTM_80 ****
multi_lstm80 = Multi_LSTM()
multi_lstm80_optimizer = tf.optimizers.Adam(0.0001)

print('--- Multi_LSTM_80 -----------')
# multi_lstm, multi_lstm_best_epoch = early_stopping(train_data80, valid_data80, multi_lstm80, multi_lstm80_optimizer)
multi_lstm80_result = train_model(train_data80, multi_lstm80, multi_lstm80_optimizer, epochs=4)

multi_lstm80_acc40 = accuracy_batch(multi_lstm80_result, test_data40)
multi_lstm80_acc80 = accuracy_batch(multi_lstm80_result, test_data80)
multi_lstm80_acc160 = accuracy_batch(multi_lstm80_result, test_data160)

print(f'(multi_lstm80_accuracy) test_40: {multi_lstm80_acc40}, test_80: {multi_lstm80_acc80}, test_160: {multi_lstm80_acc160}')
print()


# Multi_LSTM_160 ****
multi_lstm160 = Multi_LSTM()
multi_lstm160_optimizer = tf.optimizers.Adam(0.0001)

print('--- Multi_LSTM_160 -----------')
# multi_lstm160, multi_lstm160_best_epoch = early_stopping(train_data80, valid_data80, multi_lstm160, multi_lstm160_optimizer)
multi_lstm160_result = train_model(train_data160, multi_lstm160, multi_lstm160_optimizer, epochs=4)

multi_lstm160_acc40 = accuracy_batch(multi_lstm160_result, test_data40)
multi_lstm160_acc80 = accuracy_batch(multi_lstm160_result, test_data80)
multi_lstm160_acc160 = accuracy_batch(multi_lstm160_result, test_data160)

print(f'(multi_lstm160_accuracy) test_40: {multi_lstm160_acc40}, test_80: {multi_lstm160_acc80}, test_160: {multi_lstm160_acc160}')
print()







# Bidirectional LSTM -----------------------------------------------------------------------------
class BI_LSTM(tf.keras.Model):
    def __init__(self):
        super(BI_LSTM, self).__init__()
        self.embedding = tf.keras.layers.Embedding(top_words, 100)

        self.lstm_fw = tf.keras.layers.LSTM(64)
        self.lstm_bw = tf.keras.layers.LSTM(64, go_backwards=True)
        self.bi_lstm = tf.keras.layers.Bidirectional(self.lstm_fw, backward_layer=self.lstm_bw)

        self.out = layers.Dense(num_classes)
    
    def call(self, x, is_training=False):
        x = self.embedding(x)
        x = self.bi_lstm(x)
        if not is_training:
            x = tf.nn.softmax(x)
        return x


# BI_LSTM_80 ****
bi_lstm80 = BI_LSTM()
bi_lstm80_optimizer = tf.optimizers.Adam(0.0001)

print('--- BI_LSTM_80 -----------')
# bi_lstm, bi_lstm_best_epoch = early_stopping(train_data80, valid_data80, bi_lstm80, bi_lstm80_optimizer)
bi_lstm80_result = train_model(train_data80, bi_lstm80, bi_lstm80_optimizer, epochs=4)

bi_lstm80_acc40 = accuracy_batch(bi_lstm80_result, test_data40)
bi_lstm80_acc80 = accuracy_batch(bi_lstm80_result, test_data80)
bi_lstm80_acc160 = accuracy_batch(bi_lstm80_result, test_data160)

print(f'(bi_lstm80_accuracy) test_40: {bi_lstm80_acc40}, test_80: {bi_lstm80_acc80}, test_160: {bi_lstm80_acc160}')
print()


# BI_LSTM_160 ****
bi_lstm160 = BI_LSTM()
bi_lstm160_optimizer = tf.optimizers.Adam(0.0001)

print('--- BI_LSTM_160 -----------')
# bi_lstm160, bi_lstm160_best_epoch = early_stopping(train_data80, valid_data80, bi_lstm160, bi_lstm160_optimizer)
bi_lstm160_result = train_model(train_data160, bi_lstm160, bi_lstm160_optimizer, epochs=4)

bi_lstm160_acc40 = accuracy_batch(bi_lstm160_result, test_data40)
bi_lstm160_acc80 = accuracy_batch(bi_lstm160_result, test_data80)
bi_lstm160_acc160 = accuracy_batch(bi_lstm160_result, test_data160)

print(f'(bi_lstm160_accuracy) test_40: {bi_lstm160_acc40}, test_80: {bi_lstm160_acc80}, test_160: {bi_lstm160_acc160}')
print()








# result summary ----------------------------------------------------------------------------------------
print(f'(rnn80_accuracy) test_40: {format(rnn80_acc40, ".2f")}, test_80: {format(rnn80_acc80, ".2f")}, test_160: {format(rnn80_acc160, ".2f")}')
print(f'(rnn160_accuracy) test_40: {format(rnn160_acc40, ".2f")}, test_80: {format(rnn160_acc80, ".2f")}, test_160: {format(rnn160_acc160, ".2f")}')
print(f'(lstm80_accuracy) test_40: {format(lstm80_acc40, ".2f")}, test_80: {format(lstm80_acc80, ".2f")}, test_160: {format(lstm80_acc160, ".2f")}')
print(f'(lstm160_accuracy) test_40: {format(lstm160_acc40, ".2f")}, test_80: {format(lstm160_acc80, ".2f")}, test_160: {format(lstm160_acc160, ".2f")}')
print(f'(multi_lstm80_accuracy) test_40: {format(multi_lstm80_acc40, ".2f")}, test_80: {format(multi_lstm80_acc80, ".2f")}, test_160: {format(multi_lstm80_acc160, ".2f")}')
print(f'(multi_lstm160_accuracy) test_40: {format(multi_lstm160_acc40, ".2f")}, test_80: {format(multi_lstm160_acc80, ".2f")}, test_160: {format(multi_lstm160_acc160, ".2f")}')
print(f'(bi_lstm80_accuracy) test_40: {format(bi_lstm80_acc40, ".2f")}, test_80: {format(bi_lstm80_acc80, ".2f")}, test_160: {format(bi_lstm80_acc160, ".2f")}')
print(f'(bi_lstm160_accuracy) test_40: {format(bi_lstm160_acc40, ".2f")}, test_80: {format(bi_lstm160_acc80, ".2f")}, test_160: {format(bi_lstm160_acc160, ".2f")}')




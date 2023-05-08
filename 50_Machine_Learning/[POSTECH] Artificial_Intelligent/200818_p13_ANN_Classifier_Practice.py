import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

import os
import sys
sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import DS_DF_Summary, DS_OneHotEncoder, DS_LabelEncoder
from DS_OLS import *


absolute_path = 'D:/Python/★★Python_POSTECH_AI/Postech_AI 4) Aritificial_Intelligent/교재_실습_자료/'


df = pd.read_csv(absolute_path + 'BigML_Steel_Plates_Faults.csv')
df_info = DS_DF_Summary(df)


col = df.columns
y_cols = ['Fault']
X_cols = col.drop(y_cols)

X = df[X_cols]
y = df[y_cols]

# display(X.describe(include = 'all'))

x_np = np.asarray(X)
x_np

x_norm = x_np.copy()

x_min = np.min(x_norm, axis = 0)
x_norm = np.subtract(x_norm, x_min)

x_max = np.max(x_norm, axis = 0)
x_norm /= (1*x_max)

np.max(x_norm)
x_norm.shape


# 3.2. Label Encoding
# le = LabelEncoder()
# le.fit(y)
# y_class = le.transform(y).reshape(-1,1)

# le2 = DS_LabelEncoder()
# y_class = le2.fit_transform(y).to_numpy()


y
y_np = y.to_numpy()

labels = list(np.unique(y))
labels


label_dict = {}
for i, l in zip(range(len(labels)), labels):
    label_dict[l] = i
    
label_dict

y_class = np.zeros(y.shape)

for i in range(len(y)):
    y_class[i] =  label_dict[y_np[i][0]]







# 4. ANN Model  --------------------------------------------------------------------------
# 4.1. Base model

n_input = 27
n_hidden1 = 100
n_hidden2 = 50
n_output = 7


# Define Structure
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(n_hidden1, activation=tf.nn.relu, input_shape=(n_input,)),
    tf.keras.layers.Dense(n_hidden2, activation=tf.nn.relu, input_shape=(n_hidden1,)),
    tf.keras.layers.Dense(n_output, activation=tf.nn.softmax,input_shape=(n_hidden2,),)
    ])


model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])


# train_valid_split
def train_valid_split(x, y, test_rate = 0.2):
    v_idx = np.random.choice(len(x), size = int(len(x)*test_rate), replace = False)
    t_idx = np.setdiff1d(np.arange(len(x)), v_idx)
    return x[t_idx], y[t_idx], x[v_idx], y[v_idx]

train_X, train_Y, valid_X, valid_Y = train_valid_split(x_np, y_class)
train_X = train_X.astype(np.float32)
valid_X = valid_X.astype(np.float32)


train_X_n, train_Y_n, valid_X_n, valid_Y_n = train_valid_split(x_norm, y_class)
train_X_n = train_X_n.astype(np.float32)
valid_X_n = valid_X_n.astype(np.float32)


n_batch = 64    # Batch Size
n_epoch = 500   # Learning Iteration
n_prt = 100    # Print Cycle
LR = 0.001

training_records = model.fit(train_X, train_Y, batch_size=n_batch, epochs=n_epoch, verbose=0)
training_records_n = model.fit(train_X_n, train_Y_n, batch_size=n_batch, epochs=n_epoch, verbose=0)

training_records.history.keys()
training_records_n.history.keys()




not_n_alpha = 1
n_alpha = 1

    # 그래프
plt.figure(figsize=(10,8))
plt.plot(np.arange(len(training_records.history['loss'])), training_records.history['loss'], 'k-', alpha = not_n_alpha, label = 'Not normalized')
plt.plot(np.arange(len(training_records_n.history['loss'])), training_records_n.history['loss'], 'b-', alpha = n_alpha,  label = 'Normalilzed')
plt.xlabel('Epoch', fontsize = 15)
plt.ylabel('Loss', fontsize = 15)
plt.ylim([0, 5])
plt.legend(fontsize = 12)
plt.show()

    # 그래프 : 정규화 후
plt.figure(figsize=(10,8))
plt.plot(np.arange(len(training_records.history['accuracy'])), training_records.history['accuracy'], 'k-', alpha = not_n_alpha, label = 'Not normalized')
plt.plot(np.arange(len(training_records_n.history['accuracy'])), training_records_n.history['accuracy'], 'b-', alpha = n_alpha,  label = 'Normalilzed')
plt.xlabel('Epoch', fontsize = 15)
plt.ylabel('Accuracy', fontsize = 15)
plt.legend(fontsize = 12)
plt.show()




# 4.3. Batch Normalization (Feature Scaling)
# batch data를 기준으로 normalization

model_bn = tf.keras.models.Sequential([
    tf.keras.layers.Dense(n_hidden1, activation=tf.nn.relu,input_shape=(n_input,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(n_hidden2, activation=tf.nn.relu,input_shape=(n_hidden1,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(n_output, activation=tf.nn.softmax,input_shape=(n_hidden2,),)
    ])

model_bn.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

training_records_bn = model_bn.fit(train_X_n, train_Y_n, batch_size=n_batch, epochs=n_epoch, verbose=0)


not_n_alpha = 0.3
n_alpha = 0.3
bn_alpha = 1
plt.figure(figsize=(10,8))
plt.plot(np.arange(len(training_records.history['loss']))*n_prt, training_records.history['loss'], 'k-', alpha = not_n_alpha, label = 'Not normalized')
plt.plot(np.arange(len(training_records_n.history['loss']))*n_prt, training_records_n.history['loss'], 'b-', alpha = n_alpha,  label = 'Normalilzed')
plt.plot(np.arange(len(training_records_bn.history['loss']))*n_prt, training_records_bn.history['loss'], 'r-', alpha = bn_alpha,  label = 'Norm. with BN')
plt.xlabel('Epoch', fontsize = 15)
plt.ylabel('Loss', fontsize = 15)
plt.ylim([0, 5])
plt.legend(fontsize = 12)
plt.show()

plt.figure(figsize=(10,8))
plt.plot(np.arange(len(training_records.history['accuracy'])), training_records.history['accuracy'], 'k-', alpha = not_n_alpha, label = 'Not normalized')
plt.plot(np.arange(len(training_records_n.history['accuracy'])), training_records_n.history['accuracy'], 'b-', alpha = n_alpha,  label = 'Normalilzed')
plt.plot(np.arange(len(training_records_bn.history['accuracy'])), training_records_bn.history['accuracy'], 'r-', alpha = bn_alpha,  label = 'Norm. with BN')
plt.xlabel('Epoch', fontsize = 15)
plt.ylabel('Accuracy', fontsize = 15)
plt.legend(fontsize = 12)
plt.show()



# 4.4. Validation to check "Overfitting" -----------------------------------------------------
# Define Structure
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(n_hidden1, activation=tf.nn.relu, input_shape=(n_input,)),
    tf.keras.layers.Dense(n_hidden2, activation=tf.nn.relu, input_shape=(n_hidden1,)),
    tf.keras.layers.Dense(n_output, activation=tf.nn.softmax, input_shape=(n_hidden2,),)
    ])

model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

training_reocrds_v = model.fit(train_X_n, train_Y_n, validation_data = (valid_X_n, valid_Y_n), batch_size=n_batch, epochs=n_epoch, verbose=0)
training_reocrds_bn_v = model_bn.fit(train_X_n, train_Y_n, validation_data = (valid_X_n, valid_Y_n), batch_size=n_batch, epochs=n_epoch, verbose=0)

training_reocrds_v.history.keys()
training_reocrds_bn_v.history.keys()

base_alpha = 0.3
bn_alpha = 1

plt.figure(figsize=(10,8))
plt.plot(np.arange(len(training_reocrds_v.history['loss']))*n_prt, training_reocrds_v.history['loss'], 'b--', alpha = base_alpha, label = 'Base_train')
plt.plot(np.arange(len(training_reocrds_bn_v.history['loss']))*n_prt, training_reocrds_bn_v.history['loss'], 'r--', alpha = bn_alpha, label = 'BN_train')
plt.plot(np.arange(len(training_reocrds_v.history['val_loss']))*n_prt, training_reocrds_v.history['val_loss'], 'b-', alpha = base_alpha, label = 'Base_valid')
plt.plot(np.arange(len(training_reocrds_bn_v.history['val_loss']))*n_prt, training_reocrds_bn_v.history['val_loss'], 'r-', alpha = bn_alpha, label = 'BN_valid')
plt.xlabel('Epoch', fontsize = 15)
plt.ylabel('Loss', fontsize = 15)
plt.ylim([0, 5])
plt.legend(fontsize = 12)
plt.grid(alpha = 0.3)
plt.show()

plt.figure(figsize=(10,8))
plt.plot(np.arange(len(training_reocrds_v.history['accuracy'])), training_reocrds_v.history['accuracy'], 'b--', alpha = base_alpha, label = 'Base_train')
plt.plot(np.arange(len(training_reocrds_bn_v.history['accuracy'])), training_reocrds_bn_v.history['accuracy'], 'r--', alpha = bn_alpha, label = 'BN_train')
plt.plot(np.arange(len(training_reocrds_v.history['val_accuracy'])), training_reocrds_v.history['val_accuracy'], 'b-', alpha = base_alpha, label = 'Base_valid')
plt.plot(np.arange(len(training_reocrds_bn_v.history['val_accuracy'])), training_reocrds_bn_v.history['val_accuracy'], 'r-', alpha = bn_alpha, label = 'BN_valid')
plt.xlabel('Epoch', fontsize = 15)
plt.ylabel('Accuracy', fontsize = 15)
plt.legend(fontsize = 12)
plt.grid(alpha = 0.3)
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

import matplotlib
import time
import os

sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import DS_DF_Summary, DS_OneHotEncoder, DS_LabelEncoder
from DS_OLS import *

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Postech_AI 4) Aritificial_Intelligent/교재_실습_자료/'

train = pd.read_csv(absolute_path +  '/HousePrice_train.csv')
train.head()

# train_info = DS_DF_Summary(train)


print('Shape of the train data with all features:', train.shape)
train = train.select_dtypes(exclude = ['object'])
print("")
print('Shape of the train data with numerical features:', train.shape)
train2_info = DS_DF_Summary(train)
train2_info.summary

train.drop('Id', axis = 1, inplace = True)
train.fillna(0, inplace = True)

print("")
print('List of features contained our dataset:', list(train.columns))
print("")
print('Shape of the test data with numerical features:', train.shape)

# 3.1. Outliers
# In this small part we will isolate the outliers with an IsolationForest (http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).
# I tried with and without this step and I had a better performance removing these rows
# I haven't analysed the test set but I suppose that our train set looks like more at our data test without these outliers.

from sklearn.ensemble import IsolationForest

clf = IsolationForest(max_samples = 100, random_state = 42)
clf.fit(train)
y_noano = clf.predict(train)
y_noano = pd.DataFrame(y_noano, columns = ['Top'])
y_noano[y_noano['Top'] == 1].index.values

train = train.iloc[y_noano[y_noano['Top'] == 1].index.values]
train.reset_index(drop = True, inplace = True)
print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])
print("Number of rows without outliers:", train.shape[0])



# 3.2. Normalization
train = np.asarray(train)
train.shape

x_norm = train.copy()
x_norm.shape

x_min = np.min(train, axis = 0)
x_min

x_norm = np.subtract(x_norm, x_min)
x_norm


train
x_max = np.max(train, axis = 0)
x_norm /= x_max

x_max

x_norm
np.max(x_norm)
x_norm.shape


train_x_wo = train[:, :36]
train_y_wo = train[:, 36].reshape(-1,1)

train_x_norm = x_norm[:, :36]
train_y_norm = x_norm[:, 36].reshape(-1,1)

print('train_x_without normalization:', train_x_wo.shape)
print('train_y_without normalization:', train_y_wo.shape)
print('train_x_normalization:', train_x_norm.shape)
print('train_y_normalization:', train_y_norm.shape)

# tf.keras.layers.Dense?


# 4. ANN model ---------------------------------------------------------------
n_input = 36
n_hidden1 = 200
n_hidden2 = 100
n_hidden3 = 50
n_hidden4 = 25
n_output = 1


# Define Structure
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(n_hidden1, activation=tf.nn.relu, input_shape=(n_input,)),
    tf.keras.layers.Dense(n_hidden2, activation=tf.nn.relu, input_shape=(n_hidden1,)),
    tf.keras.layers.Dense(n_hidden3, activation=tf.nn.relu, input_shape=(n_hidden2,)),
    tf.keras.layers.Dense(n_hidden4, activation=tf.nn.relu,input_shape=(n_hidden3,)),
    tf.keras.layers.Dense(n_output, activation=None, input_shape=(n_hidden4,))
    ])

optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile(optimizer=optimizer,
                loss='mse',
                metrics=['mae', 'mse'])




# train_valid_split
def train_test_split(x, y, test_rate = 0.2):
    v_idx = np.random.choice(len(x), size = int(len(x)*test_rate), replace = False)
    t_idx = np.setdiff1d(np.arange(len(x)), v_idx)
    return x[t_idx], y[t_idx], x[v_idx], y[v_idx]

train_X, train_Y, test_X, test_Y = train_test_split(train_x_wo, train_y_wo)

train_X_n, train_Y_n, test_X_n, test_Y_n = train_test_split(train_x_norm, train_y_norm)

n_batch = 64    # Batch Size
n_epoch = 300   # Learning Iteration

training_records = model.fit(train_X, train_Y, batch_size=n_batch, epochs=n_epoch, verbose=0)
training_records_n = model.fit(train_X_n, train_Y_n, batch_size=n_batch, epochs=n_epoch, verbose=0)


pd.DataFrame(training_records.history)
pd.DataFrame(training_records_n.history)



    # plotting
not_n_alpha = 1
n_alpha = 1

plt.figure(figsize=(10,8))
plt.plot(np.arange(len(training_records.history['mse'])), training_records.history['mse'], 'k-', alpha = not_n_alpha, label = 'Not normalized')
plt.xlabel('Epoch', fontsize = 15)
plt.ylabel('mean_squared_error', fontsize = 15)
plt.legend(fontsize = 12)

plt.show()
plt.figure(figsize=(10,8))
plt.plot(np.arange(len(training_records_n.history['mse'])), training_records_n.history['mse'], 'b-', alpha = n_alpha,  label = 'Normalilzed')
plt.xlabel('Epoch', fontsize = 15)
plt.ylabel('mean_squared_error', fontsize = 15)
plt.legend(fontsize = 12)
plt.ylim([0, 0.01])
plt.show()






# 5. Predictions ---------------------------------------------------------------
test_x = test_X_n
test_y = test_Y_n

my_pred = model(test_x)
my_pred

pd.DataFrame(np.hstack([test_y, my_pred]), columns=['y_true', 'y_pred'])

test_y_min = x_min[36]
test_y_max = x_max[36]

test_y = (test_y * test_y_max) + test_y_min
my_pred = (my_pred * test_y_max) + test_y_min


# plotting
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30) 

fig, ax = plt.subplots(figsize=(50, 40))

plt.style.use('ggplot')
plt.plot(my_pred, test_y, 'ro')
plt.xlabel('Predictions', fontsize = 30)
plt.ylabel('Reality', fontsize = 30)
plt.title('Predictions x Reality on dataset Test', fontsize = 30)
ax.plot([test_y.min(), test_y.max()], [min(my_pred), max(my_pred)], 'k--', lw=4)
plt.show()


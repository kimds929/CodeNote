import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

# RNN Regressor (Weather Forcast)
dataset_path = r"C:\Users\Admin\Desktop\DataBase"
data = pd.read_table(f'{dataset_path}/12-22년_서울시_일단위_기온.csv', encoding='cp949')
data = pd.read_clipboard(sep='\t')
data.to_csv(f'{dataset_path}/12-22년_서울시_일단위_기온.csv', encoding='utf-8-sig',index=False)

data['일시'] = pd.to_datetime(data['일시'], format='%Y-%m-%d')


data_target = data.set_index('일시').resample('5D')['평균기온(℃)'].mean().reset_index()


# show graph
plt.figure(figsize=(15,3))
plt.plot(data_target['일시'], data_target['평균기온(℃)'])
plt.show()



# train_set
train_set = data_target[data_target['일시'] < '2022-01-01']
test_set = data_target[data_target['일시'] >= '2022-01-01']

# 결측치 확인
train_set['평균기온(℃)'].isna().sum()
X = train_set['평균기온(℃)'].values
# X = train_set['평균기온(℃)'].to_numpy()


# Recurrent DataSet
len(X)
window = 70

train_X = []
train_y = []
for idx in range(window, len(X)):
    train_X.append(X[idx-window:idx])
    train_y.append(X[idx])

train_X = np.stack(train_X)
train_y = np.stack(train_y)
print(train_X.shape, train_y.shape)

train_X_input = train_X[...,np.newaxis]
train_y_input = train_y[...,np.newaxis]
print(train_X_input.shape, train_y_input.shape)


# modeling
# l1 = tf.keras.layers.SimpleRNN(10)
# l1(train_X_input).shape

model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(50, return_sequences=True),
    tf.keras.layers.SimpleRNN(50),
    tf.keras.layers.Dense(1)
    ])
model.compile(optimizer='rmsprop', loss='mse')
model.fit(train_X_input, train_y_input, epochs=50, verbose=0)

# show model performance
pred = model.predict(train_X_input)[:,0]

plt.figure(figsize=(15,3))
plt.plot(train_set['일시'], train_set['평균기온(℃)'], label='real')
plt.plot(train_set.iloc[window:,:]['일시'], pred, label='pred', alpha=0.5)
plt.show()



# prediction new data
test_X = train_set.iloc[-window:,:]['평균기온(℃)'].to_numpy()
model.predict(test_X.reshape(1,-1,1))
test_set.head()


# multi_prediction
pred_y = None
n_pred = 70

test_Xs = []
pred = []
for _ in range(n_pred):
    if pred_y is None:
        test_X = train_set.iloc[-window:,:]['평균기온(℃)'].to_numpy()
    else:
        test_X = np.append(test_X[1:], pred_y)
    test_Xs.append(test_X)
    pred_y = model.predict(test_X.reshape(1,-1,1))[0][0]
    pred.append(pred_y)

# pd.DataFrame(np.stack(test_Xs)).T.to_clipboard()


pred_data = test_set.head(n_pred)

plt.figure(figsize=(15,3))
plt.plot(pred_data['일시'], pred_data['평균기온(℃)'], label='real')
plt.plot(pred_data['일시'], np.array(pred), label='pred', alpha=0.5)
plt.show()





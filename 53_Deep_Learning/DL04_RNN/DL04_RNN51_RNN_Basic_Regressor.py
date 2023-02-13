import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

# RNN Regressor (Weather Forcast)
url_path = 'https://raw.githubusercontent.com/kimds929/CodeNote/main/53_Deep_Learning/DL04_RNN'
data = pd.read_csv(f'{url_path}/12-22YR_Seoul_Temperature.csv', encoding='utf-8-sig')

# data = pd.read_clipboard(sep='\t')
# data.to_csv(f'{dataset_path}/12-22YR_Seoul_Temperature.csv', encoding='utf-8-sig',index=False)

data['일시'] = pd.to_datetime(data['일시'], format='%Y-%m-%d')


data_target = data.set_index('일시').resample('5D')['평균기온(℃)'].mean().reset_index()


# show graph
plt.figure(figsize=(15,3))
plt.plot(data_target['일시'], data_target['평균기온(℃)'])
plt.show()



# train_set
train_set = data_target[data_target['일시'] < '2022-01-01']
test_set = data_target[data_target['일시'] >= '2022-01-01']
# train_set.shape, test_set.shape


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








# Torch AUS Weather #################################################################################################################################
import torch

# dataset_path = r'D:\Python\★★Python_POSTECH_AI\Dataset'
# path ='/home/kimds929/DataSet/attention_mnist_small.pkl'
# mnist_small = cPickle.load(open(f"{dataset_path}/attention_mnist_small.pkl", 'rb') )

# (Python Dataset) AUS_Weather 230206
dataset_path = r'D:\작업방\업무 - 자동차 ★★★\Dataset'
df_w = pd.read_csv(f"{dataset_path}/weatherAUS_merge.csv", encoding='utf-8-sig')
# df_wine = pd.read_csv(f"{dataset_path}/wine_aroma.csv", encoding='utf-8-sig')

from datetime import datetime
# datetime.strptime?
# datetime.strftime?
df_w['Date'] = df_w['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
df_w0 = df_w.set_index('Date')
df_w0['Temp9am_tomorrow'] = df_w0['Temp9am'].shift(1)

df_w_sample1 = df_w0[(df_w0.index >= '2017-05-01') & (df_w0.index <= '2017-05-05')]
df_w_sample2 = df_w0[(df_w0.index >= '2017-05-06') & (df_w0.index <= '2017-05-10')]
X_cols = [
    # 'Date',
        'Location', 'lat', 'lng', 'Rainfall', 
 'MinTemp','MaxTemp', 'WindDir3pm',
 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed',
 'WindSpeed3pm', 'Humidity3pm', 'Pressure3pm', 'Cloud3pm', 'Temp3pm',
 'RainToday', 'RainTomorrow']
X_cols_simple = [
    # 'Date',
    'lat', 'lng', 'Humidity3pm', 'Pressure3pm', 'Temp3pm']

df_w0
y_col_reg = 'Temp9am_tomorrow'
y_col_clf = 'RainTomorrow'
y_cols = [y_col_reg, y_col_clf]

df_ws1 = df_w_sample1[y_cols + X_cols_simple]
df_ws2 = df_w_sample2[y_cols + X_cols_simple]


df_ws1.isna().sum(0)
df_ws2.isna().sum(0)
# import missingno as msno
# msno.matrix(df_ws)

df_ws1_ff = df_ws1.ffill()
df_ws2_ff = df_ws2.ffill()

# ---------------------------------------------------------------------------
df_train = df_ws1_ff.copy()
df_test = df_ws2_ff.copy()

from sklearn.preprocessing import StandardScaler, OneHotEncoder
scalerX = StandardScaler()
df_X_train = scalerX.fit_transform(df_train[X_cols_simple])
df_X_test = scalerX.transform(df_test[X_cols_simple])
# df_ws2 = pd.DataFrame(, columns=df_target.columns, index=df_target.index)

# scalerY_reg = StandardScaler()
# df_yreg_train = scalerY_reg.fit_transform(df_train[[y_col_reg]])[:,0]
# df_yreg_test = scalerY_reg.fit_transform(df_test[[y_col_reg]])[:,0]
df_yreg_train = df_train[y_col_reg]
df_yreg_test = df_test[y_col_reg]

scalerY_clf = OneHotEncoder(sparse=False, drop='first')
df_yclf_train = scalerY_clf.fit_transform(df_train[[y_col_clf]])[:,0]
df_yclf_test = scalerY_clf.fit_transform(df_test[[y_col_clf]])[:,0]

train_X = torch.Tensor(df_X_train)
train_y_reg = torch.Tensor(df_yreg_train)
train_y_clf = torch.tensor(df_yclf_train)

test_X = torch.Tensor(df_X_test)
test_y_reg = torch.Tensor(df_yreg_test)
test_y_clf = torch.tensor(df_yclf_test)

print(train_X.shape, train_y_reg.shape, train_y_clf.shape)
print(test_X.shape, test_y_reg.shape, test_y_clf.shape)

X_sample = train_X[:10]
yreg_sample = train_y_reg[:10]
yclf_sample = train_y_clf[:10]

X0 = torch.Tensor(X_sample)
y_reg0 = torch.Tensor(yreg_sample)
y_clf0 = torch.tensor(yclf_sample)
print(X0.shape, y_reg0.shape, y_clf0.shape)

train_dataset = torch.utils.data.TensorDataset(train_X, train_y_reg, train_y_clf)
test_dataset = torch.utils.data.TensorDataset(test_X, test_y_reg, test_y_clf)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)




# ------------------------------------------------------------
# a = torch.rand(1,4).requires_grad_(False) # Gradient 부여
# a.requires_grad       # gradient 여부 확인
# ------------------------------------------------------------
import time
import copy
import torch

# # 30 epoch) train_loss: 18.6642, test_loss: 8.582095
# class FC_Model2(torch.nn.Module):
#     def __init__(self, n_input):
#         super().__init__()
        
#         self.fc1 = torch.nn.Linear(n_input, 25)
#         self.fc2 = torch.nn.Linear(25, 1)
#         self.dropout = torch.nn.Dropout(0.5)
        
#     def forward(self, X):
#         self.fc1_result = self.dropout(torch.relu(self.fc1(X)))
#         self.fc2_result = self.fc2(self.fc1_result)
#         return self.fc2_result



# 10000 epoch) train_loss: 26.2743, test_loss: 27.9647
class FC_Model1(torch.nn.Module):
    def __init__(self, n_input):
        super().__init__()
        
        self.fc1 = torch.nn.Linear(n_input, 1)
        
    def forward(self, X):
        self.fc1_result = self.fc1(X)
        return self.fc1_result


# 2000 epoch) train_loss: 28.2434, test_loss: 28.24403
class FC_Model1_1(torch.nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.w = torch.nn.Parameter(torch.rand(1,n_input))
        self.b = torch.nn.Parameter(torch.rand(1))
        
    def forward(self, X):
        self.result = torch.matmul(X, self.w.T) + self.b
        return self.result

# 2000 epoch) train_loss: 27.6543, test_loss: 27.92992
class FC_Model1_2(torch.nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.rand(1,1))
        self.w_ = torch.nn.Parameter(torch.rand(1,n_input-1))
        self.b = torch.nn.Parameter(torch.rand(1))
        
    def forward(self, X):
        y1 = torch.matmul(X[:,:1], self.w1.T)
        y_ = torch.matmul(X[:,1:], self.w_.T)
        
        self.result = y1 + y_ + self.b
        return self.result

# 2000 epoch) train_loss: 28.6798, test_loss: 28.52359
class FC_Model1_3(torch.nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.rand(1,1))
        self.w_ = torch.nn.Parameter(torch.rand(1,n_input-1))
        self.b = torch.nn.Parameter(torch.rand(1))
        
    def forward(self, X):
        y1 = torch.matmul(X[:,:1], -abs(self.w1.T))
        y_ = torch.matmul(X[:,1:], self.w_.T)
        
        self.result = y1 + y_ + self.b
        return self.result

# 2000 epoch) train_loss: 28.8878, test_loss: 28.65046
class FC_Model1_4(torch.nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.w = torch.nn.Parameter(torch.rand(1,n_input))
        self.b = torch.nn.Parameter(torch.rand(1))
        
    def forward(self, X):
        y1 = torch.matmul(X[:,:1], -abs(self.w[:,:1].T))
        y_ = torch.matmul(X[:,1:], self.w[:,1:].T)
        
        self.result = y1 + y_ + self.b
        return self.result

# 2000 epoch) train_loss: 28.9768, test_loss: 28.55010
class FC_Model1_5(torch.nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.w = torch.nn.Parameter(torch.rand(1,n_input))
        self.b = torch.nn.Parameter(torch.rand(1))
        
    def forward(self, X):
        y1 = torch.matmul(X[:,:1], -abs(self.w[:,:1].T))
        y_ = torch.matmul(X[:,1:-1], self.w[:,1:-1].T)
        yf = torch.matmul(X[:,-1:], -abs(self.w[:,-1:].T))
        
        self.result = y1 + y_ + yf + self.b
        return self.result



# X0
# y_reg0, y_clf0

model_fc1 = FC_Model1(5)
model_fc1_1 = FC_Model1_1(5)
model_fc1_2 = FC_Model1_2(5)
model_fc1_3 = FC_Model1_3(5)
model_fc1_4 = FC_Model1_4(5)
model_fc1_5 = FC_Model1_5(5)
# model_fc2 = FC_Model2(5)


# torch.inverse(train_X.T@train_X) @ train_X.T @ train_y_reg
# tensor([ 2.3579, -0.1187,  1.6327, -0.3875,  2.2633])

model_fc1.state_dict()
# OrderedDict([('fc1.weight',
#               tensor([[0.3546, 0.2552, 0.4976, 0.1026, 0.5836]])),
#              ('fc1.bias', tensor([15.2487]))])
model_fc1_1.state_dict()
# OrderedDict([('w', tensor([[0.5487, 0.2236, 0.3701, 0.0541, 0.3021]])),
#              ('b', tensor([13.8503]))])
model_fc1_2.state_dict()
model_fc1_3.state_dict()
# OrderedDict([('w1', tensor([[0.0001]])),('w_', tensor([[ 0.2310,  0.3831, -0.0883,  0.6206]])),
#              ('b', tensor([13.7382]))])
model_fc1_4.state_dict()
# OrderedDict([('w', tensor([[ 0.0009,  0.2293,  0.3782, -0.0926,  0.6143]])),
#              ('b', tensor([13.6717]))])
model_fc1_5.state_dict()
# OrderedDict([('w', tensor([[ 0.0005,  0.2070,  0.1566, -0.3467,  0.0009]])),
#              ('b', tensor([13.7392]))])



# model(X0)
# model.state_dict()
# model_fc1.load_state_dict(model.state_dict())
# model_fc1_1.load_state_dict(model.state_dict())
# model_fc1_2.load_state_dict(model.state_dict())
# model_fc1_3.load_state_dict(model.state_dict())
# model_fc1_4.load_state_dict(model.state_dict())
# model_fc1_5.load_state_dict(model.state_dict())
# model_fc2.load_state_dict(model.state_dict())



# --------------------------------------------------------------
# model = model_fc1_2

model = copy.deepcopy(model_fc1_5)


loss_function = torch.nn.MSELoss()
# loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
epochs = 2000

losses = []
for epoch in range(epochs):
    model.train()
    loss_batch = []
    
    for batch_X, batch_yrge, batch_yclf in train_loader:
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = loss_function(pred, batch_yrge)
        # loss = loss_function(pred, batch_yclf)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            loss_batch.append(loss.to('cpu').detach().numpy())
    train_loss = np.mean(loss_batch)
    
    with torch.no_grad():
        model.eval()
        loss_batch = []
        for batch_X, batch_yrge, batch_yclf in test_loader:
            pred = model(batch_X)
            loss = loss_function(pred, batch_yrge)
            # loss = loss_function(pred, batch_yclf)

            with torch.no_grad():
                loss_batch.append(loss.to('cpu').detach().numpy())
    test_loss = np.mean(loss_batch)
    
    losses.append((train_loss, test_loss))
    # time.sleep(0.1)
    print(f'{epoch+1} epoch) train_loss: {losses[-1][0]:.4f}, test_loss: {losses[-1][1]:.4f}', end='\r')
# --------------------------------------------------------------


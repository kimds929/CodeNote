
####################################################################################################################
## 【 Temperature Dataset 】  ----------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

# database_path = r'C:\Users\Admin\Desktop\DataBase'
url_path = 'https://raw.githubusercontent.com/kimds929/CodeNote/main/99_DataSet'

df_temp = pd.read_csv(f'{url_path}/12-TimeSeries_12-22YR_Seoul_Temperature.csv', encoding='utf-8-sig')

df_temp['일시'] = pd.to_datetime(df_temp['일시'], format='%Y-%m-%d')

y_col = '평균기온(℃)'
X_cols = []
cols = [y_col] + X_cols

df_temp_5D = df_temp.set_index('일시').resample('5D')[cols].mean()
df_temp_10D = df_temp.set_index('일시').resample('10D')[cols].mean()
df_temp_20D = df_temp.set_index('일시').resample('20D')[cols].mean()


# df_target = df_temp.set_index('일시')[[cols]]
df_target = df_temp_10D.to_frame()
# # show graph
# plt.figure(figsize=(15,3))
# plt.plot(df_temp_5D['일시'], df_temp_5D[cols])
# plt.show()




## 【 AUC Weather Dataset 】   ----------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

# database_path = r'C:\Users\Admin\Desktop\DataBase'
url_path = 'https://raw.githubusercontent.com/kimds929/CodeNote/main/99_DataSet'
df_weather = pd.read_csv(f'{url_path}/TimeSeries_AUS_Weather_DataSet.csv', encoding='utf-8-sig')


df_weather['Date'] = pd.to_datetime(df_weather['Date'], format='%Y-%m-%d')
df_weather = df_weather.set_index('Date')
df_weather0 = df_weather.interpolate()

# X_cols = [
#     # 'Date',
#         'Location', 'lat', 'lng', 'Rainfall', 
#  'MinTemp','MaxTemp', 'WindDir3pm',
#  'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed',
#  'WindSpeed3pm', 'Humidity3pm', 'Pressure3pm', 'Cloud3pm', 'Temp3pm',
#  'RainToday']
X_cols = [
    # 'Date',
    'lat', 'lng', 'Humidity3pm', 'Pressure3pm', 'Temp3pm']

# y_cols = ['Temp9am', 'RainToday']
y_col = 'Temp9am'


df_weather1 = df_weather0[df_weather0['Location'] == 'Canberra']
print(df_weather1.index[0], df_weather1.index[-1])

df_target = df_weather1[[y_col] + X_cols]




## 【 FRED Dataset 】 ----------------------------------------------------------------------------------------------------
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # 시계열Data 
# database_path = r'C:\Users\Admin\Desktop\DataBase'
# df_fred = pd.read_csv(f"{database_path}/230210_FRED_Data.csv", encoding='utf-8-sig')
# df_fred['DATE'] = pd.to_datetime(df_fred['DATE'])
# df_fred = df_fred.set_index('DATE')
# # df_target.index = pd.date_range('1995-01-01','2023-03-31')    # Error

# import missingno as msno
# msno.matrix(df_fred)

# # # (Date Index Reset)
# # df_fred0 = df_fred.reset_index()
# # date_range = pd.date_range(df_fred.index[0], df_fred.index[-1]).to_frame()
# # date_range.columns = ['DATE']
# # df_fred1 = pd.merge(left=df_fred0, right=date_range, how='outer', on='DATE').set_index('DATE').sort_index()
# # df_fred1.to_csv(f"{database_path}/230210_FRED_Data.csv", encoding='utf-8-sig')
# df_target = df_fred.copy()
# import missingno as msno
# msno.matrix(df_target)

# pandas_interpolate_methods =  ['linear', 'time', 'index', 'values', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 
#  'barycentric', 'krogh', 'spline', 'polynomial', 'from_derivatives', 'piecewise_polynomial', 
#  'pchip', 'akima', 'cubicspline']
# interpolate = {
# 'US_M1': 'quadratic'
# ,'US_M2': 'quadratic'
# ,'USD/KRW': 'linear'
# ,'KOPSPI': 'quadratic'
# ,'KOSDAQ': 'quadratic'
# ,'S&P500': 'quadratic'
# ,'DOW': 'quadratic'
# ,'NASDAQ': 'quadratic'
# ,'QQQ': 'quadratic'
# ,'US_BASE_RATE': 'ffill'
# ,'US_GDP': 'quadratic'
# ,'VIX': 'linear'
# ,'TED_SPREAD': 'linear'
# }
# for c in df_target.columns:
#     print(c)
#     if interpolate[c] in pandas_interpolate_methods:
#         df_target[c] = df_target[c].interpolate(method=interpolate[c])
#     elif interpolate[c] == 'ffill':
#         df_target[c] = df_target[c].ffill()

# not_na_index = df_target[~df_target.isna().any(1)].index
# df_target1 = df_target[not_na_index[0]:not_na_index[-1]]
# df_target1
# df_target1.to_csv(f"{database_path}/230210_FRED_Data_interpolate_fillna.csv", encoding='utf-8-sig')

# msno.matrix(df_target1)


# for c in df_target1.columns:
#     # c = 'US_BASE_RATE'
#     plt.plot(df_target1.index, df_target1[c], label=c)
# plt.legend(bbox_to_anchor=(1,1))


# # ---------------------------------------------------------------------------------------------------------
# # df.interpolate(method='quadratic')
# # df = pd.DataFrame(np.array([[1,np.nan,np.nan,4,np.nan,np.nan,-2],[100,np.nan,300,np.nan,200,np.nan,600]]).T,columns=['A','B'])
# # df.interpolate()
# # df.interpolate(method='polynomial', order=2)
# # df.interpolate(method='spline', order=2)
# #  ['linear', 'time', 'index', 'values', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 
# #  'barycentric', 'krogh', 'spline', 'polynomial', 'from_derivatives', 'piecewise_polynomial', 
# #  'pchip', 'akima', 'cubicspline']
# # ---------------------------------------------------------------------------------------------------------


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

# database_path = r'C:\Users\Admin\Desktop\DataBase'
url_path = 'https://raw.githubusercontent.com/kimds929/CodeNote/main/99_DataSet'
df_fred_ = pd.read_csv(f"{url_path}/TimeSeries_230210_FRED_Data_interpolate_fillna.csv", encoding='utf-8-sig')
df_fred_['DATE'] = pd.to_datetime(df_fred_['DATE'])
df_fred_ = df_fred_.set_index('DATE')
print(df_fred_.index[0], df_fred_.index[-1])

import missingno as msno
msno.matrix(df_fred_)


df_fred_.columns
['US_M1', 'US_M2', 'USD/KRW', 'KOPSPI', 'KOSDAQ', 'S&P500', 'DOW',
'NASDAQ', 'QQQ', 'US_BASE_RATE', 'US_GDP', 'VIX', 'TED_SPREAD']

# y_col = 'KOPSPI'
# X_cols = ['US_M1', 'US_M2', 'USD/KRW', 'S&P500','US_BASE_RATE', 'US_GDP', 'VIX', 'TED_SPREAD']
y_col = 'S&P500'
X_cols = ['US_M1', 'US_M2', 'US_BASE_RATE', 'US_GDP', 'VIX', 'TED_SPREAD']
cols = [y_col] + X_cols

# np.where('US_M2' == df_target.columns)[0][0]

df_fred_2 = df_fred_.resample('10D')[cols].mean()

# df_target = df_fred_['2020-01-01':'2020-12-31']
df_target = df_fred_2['2010-01-01':'2022-12-31']


df_target[f"{y_col}_shift"] = (df_target[y_col].shift(-1) - df_target[y_col])/df_target[y_col]
# ----------------------------------------------------------------------------------------------------------------------







## 【 PreProcessing 】----------------------------------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
# scaler = MinMaxScaler()
scaler.fit(df_target)
df_scale = pd.DataFrame(scaler.transform(df_target),columns=df_target.columns, index=df_target.index)
df_scale
# df_scale = df_target.copy()


# graph
data_shape = df_scale.shape
plt.figure()
plt.figure(figsize=(15,3*data_shape[1]))
for e, c in enumerate(df_scale.columns):
    plt.subplot(data_shape[1],1,e+1)
    plt.plot(df_scale.index, df_scale[c], label=c,alpha=0.5)
    plt.legend(bbox_to_anchor=(1,1))
# plt.yscale('symlog')
plt.show()


cols = [y_col] + X_cols

if f"{y_col}_shift" in df_scale.columns:
    df_anal = df_scale[[f"{y_col}_shift"] + X_cols]
else:
    df_anal = df_scale[cols]
    df_anal[f"{y_col}_shift"] = df_anal[y_col].shift(-1)
df_anal = df_anal.dropna()
df_anal.shape   # 365,7
df_anal



# Dataset Transform (Time-Series Shape)
n_data = len(df_anal)
window = 70
stacked_list = []
for i_start in range(n_data - window + 1):
    stacked_list.append(df_anal.iloc[i_start:i_start + window][[f"{y_col}_shift"]+X_cols])

data_stack = np.stack(stacked_list)
data_stack.shape    ##


y = data_stack[:,:,0]
X = data_stack[:,:,1:]

print(X.shape, y.shape)


##########
plt.figure()
plt.figure(figsize=(15,3))
plt.plot(df_anal[f"{y_col}_shift"], alpha=0.5, label='total range')
plt.plot(pd.DataFrame(y[:,-1], index=df_anal.index[window - 1:n_data]), alpha=0.5, label='window range')
plt.legend()
plt.show()

# Starting Point Check
print(df_anal.index[window - 1])
print(df_anal.iloc[window - 1][[f"{y_col}_shift"]+X_cols])
print(y[0,-1], X[0,-1,:])

from sklearn.model_selection import train_test_split
test_size = 50
X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(X, y, df_anal.index[window - 1:n_data], test_size=test_size, shuffle=False)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, train_index.shape, test_index.shape)
print(train_index[[0,-1]], test_index[[0, -1]])

###
plt.figure()
plt.figure(figsize=(15,3))
plt.plot(df_anal[f"{y_col}_shift"], alpha=0.5, label='total range y')
plt.plot(train_index, y_train[:,-1], alpha=0.5, label='train y')
plt.plot(test_index, y_test[:,-1], alpha=0.5, label='test y')
plt.legend(bbox_to_anchor=(1,1))
plt.show()



################################################################################################################################################################
# 1Dim Prediction : # data prediction using only y

import torch

X_train_1D = torch.Tensor(y_train[:,:-1]).unsqueeze(2)
y_train_1D = torch.Tensor(y_train[:,-1])

X_test_1D = torch.Tensor(y_test[:,:-1]).unsqueeze(2)
y_test_1D = torch.Tensor(y_test[:,-1])
print(X_train_1D.shape, y_train_1D.shape, X_test_1D.shape, y_test_1D.shape)

###
plt.figure(figsize=(15,3))
plt.plot(train_index, X_train_1D[:,-1], label='train range X')
plt.plot(train_index, y_train_1D, label='train range y')
plt.plot(test_index, X_test_1D[:,-1], alpha=0.5, ls='--', label='test range X')
plt.plot(test_index, y_test_1D, alpha=0.5, ls='--', label='test range y')
plt.legend(bbox_to_anchor=(1,1))
plt.show()

train_set_1D = torch.utils.data.TensorDataset(X_train_1D, y_train_1D)
test_set_1D = torch.utils.data.TensorDataset(X_test_1D, y_test_1D)

train_loader = torch.utils.data.DataLoader(train_set_1D, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_set_1D, batch_size=64)


X_Sample = X_train_1D[:3]
y_Sample = y_train_1D[:3]

print(X_Sample.shape, y_Sample.shape)



# --- Model ---------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(device)
# torch.cuda.empty_cache()

import sys
sys.path.append(r'C:\Users\Admin\Desktop\DataScience\★★ DS_Library')
from DS_DeepLearning import EarlyStopping


class RNN_Basic1(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc_embed_layer = torch.nn.Linear(input_dim, 32)
        self.rnn_layer = torch.nn.RNN(32, 64, batch_first=True)
        self.fc_layer = torch.nn.Linear(64, 1)

    def forward(self, X):
        self.fc_embed = self.fc_embed_layer(X)
        self.rnn_output, self.rnn_hidden = self.rnn_layer(self.fc_embed)
        self.fc = self.fc_layer(self.rnn_hidden.squeeze(0))
        return self.fc.squeeze(1)

class RNN_Basic2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        # self.fc_embed_layer = torch.nn.Linear(input_dim, 32)
        self.rnn_init = torch.nn.RNN(1, hidden_dim, batch_first=True)
        self.tanh = torch.nn.Tanh()
        self.rnn_layers = torch.nn.ModuleList([
            torch.nn.RNN(hidden_dim, hidden_dim, batch_first=True)
            # ,torch.nn.RNN(hidden_dim, hidden_dim, batch_first=True)
        ])
        self.fc_layer = torch.nn.Linear(hidden_dim, 1)

    def forward(self, X):
        X, hidden =  self.rnn_init(X)
        for rnn in self.rnn_layers:
            X, hidden = rnn(self.tanh(X), hidden)
        self.fc = self.fc_layer(hidden.squeeze(0))
        return self.fc.squeeze(1)

class RNN_Basic3(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc_embed_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.rnn_init = torch.nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.tanh = torch.nn.Tanh()
        self.rnn_layers = torch.nn.ModuleList([
            torch.nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            # ,torch.nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        ])
        self.fc_layer = torch.nn.Linear(hidden_dim, 1)

    def forward(self, X):
        X = self.fc_embed_layer(X)
        X, (hidden, cell) = self.rnn_init(X)
        for rnn in self.rnn_layers:
            X, (hidden, cell) = rnn(self.tanh(X), (hidden, cell))
        self.fc = self.fc_layer(hidden.squeeze(0))
        return self.fc.squeeze(1)


# --- Predict ---------------------------------------

# X_1D.shape, y_1D.shape
# rnn_basic_model1 = RNN_Basic1(input_dim=1).to(device)
# rnn_basic_model2 = RNN_Basic2(input_dim=1).to(device)
rnn_basic_model3 = RNN_Basic3(input_dim=1).to(device)
# rnn_basic_model3(X_Sample.to(device))



# rnn_basic_model1.load_state_dict(model.state_dict())
# rnn_basic_model1.load_state_dict(es.optimum[2])
# rnn_basic_model2.load_state_dict(model.state_dict())
# rnn_basic_model2.load_state_dict(es.optimum[2])
# rnn_basic_model3.load_state_dict(model.state_dict())
# rnn_basic_model3.load_state_dict(es.optimum[2])


# model = copy.deepcopy(rnn_basic_model1)
# model = copy.deepcopy(rnn_basic_model2)
model = copy.deepcopy(rnn_basic_model3)
# model(X_Sample.to(device))

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters())
epochs = 50

es = EarlyStopping(patience=50)
losses = []
for epoch in range(epochs):
    model.train()
    loss_batch = []
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        pred = model(batch_X.to(device))
        loss = loss_function(pred, batch_y.to(device))
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            loss_batch.append(loss.to('cpu').detach().numpy())
    train_loss = np.mean(loss_batch)
    
    with torch.no_grad():
        model.eval()
        loss_batch = []
        for batch_X, batch_y in test_loader:
            pred = model(batch_X.to(device))
            loss = loss_function(pred, batch_y.to(device))

            with torch.no_grad():
                loss_batch.append(loss.to('cpu').detach().numpy())
    valid_loss = np.mean(loss_batch)
    
    losses.append((train_loss, valid_loss))
    # time.sleep(0.1)
    # print(f'{epoch+1} epoch) train_loss: {losses[-1][0]:.4f}, test_loss: {losses[-1][1]:.4f}', end='\r')
    early_stop = es.early_stop(score=valid_loss, reference_score=train_loss, save=model.state_dict(), verbose=2)

    if early_stop == 'break':
        break

es.plot

# plt.plot(np.stack(losses)[:,0])
# plt.plot(np.stack(losses)[:,1])


# pred_model = copy.deepcopy(rnn_basic_model1)
# pred_model = copy.deepcopy(rnn_basic_model2)
pred_model = copy.deepcopy(rnn_basic_model3)

# predict
with torch.no_grad():
    pred_model.eval()
    pred_train = pred_model(X_train_1D.to(device))
    pred_test = pred_model(X_test_1D.to(device))

plt.figure(figsize=(15,3))
plt.plot(train_index, y_train_1D, label='train_real', alpha=0.7)
plt.plot(train_index, pred_train.to('cpu'), label='train_pred', alpha=0.7)
plt.plot(test_index, y_test_1D, alpha=0.5, ls='--', label='test_real')
plt.plot(test_index, pred_test.to('cpu'), alpha=0.5, ls='--', label='test_pred')
plt.legend(bbox_to_anchor=(1,1))
plt.show()



# predict
X_input_before = X_train_1D[-1].unsqueeze(0)

pred = []
for _ in range(test_size):
    with torch.no_grad():
        pred_model.eval()
        pred_next = pred_model(X_input_before.to(device))
        pred.append(pred_next.to('cpu').detach().numpy()[0])
        X_input_before = torch.cat([X_input_before[:,1:,:].to(device), pred_next.unsqueeze(0).unsqueeze(2)], dim=1)
        # print(X_input_before.squeeze())
np.stack(pred)

plt.figure(figsize=(15,3))
plt.plot(test_index, y_test_1D, alpha=0.5, ls='--', label='test_real')
plt.plot(test_index, pred_test.to('cpu'), alpha=0.5, ls='--', label='test_pred')
plt.plot(test_index, np.stack(pred), label='test_iter_pred')
plt.legend(bbox_to_anchor=(1,1))
plt.show()



# Scaler Inverse Transform -----------------------------------------------------------------
y_mean, y_std = df_target[f'{y_col}_shift'].agg(['mean','std'])
df_y_real = df_target[y_col] * (1+df_target[f'{y_col}_shift'])

train_pred_inv
train_pred_inv = pd.Series( (pred_train.to('cpu').detach().numpy() * y_std)+y_mean, index=train_index)


def get_inv_predict(origin_data, y_col, pred_prob, scale_mean, scale_std, index):
    pred_inv = pd.Series( (pred_prob * scale_std) + scale_mean, index=index)
    pred_list = [ origin_data[y_col][pred_inv.index[0]] ]

    for time_index, next_prob in pred_inv.items():
        pred_list.append( pred_list[-1] *(1+next_prob) )
    y_pred = pd.Series(pred_list[:-1], index=index)
    return y_pred


train_y_pred = get_inv_predict(df_target, y_col, pred_train.to('cpu').detach().numpy(), y_mean, y_std, train_index)
test_y_pred = get_inv_predict(df_target, y_col, pred_test.to('cpu').detach().numpy(), y_mean, y_std, test_index)
test_pred_iter_inv = get_inv_predict(df_target, y_col, np.stack(pred), y_mean, y_std, test_index)
pd.concat([df_y_real, train_y_pred, test_y_pred, test_pred_iter_inv], axis=1).to_clipboard()

plt.figure(figsize=(15,3))
plt.plot(df_y_real, label='real_y')
plt.plot(train_y_pred, alpha=0.5, label='pred_train_y')
plt.plot(test_y_pred, alpha=0.5, ls='--', label='pred_test_y')
plt.plot(test_pred_iter_inv, alpha=0.5, ls='--', label='pred_test_y_iter')
plt.legend(bbox_to_anchor=(1,1))
plt.show()









################################################################################################################################################################
# 2Dim-1Dim Prediction : # data prediction using only y
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, train_index.shape, test_index.shape)
print(train_index[[0,-1]], test_index[[0, -1]])

import torch

X_train_2D = torch.Tensor(X_train)
y_train_2D = torch.Tensor(y_train[:,-1])

X_test_2D = torch.Tensor(X_test)
y_test_2D = torch.Tensor(y_test[:,-1])
print(X_train_2D.shape, y_train_2D.shape, X_test_2D.shape, y_test_2D.shape)


###
X_shape = X_train.shape
plt.figure(figsize=(15,3*(X_shape[-1]+1)))
plt.subplot(data_shape[-1]+1,1,1)
plt.plot(train_index, y_train_2D, label=f'train_y:{y_col}')
plt.plot(test_index, y_test_2D, ls='--', label=f'test_y:{y_col}')
plt.legend(bbox_to_anchor=(1,1))
for e, c in enumerate(X_cols):
    plt.subplot(data_shape[-1]+1,1,e+2)
    plt.plot(train_index, X_train_2D[:,-1,e], alpha=0.5, label=f"train_X: {c}")
    plt.plot(test_index, X_test_2D[:,-1,e], alpha=0.5, ls='--', label=f"train_X: {c}")
    plt.legend(bbox_to_anchor=(1,1))
plt.show()

train_set_2D = torch.utils.data.TensorDataset(X_train_2D, y_train_2D)
test_set_2D = torch.utils.data.TensorDataset(X_test_2D, y_test_2D)

train_loader = torch.utils.data.DataLoader(train_set_2D, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_set_2D, batch_size=64)

X_Sample = X_train_2D[:3]
y_Sample = y_train_2D[:3]

print(X_Sample.shape, y_Sample.shape)





# --- Model ---------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(device)
# torch.cuda.empty_cache()

import sys
sys.path.append(r'C:\Users\Admin\Desktop\DataScience\★★ DS_Library')
from DS_DeepLearning import EarlyStopping


class RNN_Basic1(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc_embed_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.rnn_layer = torch.nn.RNN(hidden_dim, hidden_dim, batch_first=True)
        self.fc_layer = torch.nn.Linear(hidden_dim, 1)

    def forward(self, X):
        self.fc_embed = self.fc_embed_layer(X)
        self.rnn_output, self.rnn_hidden = self.rnn_layer(self.fc_embed)
        self.fc = self.fc_layer(self.rnn_hidden.squeeze(0))
        return self.fc.squeeze(1)

class RNN_Basic2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc_embed_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.rnn_init = torch.nn.RNN(hidden_dim, hidden_dim, batch_first=True)
        self.tanh = torch.nn.Tanh()
        self.rnn_layers = torch.nn.ModuleList([
            torch.nn.RNN(hidden_dim, hidden_dim, batch_first=True)
            # ,torch.nn.RNN(hidden_dim, hidden_dim, batch_first=True)
        ])
        self.fc_layer = torch.nn.Linear(hidden_dim, 1)

    def forward(self, X):
        self.fc_embed = self.fc_embed_layer(X)
        X, hidden =  self.rnn_init(self.fc_embed)
        for rnn in self.rnn_layers:
            X, hidden = rnn(self.tanh(X), hidden)
        self.fc = self.fc_layer(hidden.squeeze(0))
        return self.fc.squeeze(1)

class RNN_Basic3(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc_embed_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.rnn_init = torch.nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.tanh = torch.nn.Tanh()
        self.rnn_layers = torch.nn.ModuleList([
            torch.nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            ,torch.nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        ])
        self.fc_layer = torch.nn.Linear(hidden_dim, 1)

    def forward(self, X):
        self.fc_embed = self.fc_embed_layer(X)
        X, (hidden, cell) = self.rnn_init(self.fc_embed)
        for rnn in self.rnn_layers:
            X, (hidden, cell) = rnn(self.tanh(X), (hidden, cell))
        self.fc = self.fc_layer(hidden.squeeze(0))
        return self.fc.squeeze(1)


X_shape = X_train.shape
# X_Sample.shape, y_Sample.shape
# rnn_basic_model1 = RNN_Basic1(input_dim=X_shape[-1]).to(device)
# rnn_basic_model2 = RNN_Basic2(input_dim=X_shape[-1]).to(device)
rnn_basic_model3 = RNN_Basic3(input_dim=X_shape[-1], hidden_dim=64).to(device)
# rnn_basic_model3(X_Sample.to(device))


# rnn_basic_model1.load_state_dict(model.state_dict())
# rnn_basic_model1.load_state_dict(es.optimum[2])
# rnn_basic_model2.load_state_dict(model.state_dict())
# rnn_basic_model2.load_state_dict(es.optimum[2])
# rnn_basic_model3.load_state_dict(model.state_dict())
# rnn_basic_model3.load_state_dict(es.optimum[2])


# model = copy.deepcopy(rnn_basic_model1)
# model = copy.deepcopy(rnn_basic_model2)
model = copy.deepcopy(rnn_basic_model3)
# model(X_Sample.to(device))

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters())
epochs = 50

es = EarlyStopping(patience=50)
losses = []
for epoch in range(epochs):
    model.train()
    loss_batch = []
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        pred = model(batch_X.to(device))
        loss = loss_function(pred, batch_y.to(device))
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            loss_batch.append(loss.to('cpu').detach().numpy())
    train_loss = np.mean(loss_batch)
    
    with torch.no_grad():
        model.eval()
        loss_batch = []
        for batch_X, batch_y in test_loader:
            pred = model(batch_X.to(device))
            loss = loss_function(pred, batch_y.to(device))

            with torch.no_grad():
                loss_batch.append(loss.to('cpu').detach().numpy())
    valid_loss = np.mean(loss_batch)
    
    losses.append((train_loss, valid_loss))
    # time.sleep(0.1)
    # print(f'{epoch+1} epoch) train_loss: {losses[-1][0]:.4f}, test_loss: {losses[-1][1]:.4f}', end='\r')
    early_stop = es.early_stop(score=valid_loss, reference_score=train_loss, save=model.state_dict(), verbose=2)

    if early_stop == 'break':
        break

es.plot



# pred_model = copy.deepcopy(rnn_basic_model1)
# pred_model = copy.deepcopy(rnn_basic_model2)
pred_model = copy.deepcopy(rnn_basic_model3)

# predict
with torch.no_grad():
    pred_model.eval()
    pred_train = pred_model(X_train_2D.to(device))
    pred_test = pred_model(X_test_2D.to(device))

plt.figure(figsize=(15,3))
plt.plot(train_index, y_train_2D, label='train_real', alpha=0.7)
plt.plot(train_index,pred_train.to('cpu'), label='train_pred', alpha=0.7)
plt.plot(test_index, y_test_2D, alpha=0.5, ls='--', label='test_real')
plt.plot(test_index, pred_test.to('cpu'), alpha=0.5, ls='--', label='test_pred')
plt.legend(bbox_to_anchor=(1,1))
plt.show()


### -------------------------------------------------------------------------
def get_inv_predict(origin_data, y_col, pred_prob, scale_mean, scale_std, index):
    pred_inv = pd.Series( (pred_prob * scale_std) + scale_mean, index=index)
    pred_list = [ origin_data[y_col][pred_inv.index[0]] ]

    for time_index, next_prob in pred_inv.items():
        pred_list.append( pred_list[-1] *(1+next_prob) )
    y_pred = pd.Series(pred_list[:-1], index=index)
    return y_pred


train_y_pred = get_inv_predict(df_target, y_col, pred_train.to('cpu').detach().numpy(), y_mean, y_std, train_index)
test_y_pred = get_inv_predict(df_target, y_col, pred_test.to('cpu').detach().numpy(), y_mean, y_std, test_index)
pd.concat([df_y_real, train_y_pred, test_y_pred], axis=1).to_clipboard()

plt.figure(figsize=(15,3))
plt.plot(df_y_real, label='real_y')
plt.plot(train_y_pred, alpha=0.5, label='pred_train_y')
plt.plot(test_y_pred, alpha=0.5, ls='--', label='pred_test_y')
plt.legend(bbox_to_anchor=(1,1))
plt.show()

###############





################################################################################################################################################################
# 2Dim-2Dim Prediction : # data prediction using only y
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, train_index.shape, test_index.shape)
print(train_index[[0,-1]], test_index[[0, -1]])

import torch
X_train.shape
X_train_2D = torch.Tensor(X_train)
y_train_2D = torch.Tensor(y_train)

X_test_2D = torch.Tensor(X_test)
y_test_2D = torch.Tensor(y_test[:,-1])
print(X_train_2D.shape, y_train_2D.shape, X_test_2D.shape, y_test_2D.shape)



plt.figure(figsize=(15,3))
plt.plot(train_index, X_train_2D[:,0,:], alpha=0.2)
plt.plot(train_index, y_train_2D)
plt.plot(test_index, X_test_2D[:,0,:], alpha=0.2, ls='--')
plt.plot(test_index, y_test_2D, ls='--')
plt.show()

train_set_2D = torch.utils.data.TensorDataset(X_train_2D, y_train_2D)
test_set_2D = torch.utils.data.TensorDataset(X_test_2D, y_test_2D)

train_loader = torch.utils.data.DataLoader(train_set_2D, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_set_2D, batch_size=64)

X_Sample = X_train_2D[:3]
y_Sample = y_train_2D[:3]

print(X_Sample.shape, y_Sample.shape)



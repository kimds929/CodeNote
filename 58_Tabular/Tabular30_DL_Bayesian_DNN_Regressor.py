## 【 Bayesian Deep Learning Regression 】##############################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(device)


# from sklearn.datasets import load_boston
# load_skdata = load_boston()

# data_boston = pd.DataFrame(load_skdata['data'], columns=load_skdata.feature_names)
# data_boston['Target']=load_skdata['target']

data_url = 'https://raw.githubusercontent.com/kimds929/CodeNote/main/99_DataSet/Data_Tabular/'
data_boston = pd.read_csv(data_url + "boston_house.csv")

data_boston.head(5)

data = data_boston.drop('CHAS', axis=1)


# train-valid-test split
from sklearn.model_selection import train_test_split
train_valid_idx, test_idx = train_test_split(range(len(data)), test_size=0.3)
train_idx, valid_idx = train_test_split(train_valid_idx, test_size=0.2)

train_X = data.iloc[train_idx, :-1]
train_y = data.iloc[train_idx, [-1]]
valid_X = data.iloc[valid_idx, :-1]
valid_y = data.iloc[valid_idx, [-1]]
test_X = data.iloc[test_idx, :-1]
test_y = data.iloc[test_idx, [-1]]
print(train_X.shape, train_y.shape, valid_X.shape, valid_y.shape, test_X.shape, test_y.shape)

# normalize
from sklearn.preprocessing import StandardScaler
sd_X = StandardScaler()
train_X_norm = sd_X.fit_transform(train_X)
valid_X_norm = sd_X.transform(valid_X)
test_X_norm = sd_X.transform(test_X)

sd_y = StandardScaler()
train_y_norm = sd_y.fit_transform(train_y)
valid_y_norm = sd_y.transform(valid_y)
test_y_norm = sd_y.transform(test_y)

train_X_torch = torch.Tensor(train_X_norm)
train_y_torch = torch.Tensor(train_y_norm)
valid_X_torch = torch.Tensor(valid_X_norm)
valid_y_torch = torch.Tensor(valid_y_norm)
test_X_torch = torch.Tensor(test_X_norm)
test_y_torch = torch.Tensor(test_y_norm)

# dataset, dataloader
train_dataset = torch.utils.data.TensorDataset(train_X_torch, train_y_torch)
valid_dataset = torch.utils.data.TensorDataset(valid_X_torch, valid_y_torch)
test_dataset = torch.utils.data.TensorDataset(test_X_torch, test_y_torch)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)



for batch_X, batch_y in train_loader:
    break
print(len(train_loader), batch_X.shape, batch_y.shape)


# Bayesian Deep Neural Network Regressor
class BayesianDL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(12, 32)
            ,torch.nn.ReLU()
            ,torch.nn.Linear(32, 64)
            ,torch.nn.ReLU()
            ,torch.nn.Linear(64, 2)
        )
        self.training = True
        
    def forward(self, x):
        self.output = self.layers(x)
        self.mu, self.sigma = torch.split(self.output, 1, 1)
        
        if self.training:
            # reparameterization trick
            return self.mu + self.sigma * torch.randn(self.mu.shape)
        else:
            return self.mu


bayes_dl = BayesianDL() 
# bayes_dl.train()
# bayes_dl(batch_X)



# # customize library ***---------------------
# import sys
# sys.path.append(r'C:\Users\Admin\Desktop\DataScience\★★ DS_Library')
# from DS_DeepLearning import EarlyStopping

import httpimport
remote_url = 'https://raw.githubusercontent.com/kimds929/'
with httpimport.remote_repo(f"{remote_url}/DS_Library/main/"):
    from DS_DeepLearning import EarlyStopping

es = EarlyStopping(patience=100)
# # ------------------------------------------
import time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    
    return elapsed_mins, elapsed_secs



bayes_dl = BayesianDL() 

model = copy.deepcopy(bayes_dl)

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

n_epochs = 50

train_losses = []
valid_losses = []
for e in range(n_epochs):
    start_time = time.time() # 시작 시간 기록
    model.train()
    train_epoch_loss = []
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()   # optimizer initialize
        
        
        pred = model(batch_X.to(device))          # forward
        loss = loss_function(pred, batch_y.to(device))  # loss
        
        loss.backward()         # backward
        optimizer.step()        # weight update
        
        with torch.no_grad():
            train_batch_loss = loss.to('cpu').detach().numpy()
            train_epoch_loss.append( train_batch_loss )
    
    # valid_set evaluation *
    valid_epoch_loss = []
    with torch.no_grad():
        model.eval() 
        for batch_X, batch_y in valid_loader:
            pred = model(batch_X.to(device))          # forward
            loss = loss_function(pred, batch_y.to(device))  # loss
            
            valid_batch_loss = loss.to('cpu').detach().numpy()
            valid_epoch_loss.append( valid_batch_loss )
    
    with torch.no_grad():
        train_loss = np.mean(train_epoch_loss)
        valid_loss = np.mean(valid_epoch_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        end_time = time.time() # 종료 시간 기록
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # print(f'Epoch: {e + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        # print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {np.exp(train_loss):.3f}')
        # print(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {np.exp(valid_loss):.3f}')
        # customize library ***---------------------
        early_stop = es.early_stop(score=valid_loss, reference_score=train_loss, save=model.state_dict(), verbose=2)
        if early_stop == 'break':
            break
        # ------------------------------------------

# plt.figure()
# plt.plot(train_losses, 'o-')
# plt.plot(valid_losses, 'o-')
# plt.show()

# customize library ***---------------------
es.plot     # early_stopping plot

model.load_state_dict(es.optimum[2])    # optimum model (load weights)
# ------------------------------------------
bayes_dl.load_state_dict(model.state_dict())


for batch_X, batch_y in test_loader:
    break

with torch.no_grad():
    bayes_dl.eval()
    pred = bayes_dl(batch_X)
    sigma = sd_y.inverse_transform(bayes_dl.sigma.to('cpu').detach().numpy())


plt.scatter(sd_y.inverse_transform(batch_y), sd_y.inverse_transform(pred), color='steelblue')
plt.scatter(sd_y.inverse_transform(batch_y), sd_y.inverse_transform(pred)+sigma, alpha=0.2, color='steelblue')
plt.scatter(sd_y.inverse_transform(batch_y), sd_y.inverse_transform(pred)-sigma, alpha=0.2, color='steelblue')
plt.show()

print(f"MSE: {sd_y.inverse_transform( np.array([[es.optimum[1]]]) )}")
print(f"RMSE: {np.sqrt(sd_y.inverse_transform( np.array([[es.optimum[1]]]) ))}")


# 1D Partial Dependence
estimator = copy.deepcopy(bayes_dl)
X = train_X.copy()
X_mean = X.mean()
X_cols = X.columns
scaler_X = sd_X
scaler_y = sd_y


# ['AGE', 'B', 'RM', 'CRIM', 'DIS', 'INDUS', 'LSTAT', 'NOX', 'PTRATIO', 'RAD', 'ZN', 'TAX'],
x1 = 'PTRATIO'
n_points = 50

xp = np.linspace(X[x1].min(), X[x1].max(), n_points)

grid = pd.DataFrame(np.zeros((n_points, X.shape[1])), columns=X_cols)
for xc in X_cols:
    grid[xc] = xp if xc == x1 else X_mean[xc]

grid_norm = scaler_X.transform(grid)
grid_norm_torch = torch.Tensor(grid_norm)

with torch.no_grad():
    estimator.eval()
    pred = scaler_y.inverse_transform(estimator(grid_norm_torch).to('cpu').detach().numpy())
    sigma = np.sqrt(scaler_y.inverse_transform(estimator.sigma.to('cpu').detach().numpy()))


pred_df = pd.DataFrame(np.hstack([xp.reshape(-1,1), pred, sigma]), columns=[x1, 'mu' ,'sigma'])
pred_df

plt.figure()
plt.plot(pred_df[x1], pred_df['mu'], color='steelblue', ls='-', marker='o')
plt.fill_between(pred_df[x1], pred_df['mu']-pred_df['sigma'], pred_df['mu']+pred_df['sigma'], color='steelblue', alpha=0.2)
plt.show()




















"""
import copy


# --- Model ---------------------------------------

# torch.cuda.empty_cache()

class Embedding(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.embed_layer = torch.nn.Embedding(vocab_size, emb_dim)
        
    def forward(self, X):
        self.embed_output = self.embed_layer(X)
        return self.embed_output
    
xs = torch.tensor(train_X[X_cols_str_t].to_numpy())
xn = torch.tensor(train_X[X_cols_num_t].to_numpy()).type(torch.float32)


# 3 layer relu (128, 128, 128)
# batch norm
class DNN_Regressor(torch.nn.Module):
    def __init__(self, str_cols, num_cols, emb_dim=2, output_dim=1, hidden_dims=[128,128,128], dropout=0.1):
        super().__init__()
        self.embed_layer = Embedding(str_cols+1, emb_dim)
        self.flatten_layer = torch.nn.Flatten()
        
        fc_modules = []
        for node_in, node_out in zip([(str_cols)*emb_dim + num_cols, *hidden_dims], hidden_dims+[output_dim]):
            fc_modules.append( torch.nn.Linear(node_in, node_out) )
            fc_modules.append( torch.nn.BatchNorm1d(node_out) )
            fc_modules.append( torch.nn.ReLU() )
            
        self.fc_modules = torch.nn.ModuleList(fc_modules[:-2])
        self.dropout_layer = torch.nn.Dropout(dropout)
        
    def forward(self, X):
        # X : [X(str), X(num)]
        self.embed_output = self.embed_layer(X[0])
        self.embed_flatten = self.flatten_layer(self.embed_output)
        
        self.concat = torch.cat([self.embed_flatten, X[1]], dim=-1)
        output = self.concat
        
        for e, fc_layer in enumerate(self.fc_modules):
            # print(e, fc_layer)
            if e % 3 == 2:
                output_ = fc_layer(output)
                output = self.dropout_layer(output + output_)   # residual connection
            else:
                output = fc_layer(output)
        
        return output
"""
    




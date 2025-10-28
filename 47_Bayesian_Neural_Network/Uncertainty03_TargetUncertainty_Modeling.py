import os
import sys
sys.path.append('/home/pd299370/DataScience/DS_Library')
sys.path.append(r'D:\DataScience\00_DataAnalysis_Basic')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
import missingno as msno
from DS_Basic_Module import DF_Summary, ScalerEncoder

try:
    from DS_Torch import TorchDataLoader, TorchModeling, AutoML, early
    from DS_DeepLearning import EarlyStopping
except:
    remote_library_url = 'https://raw.githubusercontent.com/kimds929/'
    try:
        import httpimport
        with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
            from DS_Torch import TorchDataLoader, TorchModeling, AutoML
            from DS_DeepLearning import EarlyStopping
    except:
        import requests
        response = requests.get(f"{remote_library_url}/DS_Library/main/DS_Torch.py", verify=False)
        exec(response.text)
        
        response = requests.get(f"{remote_library_url}/DS_Library/main/DS_DeepLearning.py", verify=False)
        exec(response.text)

db_path = r'D:\DataScience\DataBase'

# df00 = pd.read_csv(f"{db_path}/Data_Tabular/datasets_Titanic_original.csv", encoding='utf-8-sig')
df00 = pd.read_csv(f"{db_path}/Data_Tabular/datasets_Boston_house.csv", encoding='utf-8-sig')
df00.info()
df00.head()

df_summary = DF_Summary(df00)
msno.matrix(df00)


# df00.columns
y_col = ['Target']
X_cols_num = ['AGE', 'B', 'RM', 'CRIM', 'DIS', 'INDUS', 'LSTAT', 'NOX', 'PTRATIO', 'ZN', 'TAX']
X_cols_cat = ['RAD', 'CHAS']


# --------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
df01 = df00.copy()
train_valid_idx, tests_idx = train_test_split(df01.index, test_size=0.2, shuffle=True)
train_idx, valid_idx = train_test_split(train_valid_idx, test_size=0.2, shuffle=True)

train_idx = sorted(train_idx)
valid_idx = sorted(valid_idx)
tests_idx = sorted(tests_idx)


# --------------------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler

ss_X = StandardScaler()
train_X_num_norm = ss_X.fit_transform(df01.loc[train_idx][X_cols_num])
valid_X_num_norm = ss_X.transform(df01.loc[valid_idx][X_cols_num])
tests_X_num_norm = ss_X.transform(df01.loc[tests_idx][X_cols_num])

train_X_cat = np.array(df01.loc[train_idx][X_cols_cat])
valid_X_cat = np.array(df01.loc[valid_idx][X_cols_cat])
tests_X_cat = np.array(df01.loc[tests_idx][X_cols_cat])


ss_y = StandardScaler()
train_y_norm = ss_y.fit_transform(df01.loc[train_idx][y_col])
valid_y_norm = ss_y.transform(df01.loc[valid_idx][y_col])
tests_y_norm = ss_y.transform(df01.loc[tests_idx][y_col])

# --------------------------------------------------------------------------------------
from torch.utils.data import TensorDataset, DataLoader
train_tensor = (torch.FloatTensor(train_X_num_norm), torch.LongTensor(train_X_cat), torch.FloatTensor(train_y_norm))
valid_tensor = (torch.FloatTensor(valid_X_num_norm), torch.LongTensor(valid_X_cat), torch.FloatTensor(valid_y_norm))
tests_tensor = (torch.FloatTensor(tests_X_num_norm), torch.LongTensor(tests_X_cat), torch.FloatTensor(tests_y_norm))

train_dataset = TensorDataset(*train_tensor)
valid_dataset = TensorDataset(*valid_tensor)
tests_dataset = TensorDataset(*tests_tensor)


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
tests_loader = DataLoader(tests_dataset, batch_size=64, shuffle=True)


####################################################################################################
####################################################################################################
####################################################################################################

# --------------------------------------------------------------------------------------

class PseudoData():
    def __init__(self, max_categories=None, window_alpha=2, window_beta=0.5, feature_gamma=0.5,   max_samples=1e+6):
        """
        window_alpha : 얼마나 넓은영역까지 data generation할지? max-min ± (max - min) * windonw_alpha
        window_beta : data generation - window 가중치
        feature_gamma : data generation - feature dimension window 가중치
        max_categories : categorical data (max_categories)
        """
        self.max_categories = max_categories.amax(dim=tuple(range(max_categories.ndim - 1))).tolist()  if 'torch.Tensor' in str(type(max_categories)) else max_categories
        self.window_alpha = window_alpha 
        self.window_beta = window_beta
        self.feature_gamma = feature_gamma
        
        self.max_samples = max_samples
        
    def gen_num_pseudo_data(self, x_num, n_samples=None, window_alpha=None, window_beta=None, feature_gamma=None, max_samples=None):
        window_alpha = self.window_alpha if window_alpha is None else window_alpha
        window_beta = self.window_beta if window_beta is None else window_beta
        feature_gamma = self.feature_gamma if feature_gamma is None else feature_gamma
        max_samples = self.max_samples if max_samples is None else max_samples

        x_num_shape = x_num.shape
        device = x_num.device
        dtype = x_num.dtype
        
        if n_samples is None:
            n_samples = int( min(max_samples, x_num_shape[0] * (window_alpha ** window_beta) * (x_num_shape[1] ** feature_gamma) ) )
        gen_shape = [n_samples, *x_num_shape[1:]]
        
        # 모든 연산은 x와 동일 device/dtype에서 진행
        reduce_dims = torch.arange(x_num.ndim, device=device).tolist()[:-1]  # 마지막 feature 축 제외
        x_min = x_num.amin(dim=reduce_dims)
        x_max = x_num.amax(dim=reduce_dims)
        x_window = x_max - x_min

        x_windowmin = x_min - x_window * window_alpha
        # x_windowmax = x_max + x_window * window_alpha
        gen_rand = torch.rand(gen_shape, device=device, dtype=dtype)    # # 난수 생성 시점에 device/dtype을 명시
        x_gen = ( x_windowmin + gen_rand * x_window *(1 + 2 *window_alpha) ).to(device)
        return x_gen

    def gen_cat_pseudo_data(self, x_cat, n_samples=None, max_categories=None, window_alpha=None, window_beta=None, feature_gamma=None, max_samples=None):
        """
        x_cat: torch.Tensor (정수형 카테고리 값, shape: [N, num_features])
        n_samples: 생성할 샘플 수 (None이면 x_cat.shape[0]과 동일)
        """
        window_alpha = self.window_alpha if window_alpha is None else window_alpha
        window_beta = self.window_beta if window_beta is None else window_beta
        feature_gamma = self.feature_gamma if feature_gamma is None else feature_gamma
        max_categories = self.max_categories if max_categories is None else max_categories
        max_samples = self.max_samples if max_samples is None else max_samples

        x_cat_shape = x_cat.shape
        device = x_cat.device
        dtype = x_cat.dtype  # 일반적으로 torch.long일 가능성이 높음

        max_categories = x_cat.amax(dim=tuple(range(x_cat.ndim - 1))).tolist() if max_categories is None else max_categories    
        max_cat_cols = sum(max_categories) + len(max_categories)

        if n_samples is None:
            n_samples = int( min(max_samples, x_cat_shape[0] * (window_alpha ** window_beta) * (max_cat_cols ** feature_gamma) ) )

        gen_cols = []
        for max_cat in max_categories:
            max_cat_n = max_cat + 1
            unique_vals = torch.arange(max_cat_n)
            uniform_dist = torch.ones(max_cat_n)/max_cat_n
            idx = torch.multinomial(uniform_dist, n_samples, replacement=True)
            gen_col = unique_vals[idx].to(device=device, dtype=dtype)
            gen_cols.append(gen_col.unsqueeze(-1))
        return torch.cat(gen_cols, dim=-1)          

    def gen_pseudo_data(self, *x, n_samples=None, max_categories=None, window_alpha=None, window_beta=None, feature_gamma=None, max_samples=None):
        """
        *x: torch.Tensor (shape: [N, num_features])
        n_samples: 생성할 샘플 수 (None이면 x_cat.shape[0]과 동일)
        """
        window_alpha = self.window_alpha if window_alpha is None else window_alpha
        window_beta = self.window_beta if window_beta is None else window_beta
        feature_gamma = self.feature_gamma if feature_gamma is None else feature_gamma
        max_categories = self.max_categories if max_categories is None else max_categories
        max_samples = self.max_samples if max_samples is None else max_samples
        
        # error validate
        if len(x) ==0:
            raise ("no input data.")
        else:
            if any(len(sub_x) != len(x[0]) for sub_x in x):
                raise ("input data have same length.")
            if any(sub_x.device != x[0].device for sub_x in x):
                raise ("input data have same device.")

        max_cols = 0
        for sub_x in x:
            if 'int' in str(sub_x.dtype):
                max_categories = sub_x.amax(dim=tuple(range(sub_x.ndim - 1))).tolist() if max_categories is None else max_categories    
                max_cat_cols = sum(max_categories) + len(max_categories)
                max_cols = max_cols if max_cols > max_cat_cols else max_cat_cols
            elif 'float' in str(sub_x.dtype):
                max_num_cols = max(sub_x.shape[-1] for sub_x in x)
                max_cols = max_cols if max_cols > max_num_cols else max_num_cols

        # shape
        x_shape = x[0].shape
        
        # n_samples
        if n_samples is None:
            n_samples = int( min(max_samples, x_shape[0] * (window_alpha ** window_beta) * (max_cols ** feature_gamma) ) )
        
        # gen_pseudo_Data
        pseudo_data_list = []
        for sub_x in x:
            if 'float' in str(sub_x.dtype):
                pseudo_data = self.gen_num_pseudo_data(sub_x, n_samples, window_alpha, window_beta, feature_gamma, max_samples)
            elif 'int' in str(sub_x.dtype):
                pseudo_data = self.gen_cat_pseudo_data(sub_x, n_samples, max_categories, window_alpha, window_beta, feature_gamma, max_samples)
            pseudo_data_list.append(pseudo_data)
        
        # return
        if len(pseudo_data_list) == 1:
            return pseudo_data_list[0]
        else:
            return tuple(pseudo_data_list)

    def __call__(self, *x, n_samples=None, max_categories=None, window_alpha=None, window_beta=None, feature_gamma=None, max_samples=None):
        """
        *x: torch.Tensor (shape: [N, num_features])
        n_samples: 생성할 샘플 수 (None이면 x_cat.shape[0]과 동일)
        """
        return self.gen_pseudo_data(*x, n_samples=n_samples, max_categories=max_categories, 
                                    window_alpha=window_alpha, window_beta=window_beta, 
                                    feature_gamma=feature_gamma, max_samples=max_samples)


# pseudo_data = PseudoData()
# pseudo_data = PseudoData( max_categories=torch.LongTensor(train_X_cat) )
# pseudo_data = PseudoData( max_categories=[9,2] )
# x1 = torch.rand(5,3)    # float
# x2 = x_cat = torch.cat([torch.randint(0,4, size=(5,1)), torch.randint(0,2, size=(5,1))], dim=-1)     # Long
# pseudo_data(x1)
# pseudo_data(x2)
# pseudo_data(x1, x2)
# pseudo_data(x2, x1)




# --------------------------------------------------------------------------------------
# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"torch device : {device}")

# --------------------------------------------------------------------------------------
class CategoricalEmbeddingLayer(nn.Module):
    def __init__(self, n_features, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding_weights = nn.Parameter(torch.randn((n_features, num_embeddings, embedding_dim)))
    
    def forward(self, x):
        x_shape = x.shape
        n_features = x_shape[-1]
        
        feature_idx = torch.arange(n_features).view(*([1] * (x.ndim - 1)), n_features).expand(*x_shape)
        return self.embedding_weights[feature_idx, x]


class EnsembleNN(nn.Module):
    def __init__(self, num_input_dim, cat_input_dim, hidden_dim=32, num_embeddings=10, n_cat_embed=2, n_ensemble=5):
        super().__init__()

        self.cat_embedding_block = nn.Sequential(
                CategoricalEmbeddingLayer(cat_input_dim, num_embeddings=num_embeddings, embedding_dim=n_cat_embed),
                nn.Linear(n_cat_embed, n_cat_embed),
                nn.ReLU(),
            )
                
        self.num_fc_block = nn.Sequential(
                # EmbeddingBlock(input_dim, flatten=True),
                # nn.ReLU(),
                nn.Linear(num_input_dim, hidden_dim),
                nn.ReLU(),
            )
        
        self.concat_fc_block = nn.Sequential(
            nn.Linear(hidden_dim+(cat_input_dim*n_cat_embed), hidden_dim*2),
            nn.ReLU(),
        )
        
        self.ensemble_blocks = nn.ModuleList()
        for _ in range(n_ensemble):
            block = nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim*2),
                nn.ReLU(),
                # nn.Linear(hidden_dim*2, hidden_dim*2),
                # nn.ReLU(),
                nn.Linear(hidden_dim*2, 1)
            )
            self.ensemble_blocks.append(block)

        # 모든 Linear 레이어 weight를 uniform 초기화
        for block in self.ensemble_blocks:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.uniform_(layer.weight, a=-0.1, b=0.1)  # 범위 [-0.1, 0.1]
                    torch.nn.init.zeros_(layer.bias)  # bias는 0으로 초기화

    def forward(self, x_num, x_cat):
        # embedding
        cat_x_embed = self.cat_embedding_block(x_cat).view(x_cat.size(0), -1)
        num_x_embed = self.num_fc_block(x_num)
        
        # concat
        x_embed = torch.cat([num_x_embed, cat_x_embed], dim=-1)
        x_latent = self.concat_fc_block(x_embed)

        # shared
        outputs = []
        for block in self.ensemble_blocks:
            outputs.append(block(x_latent))

        outputs_cat = torch.cat(outputs, dim=-1)  # (batch, n_ensemble)

        # mu, std
        mu = outputs_cat.mean(dim=-1, keepdims=True)
        std = outputs_cat.std(dim=-1, keepdims=True)

        return mu, std

# for batch in train_loader:
#     break
# batch[0].shape
# batch[1].shape

model = EnsembleNN(num_input_dim=11, cat_input_dim=2, hidden_dim=32, 
                   num_embeddings=100, n_cat_embed=2, n_ensemble=5).to(device)
# sum(p.numel() for p in model.parameters() if p.requires_grad)
# model(torch.rand(10,11), torch.randint(0,2, size=(10,2)) )

def loss_function(model, batch, optimizer=None):
    # --------------------------------------------------
    X_num, X_cat, y = batch
    mu, std = model(X_num, X_cat)
    
    mse_loss = lambda pred,y : ((pred - y)**2).mean()

    loss_truth = mse_loss(mu, y)
    # loss_truth = nn.functional.gaussian_nll_loss(mu, y, std**2)
    # --------------------------------------------------
    pseudo_num_X, pseudo_cat_X = pseudo_data(X_num, X_cat)
    
    pseudo_mu, pseudo_std = model(pseudo_num_X, pseudo_cat_X)
    
    gamma_ = 0.1     # observe uncertainty coefficient
    lambda_ = 0.3   # unobserve uncertainty coefficient
    p = 0.95
    
    X_true = torch.cat([X_num, X_cat], dim=-1)
    X_true_mean = X_true.mean(dim=torch.arange(X_true.ndim)[:-1].tolist(), keepdims=True)
    X_pseudo = torch.cat([pseudo_num_X, pseudo_cat_X], dim=-1)
    std_target = gamma_* y.std() 
    
    pseudo_std_target = lambda_*( y.std() + torch.abs((X_pseudo - X_true_mean).mean())* pseudo_data.window_alpha )
    loss_pseudo = p* mse_loss(std, std_target)+ (1-p)*mse_loss(pseudo_std, pseudo_std_target)
    # --------------------------------------------------
    loss = p * loss_truth + (1-p)* loss_pseudo

    return loss

pseudo_data = PseudoData( max_categories=torch.LongTensor(train_X_cat) )

tm1 = TorchModeling(model, device=device)
tm1.compile(optimizer=optim.Adam(model.parameters(), lr=1e-3) 
            ,loss_function = loss_function
            # ,loss_function = weighted_gaussian_loss
            # , scheduler=scheduler
            , early_stop_loss = EarlyStopping(patience=100)
            )
tm1.early_stop_loss.reset_patience_scores()
tm1.train_model(train_loader=train_loader, valid_loader=valid_loader, epochs=500)

# tm1.early_stop_loss.plot
# tm1.early_stop_loss.metrics_frame
# tm1.set_best_model()
# test_loss = np.sqrt(tm1.test_model(tests_loader)['test_loss']).item()      # test evaluate
# print(f"Test Loss : {test_loss:.3f}")

test_mse_losses = []
with torch.no_grad():
    model.eval()
    for batch in tests_loader:
        X_num, X_cat, y = batch
        X_num, X_cat, y = X_num.to(device), X_cat.to(device), y.to(device)
        mu, std = model(X_num, X_cat)
        mse_loss = lambda pred, y : ((pred - y)**2).mean()
        loss = mse_loss(mu, y)
        test_mse_losses.append( loss.detach().to('cpu').numpy() )
test_rmse = np.mean([np.sqrt(loss) for loss in test_mse_losses]).item()
print(f"Test RMSE : {test_rmse:.3f}")



####################################################################################################
# visualize util function
def gen_mesh_data(col_idx, X_num, X_cat, n_points=50):
    X_num = torch.FloatTensor(train_X_num_norm)
    X_cat = torch.LongTensor(train_X_cat)

    X_num_shape = X_num.shape
    X_cat_shape = X_cat.shape

    # Generate Partial Dependendence Data
    if col_idx < X_num_shape[1]:
        X_obs = torch.linspace(-6, 6, steps=n_points)
    else:
        max_cat = (X_cat.amax(axis=0)+1)[col_idx - X_num_shape[1]]
        X_obs = torch.arange(max_cat.item())
    X_obs_size = X_obs.shape[0]

    # Generate Mean & Mode Data 
    X_cat_mode = torch.tensor([
        torch.nonzero((lambda c: c == c.max())(torch.bincount(X_cat[:, col]))).flatten()
        [torch.randint(0, torch.sum(torch.bincount(X_cat[:, col]) == torch.bincount(X_cat[:, col]).max()), (1,))].item()
        for col in range(X_cat.shape[1])
    ])
    X_gen_num = torch.zeros((1, X_num_shape[-1])).expand(X_obs_size, -1).clone()
    X_gen_cat = (torch.ones((1, X_cat_shape[-1]), dtype=torch.int64) * X_cat_mode).expand(X_obs_size, -1).clone()
    # X_gen = torch.cat([X_gen_num, X_gen_cat], dim=-1)

    # Replace Partial Dependence Feature
    if col_idx < X_num_shape[1]:
        X_gen_num[:, col_idx] = X_obs
    else:
        X_gen_cat[:, col_idx - X_num_shape[1]] = X_obs
    return X_gen_num, X_gen_cat

def partial_dependence_uncertainty_plot(model, index, train_X_num, train_X_cat, train_y, sigma=2,
                                        scaler_X=None, scaler_y=None, return_plot=True):
    train_X_num_np = np.array(train_X_num)
    train_X_cat_np = np.array(train_X_cat)
    train_y_np = np.array(train_y)
    
    X_num_shape = train_X_num_np.shape
    X_cat_shape = train_X_cat_np.shape

    # (function) gen_mesh_data  
    X_gen_num, X_gen_cat = gen_mesh_data(index, train_X_num_np, train_X_cat_np, n_points=100)


    # model predict
    with torch.no_grad():
        model.eval()
        mu_pred_gen, sigma_pred_gen = model(X_gen_num.to(device), X_gen_cat.to(device))
        mu_pred_gen = mu_pred_gen.view(-1)
        sigma_pred_gen = sigma_pred_gen.view(-1)
    
    # calculate uncertainty        
    mu_cpu_np = mu_pred_gen.detach().cpu().numpy()
    sig_cpu_np = sigma_pred_gen.detach().cpu().numpy()
    lower_cpu_np = (mu_cpu_np - sigma * sig_cpu_np)
    upper_cpu_np = (mu_cpu_np + sigma * sig_cpu_np)

    X_gen_num_np = X_gen_num.detach().to('cpu').numpy()
    X_gen_cat_np = X_gen_cat.detach().to('cpu').numpy()

    # scaler inverse
    if scaler_X is not None:
        X_gen_num_np = scaler_X.inverse_transform(X_gen_num_np)
        train_X_num_np = scaler_X.inverse_transform(train_X_num_np)
    if scaler_y is not None:
        mu_cpu_np = scaler_y.inverse_transform(mu_cpu_np.reshape(-1,1)).ravel()
        lower_cpu_np = scaler_y.inverse_transform(lower_cpu_np.reshape(-1,1)).ravel()
        upper_cpu_np = scaler_y.inverse_transform(upper_cpu_np.reshape(-1,1)).ravel()
        train_y_np = scaler_y.inverse_transform(train_y_np).ravel()

    # observe feature selection
    if index < X_num_shape[1]:
        x_obs = X_gen_num_np[:, index]
        train_x_np = train_X_num_np[:, index] 
    else:
        x_obs = X_gen_cat_np[:, index - X_num_shape[1]]
        train_x_np = train_X_cat_np[:, index - X_num_shape[1]] 

    # figure
    fig = plt.figure()
    plt.plot(x_obs, mu_cpu_np, color='orange')
    # plt.scatter(x_obs, mu_cpu, color='steelblue', edgecolors='white', alpha=0.5)
    plt.scatter(train_x_np, train_y_np, color='steelblue', edgecolors='white', alpha=0.5)
    plt.fill_between(
        x_obs,
        lower_cpu_np,
        upper_cpu_np,
        color='orange',
        alpha=0.2,
        label="Uncertainty (±2 std)"
    )
    plt.legend(loc='upper right')
    
    if return_plot:
        plt.close()
        return fig
    else:
        plt.show()

# visualize
partial_dependence_uncertainty_plot(model, 11, train_X_num_norm, train_X_cat, train_y_norm, 
                                sigma=2, scaler_X=ss_X, scaler_y=ss_y)





################################################################################

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

train_X_concat =np.concat([train_X_num_norm, train_X_cat], axis= -1)
tests_X_concat =np.concat([tests_X_num_norm, tests_X_cat], axis= -1)

# ---------------------------------------------------------------------------
oh = OneHotEncoder(sparse_output=True)
train_X_cat_oh = oh.fit_transform(train_X_cat).toarray()
tests_X_cat_oh = oh.fit_transform(tests_X_cat).toarray()

train_X_concat = np.concat([train_X_num_norm, train_X_cat_oh], axis= -1)
tests_X_concat = np.concat([tests_X_num_norm, tests_X_cat_oh], axis= -1)
# ---------------------------------------------------------------------------

# RandomForest
RF = RandomForestRegressor()
RF.fit(train_X_concat, train_y_norm)

RF_pred = RF.predict(train_X_concat)
np.sqrt(mean_squared_error(train_y_norm, RF_pred))


RF_pred_tests = RF.predict(tests_X_concat)
RF_rmse_tests = np.sqrt(mean_squared_error(tests_y_norm, RF_pred_tests))
print(f"RF_rmse_tests : {RF_rmse_tests:.3f}")

# ---------------------------------------------------------------------------

# GradientBoosting
GB = GradientBoostingRegressor()
GB.fit(train_X_concat, train_y_norm)

GB_pred = GB.predict(train_X_concat)
np.sqrt(mean_squared_error(train_y_norm, GB_pred))


GB_pred_tests = GB.predict(tests_X_concat)
GB_rmse_tests = np.sqrt(mean_squared_error(tests_y_norm, GB_pred_tests))
print(f"GB_rmse_tests : {GB_rmse_tests:.3f}")
# ---------------------------------------------------------------------------







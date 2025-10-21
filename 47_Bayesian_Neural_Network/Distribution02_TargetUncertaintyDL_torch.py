import os
import sys
sys.path.append('/home/pd299370/DataScience/DS_Library')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

try:
    from DS_Torch import TorchDataLoader, TorchModeling, AutoML
except:
    try:
        import httpimport
        with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
            from DS_Torch import TorchDataLoader, TorchModeling, AutoML
    except:
        import requests
        remote_library_url = 'https://raw.githubusercontent.com/kimds929/'
        response = requests.get(f"{remote_library_url}/DS_Library/main/DS_Torch.py", verify=False)

#########################################################################################
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.device_count()
# torch.autograd.set_detect_anomaly(False)

succeed = True
fail = False
#########################################################################################

# 1차원 Dataset 생성 ------------------------------------------------------------
sample_size=300
group_size = 3

# rng = np.random.RandomState(5)
rng = np.random.RandomState(1)
mu_list = rng.rand(group_size)*8 -4
sigma_list = rng.rand(group_size)*0.5

data_sample = []
for mu, sigma in zip(mu_list, sigma_list):
    data_sample.append(rng.normal(loc=mu, scale=sigma, size=sample_size))

plt.hist(data_sample, bins=30)
plt.show()

data_np_cat = np.stack(data_sample).reshape(-1,1)

X_train = torch.FloatTensor(data_np_cat)



# (sample data) x_train / y_train --------------------------------------------------
class UnknownFuncion():
    def __init__(self, n_polynorm=1, theta_scale=1, error_scale=None, normalize=True):
        self.n_polynorm = n_polynorm
        self.normalize = normalize
        self.y_mu = None
        self.y_std = None

        self.true_theta = torch.randn((self.n_polynorm+1,1)) * theta_scale
        self.error_scale = torch.rand((1,))*0.3+0.1 if error_scale is None else error_scale
    
    def normalize_setting(self, train_x):
        if (self.y_mu is None) and (self.y_std is None):
            outputs = self.true_f(train_x)
            self.y_mu = torch.mean(outputs)
            self.y_std = torch.std(outputs)

    def true_f(self, x):
        for i in range(self.n_polynorm+1):
            response = self.true_theta[i] * (x**i)

            if (self.normalize) and (self.y_mu is not None) and (self.y_std is not None):
                response = (response - self.y_mu)/self.y_std
        return response

    def forward(self, x):
        if (self.normalize) and (self.y_mu is not None) and (self.y_std is not None):
            return self.true_f(x) + self.error_scale * torch.randn((x.shape[0],1))
        else:
            return self.true_f(x) + self.true_f(x).mean()*self.error_scale * torch.randn((x.shape[0],1))

    def __call__(self, x):
        return self.forward(x)


# f = UnknownFuncion()
f = UnknownFuncion(n_polynorm=2)
# f = UnknownFuncion(n_polynorm=3)
# f = UnknownFuncion(n_polynorm=4)
# f = UnknownFuncion(n_polynorm=5)
# f = UnknownFuncion(n_polynorm=6)
# f = UnknownFuncion(n_polynorm=7)
# f = UnknownFuncion(n_polynorm=8)
# f = UnknownFuncion(n_polynorm=9)
# f = RewardFunctionTorch()
# f = UnknownBernoulliFunction()
f.normalize_setting(X_train)


y_true = f.true_f(X_train)
y_train = f(X_train)

# visualize
# x_lin = torch.linspace(X_train.min(),X_train.max(),300).reshape(-1,1)
x_lin = torch.linspace(-5,5,300).reshape(-1,1)

x_lin_add_const = torch.concat([x_lin, torch.ones_like(x_lin)], axis=1)

plt.figure()
plt.scatter(X_train, y_train, label='obs', edgecolor='white', alpha=0.5)
plt.plot(x_lin, f.true_f(x_lin), color='orange', label='true')
plt.legend()
plt.show()


###################################################################################################
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)



###################################################################################################
def visualize_validate(model, X_train, y_train, xmin=-6, xmax=6):
    # 예측 및 불확실성 계산
    model.eval()
    X_test = torch.linspace(xmin, xmax, steps=100).unsqueeze(1)

    mu_pred, sigma_pred = model(X_test)
    # mu_pred, sigma_pred = model(X_test, alpha=0)
    mu_pred = mu_pred.view(-1)
    sigma_pred = sigma_pred.view(-1)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(X_train, y_train, 'o', label="Training Data")
    plt.plot(X_test, mu_pred.detach(), color='steelblue', label="Mean Prediction")
    plt.plot(X_test, f.true_f(X_test), color='orange', label='true')
    plt.fill_between(
        X_test.squeeze().detach(),
        (mu_pred.view(-1) - 2 * sigma_pred).detach(),
        (mu_pred.view(-1) + 2 * sigma_pred).detach(),
        alpha=0.2,
        label="Uncertainty (±2 std)"
    )
    plt.legend()
    return fig

###################################################################################################




########################################################################################
class EmbeddingLayer(nn.Module):
    def __init__(self, feature_dim, embed_dim, bias=True, sine=False, sine_bias=True, independent=False, expand=True):
        super().__init__()
        self.sine = sine
        self.bias = bias
        self.sine_bias = sine_bias
        self.independent = independent
        self.expand = expand
        self.embed_dim = embed_dim

        if independent is False:
            self.weight = nn.Parameter(torch.randn(feature_dim, embed_dim))  # (feature_dim, embed_dim, 1)
            nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='linear')

        if (independent is True) or (bias is True):
            self.weight_bias = nn.Parameter(torch.randn(feature_dim, embed_dim))       # (feature_dim, embed_dim)
            nn.init.uniform_(self.weight_bias, -1, 1)

        # if sine is True:
        #     self.weight_sine = nn.Parameter(torch.ones(feature_dim, embed_dim))      # (feature_dim, embed_dim)
        #     nn.init.kaiming_normal_(self.weight_sine, mode='fan_in', nonlinearity='linear')
            
        #     if sine_bias is True:
        #         self.weight_sine_bias = nn.Parameter(torch.randn(feature_dim, embed_dim))
        #         nn.init.uniform_(self.weight_sine_bias, -1, 1)

    def forward(self, x):
        *x_shape, f_dim = x.shape   # (batch_dim, feature_dim)
        x_unsqueezed = x.unsqueeze(-1)
        if self.independent:
            x_embed = self.weight_bias     # (feature_dim, embed_dim)
            if self.expand:
                x_embed = x_embed.expand(*x_shape, f_dim, self.embed_dim)
        else:
            x_embed = self.weight * x_unsqueezed
            if self.bias:
                x_embed += self.weight_bias
        
        if self.sine:
            x_embed = torch.sin(x_embed) 
            # x_embed = self.weight_sine * torch.sin(x_embed) 
            # if self.sine_bias:
            #     x_embed += self.weight_sine_bias
        
        return x_embed



class EmbeddingBlock(nn.Module):
    def __init__(self, input_dim, flatten=False):
        super().__init__()
        self.flatten = flatten

        self.ind_embedding = EmbeddingLayer(input_dim, 1, independent=True)
        self.sin_embedding = EmbeddingLayer(input_dim, 1, sine=True)

    def forward(self, x):
        x_shape = x.shape
        x_embed = x.unsqueeze(-1)
        ind_x = self.ind_embedding(x)
        sin_x = self.sin_embedding(x)
        embed_output = torch.cat([x_embed, ind_x, sin_x], axis=-1)
        if self.flatten:
            embed_output = embed_output.view(x_shape[0],-1)
        return embed_output

#############################################################################

def gen_pseudo_data(x, window_alpha=3, max_ord=5, max_samples=1e+6):
    x_shape = x.shape
    # n_of_gen = int( min(max_samples, (x_shape[0]*2*window_alpha) ** min( torch.sqrt(torch.tensor(x_shape[-1])), torch.tensor(max_ord)).item() ) )
    n_of_gen = int( x_shape[0] * np.sqrt(x_shape[1]) )
    # print(n_of_gen)
    gen_shape = [n_of_gen, *x_shape[1:]]
    # print(gen_shape)
    
    x_min = x.amin(dim=torch.arange(x.ndim)[:-1].tolist())
    x_max = x.amax(dim=torch.arange(x.ndim)[:-1].tolist())
    x_window = x_max - x_min

    x_windowmin = x_min - x_window * window_alpha
    x_windowmax = x_max + x_window * window_alpha
    x_gen = ( x_windowmin + torch.rand(gen_shape) * x_window *(1 + 2 *window_alpha) ).to(x.device)
    return x_gen




#############################################################################
# 1D Ensemble Model
if succeed:
    class EnsembleNN01(nn.Module):
        def __init__(self, input_dim, hidden_dim=32, n_ensemble=5):
            super().__init__()

            self.ensemble_blocks = nn.ModuleList()
            for _ in range(n_ensemble):
                block = nn.Sequential(
                    # EmbeddingBlock(input_dim, flatten=True),
                    # nn.ReLU(),
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim*2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim*2, 1),
                )
                self.ensemble_blocks.append(block)

            # 모든 Linear 레이어 weight를 uniform 초기화
            for block in self.ensemble_blocks:
                for layer in block:
                    if isinstance(layer, nn.Linear):
                        torch.nn.init.uniform_(layer.weight, a=-0.1, b=0.1)  # 범위 [-0.1, 0.1]
                        torch.nn.init.zeros_(layer.bias)  # bias는 0으로 초기화

        def forward(self, x):
            outputs = []
            for block in self.ensemble_blocks:
                outputs.append(block(x))

            outputs_cat = torch.cat(outputs, dim=-1)  # (batch, n_ensemble)

            mu = outputs_cat.mean(dim=-1, keepdims=True)
            std = outputs_cat.std(dim=-1, keepdims=True)

            return mu, std


    model = EnsembleNN01(1,64).to(device)
    sum(p.numel() for p in model.parameters() if p.requires_grad)
    # model(torch.rand(10,2).to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-3) 

    def loss_function(model, x, y):
        # --------------------------------------------------
        mu, std = model(x)
        # loss_truth = torch.nn.functional.mse_loss(mu, y)
        # loss_truth = 1/2 * (mu - y)**2

        # loss_truth = torch.nn.functional.gaussian_nll_loss(mu, y, std**2)
        loss_truth = ( 0.5 * torch.log(2 * torch.pi * std**2) + (y - mu)**2 / (2 * std**2) ).mean()
        # return loss_truth
        # --------------------------------------------------
        pseudo_X = gen_pseudo_data(x)
        pseudo_mu, pseudo_std = model(pseudo_X)
        y_pseudo_true = torch.ones_like(pseudo_std) * torch.quantile(pseudo_std, 0.75).detach().item()
        loss_pseudo = (pseudo_std - y_pseudo_true) **2      # MSE Loss
        max_loss_pseudo = loss_pseudo.detach().max()
        if max_loss_pseudo == 0:
            loss_pseudo = torch.mean(loss_pseudo)  
            p=1
        else:
            loss_pseudo = torch.mean( loss_pseudo/max_loss_pseudo * torch.abs(loss_truth.detach().mean()) )
            p=0.9
        # --------------------------------------------------
        loss = p * loss_truth + (1-p)* loss_pseudo

        return loss


    tm1 = TorchModeling(model, device=device)
    tm1.compile(optimizer=optimizer
                ,loss_function = loss_function
                # ,loss_function = weighted_gaussian_loss
                # , scheduler=scheduler
                # , early_stop_loss = EarlyStopping(patience=5)
                )
    tm1.train_model(train_loader=train_loader, epochs=100)

    visualize_validate(model, X_train, y_train, xmin=-6, xmax=6)
    ##################################################################





class EnsembleNN02(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, n_ensemble=5):
        super().__init__()

        self.shared_block = nn.Sequential(
                # EmbeddingBlock(input_dim, flatten=True),
                # nn.ReLU(),
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            )
        self.ensemble_blocks = nn.ModuleList()
        for _ in range(n_ensemble):
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim*2),
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

    def forward(self, x):
        latent = self.shared_block(x)
        outputs = []
        for block in self.ensemble_blocks:
            outputs.append(block(latent))

        outputs_cat = torch.cat(outputs, dim=-1)  # (batch, n_ensemble)

        mu = outputs_cat.mean(dim=-1, keepdims=True)
        std = outputs_cat.std(dim=-1, keepdims=True)

        return mu, std

model = EnsembleNN02(1,64).to(device)
sum(p.numel() for p in model.parameters() if p.requires_grad)
# model(torch.rand(10,2).to(device))
optimizer = optim.Adam(model.parameters(), lr=1e-3) 

def loss_function(model, x, y):
    # --------------------------------------------------
    mu, std = model(x)
    # loss_truth = torch.nn.functional.mse_loss(mu, y)
    # loss_truth = 1/2 * (mu - y)**2

    # loss_truth = torch.nn.functional.gaussian_nll_loss(mu, y, std**2)
    loss_truth = ( 0.5 * torch.log(2 * torch.pi * std**2) + (y - mu)**2 / (2 * std**2) ).mean()
    # return loss_truth
    # --------------------------------------------------
    pseudo_X = gen_pseudo_data(x)
    pseudo_mu, pseudo_std = model(pseudo_X)
    y_pseudo_true = torch.ones_like(pseudo_std) * torch.quantile(pseudo_std, 0.75).detach().item()
    loss_pseudo = (pseudo_std - y_pseudo_true) **2      # MSE Loss
    max_loss_pseudo = loss_pseudo.detach().max()
    if max_loss_pseudo == 0:
        loss_pseudo = torch.mean(loss_pseudo)  
        p=1
    else:
        loss_pseudo = torch.mean( loss_pseudo/max_loss_pseudo * torch.abs(loss_truth.detach().mean()) )
        p=0.9
    # --------------------------------------------------
    loss = p * loss_truth + (1-p)* loss_pseudo

    return loss


tm1 = TorchModeling(model, device=device)
tm1.compile(optimizer=optimizer
            ,loss_function = loss_function
            # ,loss_function = weighted_gaussian_loss
            # , scheduler=scheduler
            # , early_stop_loss = EarlyStopping(patience=5)
            )
tm1.train_model(train_loader=train_loader, epochs=100)

visualize_validate(model, X_train, y_train, xmin=-6, xmax=6)
##################################################################









####################################################################################
# 2D Ensemble Model
if succeed:
    # 2차원 Dataset 생성 ------------------------------------------------------------
    sample_size = 300
    group_size = 3
    mu_list = np.random.rand(group_size, 2) * 8 - 4   # shape: (group_size, 2)

    # 2차원 공분산 행렬 (scale) 랜덤 생성
    sigma_list = []
    for _ in range(group_size):
        # 대각선만 있는 단순한 공분산 (각 축별로 std)
        stds = np.random.rand(2) * 0.5 + 0.1
        cov = np.diag(stds**2)
        sigma_list.append(cov)

    data_sample = []
    for mu, cov in zip(mu_list, sigma_list):
        data_sample.append(np.random.multivariate_normal(mean=mu, cov=cov, size=sample_size))

    data_np_cat = np.vstack(data_sample)   # shape: (sample_size*group_size, 2)

    plt.figure(figsize=(6,6))
    plt.scatter(data_np_cat[:,0], data_np_cat[:,1], s=10, alpha=0.5)
    plt.title("2D Gaussian Mixture Samples")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()



    # ----------------------------------------
    # 2차원 True Function 정의
    class UnknownFuncion2D():
        def __init__(self):
            pass

        def true_f(self, X):
            """
            X: shape (N, 2)  -> columns: x1, x2
            """
            x1 = X[:, 0]
            x2 = X[:, 1]
            # 예시: 원형 + 사인 패턴
            r = np.sqrt(x1**2 + x2**2)
            return (np.sin(r) + 0.1 * x1 + 0.05 * x2).reshape(-1,1)

        def forward(self, X):
            true_y = self.true_f(X)
            obs = true_y + np.random.normal(loc=0, scale=0.15, size=true_y.shape)
            return obs


    f_2D = UnknownFuncion2D()


    # visualize
    n_grid = 30
    x1 = np.linspace(-5, 5, n_grid)
    x2 = np.linspace(-5, 5, n_grid)
    xx1, xx2 = np.meshgrid(x1, x2)
    grid_points = np.stack([xx1.ravel(), xx2.ravel()], axis=1)  # # shape: (n_grid*n_grid, 2)


    y_true = f_2D.true_f(grid_points)
    y_true_contour = y_true.reshape(xx1.shape)

    y_obs = f_2D.forward(grid_points)
    y_obs_contour = y_obs.reshape(xx1.shape)

    plt.figure(figsize=(6,6))
    # contour = plt.contourf(xx1, xx2, y_true_contour, cmap='coolwarm', alpha=0.3)
    contour = plt.contourf(xx1, xx2, y_obs_contour, cmap='coolwarm', alpha=0.3)
    plt.scatter(data_np_cat[:,0], data_np_cat[:,1], s=10, alpha=0.5, edgecolors='white')
    plt.colorbar(contour)
    plt.title("2D Gaussian Mixture Samples")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


    X_train = torch.FloatTensor(data_np_cat)
    y_train = torch.FloatTensor(f_2D.true_f(data_np_cat))

    #########################################################

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    #########################################################


    model = EnsembleNN01(2,128)
    sum(p.numel() for p in model.parameters() if p.requires_grad)
    # model(torch.rand(10,2))
    optimizer = optim.Adam(model.parameters(), lr=1e-3) 


    def loss_function(model, x, y):
        # --------------------------------------------------
        mu, std = model(x)
        # loss_truth = torch.nn.functional.mse_loss(mu, y)
        loss_truth = torch.nn.functional.gaussian_nll_loss(mu, y, std**2)
        # return loss_truth
        # --------------------------------------------------
        pseudo_X = gen_pseudo_data(x)
        pseudo_mu, pseudo_std = model(pseudo_X)
        y_pseudo_true = torch.ones_like(pseudo_std) * torch.quantile(pseudo_std, 0.75).detach().item()
        loss_pseudo = F.mse_loss(pseudo_std, y_pseudo_true)    
        loss_pseudo = loss_pseudo/loss_pseudo.detach().max() * torch.abs(loss_truth.detach().mean())
        # --------------------------------------------------
        p=0.9
        loss = p * loss_truth + (1-p)* loss_pseudo
        return loss


    tm1 = TorchModeling(model)
    tm1.compile(optimizer=optimizer
                ,loss_function = loss_function
                # ,loss_function = weighted_gaussian_loss
                # , scheduler=scheduler
                # , early_stop_loss = EarlyStopping(patience=5)
                )
    tm1.train_model(train_loader=train_loader, epochs=50)

    ##################################################################



    # visualize
    n_grid = 30
    x1 = np.linspace(-5, 5, n_grid)
    x2 = np.linspace(-5, 5, n_grid)
    xx1, xx2 = np.meshgrid(x1, x2)
    grid_points = np.stack([xx1.ravel(), xx2.ravel()], axis=1)  # # shape: (n_grid*n_grid, 2)


    y_true = f_2D.true_f(grid_points)
    y_true_contour = y_true.reshape(xx1.shape)

    y_obs = f_2D.forward(grid_points)
    y_obs_contour = y_obs.reshape(xx1.shape)

    with torch.no_grad():
        model.eval()
        pred_mu, pred_std = model(torch.FloatTensor(grid_points))
    pred_mu_contour = pred_mu.view(xx1.shape)
    pred_std_contour = pred_std.view(xx1.shape)

    plt.figure(figsize=(6,6))
    # contour = plt.contourf(xx1, xx2, y_true_contour, cmap='coolwarm', alpha=0.3)
    # contour = plt.contourf(xx1, xx2, y_obs_contour, cmap='coolwarm', alpha=0.3)
    # contour = plt.contourf(xx1, xx2, pred_mu_contour, cmap='coolwarm', alpha=0.3)
    contour = plt.contourf(xx1, xx2, pred_std_contour, cmap='coolwarm', alpha=0.3)
    plt.scatter(data_np_cat[:,0], data_np_cat[:,1], s=10, alpha=0.5, edgecolors='white')
    plt.colorbar(contour)
    plt.title("2D Gaussian Mixture Samples")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()



    def visualize_slice(model, f_true, X_train, y_train, fixed_dim=1, fixed_value=0.0,
                        xmin=-5, xmax=5, n_points=200, atol=0.2):
        """
        fixed_dim: 0이면 x1을 고정하고 x2를 변화, 1이면 x2를 고정하고 x1을 변화
        fixed_value: 고정할 값
        atol: slice 근처 데이터 허용 오차
        """
        # 1D grid 생성
        x_var = np.linspace(xmin, xmax, n_points)
        if fixed_dim == 0:
            grid_points = np.stack([np.ones_like(x_var)*fixed_value, x_var], axis=1)
        else:
            grid_points = np.stack([x_var, np.ones_like(x_var)*fixed_value], axis=1)

        # 모델 예측
        with torch.no_grad():
            mu, std = model(torch.FloatTensor(grid_points))
        mu_np = mu.numpy().flatten()
        std_np = std.numpy().flatten()

        # True function 값
        y_true = f_true(grid_points).flatten()

        # slice 근처 데이터 마스크
        mask = np.isclose(X_train[:, fixed_dim].numpy(), fixed_value, atol=atol)
        X_slice = X_train[mask]
        y_slice = y_train[mask]

        # 시각화
        plt.figure(figsize=(8,5))
        # 전체 관측 데이터 (연한 회색)
        plt.scatter(X_train[:, 1-fixed_dim], y_train, s=15, alpha=0.3, color='gray', label="All Training Data")
        # slice 근처 데이터 (파란색)
        plt.scatter(X_slice[:, 1-fixed_dim], y_slice, s=20, alpha=0.8, color='blue', label="Slice Data")
        # 예측 평균
        plt.plot(x_var, mu_np, color='orange', label="Mean Prediction")
        # True function
        plt.plot(x_var, y_true, color='green', label="True Function")
        # 불확실성 영역
        plt.fill_between(x_var, mu_np - 2*std_np, mu_np + 2*std_np,
                        color='skyblue', alpha=0.2, label="Uncertainty (+/- 2 std)")

        plt.xlabel("x{}".format(2-fixed_dim))
        plt.ylabel("y")
        plt.legend(loc='right', bbox_to_anchor=(1,1))
        plt.title(f"Slice at x{fixed_dim+1} = {fixed_value}")
        plt.show()



    # x1을 0으로 고정하고 x2 방향으로 slice
    visualize_slice(model, f_2D.true_f, X_train, y_train, 
                    fixed_dim=0, fixed_value=-2)

    # x1을 0으로 고정하고 x2 방향으로 slice
    visualize_slice(model, f_2D.true_f, X_train, y_train, 
                    fixed_dim=0, fixed_value=0.0)

    # x2를 -4으로 고정하고 x1 방향으로 slice
    visualize_slice(model, f_2D.true_f, X_train, y_train, 
                    fixed_dim=1, fixed_value=4)

    # x2를 0으로 고정하고 x1 방향으로 slice
    visualize_slice(model, f_2D.true_f, X_train, y_train, 
                    fixed_dim=1, fixed_value=0)






####################################################################################
####################################################################################

# Variational (Mu, Std)  Network
if fail:
    class VariationalNetwork(nn.Module):
        def __init__(self, input_dim, hidden_dim=32, n_ensemble=5):
            super().__init__()
            self.share_block = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim*2),
                nn.ReLU(),
            )

            self.mu_block = nn.Sequential(
                    nn.Linear(hidden_dim*2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )
            self.std_block = nn.Sequential(
                    nn.Linear(hidden_dim*2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )
                
            # 모든 Linear 레이어 weight를 uniform 초기화
            for block in [self.share_block, self.mu_block, self.std_block]:
                for layer in block:
                    if isinstance(layer, nn.Linear):
                        torch.nn.init.uniform_(layer.weight, a=-0.1, b=0.1)  # 범위 [-0.1, 0.1]
                        # torch.nn.init.uniform_(layer.weight, a=-0.1, b=0.1)  # 범위 [-0.1, 0.1]
                        torch.nn.init.zeros_(layer.bias)  # bias는 0으로 초기화

        def forward(self, x):
            latent = self.share_block(x)
            mu = self.mu_block(latent)
            std = torch.nn.functional.softplus( self.std_block(latent) ) 
            return mu, std



    model = VariationalNetwork(1,64).to(device)
    sum(p.numel() for p in model.parameters() if p.requires_grad)
    # model(torch.rand(10,1).to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-3) 


    def loss_function(model, x, y):
        # --------------------------------------------------
        mu, std = model(x)
        # loss_truth = torch.nn.functional.mse_loss(mu, y)
        # loss_truth = 1/2 * (mu - y)**2

        # loss_truth = torch.nn.functional.gaussian_nll_loss(mu, y, std**2)
        loss_truth = ( 0.5 * torch.log(2 * torch.pi * std**2) + (y - mu)**2 / (2 * std**2) ).mean()
        # return loss_truth
        # --------------------------------------------------
        pseudo_X = gen_pseudo_data(x)
        pseudo_mu, pseudo_std = model(pseudo_X)
        y_pseudo_true = torch.ones_like(pseudo_std) * torch.quantile(pseudo_std, 0.75).detach().item()
        loss_pseudo = F.mse_loss(pseudo_std, y_pseudo_true)    
        loss_pseudo = loss_pseudo/loss_pseudo.detach().max() * torch.abs(loss_truth.detach().mean())
        # --------------------------------------------------
        p=0.9
        loss = p * loss_truth + (1-p)* loss_pseudo
        return loss


    tm1 = TorchModeling(model, device=device)
    tm1.compile(optimizer=optimizer
                ,loss_function = loss_function
                # ,loss_function = weighted_gaussian_loss
                # , scheduler=scheduler
                # , early_stop_loss = EarlyStopping(patience=5)
                )
    tm1.train_model(train_loader=train_loader, epochs=300)

    visualize_validate(model, X_train, y_train, xmin=-8, xmax=8)








####################################################################################
####################################################################################

# ------------------------------------------------------------------------------
# Bayesian Linear Layer
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, n_models=5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_models = n_models
        
        # Weight mean and log variance
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_std = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Bias mean and log variance
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_std = nn.Parameter(torch.Tensor(out_features))

        # # initialize parameters
        nn.init.uniform_(self.weight_mu, -0.1, 0.1)
        nn.init.zeros_(self.weight_std)

        nn.init.uniform_(self.bias_mu, -0.1, 0.1)
        nn.init.zeros_(self.bias_std)
    
    def forward(self, x):
        return self.forward_stochastic(x)

    def forward_stochastic(self, x):
        # Reparameterization trick
        weight_std = nn.functional.softplus(self.weight_std)
        bias_std = nn.functional.softplus(self.bias_std) 
        
        # Sample from normal distribution
        weight_eps = torch.randn_like(self.weight_mu)
        bias_eps = torch.randn_like(self.bias_mu)
            
        weight = self.weight_mu + weight_std * weight_eps   # re-parameterization_trick
        bias = self.bias_mu + bias_std * bias_eps      # re-parameterization_trick
        return x @ weight.T + bias

    def forward_deterministic(self, x):
        return x @ self.weight_mu.T + self.bias_mu

    def forward_ensemble(self, x, n_models=None):
        n_models = self.n_models if n_models is None else n_models
        outputs = []
        for _ in range(n_models):
            outputs.append( self.forward_stochastic(x) )
        outputs_stack = torch.stack(outputs, dim=-1)
        mu = outputs_stack.mean(dim=-1)
        std = outputs_stack.std(dim=-1)
        return mu, std
    


# BNN model : ensemble only when evaluation
class BNN_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_models=5):
        super().__init__()
        self.n_models = n_models
        self.bayes_block = nn.Sequential(
                # EmbeddingBlock(input_dim, flatten=True),
                # nn.ReLU(),
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                # BayesianLinear(hidden_dim, hidden_dim*2),
                nn.Linear(hidden_dim, hidden_dim*2),
                nn.ReLU(),
            )
        self.bayes_head = BayesianLinear(hidden_dim*2, 1, n_models)
    
    def forward(self, x, n_models=None):
        n_models = self.n_models if n_models is None else n_models
        latent = self.bayes_block(x)
        mu, std = self.bayes_head.forward_ensemble(latent)
        return mu, std



model = BNN_Model(1,64).to(device)
sum(p.numel() for p in model.parameters() if p.requires_grad)
# model(torch.rand(10,1))
optimizer = optim.Adam(model.parameters(), lr=1e-3) 


def loss_function(model, x, y):
    # --------------------------------------------------
    mu, std = model(x)
    # loss_truth = torch.nn.functional.mse_loss(mu, y)
    # loss_truth = 1/2 * (mu - y)**2

    # loss_truth = torch.nn.functional.gaussian_nll_loss(mu, y, std**2)
    loss_truth = ( 0.5 * torch.log(2 * torch.pi * std**2) + (y - mu)**2 / (2 * std**2) ).mean()
    return loss_truth
    # # --------------------------------------------------
    # pseudo_X = gen_pseudo_data(x)
    # pseudo_mu, pseudo_std = model(pseudo_X)
    # y_pseudo_true = torch.ones_like(pseudo_std) * torch.quantile(pseudo_std, 0.75).detach().item()
    # loss_pseudo = (pseudo_std - y_pseudo_true) **2      # MSE Loss
    # max_loss_pseudo = loss_pseudo.detach().max()
    # if max_loss_pseudo == 0:
    #     loss_pseudo = torch.mean(loss_pseudo)  
    #     p=1
    # else:
    #     loss_pseudo = torch.mean( loss_pseudo/max_loss_pseudo * torch.abs(loss_truth.detach().mean()) )
    #     p=0.9
    # # --------------------------------------------------
    # loss = p * loss_truth + (1-p)* loss_pseudo

    # return loss


tm1 = TorchModeling(model, device=device)
tm1.compile(optimizer=optimizer
            ,loss_function = loss_function
            # ,loss_function = weighted_gaussian_loss
            # , scheduler=scheduler
            # , early_stop_loss = EarlyStopping(patience=5)
            )
tm1.train_model(train_loader=train_loader, epochs=300)

visualize_validate(model, X_train, y_train, xmin=-6, xmax=6)












########################################################################
class EnsembleBayesianNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, n_ensemble=5):
        super().__init__()
        self.shared_block = nn.Sequential(
                # EmbeddingBlock(input_dim, flatten=True),
                # nn.ReLU(),
                BayesianLinear(input_dim, hidden_dim),
                nn.ReLU(),
                
            )
        self.ensemble_blocks = nn.ModuleList()
        for _ in range(n_ensemble):
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, 1),
            )
            self.ensemble_blocks.append(block)

        # 모든 Linear 레이어 weight를 uniform 초기화
        for block in self.ensemble_blocks:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.uniform_(layer.weight, a=-0.1, b=0.1)  # 범위 [-0.1, 0.1]
                    torch.nn.init.zeros_(layer.bias)  # bias는 0으로 초기화

    def forward(self, x):
        latent = self.shared_block(x)
        outputs = []
        for block in self.ensemble_blocks:
            outputs.append(block(latent))

        outputs_cat = torch.cat(outputs, dim=-1)  # (batch, n_ensemble)

        mu = outputs_cat.mean(dim=-1, keepdims=True)
        std = outputs_cat.std(dim=-1, keepdims=True)

        return mu, std

model = EnsembleBayesianNN(1,64).to(device)
sum(p.numel() for p in model.parameters() if p.requires_grad)
# model(torch.rand(10,1).to(device))
optimizer = optim.Adam(model.parameters(), lr=1e-3) 

def loss_function(model, x, y):
    # --------------------------------------------------
    mu, std = model(x)
    # loss_truth = torch.nn.functional.mse_loss(mu, y)
    # loss_truth = 1/2 * (mu - y)**2

    # loss_truth = torch.nn.functional.gaussian_nll_loss(mu, y, std**2)
    loss_truth = ( 0.5 * torch.log(2 * torch.pi * std**2) + (y - mu)**2 / (2 * std**2) ).mean()
    # return loss_truth
    # --------------------------------------------------
    pseudo_X = gen_pseudo_data(x)
    pseudo_mu, pseudo_std = model(pseudo_X)
    y_pseudo_true = torch.ones_like(pseudo_std) * torch.quantile(pseudo_std, 0.75).detach().item()
    loss_pseudo = (pseudo_std - y_pseudo_true) **2      # MSE Loss
    max_loss_pseudo = loss_pseudo.detach().max()
    if max_loss_pseudo == 0:
        loss_pseudo = torch.mean(loss_pseudo)  
        p=1
    else:
        loss_pseudo = torch.mean( loss_pseudo/max_loss_pseudo * torch.abs(loss_truth.detach().mean()) )
        p=0.9
    # --------------------------------------------------
    loss = p * loss_truth + (1-p)* loss_pseudo

    return loss


tm1 = TorchModeling(model, device=device)
tm1.compile(optimizer=optimizer
            ,loss_function = loss_function
            # ,loss_function = weighted_gaussian_loss
            # , scheduler=scheduler
            # , early_stop_loss = EarlyStopping(patience=5)
            )
tm1.train_model(train_loader=train_loader, epochs=1000)

visualize_validate(model, X_train, y_train, xmin=-6, xmax=6)
##################################################################





import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

# data1 = np.random.normal(loc=1.5, scale=0.2, size=1000)
# data2 = np.random.normal(loc=-2, scale=0.3, size=1000)

# data_sample = [data1, data2]



sample_size=300
group_size = 4

# 1차원 Dataset 생성 ------------------------------------------------------------
mu_list = np.random.rand(group_size)*8 -4
sigma_list = np.random.rand(group_size)*0.5

data_sample = []
for mu, sigma in zip(mu_list, sigma_list):
    data_sample.append(np.random.normal(loc=mu, scale=sigma, size=sample_size))

plt.hist(data_sample, bins=30)
plt.show()

data_np_cat = np.stack(data_sample).reshape(-1,1)



# 2차원 Dataset 생성 ------------------------------------------------------------
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



# 3차원 Dataset 생성 ------------------------------------------------------------
sample_size = 300
group_size = 4

# 3차원 mean (중심) 랜덤 생성
mu_list = np.random.rand(group_size, 3) * 8 - 4   # shape: (group_size, 3)

# 3차원 공분산 행렬 (scale) 랜덤 생성
sigma_list = []
for _ in range(group_size):
    stds = np.random.rand(3) * 0.5 + 0.1
    cov = np.diag(stds**2)
    sigma_list.append(cov)

data_sample = []
for mu, cov in zip(mu_list, sigma_list):
    data_sample.append(np.random.multivariate_normal(mean=mu, cov=cov, size=sample_size))

data_np_cat = np.vstack(data_sample)   # shape: (sample_size*group_size, 3)


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_np_cat[:,0], data_np_cat[:,1], data_np_cat[:,2], s=10, alpha=0.5)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
plt.title("3D Gaussian Mixture Samples")
plt.show()





########################################################################################
########################################################################################
# Dataset & DataLoader #################################################################
data_torch = torch.FloatTensor(data_np_cat)
dataset = TensorDataset(data_torch)
dataloader = DataLoader(dataset, shuffle=True, batch_size=128)
train_loader = deepcopy(dataloader)

for batch in train_loader:
    break






########################################################################################
def gen_pseudo_data(x, window_alpha=3, max_ord=5, max_samples=1e+6):
    x_shape = x.shape
    n_of_gen = int( min(max_samples, (x_shape[0]*2*window_alpha) ** min( torch.sqrt(torch.tensor(x_shape[-1])), torch.tensor(max_ord)).item() ) )
    print(n_of_gen)
    gen_shape = [n_of_gen, *x_shape[1:]]
    # print(gen_shape)
    
    x_min = x.amin(dim=torch.arange(x.ndim)[:-1].tolist())
    x_max = x.amax(dim=torch.arange(x.ndim)[:-1].tolist())
    x_window = x_max - x_min

    x_windowmin = x_min - x_window * window_alpha
    x_windowmax = x_max + x_window * window_alpha
    x_gen = ( x_windowmin + torch.rand(gen_shape) * x_window *(1 + 2 *window_alpha) ).to(x.device)
    return x_gen

########################################################################################

class Model01(nn.Module):
    def __init__(self, input_dim=1,
                scale_beta=1, forgetting_gamma=0.1,
                window_alpha=3, max_ord=5, max_sigma=6, max_samples=1e+6):
        super().__init__()
        self.window_alpha = window_alpha
        self.scale_beta = scale_beta
        self.forgetting_gamma = forgetting_gamma
        self.max_ord = max_ord
        self.max_samples = max_samples
        self.max_sigma = max_sigma
        

        self.block = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.block(x)
        x = F.sigmoid(x)*self.max_sigma
        return x
    
    def gen_pseudo_data(self, x):
        x_shape = x.shape
        n_of_gen = int( min(self.max_samples, (x_shape[0]*2*self.window_alpha) ** min( torch.sqrt(torch.tensor(x_shape[-1])), torch.tensor(self.max_ord)).item() ) )
        gen_shape = [n_of_gen, *x_shape[1:]]
        # print(gen_shape)
        
        x_min = x.amin(dim=torch.arange(x.ndim)[:-1].tolist())
        x_max = x.amax(dim=torch.arange(x.ndim)[:-1].tolist())
        x_window = x_max - x_min

        x_windowmin = x_min - x_window * self.window_alpha
        x_windowmax = x_max + x_window * self.window_alpha
        x_gen = ( x_windowmin + torch.rand(gen_shape) * x_window *(1 + 2 *self.window_alpha) ).to(x.device)
        return x_gen

model = Model01(1)
# model(torch.rand(10,1))
# a = model.gen_pseudo_data(torch.rand(10,1)).detach()

torch.zeros_like(torch.rand(10,1))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 100
for epoch in range(EPOCHS):
    for batch in train_loader:
        batch_X = batch[0]
        
        ############################################
        X_shape = batch_X.shape
        y_pred = model(batch_X)

        y_true = torch.zeros_like(y_pred)       # pseudo label of True Data
        ############################################
        X_pseudo = model.gen_pseudo_data(batch_X)
        X_pseudo_shape = X_pseudo.shape

        y_pseudo_pred = model(X_pseudo)
        y_pseudo_true = torch.ones_like(y_pseudo_pred) * model.max_sigma    # pseudo label of Pseudo Data
        ############################################
        marginal = X_shape[0] + X_pseudo_shape[0] * 0.3
        p1 = X_shape[0] / marginal
        p2 = X_pseudo_shape[0] * 0.3 / marginal
        
        # ----------------------------------------------
        loss_true = F.mse_loss(y_pred, y_true)
        loss_pseudo = F.mse_loss(y_pseudo_pred, y_pseudo_true)
        loss = p1 * loss_true + p2* loss_pseudo
        #############################################
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"\r{epoch}) {loss.item():.3f}, {loss_pseudo.item():.3f}", end='')





data_inf = torch.FloatTensor(np.linspace(-5,5, num=1000).reshape(-1,1))

pred_data_inf = model(data_inf).detach()
normal_dist = torch.distributions.Normal(0, 1)
pred_prob = (1-normal_dist.cdf(pred_data_inf))*2



fig, ax1 = plt.subplots()
# ax1.scatter(data_inf, -pred_data_inf, alpha=0.5, color='orange')
ax1.scatter(data_inf, pred_prob, alpha=0.5, color='orange')
ax1.set_ylim(0, 1)
ax2 = ax1.twinx()
ax2.hist(data_np_cat.ravel(), bins=30, alpha=0.5,density=True)
plt.show()







########################################################################################

class LinearEmbedding(nn.Module):
    def __init__(self, feature_dim, embed_dim):
        super().__init__()
        # 각 feature별로 weight와 bias를 직접 파라미터로 선언
        self.weight = nn.Parameter(torch.randn(feature_dim, embed_dim, 1))  # (feature_dim, embed_dim, 1)
        self.bias = nn.Parameter(torch.randn(feature_dim, embed_dim))       # (feature_dim, embed_dim)
        self.embed_dim = embed_dim

        # weight 초기화
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='linear')
        fan_in = torch.tensor(1.0)  # 각 feature별로 입력이 1개이므로
        bound = 1.0 / torch.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x: (batch_dim, feature_dim)
        x = x.unsqueeze(-1).unsqueeze(-1)  # (batch_dim, feature_dim, 1, 1)
        out = torch.matmul(x, self.weight.transpose(1, 2))  # (batch_dim, feature_dim, 1, embed_dim)
        out = out.squeeze(-2) + self.bias      # (batch_dim, feature_dim, embed_dim)
        return out

# le = LinearEmbedding(3,2)
# le(torch.rand(10,3)).shape


class SineEmbedding(nn.Module):
    def __init__(self, feature_dim, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # 각 feature별로 k개의 진폭, 주기, 위상 파라미터 선언
        self.amplitude = nn.Parameter(torch.ones(feature_dim, embed_dim))      # (feature_dim, embed_dim)
        self.frequency = nn.Parameter(torch.ones(feature_dim, embed_dim))      # (feature_dim, embed_dim)
        self.phase     = nn.Parameter(torch.zeros(feature_dim, embed_dim))     # (feature_dim, embed_dim)

        # 진폭, 주기, 위상 초기화
        nn.init.normal_(self.amplitude, mean=1.0, std=0.1)
        nn.init.normal_(self.frequency, mean=1.0, std=0.1)
        nn.init.normal_(self.phase, mean=0.0, std=0.1)

    def forward(self, x):
        # x: (batch_dim, feature_dim)
        x = x.unsqueeze(-1)  # (batch_dim, feature_dim, 1)
        # (batch_dim, feature_dim, 1) * (feature_dim, embed_dim) -> (batch_dim, feature_dim, embed_dim)
        out = self.amplitude * torch.sin(self.frequency * x + self.phase)  # (batch_dim, feature_dim, embed_dim)
        return out

# se = SineEmbedding(3,2)
# se(torch.rand(3))

########################################################################################








class Model02(nn.Module):
    def __init__(self, input_dim=1,
                scale_beta=1, forgetting_gamma=0.1,
                window_alpha=3, max_ord=5, max_sigma=6, max_samples=1e+6):
        super().__init__()
        self.window_alpha = window_alpha
        self.scale_beta = scale_beta
        self.forgetting_gamma = forgetting_gamma
        self.max_ord = max_ord
        self.max_samples = max_samples
        self.max_sigma = max_sigma
        self.norm_dist = torch.distributions.Normal(0, 1)

        self.lin_embedding = LinearEmbedding(input_dim, 1)
        self.sin_embedding = SineEmbedding(input_dim, 2)
        self.embed_dim = self.lin_embedding.embed_dim + self.sin_embedding.embed_dim

        self.block = nn.Sequential(
            nn.Linear(input_dim*self.embed_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x_shape = x.shape
        emb_1 = self.lin_embedding(x)
        emb_2 = self.sin_embedding(x)
        emb_x = torch.cat([emb_1, emb_2], dim=-1)
        emb_x_flat = emb_x.view(x_shape[0], -1)

        x = self.block(emb_x_flat)
        x = F.sigmoid(x)*self.max_sigma
        return x

    def predict_probs(self, x):
        pred_sigma = self.forward(x)
        pred_prob = (1 - self.norm_dist.cdf(pred_sigma))*2
        return pred_prob

    def gen_pseudo_data(self, x):
        x_shape = x.shape
        n_of_gen = int( min(self.max_samples, (x_shape[0]*2*self.window_alpha) ** min( torch.sqrt(torch.tensor(x_shape[-1])), torch.tensor(self.max_ord)).item() ) )
        gen_shape = [n_of_gen, *x_shape[1:]]
        # print(gen_shape)
        
        x_min = x.amin(dim=torch.arange(x.ndim)[:-1].tolist())
        x_max = x.amax(dim=torch.arange(x.ndim)[:-1].tolist())
        x_window = x_max - x_min

        x_windowmin = x_min - x_window * self.window_alpha
        x_windowmax = x_max + x_window * self.window_alpha
        x_gen = ( x_windowmin + torch.rand(gen_shape) * x_window *(1 + 2 *self.window_alpha) ).to(x.device)
        return x_gen

model = Model02(3)
# model(torch.rand(10,1))
# a = model.gen_pseudo_data(torch.rand(10,1)).detach()

torch.zeros_like(torch.rand(10,1))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 30
for epoch in range(EPOCHS):
    for batch in train_loader:
        batch_X = batch[0]
        
        ############################################
        X_shape = batch_X.shape
        y_pred = model(batch_X)

        y_true = torch.zeros_like(y_pred)       # pseudo label of True Data
        # ############################################
        X_pseudo = model.gen_pseudo_data(batch_X)
        X_pseudo_shape = X_pseudo.shape

        y_pseudo_pred = model(X_pseudo)
        y_pseudo_true = torch.ones_like(y_pseudo_pred) * model.max_sigma    # pseudo label of Pseudo Data
        # ############################################
        # marginal = X_shape[0] + X_pseudo_shape[0] * 0.05
        # p1 = X_shape[0] / marginal
        # p2 = X_pseudo_shape[0] * 0.05 / marginal
        p1=0.7
        p2=1-p1
        
        # # ----------------------------------------------
        loss_true = F.mse_loss(y_pred, y_true)
        loss_pseudo = F.mse_loss(y_pseudo_pred, y_pseudo_true)
        loss = p1 * loss_true + p2* loss_pseudo
        #############################################
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"\r{epoch}) {loss.item():.3f}, {loss_pseudo.item():.3f}", end='')



# 1차원 시각화 검증  -----------------------------------------------------------
# data_inf = torch.FloatTensor(np.linspace(-5,5, num=1000).reshape(-1,1))

# # pred_data_inf = model(data_inf).detach()
# # normal_dist = torch.distributions.Normal(0, 1)
# # pred_prob = (1-normal_dist.cdf(pred_data_inf))*2

# pred_prob = model.predict_probs(data_inf).detach()



# fig, ax1 = plt.subplots()
# # ax1.scatter(data_inf, -pred_data_inf, alpha=0.5, color='orange')
# ax1.scatter(data_inf, pred_prob, alpha=0.5, color='orange')
# ax1.set_ylim(0, 1)
# ax2 = ax1.twinx()
# ax2.hist(data_np_cat.ravel(), bins=30, alpha=0.5,density=True)
# plt.show()



# # 2차원 시각화 검증 -----------------------------------------------------------
# # grid resolution 설정
# n_grid = 100
# x1 = np.linspace(-5, 5, n_grid)
# x2 = np.linspace(-5, 5, n_grid)
# xx1, xx2 = np.meshgrid(x1, x2)
# grid_points = np.stack([xx1.ravel(), xx2.ravel()], axis=1)  # # shape: (n_grid*n_grid, 2)


# grid_tensor = torch.FloatTensor(grid_points)
# with torch.no_grad():
#     pred_prob = model.predict_probs(grid_tensor).cpu().numpy().reshape(xx1.shape)


# plt.figure(figsize=(8,6))
# # contour map: pred_prob 값이 높을수록 uncertainty가 높음
# contour = plt.contourf(xx1, xx2, pred_prob, levels=30, cmap='coolwarm')
# plt.colorbar(contour, label='Predicted Uncertainty (prob)')

# # 실제 데이터 산점도
# plt.scatter(data_np_cat[:,0], data_np_cat[:,1], s=10, c='black', alpha=0.3, label='Data')

# plt.title("Uncertainty Contour Map over 2D Space")
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.legend()
# plt.show()



# 3차원 시각화 검증 -----------------------------------------------------------
n_grid = 30  # 3D에서는 너무 크면 메모리 이슈가 있을 수 있으니 30~50 추천
x1 = np.linspace(-5, 5, n_grid)
x2 = np.linspace(-5, 5, n_grid)
x3 = np.linspace(-5, 5, n_grid)
xx1, xx2, xx3 = np.meshgrid(x1, x2, x3)
grid_points = np.stack([xx1.ravel(), xx2.ravel(), xx3.ravel()], axis=1)  # shape: (n_grid^3, 3)

grid_tensor = torch.FloatTensor(grid_points)
with torch.no_grad():
    pred_prob = model.predict_probs(grid_tensor).cpu().numpy().ravel()  # shape: (n_grid^3,)

from mpl_toolkits.mplot3d import Axes3D

# alpha 값 계산: pred_prob이 높을수록 alpha가 높게
# (예: 0.2~1.0 사이로 정규화)
alpha_vals = 0.2 + 0.8 * (pred_prob - pred_prob.min()) / (pred_prob.max() - pred_prob.min())

threshold = 0.8  # uncertainty가 0.6 이상인 점만 표시
mask = pred_prob > threshold
fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(grid_points[mask,0], grid_points[mask,1], grid_points[mask,2],
               c=pred_prob[mask], cmap='Reds', s=10,
               alpha=0.5)
            #    alpha=alpha_vals[mask])
ax.scatter(data_np_cat[:,0], data_np_cat[:,1], data_np_cat[:,2], s=15, c='black', alpha=0.5, label='Data')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
plt.title("Uncertainty in 3D Space (filtered)")
plt.legend()
plt.show()











import os
import sys
if 'd:' in os.getcwd().lower():
    os.chdir("D:/DataScience/")
sys.path.append(f"{os.getcwd()}/DataScience/00_DataAnalysis_Basic")
sys.path.append(f"{os.getcwd()}/DataScience/DS_Library")
sys.path.append(r'D:\DataScience\00_DataAnalysis_Basic')


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f"torch_device : {device}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




try:
    from DS_MachineLearning import DS_LabelEncoder, DataPreprocessing
    from DS_DeepLearning import TorchDataLoader, TorchModeling, AutoML, EarlyStopping
    from DS_TorchModule import CategoricalEmbedding, EmbeddingLinear, ContinuousEmbeddingBlock
    from DS_TorchModule import PositionalEncoding, LearnablePositionalEncoding, FeatureWiseEmbeddingNorm
    from DS_TorchModule import ScaledDotProductAttention, MultiheadAttention, PreLN_TransformerEncoderLayer, AttentionPooling
    from DS_TorchModule import KwargSequential, ResidualConnection
    from DS_TorchModule import MaskedConv1d
    from DS_TimeSeries import pad_series_list_1d, pad_series_list_2d, series_smoothing
    
except:
    remote_library_url = 'https://raw.githubusercontent.com/kimds929/'
    try:
        import httpimport
        with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
            from DS_MachineLearning import DS_LabelEncoder, DataPreprocessing, TorchDataLoader, TorchModeling, AutoML
            from DS_DeepLearning import EarlyStopping
            from DS_TorchModule import CategoricalEmbedding, EmbeddingLinear, ContinuousEmbeddingBlock
            from DS_TorchModule import PositionalEncoding, LearnablePositionalEncoding, FeatureWiseEmbeddingNorm
            from DS_TorchModule import ScaledDotProductAttention, MultiheadAttention, PreLN_TransformerEncoderLayer, AttentionPooling
            from DS_TorchModule import KwargSequential, ResidualConnection
            from DS_TorchModule import MaskedConv1d
            from DS_TimeSeries import pad_series_list_1d, pad_series_list_2d, series_smoothing
    except:
        import requests
        response = requests.get(f"{remote_library_url}/DS_Library/main/DS_TimeSeries.py", verify=False)
        exec(response.text)
        
        response = requests.get(f"{remote_library_url}/DS_Library/main/DS_MachineLearning.py", verify=False)
        exec(response.text)
        
        response = requests.get(f"{remote_library_url}/DS_Library/main/DS_DeepLearning.py", verify=False)
        exec(response.text)
        
        response = requests.get(f"{remote_library_url}/DS_Library/main/DS_TorchModule.py", verify=False)
        exec(response.text)


# ---------------------------
# 기본 유틸
# ---------------------------
def _sample_length(mean_len=250, std_len=15, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    L = int(rng.normal(mean_len, std_len))
    return max(L, 32)  # 안정성 확보

# ---- (보조) 동일 파라미터로 t 벡터를 받아 수렴곡선 계산 ----
def _first_order_response_with_t(t: np.ndarray, target: float, k: float, y0: float = 0.0):
    # y(t) = target - (target - y0) * exp(-k * t)
    return target - (target - y0) * np.exp(-k * t)


# ---------------------------
# 정상/이상 시계열 생성기
# ---------------------------
def generate_series(
    mean_len=250, std_len=15, rng=None,
    series_type='normal',
    p_range = None,
    noise_scale=0.1
):
    default_normal_p = (0.85, 1.15)       # normal distribution 
    default_early_p = (0.35, 0.55)        # early distribution
    default_late_p = (1.55, 1.75)         # late distribution
    
    rng = np.random.default_rng() if rng is None else rng
    L = _sample_length(mean_len, std_len, rng)

    # 공통 파라미터
    t = np.linspace(0.0, 1.0, L)
    k = rng.uniform(5.0, 7.0)
    y0 = 0.0
    target = rng.uniform(0.9, 1.1)
    

    # series_type에 따라 p 선택
    if series_type in ['normal', 'early_surge', 'late_surge']:
        
        if series_type == 'normal':
            p_range = default_normal_p if p_range is None else p_range
        elif series_type == 'early_surge':
            p_range = default_early_p if p_range is None else p_range
        elif series_type == 'late_surge':
            p_range = default_late_p if p_range is None else p_range
        p = float(rng.uniform(*p_range))
    else:
        raise ValueError("series_type은 'normal', 'early_surge', 'late_surge'만 지원합니다.")

    # 시간 워핑
    t_warp = np.power(t, p)

    # 기본 수렴 곡선
    y = _first_order_response_with_t(t_warp, target=target, k=k, y0=y0)

    # 감쇠 진동 + 상관 잡음
    freq = rng.uniform(0.5, 1.0)
    damp = rng.uniform(2.0, 4.0)
    oscill = 0.05 * np.sin(2 * np.pi * freq * t) * np.exp(-damp * t)

    eps = rng.normal(0, 0.02 * noise_scale, size=L)
    ar = np.zeros(L)
    phi = rng.uniform(0.2, 0.5)
    for i in range(1, L):
        ar[i] = phi * ar[i-1] + eps[i]

    y = y + oscill + ar
    y_clip = np.clip(y, -5, 5)
    return y_clip, series_type, {"p": p}



# ---------------------------
# 데이터셋 대량 생성기
# ---------------------------
def generate_dataset(n_normal=100, n_early=10, n_late=10, mean_len=250, std_len=15, seed=None):
    """
    반환:
      series_list: 길이가 제각각인 1D ndarray들의 리스트
      labels     : 0(정상) / 1(이상)
      meta       : 각 샘플의 메타정보(dict, anomaly_type 포함)
    """
    rng = np.random.default_rng(seed)
    series_list, labels, meta = [], [], []

    gen_dict = {'normal':n_normal, 'early_surge':n_early, 'late_surge':n_late}
    for series_type, n_gen in gen_dict.items():
        for _ in range(n_gen):
            y, series_type, p_info  = generate_series(mean_len, std_len, rng, series_type=series_type)
            series_list.append(y)
            labels.append(series_type)
            meta.append(p_info)

    return series_list, labels, meta



n_normal = 200
n_early = 200
n_late = 200

# early_surge, late_surge
X, y, info = generate_dataset(n_normal=n_normal, n_early=n_early, n_late=n_late)
print(len(X), "series generated.")



# 시각화
plt.figure()
colors = {'normal': 'steelblue', 'early_surge':'orange', 'late_surge':'green'}
for Xi, yi in zip(X, y):
    plt.plot(Xi, alpha=0.5, color=colors[yi], label=yi)
# plt.legend()
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.show()


def pad_series_list(series_list, pad_value=np.nan):
    """
    서로 다른 길이의 시계열을 최대 길이에 맞춰 zero padding.
    Args:
        series_list (list[np.ndarray]): 각 시계열 (길이 다름)
        pad_value (float): 패딩값 (기본 0)
    Returns:
        np.ndarray: shape = (N, max_len)
    """
    max_len = max(len(s) for s in series_list)
    padded = np.full((len(series_list), max_len), pad_value, dtype=float)
    for i, s in enumerate(series_list):
        padded[i, :len(s)] = s
    return padded


############################################################################################################
normal_idx = (np.array(y) == 'normal')
y_labels_dict = {'normal': 0, 'early_surge':1, 'late_surge':1}
y_transformed = np.array([y_labels_dict[yi] for yi in y])
X_Series = pad_series_list(X, pad_value=np.nan)





############################################################################################################
normal_idx = (np.array(y) == 'normal')
y_labels_dict = {'normal': 0, 'early_surge':1, 'late_surge':1}
y_transformed = np.array([y_labels_dict[yi] for yi in y])
X_Series = pad_series_list(X, pad_value=np.nan)


# ------------------------------------------------------------------------------------
def compute_tail_mean(x_np: np.ndarray, mask_np: np.ndarray, tail_frac=0.10):
    """각 시계열 뒤 10%(tail_frac) 유효구간의 평균 → setpoint 근사"""
    N, L = x_np.shape
    
    tail_mean = np.zeros(N, dtype=np.float32)
    for i in range(N):
        valid_indices = np.where(mask_np[i])[0]
        
        if len(valid_indices) == 0:
            tail_mean[i] = 0.0
            continue
        else:
            compute_len = max(1, int(L*tail_frac))
            tail_mean[i] =x_np[i][valid_indices[-compute_len:]].mean()
    return tail_mean

def timeseries_processing(x, masking_value=np.nan, replace_value=None, tail_frac=0.1, to_tensor=True):
    if np.isnan(masking_value):
        mask = ~np.isnan(x)
        x_transform = np.nan_to_num(x, nan=0.0) if replace_value is not None else x.copy()
    else:
        mask = x != masking_value
        x_transform = x.copy()
        if replace_value is not None:
            replace_mask = (x_transform == masking_value)
            x_transform[replace_mask] = replace_value
    
    # valid_seq_len
    valid_seq_len = np.where((~mask).any(axis=-1), (~mask).argmax(axis=-1), mask.shape[-1])
    
    # tail_mean
    tail_mean = compute_tail_mean(x_transform, mask, tail_frac=tail_frac)
    
    if to_tensor:
        return torch.FloatTensor(x_transform), torch.tensor(mask, dtype=bool), torch.LongTensor(valid_seq_len), torch.FloatTensor(tail_mean)
    else:
        return x_transform, mask, valid_seq_len, tail_mean

# ------------------------------------------------------------------------------------
############################################################################################################






























############################################################################################################
############################################################################################################
############################################################################################################

# ------------------------------------------------------------------------------------
# (Slicing)
# X_ = X_Series[:,::10]
slicing_num = 20
slice_idx = np.linspace(0, X_Series.shape[1]-1, num=slicing_num).astype(int)
X_slice = X_Series[:, slice_idx]


# visualize slicing
plt.plot(X_slice[:n_normal].T, alpha=0.3, color='steelblue')
plt.plot(X_slice[n_normal:].T, alpha=0.3, color='orange')
plt.show()


# ------------------------------------------------------------------------------------
# (Train Test Split)
from sklearn.model_selection import train_test_split

random_state = 0
train_valid_idx ,tests_idx = train_test_split(np.arange(len(X_slice)), test_size=0.3, stratify=y, random_state=random_state)
train_idx, valid_idx = train_test_split(train_valid_idx, test_size=0.3, stratify=y_transformed[train_valid_idx], random_state=random_state)


train_idx = sorted(train_idx)
valid_idx = sorted(valid_idx)
tests_idx = sorted(tests_idx)

train_X = X_slice[train_idx]
valid_X = X_slice[valid_idx]
tests_X = X_slice[tests_idx]

train_y = y_transformed[train_idx]
valid_y = y_transformed[valid_idx]
tests_y = y_transformed[tests_idx]


# ------------------------------------------------------------------------------------
# (Normalizing : StandardScaler)
from sklearn.preprocessing import StandardScaler
X_mean = np.nanmean(train_X)
X_std = np.nanstd(train_X)
train_X_norm = (train_X - X_mean) / X_std
valid_X_norm = (valid_X - X_mean) / X_std
tests_X_norm = (tests_X - X_mean) / X_std



# ss_X = StandardScaler()
# train_X_norm = ss_X.fit_transform(train_X)
# valid_X_norm = ss_X.transform(valid_X)
# tests_X_norm = ss_X.transform(tests_X)




train_X_tensor_set = timeseries_processing(train_X_norm, replace_value=0)
valid_X_tensor_set = timeseries_processing(valid_X_norm, replace_value=0)
tests_X_tensor_set = timeseries_processing(tests_X_norm, replace_value=0)

train_y_tensor = torch.LongTensor(train_y)
valid_y_tensor = torch.LongTensor(valid_y)
tests_y_tensor = torch.LongTensor(tests_y)

# visualize tail mean
# train_set
trainX_tensor, trainX_mask, trainX_valid_seq_len, trainX_tailmean = train_X_tensor_set
len_normal_train = np.argmax(train_y)

plt.plot(trainX_tensor[:len_normal_train].T, alpha=0.1, color='steelblue')
plt.scatter(trainX_valid_seq_len[:len_normal_train]-1, trainX_tailmean[:len_normal_train], color='steelblue', edgecolor='blue', s=5)
plt.plot(trainX_tensor[len_normal_train:].T, alpha=0.5, color='orange')
plt.scatter(trainX_valid_seq_len[len_normal_train:]-1, trainX_tailmean[len_normal_train:], color='orange', edgecolor='red', s=5)
plt.show()



#################################################################################################
from torch.utils.data import Dataset, TensorDataset, DataLoader

train_dataset = TensorDataset(*train_X_tensor_set, train_y_tensor)
valid_dataset = TensorDataset(*valid_X_tensor_set, valid_y_tensor)
tests_dataset = TensorDataset(*tests_X_tensor_set, tests_y_tensor)

BATCH = 128
train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH, shuffle=True)
tests_loader = DataLoader(tests_dataset, batch_size=BATCH, shuffle=True)

for batch in train_loader:
    break
len(batch)        # 5 : X, mask, valid_seq_len, tail_mean, y
batch[0]    # time-series data (Batch, Seq)
batch[1]    # masking data (Batch, Seq)
batch[2]    # valid sequence length (Batch)
# batch[3].shape
# batch[4].shape

#################################################################################################












#################################################################################################
#################################################################################################
#################################################################################################
class TimeSeriesConv1dVAE(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, latent_dim=8):
        super().__init__()
        self.register_buffer('T', torch.ones(()) )
        
        # (convolution encoder)
        self.encoder = KwargSequential(
            MaskedConv1d(in_channels, hidden_channels, kernel_size=5, padding=2)
            ,nn.ReLU()
            ,ResidualConnection(
                KwargSequential(
                    MaskedConv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
                    ,nn.BatchNorm1d(hidden_channels)
                    ,nn.ReLU()
                )
            )
            ,ResidualConnection(
                KwargSequential(
                    MaskedConv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
                    ,nn.BatchNorm1d(hidden_channels)
                    ,nn.ReLU()
                )
            )
        )
        
        self.attn_pool_layer = AttentionPooling(hidden_channels)
        
        self.mu = None
        self.logvar = None
        self.std = None
        
        # (latent dim)
        self.fc_mu = nn.Linear(hidden_channels, latent_dim)
        self.fc_logvar = nn.Linear(hidden_channels, latent_dim)
        
        # (convolution decoder)
        self.decode_broadcast = nn.Linear(latent_dim, hidden_channels)
        
        self.decoder = KwargSequential(
            ResidualConnection(
                KwargSequential(
                    MaskedConv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
                    ,nn.BatchNorm1d(hidden_channels)
                    ,nn.ReLU()
                )
            )
            ,ResidualConnection(
                KwargSequential(
                    MaskedConv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
                    ,nn.BatchNorm1d(hidden_channels)
                    ,nn.ReLU()
                )
            ),
            MaskedConv1d(hidden_channels, in_channels, kernel_size=5, padding=2)
        )
        
    def encode(self, x, mask):
        # x.shape   # (B, T)
        # mask.shape # (B, T)   # valid: True
        x_T = x.unsqueeze(-2)   # (B, E, T)
        
        # (convolution encoder)
        encoder_out = self.encoder(x_T, mask=mask)     # (B, E, T)
        
        # (masking)
        mask_unsqueeze = mask.unsqueeze(-2).to(encoder_out.dtype)   # (B, 1, T)
        encoder_out_mask = encoder_out * mask_unsqueeze     # (B, E, T)
        
        # (information pooling)
        pool_out, _ = self.attn_pool_layer(encoder_out_mask.transpose(-2,-1), mask)    # (B, E) : Attention Pooling
        
        mu = self.fc_mu(pool_out)
        logvar = self.fc_logvar(pool_out)
        
        return mu, logvar
    
    def reparamerize(self, mu, std):
        # latent_z = mu + std * eps
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def decode(self, latent_z, mask):
        T = self.T.type(torch.int64).item()
        
        # latent_z: (B, z_dim) → (B, hidden) → (B, hidden, T)
        broadcast_out = self.decode_broadcast(latent_z)    # (B, z_dim) → (B, hidden) 
        
        broadcast_out_T = broadcast_out.unsqueeze(-1).expand(-1, -1, T) # broadcast :  (B, hidden) → (B, hidden, T)
        
        X_recon = self.decoder(broadcast_out_T, mask=mask)
        return X_recon.squeeze(1)
    
    def forward(self, x, mask):
        # x.shape   # (B, T)
        # mask.shape # (B, T)   # valid: True
        B, T = x.shape
        self.T = torch.tensor(T).type(torch.float32)
        
        # encode
        self.mu, self.logvar = self.encode(x, mask)
        self.std = torch.exp(0.5 * self.logvar)
        
        # reparamerize
        latent_z = self.reparamerize(self.mu, self.std)
        
        # decode
        X_recon = self.decode(latent_z, mask)
        return X_recon, self.mu, self.logvar

##########################################################################################
model = TimeSeriesConv1dVAE(in_channels=1, hidden_channels=64, latent_dim=2).to(device)
# f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
# model(batch[0].to(device), batch[1].to(device))

# load_state = torch.load("D:/DataScience/Model/TimeSeriesConv1dVAE_params.pth")
# model.load_state_dict(load_state)


class ConvVAELoss():
    def __init__(self, beta=1.0):
        self.beta = beta
        
    def loss_function(self, model, batch, optimizer=None):
        batch_X, mask, valid_seq_len, tail_mean, batch_y = batch
        
        X_recon, mu, logvar = model(batch_X, mask)
        
        valid_seq_len = mask.sum(dim=-1).clamp_min(1.0)   # (B, ) : valid seq len
        
        # reconstruction loss
        mse_elements = (batch_X - X_recon ) ** 2    # (B, T)
        loss_recon_batch = (mse_elements * mask).sum(dim=-1) / valid_seq_len  # (B, )
        loss_recon =  loss_recon_batch.mean()    # scalar
        
        # KL-Divergence loss
        loss_kl_batch = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=-1)      # (B, )
        loss_kl = loss_kl_batch.mean()
        
        # total loss
        loss = loss_recon + self.beta * loss_kl
        
        return loss
    
conv_vae_loss = ConvVAELoss()

tm = TorchModeling(model, device)
tm.compile(optim.Adam(model.parameters(), lr=1e-3),
           early_stop_loss = EarlyStopping(min_iter=200, patience=100))
tm.train_model(train_loader, valid_loader, epochs=500, loss_function=conv_vae_loss.loss_function)

tm.early_stop_loss.plot
# torch.save(model.state_dict(), "D:/DataScience/Model/TimeSeriesConv1dVAE_params.pth")


# ----------------------------------------------------------------------------------------------------------------------------------

model.eval()
mask = (batch[1].type(torch.float32).mean(0, keepdim=True) > 0.5)
pred = ( model.decode(torch.zeros(2).unsqueeze(0).to(device), mask.to(device)).detach().to('cpu') * mask.type(torch.float32).masked_fill(~mask, torch.nan) ).numpy()[0]



# mean timeseries plot
plt.plot(trainX_tensor[:len_normal_train].T, alpha=0.1, color='steelblue')
plt.plot(trainX_tensor[len_normal_train:].T, alpha=0.1, color='orange')
plt.plot(pred, alpha=1, color='red')
plt.show()


# latent_z plot
mu, logvar = model.encode(batch[0].to(device), batch[1].to(device))

plt.scatter(*mu.detach().to('cpu').numpy().T)
plt.axvline(0, color='black')
plt.axhline(0, color='black')
plt.xlim(-0.1,0.1)
plt.ylim(-0.1,0.1)































#################################################################################################
#################################################################################################
#################################################################################################
class TimeSeriesBatchConv1dVAE(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_channels=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.register_buffer('T', torch.ones(()) )
        
        # (convolution encoder)
        self.encoder = KwargSequential(
            MaskedConv1d(in_channels, hidden_channels, kernel_size=5, padding=2)
            ,nn.ReLU()
            ,ResidualConnection(
                KwargSequential(
                    MaskedConv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
                    ,nn.BatchNorm1d(hidden_channels)
                    ,nn.ReLU()
                )
            )
            ,ResidualConnection(
                KwargSequential(
                    MaskedConv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
                    ,nn.BatchNorm1d(hidden_channels)
                    ,nn.ReLU()
                )
            )
        )        
        self.attn_pool_layer = AttentionPooling(hidden_channels)
        
        # (latent head)
        self.latent_head = nn.Linear(hidden_channels, latent_dim)
        
        # mu, cov
        self.eps = 1e-3
        self.n = 0
        
        # register_buffer : 모델 안에 학습 대상은 아니지만 저장/로드 시 함께 관리해야 하는 텐서를 등록하는 기능
        self.register_buffer("mu", torch.zeros(latent_dim), persistent=True)        
        self.register_buffer("cov", torch.eye(latent_dim) * self.eps, persistent=True)
        
        # (convolution decoder)
        self.decode_broadcast = nn.Linear(latent_dim, hidden_channels)
        
        self.decoder = KwargSequential(
            ResidualConnection(
                KwargSequential(
                    MaskedConv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
                    ,nn.BatchNorm1d(hidden_channels)
                    ,nn.ReLU()
                )
            )
            ,ResidualConnection(
                KwargSequential(
                    MaskedConv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
                    ,nn.BatchNorm1d(hidden_channels)
                    ,nn.ReLU()
                )
            ),
            MaskedConv1d(hidden_channels, in_channels, kernel_size=5, padding=2)
        )
        
    def reset_params(self):
        self.n = 0
        self.mu.zero_()
        self.cov.copy_(torch.eye(self.latent_dim, device=self.cov.device) * self.eps)
    
    @torch.no_grad()
    def update_params(self, n_batch, mu, cov):
        self.mu = (self.n * self.mu + n_batch * mu) / (self.n + n_batch)
        self.cov = (self.n * self.cov + n_batch * cov) / (self.n + n_batch)
        self.n += n_batch
    
    def encode(self, x, mask):
        # x.shape   # (B, T)
        # mask.shape # (B, T)   # valid: True
        x_T = x.unsqueeze(-2)   # (B, E, T)
        
        # (convolution encoder)
        encoder_out = self.encoder(x_T, mask=mask)     # (B, E, T)
        
        # (masking)
        mask_unsqueeze = mask.unsqueeze(-2).to(encoder_out.dtype)   # (B, 1, T)
        encoder_out_mask = encoder_out * mask_unsqueeze     # (B, E, T)
        
        # (information pooling)
        pool_out, _ = self.attn_pool_layer(encoder_out_mask.transpose(-2,-1), mask)    # (B, E) : Attention Pooling
        # pool_out = self.masked_mean(encoder_out_mask, mask_unsqueeze)   # (B, E) : Mean Pooling
        
        # (classification head)
        latent_z = self.latent_head(pool_out)   # latent : (B, latent_dim)
        return torch.tanh(latent_z)*8
    
    def encode_gaussian(self, x, mask, batch_seq=-1):
        n_batch = x.size(0)
        latent_z = self.encode(x, mask)
        K = latent_z.size(-1)
        
        # mu, covariance
        mu = latent_z.mean(dim=0)
        cov = ((latent_z-mu).T @ (latent_z-mu)) / n_batch
        cov += self.eps * torch.eye(K, device=cov.device)
        
        if batch_seq > 0:
            if batch_seq == 0:
                self.reset_params()
            self.update_params(n_batch, mu.detach(), cov.detach())
        return latent_z, mu, cov
    
    def decode(self, latent_z, mask):
        T = self.T.type(torch.int64).item()
        
        # latent_z: (B, z_dim) → (B, hidden) → (B, hidden, T)
        broadcast_out = self.decode_broadcast(latent_z)    # (B, z_dim) → (B, hidden) 
        
        broadcast_out_T = broadcast_out.unsqueeze(-1).expand(-1, -1, T) # broadcast :  (B, hidden) → (B, hidden, T)
        
        X_recon = self.decoder(broadcast_out_T, mask=mask)
        return X_recon.squeeze(1)
    
    def decode_masking(self, latent_z, mask):
        X_recon = self.decode(latent_z, mask)
        return X_recon.masked_fill(~mask, torch.nan)
    
    def forward(self, x, mask, batch_seq=-1):
        # x.shape   # (B, T)
        # mask.shape # (B, T)   # valid: True
        B, T = x.shape
        self.T = torch.tensor(T).type(torch.float32)
        
        # encode
        latent_z, mu, cov = self.encode_gaussian(x, mask, batch_seq)
        
        # decode
        X_recon = self.decode(latent_z, mask)
        return X_recon, mu, cov
##########################################################################################

model = TimeSeriesBatchConv1dVAE(in_channels=1, hidden_channels=64, latent_dim=2).to(device)
# f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
# model(batch[0], batch[1])


# load_state = torch.load("D:/DataScience/Model/TimeSeriesBatchConv1dVAE_params.pth")
# model.load_state_dict(load_state)


class Batch_KL_Divergece():
    def __init__(self, train_loader, beta=1.0):
        self.beta = beta
        self.len_batch = len(train_loader)
        self.count = 0
        
    def batch_counter(self):
        self.count = self.count % self.len_batch
    
    def loss_function(self, model, batch, optimizer=None):
        self.batch_counter()
        batch_X, mask, valid_seq_len, tail_mean, batch_y = batch
        X_recon, mu, cov = model(batch_X, mask, self.count)
        
        # reconstruction loss
        mse_elements = (batch_X - X_recon ) ** 2    # (B, T)
        loss_recon_batch = (mse_elements * mask).sum(dim=-1) / valid_seq_len  # (B, )
        loss_recon =  loss_recon_batch.mean()    # scalar
        
        # KL-Divergence loss
        K = mu.size(-1)
        sign, logdet = torch.linalg.slogdet(cov)
        loss_kl = 0.5 * (torch.trace(cov) + mu @ mu - K - logdet)
        
        # total loss
        loss = loss_recon + self.beta * loss_kl
        
        self.count += 1
        return loss


batch_kl_loss = Batch_KL_Divergece(train_loader)

tm = TorchModeling(model, device)
tm.compile(optim.Adam(model.parameters(), lr=1e-3),
           early_stop_loss = EarlyStopping(min_iter=200, patience=100))
tm.train_model(train_loader, valid_loader, epochs=500, loss_function=batch_kl_loss.loss_function)

tm.early_stop_loss.plot
# torch.save(model.state_dict(), "D:/DataScience/Model/TimeSeriesBatchConv1dVAE_params.pth")



# ----------------------------------------------------------------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    mask = (batch[1].type(torch.float32).mean(0, keepdim=True) > 0.5)
    pred = ( model.decode(model.mu.unsqueeze(0), mask.to(device)).detach().to('cpu') * mask.type(torch.float32).masked_fill(~mask, torch.nan) ).numpy()[0]



# ----------------------------------------------------------------------------------------------------------------------------------
# mean timeseries plot
plt.plot(trainX_tensor[:len_normal_train].T, alpha=0.1, color='steelblue')
plt.plot(trainX_tensor[len_normal_train:].T, alpha=0.1, color='orange')
plt.plot(pred, alpha=1, color='red')
plt.show()



# mean timeseries plot with latent sampling
dist = torch.distributions.MultivariateNormal(model.mu, covariance_matrix=model.cov)
with torch.no_grad():
    pred_plot = model.decode_masking(dist.sample((1000,)).to(device), torch.BoolTensor(~np.isnan(pred)).expand(1000,-1).to(device)).detach().to('cpu').numpy()

plt.plot(pred_plot.T, alpha=0.01, color='steelblue')
plt.plot(pred, alpha=1, color='red')
plt.show()


# ----------------------------------------------------------------------------------------------------------------------------------
# latent_z plot
train_timeseries, train_mask, train_valid_len, train_tail_mean, train_y = train_dataset.tensors
sample_batch = train_timeseries[:]
sample_mask =  train_mask[:]

# tests_timeseries, tests_mask, tests_valid_len, tests_tail_mean, tests_y = tests_dataset.tensors
# sample_batch = tests_timeseries[:]
# sample_mask =  tests_mask[:]

sample_batch_unmask = sample_batch.masked_fill(sample_mask == 0, float('nan'))
sample_origin_scale = (sample_batch_unmask * X_std + X_mean).numpy()

with torch.no_grad():
    pred_batch = model.encode(sample_batch.to(device), sample_mask.to(device)).detach().to('cpu')
    pred_batch_np = pred_batch.numpy()
# model.mu
# model.cov

maha_dists = torch.sqrt( torch.einsum("nd,nd->n", torch.einsum("nd, de->ne", pred_batch, torch.linalg.inv(model.cov.to('cpu'))), pred_batch).reshape(-1,1) )

# Latent Dimension Latent Plot
threshold = 3
# Visualize 2-dim
plt.scatter(pred_batch_np[:, 0], pred_batch_np[:, 1], c=maha_dists.numpy().ravel(), cmap='jet', vmax=3)
plt.xlim(-4.5, 4.5)
plt.ylim(-4.5, 4.5)
circle = plt.Circle((0, 0), threshold, color='red', fill=False) 
plt.gca().add_patch(circle)
plt.colorbar()
plt.show()


# ----------------------------------------------------------------------------------------------------------------------------------
import matplotlib.cm as cm

plt.figure(figsize=(10, 6))
norm = plt.Normalize(vmin=maha_dists.min(), vmax=threshold)
cmap = cm.get_cmap('jet')
for i in range(len(sample_origin_scale)):
    color_val = cmap(norm(maha_dists.ravel()[i]))
    plt.plot(sample_origin_scale[i, :], color=color_val, linewidth=1.5, alpha=0.2)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([]) # 실제 데이터 없이 범위만 지정
current_ax = plt.gca() 
plt.colorbar(sm, ax=current_ax, label='Mahalanobis Distance') 
plt.show()






########################################################################################################################
# Gaussian Mixture Model
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3, covariance_type="full", reg_covar=1e-8).fit(pred_batch_np)

pred_y = gmm.predict(pred_batch_np)


plt.scatter(pred_batch_np[:, 0], pred_batch_np[:, 1], c=pred_y)
plt.xlim(-4.5, 4.5)
plt.ylim(-4.5, 4.5)
circle = plt.Circle((0, 0), threshold, color='red', fill=False) 
plt.gca().add_patch(circle)
plt.colorbar()
plt.show()


import matplotlib.cm as cm
colors = cm.get_cmap('tab10').colors

plt.figure(figsize=(10, 6))
for i in range(trainX_tensor.shape[0]):  # 20개의 라인
    plt.plot(trainX_tensor[i], color=colors[pred_y[i]], alpha=0.1, label=pred_y[i])

handles, labels = plt.gca().get_legend_handles_labels()     # 현재 범례 항목 가져오기
unique = dict(zip(labels, handles))     # 중복 제거 (순서 유지)
plt.legend(unique.values(), unique.keys(), loc='upper right', bbox_to_anchor=(1,1))
plt.show()




# ---------------------------------------------------------------------------------------------------
# Hierarchical
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

Z = linkage(pred_batch_np, method='single')  # ward는 유클리드 기반에서 가장 흔함
dend = dendrogram(Z, truncate_mode='lastp', p=30)  

hierarchy = AgglomerativeClustering(n_clusters=3, linkage="ward")   # 합칠 때, 군집 내 분산(SSE) 증가량을 최소화
pred_y = hierarchy.fit_predict(pred_batch_np)

plt.scatter(pred_batch_np[:, 0], pred_batch_np[:, 1], c=pred_y)
plt.xlim(-4.5, 4.5)
plt.ylim(-4.5, 4.5)
circle = plt.Circle((0, 0), threshold, color='red', fill=False) 
plt.gca().add_patch(circle)
plt.colorbar()
plt.show()


import matplotlib.cm as cm
colors = cm.get_cmap('tab10').colors

plt.figure(figsize=(10, 6))
for i in range(trainX_tensor.shape[0]):  # 20개의 라인
    plt.plot(trainX_tensor[i], color=colors[pred_y[i]], alpha=0.1, label=pred_y[i])

handles, labels = plt.gca().get_legend_handles_labels()     # 현재 범례 항목 가져오기
unique = dict(zip(labels, handles))     # 중복 제거 (순서 유지)
plt.legend(unique.values(), unique.keys(), loc='upper right', bbox_to_anchor=(1,1))
plt.show()


# ---------------------------------------------------------------------------------------------------
# Spectral Clustering
from sklearn.cluster import SpectralClustering

spectral = SpectralClustering(
    n_clusters=3,
    affinity="nearest_neighbors",  # 또는 "rbf"
    n_neighbors=10,
    assign_labels="kmeans",
    random_state=0
)

pred_y = spectral.fit_predict(pred_batch_np)

plt.scatter(pred_batch_np[:, 0], pred_batch_np[:, 1], c=pred_y)
plt.xlim(-4.5, 4.5)
plt.ylim(-4.5, 4.5)
circle = plt.Circle((0, 0), threshold, color='red', fill=False) 
plt.gca().add_patch(circle)
plt.colorbar()
plt.show()


import matplotlib.cm as cm
colors = cm.get_cmap('tab10').colors

plt.figure(figsize=(10, 6))
for i in range(trainX_tensor.shape[0]):  # 20개의 라인
    plt.plot(trainX_tensor[i], color=colors[pred_y[i]], alpha=0.1, label=pred_y[i])

handles, labels = plt.gca().get_legend_handles_labels()     # 현재 범례 항목 가져오기
unique = dict(zip(labels, handles))     # 중복 제거 (순서 유지)
plt.legend(unique.values(), unique.keys(), loc='upper right', bbox_to_anchor=(1,1))
plt.show()


























#################################################################################################
#################################################################################################
#################################################################################################
class TimeSeriesBatchConv1dVAE_SeqLen(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_channels=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.register_buffer('T', torch.ones(()) )
        
        # (convolution encoder)
        self.encoder = KwargSequential(
            MaskedConv1d(in_channels, hidden_channels, kernel_size=5, padding=2)
            ,nn.ReLU()
            ,ResidualConnection(
                KwargSequential(
                    MaskedConv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
                    ,nn.BatchNorm1d(hidden_channels)
                    ,nn.ReLU()
                )
            )
            ,ResidualConnection(
                KwargSequential(
                    MaskedConv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
                    ,nn.BatchNorm1d(hidden_channels)
                    ,nn.ReLU()
                )
            )
        )        
        self.attn_pool_layer = AttentionPooling(hidden_channels)
        
        # (latent head)
        self.latent_head = nn.Linear(hidden_channels, latent_dim)
        
        # mu, cov
        self.eps = 1e-3
        self.n = 0
        
        # register_buffer : 모델 안에 학습 대상은 아니지만 저장/로드 시 함께 관리해야 하는 텐서를 등록하는 기능
        self.register_buffer('mu', torch.zeros(latent_dim))        
        self.register_buffer('cov', torch.eye(latent_dim) * self.eps)
        self.register_buffer('mu_seqlen', torch.zeros(()) )
        self.register_buffer('std_seqlen', torch.ones(()) * self.eps)
        
        # (convolution decoder)
        self.decode_broadcast = nn.Linear(latent_dim-1, hidden_channels)
        
        self.decoder = KwargSequential(
            ResidualConnection(
                KwargSequential(
                    MaskedConv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
                    ,nn.BatchNorm1d(hidden_channels)
                    ,nn.ReLU()
                )
            )
            ,ResidualConnection(
                KwargSequential(
                    MaskedConv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
                    ,nn.BatchNorm1d(hidden_channels)
                    ,nn.ReLU()
                )
            ),
            MaskedConv1d(hidden_channels, in_channels, kernel_size=5, padding=2)
        )
        
    def reset_params(self):
        self.n = 0
        self.mu.zero_()
        self.cov.copy_(torch.eye(self.latent_dim, device=self.cov.device) * self.eps)
        self.mu_seqlen.zero_()
        self.std_seqlen.fill_(self.eps)
    
    def update_params(self, n_batch, mu, cov, mu_seqlen, std_seqlen):
        self.mu = (self.n * self.mu + n_batch * mu) / (self.n + n_batch)
        self.cov = (self.n * self.cov + n_batch * cov) / (self.n + n_batch)
        self.mu_seqlen = (self.n * self.mu_seqlen + n_batch * mu_seqlen) / (self.n + n_batch)
        self.std_seqlen = (self.n * self.std_seqlen + n_batch * std_seqlen) / (self.n + n_batch)
        self.n += n_batch
    
    def encode(self, x, mask):
        # x.shape   # (B, T)
        # mask.shape # (B, T)   # valid: True
        x_T = x.unsqueeze(-2)   # (B, E, T)
        
        # (convolution encoder)
        encoder_out = self.encoder(x_T, mask=mask)     # (B, E, T)
        
        # (masking)
        mask_unsqueeze = mask.unsqueeze(-2).to(encoder_out.dtype)   # (B, 1, T)
        encoder_out_mask = encoder_out * mask_unsqueeze     # (B, E, T)
        
        # (information pooling)
        pool_out, _ = self.attn_pool_layer(encoder_out_mask.transpose(-2,-1), mask)    # (B, E) : Attention Pooling
        # pool_out = self.masked_mean(encoder_out_mask, mask_unsqueeze)   # (B, E) : Mean Pooling
        
        # (classification head)
        latent_z = self.latent_head(pool_out)   # latent : (B, latent_dim)
        return torch.tanh(latent_z)*8
    
    def encode_gaussian(self, x, mask, batch_seq=-1):
        n_batch = x.size(0)
        latent_z = self.encode(x, mask)
        K = latent_z.size(-1)
        
        # (latent) mu, covariance
        mu = latent_z.mean(dim=0)
        cov = ((latent_z-mu).T @ (latent_z-mu)) / n_batch
        cov += self.eps * torch.eye(K, device=cov.device)
        
        
        # (seq_len) mu, std 
        seqlen = mask.sum(dim=1, keepdim=True).type(torch.float32)
        mu_seqlen = seqlen.mean()
        std_seqlen = seqlen.std()
        
        if batch_seq > 0:
            if batch_seq == 0:
                self.reset_params()
            self.update_params(
                n_batch,
                mu.detach(), cov.detach(),
                mu_seqlen.detach(), std_seqlen.detach()
            )
        return latent_z, mu, cov, mu_seqlen, std_seqlen
    
    def decode(self, latent_z_pattern, mask):
        T = self.T.type(torch.int64).item()
        
        # latent_z_pattern: (B, z_dim-1) → (B, hidden) → (B, hidden, T)
        broadcast_out = self.decode_broadcast(latent_z_pattern)    # (B, z_dim-1) → (B, hidden) 
        
        broadcast_out_T = broadcast_out.unsqueeze(-1).expand(-1, -1, T) # broadcast :  (B, hidden) → (B, hidden, T)
        
        X_recon = self.decoder(broadcast_out_T, mask=mask)
        return X_recon.squeeze(1)
    
    def decode_masking(self, latent_z):
        B = len(latent_z)
        T = self.T.type(torch.int64).item()
        
        latent_z_pattern = latent_z[...,:-1]
        latent_z_seqlen = latent_z[...,[-1]]
        
        pred_seq_len = torch.round(latent_z_seqlen * self.std_seqlen + self.mu_seqlen, decimals=0).type(torch.int64)
        
        idx = torch.arange(T, device=latent_z.device).unsqueeze(0).expand(B, T) 
        mask = idx < pred_seq_len
        
        X_recon = self.decode(latent_z_pattern, mask)
        
        return X_recon.masked_fill(~mask, torch.nan)
    
    def forward(self, x, mask, batch_seq=-1):
        # x.shape   # (B, T)
        # mask.shape # (B, T)   # valid: True
        B, T = x.shape
        self.T = torch.tensor(T).type(torch.float32)
        
        # encode
        latent_z, mu, cov, mu_seqlen, std_seqlen = self.encode_gaussian(x, mask, batch_seq)
        
        
        # sequence length
        seqlen_recon = latent_z[...,[-1]]
        
        # decode
        X_recon = self.decode(latent_z[...,:-1], mask)
        return X_recon, mu, cov, seqlen_recon, mu_seqlen, std_seqlen
    
##########################################################################################
model = TimeSeriesBatchConv1dVAE_SeqLen(in_channels=1, hidden_channels=64, latent_dim=3).to(device)
# f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
# model(batch[0], batch[1])


# load_state = torch.load("D:/DataScience/Model/TimeSeriesBatchConv1dVAE_SeqLen_params.pth")
# model.load_state_dict(load_state)


class Batch_KL_Divergece():
    def __init__(self, train_loader, beta=1.0, gamma=0.5):
        self.beta = beta
        self.gamma = gamma
        self.len_batch = len(train_loader)
        self.count = 0
        
    def batch_counter(self):
        self.count = self.count % self.len_batch
    
    def loss_function(self, model, batch, optimizer=None):
        self.batch_counter()
        batch_X, mask, valid_seq_len, tail_mean, batch_y = batch
        X_recon, mu, cov, X_recon_seqlen_norm, mu_seqlen, std_seqlen = model(batch_X, mask, self.count)
        
        # reconstruction loss
        mse_recon_elements = (batch_X - X_recon ) ** 2    # (B, T)
        loss_recon_batch = (mse_recon_elements * mask).sum(dim=-1) / valid_seq_len  # (B, )
        loss_recon =  loss_recon_batch.mean()    # scalar
        
        # KL-Divergence loss
        K = mu.size(-1)
        sign, logdet = torch.linalg.slogdet(cov)
        loss_kl = 0.5 * (torch.trace(cov) + mu @ mu - K - logdet)
        
        # sequence length loss
        seqlen = mask.sum(dim=1, keepdim=True).type(torch.float32)
        seqlen_norm = (seqlen  - mu_seqlen) / std_seqlen
        
        mse_recon_seqlen = (seqlen_norm - X_recon_seqlen_norm ) ** 2    # (B, 1)
        loss_seqlen = mse_recon_seqlen.mean()
        
        # total loss
        loss = loss_recon + self.beta * loss_kl + self.gamma * loss_seqlen
        
        self.count += 1
        return loss


batch_kl_loss = Batch_KL_Divergece(train_loader)

tm = TorchModeling(model, device)
tm.compile(optim.Adam(model.parameters(), lr=1e-3),
           early_stop_loss = EarlyStopping(min_iter=200, patience=100))
tm.train_model(train_loader, valid_loader, epochs=500, loss_function=batch_kl_loss.loss_function)

tm.early_stop_loss.plot
# torch.save(model.state_dict(), "D:/DataScience/Model/TimeSeriesBatchConv1dVAE_SeqLen_params.pth")



# ----------------------------------------------------------------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    pred = model.decode_masking(model.mu.unsqueeze(0)).detach().to('cpu').numpy()[0]



# ----------------------------------------------------------------------------------------------------------------------------------
# mean timeseries plot
plt.plot(trainX_tensor[:len_normal_train].T, alpha=0.1, color='steelblue')
plt.plot(trainX_tensor[len_normal_train:].T, alpha=0.1, color='orange')
plt.plot(pred, alpha=1, color='red')
plt.show()


# # mean timeseries plot with latent sampling
dist = torch.distributions.MultivariateNormal(model.mu, covariance_matrix=model.cov)
dist_sample = dist.sample((1000,))

with torch.no_grad():
    pred_plot = model.decode_masking(dist_sample)

plt.plot(pred_plot.detach().to('cpu').numpy().T, alpha=0.01, color='steelblue')
plt.plot(pred, alpha=1, color='red')
plt.show()


# ----------------------------------------------------------------------------------------------------------------------------------
# latent_z plot
train_timeseries, train_mask, train_valid_len, train_tail_mean, train_y = train_dataset.tensors
sample_batch = train_timeseries[:]
sample_mask =  train_mask[:]

# tests_timeseries, tests_mask, tests_valid_len, tests_tail_mean, tests_y = tests_dataset.tensors
# sample_batch = tests_timeseries[:]
# sample_mask =  tests_mask[:]

sample_batch_unmask = sample_batch.masked_fill(sample_mask == 0, float('nan'))
sample_origin_scale = (sample_batch_unmask * X_std + X_mean).numpy()

with torch.no_grad():
    pred_batch = model.encode(sample_batch.to(device), sample_mask.to(device)).detach().to('cpu')
pred_batch_np = pred_batch.numpy()
# model.mu
# model.cov
maha_dists = torch.sqrt( torch.einsum("nd,nd->n", torch.einsum("nd, de->ne", pred_batch, torch.linalg.inv(model.cov.to('cpu'))), pred_batch).reshape(-1,1) )



# ----------------------------------------------------------------------------------------------------------------------------------
import matplotlib.cm as cm

plt.figure(figsize=(10, 6))
norm = plt.Normalize(vmin=maha_dists.min(), vmax=threshold)
cmap = cm.get_cmap('jet')
for i in range(len(sample_origin_scale)):
    color_val = cmap(norm(maha_dists.ravel()[i]))
    plt.plot(sample_origin_scale[i, :], color=color_val, linewidth=1.5, alpha=0.2)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([]) # 실제 데이터 없이 범위만 지정
current_ax = plt.gca() 
plt.colorbar(sm, ax=current_ax, label='Mahalanobis Distance') 
plt.show()


# ----------------------------------------------------------------------------------------------------------------------------------

# Latent Dimension Latent Plot
threshold = 3
# Visualize 2-dim
plt.scatter(pred_batch_np[:, 0], pred_batch_np[:, 1], c=maha_dists.numpy().ravel(), cmap='jet', vmax=3)
# plt.scatter(pred_batch_np[:, 0], pred_batch_np[:, 2], c=maha_dists.numpy().ravel(), cmap='jet', vmax=3)
# plt.scatter(pred_batch_np[:, 1], pred_batch_np[:, 2], c=maha_dists.numpy().ravel(), cmap='jet', vmax=3)
plt.xlim(-4.5, 4.5)
plt.ylim(-4.5, 4.5)
circle = plt.Circle((0, 0), threshold, color='red', fill=False) 
plt.gca().add_patch(circle)
plt.colorbar()
plt.show()



# ----------------------------------------------------------------------------------------------------------------------------------

from sklearn.decomposition import PCA
# PCA
pca = PCA(n_components=2)
x_emb_pca = pca.fit_transform(pred_batch_np)

plt.figure(figsize=(8, 6))
plt.title('PCA')
scatter = plt.scatter(x_emb_pca[:, 0], x_emb_pca[:, 1], c=maha_dists.numpy().ravel(), cmap='jet', vmax=3)
circle = plt.Circle((0, 0), threshold, color='red', fill=False) 
plt.gca().add_patch(circle)
plt.colorbar()
plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from ipywidgets import interact
# Not-interactive
min_val, max_val = pred_batch_np.min().min(), pred_batch_np.max().max()
# jitter = pd.Series(np.random.rand(len(X_train)) * 0.1 - 0.05, index=X_train.index, name='jitter')
jitter = 0

def update_view(elev=20, azim=45):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(pred_batch_np[:,0]+jitter, pred_batch_np[:,1], pred_batch_np[:,2],
            c=maha_dists.numpy().ravel(), cmap='jet', vmax=3,
            s=15, edgecolors='lightgray', alpha=0.5, linewidths=0.5)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_zlim(min_val, max_val)
    ax.view_init(elev=elev, azim=azim)
    plt.colorbar(sc, ax=ax)  # scatter 객체를 mappable로 전달
    plt.show()

interact(update_view, elev=(0,90), azim=(0,360))











########################################################################################################################
# Gaussian Mixture Model
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3, covariance_type="full", reg_covar=1e-8).fit(pred_batch_np)

pred_y = gmm.predict(pred_batch_np)


# PCA
plt.scatter(x_emb_pca[:, 0], x_emb_pca[:, 1], c=pred_y)
plt.xlim(-4.5, 4.5)
plt.ylim(-4.5, 4.5)
circle = plt.Circle((0, 0), threshold, color='red', fill=False) 
plt.gca().add_patch(circle)
plt.colorbar()
plt.show()


import matplotlib.cm as cm
colors = cm.get_cmap('tab10').colors

plt.figure(figsize=(10, 6))
for i in range(trainX_tensor.shape[0]):  # 20개의 라인
    plt.plot(trainX_tensor[i], color=colors[pred_y[i]], alpha=0.1, label=pred_y[i])

handles, labels = plt.gca().get_legend_handles_labels()     # 현재 범례 항목 가져오기
unique = dict(zip(labels, handles))     # 중복 제거 (순서 유지)
plt.legend(unique.values(), unique.keys(), loc='upper right', bbox_to_anchor=(1,1))
plt.show()


from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from ipywidgets import interact
# Not-interactive
min_val, max_val = pred_batch_np.min().min(), pred_batch_np.max().max()
# jitter = pd.Series(np.random.rand(len(X_train)) * 0.1 - 0.05, index=X_train.index, name='jitter')
jitter = 0

def update_view(elev=20, azim=45):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(pred_batch_np[:,0]+jitter, pred_batch_np[:,1], pred_batch_np[:,2],
            c=pred_y, s=15, edgecolors='lightgray', alpha=0.5, linewidths=0.5)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_zlim(min_val, max_val)
    ax.view_init(elev=elev, azim=azim)
    plt.colorbar(sc, ax=ax)  # scatter 객체를 mappable로 전달
    plt.show()

interact(update_view, elev=(0,90), azim=(0,360))



# ---------------------------------------------------------------------------------------------------
# Hierarchical
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

Z = linkage(pred_batch_np, method='single')  # ward는 유클리드 기반에서 가장 흔함
dend = dendrogram(Z, truncate_mode='lastp', p=30)  

hierarchy = AgglomerativeClustering(n_clusters=3, linkage="ward")   # 합칠 때, 군집 내 분산(SSE) 증가량을 최소화
pred_y = hierarchy.fit_predict(pred_batch_np)

plt.scatter(x_emb_pca[:, 0], x_emb_pca[:, 1], c=pred_y)
plt.xlim(-4.5, 4.5)
plt.ylim(-4.5, 4.5)
circle = plt.Circle((0, 0), threshold, color='red', fill=False) 
plt.gca().add_patch(circle)
plt.colorbar()
plt.show()


import matplotlib.cm as cm
colors = cm.get_cmap('tab10').colors

plt.figure(figsize=(10, 6))
for i in range(trainX_tensor.shape[0]):  # 20개의 라인
    plt.plot(trainX_tensor[i], color=colors[pred_y[i]], alpha=0.1, label=pred_y[i])

handles, labels = plt.gca().get_legend_handles_labels()     # 현재 범례 항목 가져오기
unique = dict(zip(labels, handles))     # 중복 제거 (순서 유지)
plt.legend(unique.values(), unique.keys(), loc='upper right', bbox_to_anchor=(1,1))
plt.show()


from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from ipywidgets import interact
# Not-interactive
min_val, max_val = pred_batch_np.min().min(), pred_batch_np.max().max()
# jitter = pd.Series(np.random.rand(len(X_train)) * 0.1 - 0.05, index=X_train.index, name='jitter')
jitter = 0

def update_view(elev=20, azim=45):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(pred_batch_np[:,0]+jitter, pred_batch_np[:,1], pred_batch_np[:,2],
            c=pred_y, s=15, edgecolors='lightgray', alpha=0.5, linewidths=0.5)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_zlim(min_val, max_val)
    ax.view_init(elev=elev, azim=azim)
    plt.colorbar(sc, ax=ax)  # scatter 객체를 mappable로 전달
    plt.show()

interact(update_view, elev=(0,90), azim=(0,360))




# ---------------------------------------------------------------------------------------------------
# Spectral Clustering
from sklearn.cluster import SpectralClustering

spectral = SpectralClustering(
    n_clusters=3,
    affinity="nearest_neighbors",  # 또는 "rbf"
    n_neighbors=10,
    assign_labels="kmeans",
    random_state=0
)

pred_y = spectral.fit_predict(pred_batch_np)

plt.scatter(x_emb_pca[:, 0], x_emb_pca[:, 1], c=pred_y)
plt.xlim(-4.5, 4.5)
plt.ylim(-4.5, 4.5)
circle = plt.Circle((0, 0), threshold, color='red', fill=False) 
plt.gca().add_patch(circle)
plt.colorbar()
plt.show()


import matplotlib.cm as cm
colors = cm.get_cmap('tab10').colors

plt.figure(figsize=(10, 6))
for i in range(trainX_tensor.shape[0]):  # 20개의 라인
    plt.plot(trainX_tensor[i], color=colors[pred_y[i]], alpha=0.1, label=pred_y[i])

handles, labels = plt.gca().get_legend_handles_labels()     # 현재 범례 항목 가져오기
unique = dict(zip(labels, handles))     # 중복 제거 (순서 유지)
plt.legend(unique.values(), unique.keys(), loc='upper right', bbox_to_anchor=(1,1))
plt.show()


from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from ipywidgets import interact
# Not-interactive
min_val, max_val = pred_batch_np.min().min(), pred_batch_np.max().max()
# jitter = pd.Series(np.random.rand(len(X_train)) * 0.1 - 0.05, index=X_train.index, name='jitter')
jitter = 0

def update_view(elev=20, azim=45):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(pred_batch_np[:,0]+jitter, pred_batch_np[:,1], pred_batch_np[:,2],
            c=pred_y, s=15, edgecolors='lightgray', alpha=0.5, linewidths=0.5)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_zlim(min_val, max_val)
    ax.view_init(elev=elev, azim=azim)
    plt.colorbar(sc, ax=ax)  # scatter 객체를 mappable로 전달
    plt.show()

interact(update_view, elev=(0,90), azim=(0,360))

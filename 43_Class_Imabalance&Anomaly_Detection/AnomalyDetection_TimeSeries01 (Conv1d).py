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
    from DS_Torch import TorchDataLoader, TorchModeling, AutoML
    from DS_MachineLearning import LabelEncoder2D, DataPreprocessing
    from DS_DeepLearning import EarlyStopping
    from DS_TorchModule import CategoricalEmbedding, EmbeddingLinear, ContinuousEmbeddingBlock
    from DS_TorchModule import PositionalEncoding, LearnablePositionalEncoding, FeatureWiseEmbeddingNorm
    from DS_TorchModule import ScaledDotProductAttention, MultiheadAttention, PreLN_TransformerEncoderLayer, AttentionPooling
    from DS_TorchModule import KwargSequential, ResidualConnection
    from DS_TorchModule import MaskedConv1d
    from DS_TimeSeries import pad_series_list_1d, pad_series_list_2d, smoothing
    
except:
    remote_library_url = 'https://raw.githubusercontent.com/kimds929/'
    try:
        import httpimport
        with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
            from DS_Torch import TorchDataLoader, TorchModeling, AutoML
            from DS_MachineLearning import LabelEncoder2D, DataPreprocessing
            from DS_DeepLearning import EarlyStopping
            from DS_TorchModule import CategoricalEmbedding, EmbeddingLinear, ContinuousEmbeddingBlock
            from DS_TorchModule import PositionalEncoding, LearnablePositionalEncoding, FeatureWiseEmbeddingNorm
            from DS_TorchModule import ScaledDotProductAttention, MultiheadAttention, PreLN_TransformerEncoderLayer, AttentionPooling
            from DS_TorchModule import KwargSequential, ResidualConnection
            from DS_TorchModule import MaskedConv1d
            from DS_TimeSeries import pad_series_list_1d, pad_series_list_2d, smoothing
    except:
        import requests
        response = requests.get(f"{remote_library_url}/DS_Library/main/DS_TimeSeries.py", verify=False)
        exec(response.text)
        
        response = requests.get(f"{remote_library_url}/DS_Library/main/DS_Torch.py", verify=False)
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
    default_normal_p = (0.8, 1.2)
    default_early_p = (0.4, 0.7)
    default_late_p = (1.3, 1.6)
    
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



n_normal = 736
n_early = 15
n_late = 8

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
class AnomalyTimeSeriesConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64):
        super().__init__()
        self.out_channels = out_channels
        
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
        
        # (information pooling)
        self.attn_pool_layer = AttentionPooling(hidden_channels)
        
        # (latent head)
        self.latent_head = nn.Linear(hidden_channels, out_channels)
        
        # mu, cov
        self.eps = 1e-3
        self.n = 0
        self.mu = torch.zeros(self.out_channels)
        self.cov = torch.eye(self.out_channels) * self.eps
    
    def masked_mean(self, x, mask, ):
        """
        x:    (B, T, E)  (float)
        mask: (B, T, E)  (bool)
        """
        sum_x = x.sum(dim=-1)                  # (B, E)
        len_x = mask.sum(dim=-1)               # (B, 1)
        return sum_x / (len_x + self.eps)           # (B, E)

    def reset_params(self):
        self.n = 0
        self.mu = torch.zeros(self.out_channels)
        self.cov = torch.eye(self.out_channels) * self.eps
    
    def update_params(self, n_batch, mu, cov):
        self.mu = (self.n * self.mu + n_batch * mu) / (self.n + n_batch)
        self.cov = (self.n * self.cov + n_batch * cov) / (self.n + n_batch)
        self.n += n_batch
    
    def forward(self, x, mask):
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
        latent = self.latent_head(pool_out)   # latent : (B, out_channel)
        return torch.tanh(latent)*6
    
    def forward_gaussian(self, x, mask, batch_seq=-1):
        n_batch = x.size(0)
        latent = self.forward(x, mask)
        K = latent.size(-1)
        
        # mu, covariance
        mu = latent.mean(dim=0)
        cov = ((latent-mu).T @ (latent-mu)) / n_batch
        cov += self.eps * torch.eye(K, device=cov.device)
        
        if self.training:
            if batch_seq == 0:
                self.reset_params()
            self.update_params(n_batch, mu.detach().to('cpu'), cov.detach().to('cpu'))
        return mu, cov
##########################################################################################


model = AnomalyTimeSeriesConv1d(in_channels=1, out_channels=2, hidden_channels=64)
f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
# model(batch[0], batch[1]).shape


class Batch_KL_Divergece():
    def __init__(self, train_loader):
        self.len_batch = len(train_loader)
        self.count = 0
        
    def batch_counter(self):
        self.count = self.count % self.len_batch
    
    def batch_kl_divergence_loss(self, model, batch, optimizer=None):
        self.batch_counter()
        batch_X, mask, valid_seq_len, tail_mean, batch_y = batch
        mu, cov = model.forward_gaussian(batch_X, mask, self.count)
        
        # trace term
        K = mu.size(0)
        sign, logdet = torch.linalg.slogdet(cov)
        kl_loss = 0.5 * (torch.trace(cov) + mu @ mu - K - logdet)
        self.count += 1
        return kl_loss

Batch_KL_Loss = Batch_KL_Divergece(train_loader)

tm = TorchModeling(model, device)
tm.compile(optim.Adam(model.parameters(), lr=3e-4),
           early_stop_loss = EarlyStopping(min_iter=100, patience=50))
tm.train_model(train_loader, valid_loader, epochs=300, loss_function=Batch_KL_Loss.batch_kl_divergence_loss)





##############################################################################################

train_timeseries, tain_mask, train_valid_len, train_tail_mean, train_y = train_dataset.tensors
sample_batch = train_timeseries[:]
sample_mask =  tain_mask[:]
sample_batch_unmask = sample_batch.masked_fill(sample_mask == 0, float('nan'))
sample_origin_scale = (sample_batch_unmask * X_std + X_mean).numpy()

pred_batch = model(sample_batch.to(device), sample_mask.to(device)).detach().to('cpu')
pred_batch_np = pred_batch.numpy()
# model.mu
# model.cov
maha_dists = torch.sqrt( torch.einsum("nd,nd->n", torch.einsum("nd, de->ne", pred_batch, torch.linalg.inv(model.cov)), pred_batch).reshape(-1,1) )




# Visualize 2-dim
threshold = 2.5
plt.scatter(pred_batch_np[:, 0], pred_batch_np[:, 1], c=maha_dists.numpy().ravel(), cmap='jet')
circle = plt.Circle((0, 0), threshold, color='red', fill=False) 
plt.gca().add_patch(circle)
plt.colorbar()
plt.show()

# ---------------------------------------------------------------------------------------
# from sklearn.manifold import TSNE
# tsne = TSNE(n_components=2, perplexity=30, random_state=1)
# X_embedded = tsne.fit_transform(pred_batch_np)

# plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=maha_dists.numpy().ravel(), cmap='jet')
# plt.colorbar()
# # for xy_point, maha_dist in zip(X_embedded, maha_dists):
# #     plt.text(xy_point[0], xy_point[1], round(maha_dist.item(),3))
# plt.show()
# ---------------------------------------------------------------------------------------
import matplotlib.cm as cm

plt.figure(figsize=(10, 6))
norm = plt.Normalize(vmin=maha_dists.min(), vmax=3)
cmap = cm.get_cmap('jet')
for i in range(len(sample_origin_scale)):
    color_val = cmap(norm(maha_dists.ravel()[i]))
    plt.plot(sample_origin_scale[i, :], color=color_val, linewidth=1.5, alpha=0.2)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([]) # 실제 데이터 없이 범위만 지정
current_ax = plt.gca() 
plt.colorbar(sm, ax=current_ax, label='Mahalanobis Distance') 
plt.show()



# X_mean, X_std
pred_normal_idx = torch.where(maha_dists.ravel() < threshold/5)[0]
normal_samples = sample_batch[pred_normal_idx].masked_fill(sample_mask[pred_normal_idx] == 0, float('nan'))
normal_samples = normal_samples * X_std + X_mean

pred_abnormal_idx = torch.where(maha_dists.ravel() > threshold)[0]
abnormal_samples = sample_batch[pred_abnormal_idx].masked_fill(sample_mask[pred_abnormal_idx] == 0, float('nan'))
abnormal_samples = abnormal_samples * X_std + X_mean

print(len(pred_normal_idx), len(pred_abnormal_idx))



# visualize subplot
plt.figure(figsize=(15,8))
plt.plot(train_X.T, color='steelblue', alpha=0.05, label='overall')
plt.plot(normal_samples.T, color='black', alpha=0.2, label='normal')
plt.plot(abnormal_samples.T, color='red', alpha=0.2, label='abnormal')
# 중복 제거
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right')
plt.show()




##############################################################################################






























#################################################################################################
#################################################################################################
#################################################################################################
# (Only Train with Normal Data) #################################################################

# normal, abnormal split
X_normal = X_Series[normal_idx]
X_abnormal = X_Series[~normal_idx]

X_normal.shape
X_abnormal.shape


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

train_valid_idx, tests_idx = train_test_split(np.arange(normal_idx.sum()), test_size=0.1, random_state=random_state)
train_idx, valid_idx = train_test_split(np.arange(len(train_valid_idx)), test_size=0.2, random_state=random_state)

train_idx = sorted(train_idx)
valid_idx = sorted(valid_idx)
tests_idx = sorted( np.concatenate([tests_idx, np.where(normal_idx == False)[0]]) )


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

train_X_tensor_set = timeseries_processing(train_X_norm, replace_value=0)
valid_X_tensor_set = timeseries_processing(valid_X_norm, replace_value=0)
tests_X_tensor_set = timeseries_processing(tests_X_norm, replace_value=0)

train_y_tensor = torch.LongTensor(train_y)
valid_y_tensor = torch.LongTensor(valid_y)
tests_y_tensor = torch.LongTensor(tests_y)

# visualize tail mean
# train_set
plt.plot(train_X_norm.T, alpha=0.05, color='steelblue')
plt.plot(tests_X_norm.T, alpha=0.05, color='orange')
# plt.scatter(trainX_valid_seq_len-1, trainX_tailmean, color='steelblue', edgecolor='blue', s=5)
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
class AnomalyTimeSeriesConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64):
        super().__init__()
        self.out_channels = out_channels
        
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
        
        # (information pooling)
        self.attn_pool_layer = AttentionPooling(hidden_channels)
        
        # (latent head)
        self.latent_head = nn.Linear(hidden_channels, out_channels)
        
        # mu, cov
        self.eps = 1e-3
        self.n = 0
        self.mu = torch.zeros(self.out_channels)
        self.cov = torch.eye(self.out_channels) * self.eps
    
    def masked_mean(self, x, mask, ):
        """
        x:    (B, T, E)  (float)
        mask: (B, T, E)  (bool)
        """
        sum_x = x.sum(dim=-1)                  # (B, E)
        len_x = mask.sum(dim=-1)               # (B, 1)
        return sum_x / (len_x + self.eps)           # (B, E)

    def reset_params(self):
        self.n = 0
        self.mu = torch.zeros(self.out_channels)
        self.cov = torch.eye(self.out_channels) * self.eps
    
    def update_params(self, n_batch, mu, cov):
        self.mu = (self.n * self.mu + n_batch * mu) / (self.n + n_batch)
        self.cov = (self.n * self.cov + n_batch * cov) / (self.n + n_batch)
        self.n += n_batch
    
    def forward(self, x, mask):
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
        latent = self.latent_head(pool_out)   # latent : (B, out_channel)
        return torch.tanh(latent)*6
    
    def forward_gaussian(self, x, mask, batch_seq=-1):
        n_batch = x.size(0)
        latent = self.forward(x, mask)
        K = latent.size(-1)
        
        # mu, covariance
        mu = latent.mean(dim=0)
        cov = ((latent-mu).T @ (latent-mu)) / n_batch
        cov += self.eps * torch.eye(K, device=cov.device)
        
        if self.training:
            if batch_seq == 0:
                self.reset_params()
            self.update_params(n_batch, mu.detach().to('cpu'), cov.detach().to('cpu'))
        return mu, cov
##########################################################################################


model = AnomalyTimeSeriesConv1d(in_channels=1, out_channels=2, hidden_channels=64)
f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
# model(batch[0], batch[1]).shape


class Batch_KL_Divergece():
    def __init__(self, train_loader):
        self.len_batch = len(train_loader)
        self.count = 0
        
    def batch_counter(self):
        self.count = self.count % self.len_batch
    
    def batch_kl_divergence_loss(self, model, batch, optimizer=None):
        self.batch_counter()
        batch_X, mask, valid_seq_len, tail_mean, batch_y = batch
        mu, cov = model.forward_gaussian(batch_X, mask, self.count)
        
        # trace term
        K = mu.size(-1)
        sign, logdet = torch.linalg.slogdet(cov)
        kl_loss = 0.5 * (torch.trace(cov) + mu @ mu - K - logdet)
        self.count += 1
        return kl_loss

Batch_KL_Loss = Batch_KL_Divergece(train_loader)


tm = TorchModeling(model, device)
tm.compile(optim.Adam(model.parameters(), lr=1e-3),
           early_stop_loss = EarlyStopping(min_iter=100, patience=50))
tm.train_model(train_loader, valid_loader, epochs=300, loss_function=Batch_KL_Loss.batch_kl_divergence_loss)

##############################################################################################


train_timeseries, train_mask, train_valid_len, train_tail_mean, train_y = train_dataset.tensors
sample_batch = train_timeseries[:]
sample_mask =  train_mask[:]

# tests_timeseries, tests_mask, tests_valid_len, tests_tail_mean, tests_y = tests_dataset.tensors
# sample_batch = tests_timeseries[:]
# sample_mask =  tests_mask[:]

sample_batch_unmask = sample_batch.masked_fill(sample_mask == 0, float('nan'))
sample_origin_scale = (sample_batch_unmask * X_std + X_mean).numpy()


pred_batch = model(sample_batch.to(device), sample_mask.to(device)).detach().to('cpu')
pred_batch_np = pred_batch.numpy()
# model.mu
# model.cov
maha_dists = torch.sqrt( torch.einsum("nd,nd->n", torch.einsum("nd, de->ne", pred_batch, torch.linalg.inv(model.cov)), pred_batch).reshape(-1,1) )


threshold = 3
# Visualize 2-dim
plt.scatter(pred_batch_np[:, 0], pred_batch_np[:, 1], c=maha_dists.numpy().ravel(), cmap='jet', vmax=3)
plt.xlim(-4.5, 4.5)
plt.ylim(-4.5, 4.5)
circle = plt.Circle((0, 0), threshold, color='red', fill=False) 
plt.gca().add_patch(circle)
plt.colorbar()
plt.show()

# ---------------------------------------------------------------------------------------
# from sklearn.manifold import TSNE
# tsne = TSNE(n_components=2, perplexity=30, random_state=1)
# X_embedded = tsne.fit_transform(pred_batch_np)

# plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=maha_dists.numpy().ravel(), cmap='jet')
# plt.colorbar()
# # for xy_point, maha_dist in zip(X_embedded, maha_dists):
# #     plt.text(xy_point[0], xy_point[1], round(maha_dist.item(),3))
# plt.show()
# ---------------------------------------------------------------------------------------
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



# X_mean, X_std
pred_normal_idx = torch.where(maha_dists.ravel() < threshold)[0]
normal_samples = sample_batch[pred_normal_idx].masked_fill(sample_mask[pred_normal_idx] == 0, float('nan'))
normal_samples = normal_samples * X_std + X_mean

pred_abnormal_idx = torch.where(maha_dists.ravel() > threshold)[0]
abnormal_samples = sample_batch[pred_abnormal_idx].masked_fill(sample_mask[pred_abnormal_idx] == 0, float('nan'))
abnormal_samples = abnormal_samples * X_std + X_mean

print(len(pred_normal_idx), len(pred_abnormal_idx))

# visualize subplot
plt.figure(figsize=(15,8))
plt.plot(normal_samples.T, color='steelblue', alpha=0.2, label='normal')
plt.plot(abnormal_samples.T, color='red', alpha=0.2, label='abnormal')
# 중복 제거
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right')
plt.show()




##############################################################################################
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, classification_report, roc_auc_score
# threshold = 2.5
with torch.no_grad():
    model.eval()
    pred_test_y = model(tests_X_tensor_set[0].to(device), tests_X_tensor_set[1].to(device)).detach().to('cpu')
    maha_dists = torch.einsum("nd,nd->n", torch.einsum("nd, de->ne", pred_test_y, torch.linalg.inv(model.cov)), pred_test_y).reshape(-1,1)
    
    pred_torch = (maha_dists > threshold).to(torch.int64)
    true_torch = tests_y_tensor.view(-1,1).to('cpu')
    conf = confusion_matrix(true_torch, pred_torch)
    
    precision, recall, _ = precision_recall_curve(true_torch, pred_torch)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(true_torch, pred_torch)
    
    
    cls_report = classification_report(true_torch, pred_torch)
    
    print('confusion_matrix')
    print(conf.T)
    print(f" - PR-AUC : {pr_auc:.3f}")
    print(f" - ROC-AUC : {roc_auc:.3f}")
    print(cls_report)


tests_x_np_norm = tests_X_tensor_set[0].numpy().copy()
tests_mask = tests_X_tensor_set[1]
tests_x_np_norm[~tests_mask] = np.nan
test_x_np = (tests_x_np_norm * X_std) + X_mean
# test_x_np = ss_X.inverse_transform(tests_x_np_norm)
last_values = test_x_np[np.arange(len(tests_X_tensor_set[0])), tests_X_tensor_set[2]-1]



plt.figure(figsize=(7, 4))
TP, FN, TN, FP = [0]*4
for lx, ly, mh_dist, tests_x, true_y in zip(tests_X_tensor_set[2].numpy(), last_values, maha_dists.numpy().reshape(-1), test_x_np, tests_y_tensor.numpy()):
    label = None
    if true_y == 1:
        if mh_dist < threshold:
            if FN == 0:
                label='FN'
            plt.plot(tests_x, color='red', alpha=0.3, label=label)
            plt.scatter(lx-1, ly, s=3, color='darkred')
            plt.text(lx-1, ly, f"{mh_dist:.2f}", color='red')
            FN += 1
        else:
            if TP == 0:
                label='TP'
            plt.plot(tests_x, color='orange', alpha=0.3, label=label)
            # plt.scatter(lx-1, ly, s=3, color='brown')
            # plt.text(lx-1, ly, f"{mh_dist:.2f}")
            TP += 1
    else:
        if mh_dist > threshold:
            if FP == 0:
                label='FP'
            plt.plot(tests_x, color='purple', alpha=0.3, label=label)
            plt.scatter(lx-1, ly, s=3, color='darkred')
            plt.text(lx-1, ly, f"{mh_dist:.2f}")
            FP +=1
        else:
            if TN == 0:
                label='TN'
            plt.plot(tests_x, color='steelblue',alpha=0.1, label=label)
            # plt.scatter(lx-1, ly, s=3, color='brown')
            # plt.text(lx-1, ly, f"{mh_dist:.2f}")
            TN +=1
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=4)
plt.xticks([])
plt.yticks([])
plt.show()

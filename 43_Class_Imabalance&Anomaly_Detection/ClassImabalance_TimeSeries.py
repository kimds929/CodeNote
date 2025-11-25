import os
import sys
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
y_labels_dict = {'normal': 0, 'early_surge':1, 'late_surge':1}
y_transformed = np.array([y_labels_dict[yi] for yi in y])
X_Series = pad_series_list(X, pad_value=np.nan)



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

BATCH = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH, shuffle=True)
tests_loader = DataLoader(tests_dataset, batch_size=BATCH, shuffle=True)

for batch in train_loader:
    break
len(batch)        # 5 : X, mask, valid_seq_len, tail_mean, y
batch[0].shape
batch[1].shape
batch[2].shape
# batch[3].shape
# batch[4].shape
batch[0]    # time-series data (Batch, Seq)
batch[1]    # masking data (Batch, Seq)
batch[2]    # valid sequence length (Batch)

#################################################################################################













#################################################################################################
#################################################################################################
#################################################################################################
# ----------------------------------------------------------------------------------------
class PositionalEncodingLayer(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, L, d_model)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_enc = self.pos_embedding(positions)
        return x + pos_enc

# ----------------------------------------------------------------------------------------

class PreLN_TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, batch_first=True):
        super().__init__()
        
        self.layer_norm1 = nn.LayerNorm(d_model)    # layer_norm1
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ff_layer = nn.Sequential(
            nn.LayerNorm(d_model),      # layer_norm2
            nn.Linear(d_model, dim_feedforward),     # FF_linear1
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),     # FF_linear2
            
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, is_causal=False, src_key_padding_mask=None):
        # Pre-LN before MHA
        src_norm = self.layer_norm1(src)
        attn_output, _ = self.self_attn(src_norm, src_norm, src_norm,
                                        attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask,
                                        is_causal=is_causal)    # is_causal : 미래정보차단여부 (src_mask를 안넣어도 자동으로 차단해줌)
        src = src + self.dropout1(attn_output)

        # Pre-LN before FFN
        src = src + self.ff_layer(src)
        return src


# ----------------------------------------------------------------------------------------
class AttentionPooling(nn.Module):
    def __init__(self, d_model, learnable_threshold=False, eps=1e-8):
        super().__init__()
        self.learnable_threshold = learnable_threshold
        self.eps = eps
        
        self.query = nn.Parameter(torch.randn(d_model)/d_model)  # 학습 가능한 Query (d_model, ) : 어떤 방식으로 요약할까?, 무엇이 중요한 시점인지를 학습하기 위함
        
        if self.learnable_threshold:
            self.threshold = nn.Parameter(torch.randn(1)/d_model)

    def forward(self, x, mask=None):  # x: (B, T, d_model)
        # Attention score 계산
        attn_scores = torch.matmul(x, self.query)  # (B, T)
        
        # Attention Mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T)
        
        # Learnable Threshold : sparcity
        if self.learnable_threshold:
            tau = torch.sigmoid(self.threshold)
            attn_weights = torch.clamp_min(attn_weights - tau, 0.0)
            denom = attn_weights.sum(dim=-2, keepdim=True) + self.eps
            attn_weights = attn_weights / denom
        
        # Weighted Sum
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=-2)  # (B,T,d_modl) * (B,T,1) = (B,T,d_modl) → sum → (B, d_model)
        return pooled, attn_weights
# ----------------------------------------------------------------------------------------


def compute_class_weights(train_y: torch.Tensor) -> torch.Tensor:
    """
    train_y: shape (N,), LongTensor, 각 값은 클래스 인덱스
    return: shape (num_classes,), FloatTensor
    """
    # 클래스 개수
    num_classes = train_y.max().item() + 1
    
    # 각 클래스별 샘플 수 계산
    class_counts = torch.bincount(train_y, minlength=num_classes).float()
    
    # 가중치 계산: inverse frequency
    weights = train_y.size(0) / (num_classes * class_counts)
    # weights = torch.nn.functional.softmax(weights, dim=-1)    # softmax : 불균형 보정 효과 약화
    return weights
# ----------------------------------------------------------------------------------------


########################################################################################


class TimeSeriesModel(nn.Module):
    def __init__(self, d_model, nhead=2, dim_ff=128, num_layers=1, max_len=4096):
        super().__init__()
        
        # (embedding)
        self.embedding_layer = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU()
        )
        # self.pe = PositionalEncodingLayer(d_model, max_len=max_len)
        self.pe = LearnablePositionalEncoding(d_model, max_len=max_len)
        
        # (encoder) : MHA +(ResidualConnection) + LayerNorm + FF + (ResidualConnection) + LayerNorm
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True)
        self.encoder_layer = PreLN_TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # (information pooling)
        self.attn_pool_layer = AttentionPooling(d_model)
        
        # (classification head)
        self.classification_head = nn.Linear(d_model, 1)
        
    def forward(self, x, mask=None):
        # x : (B, T)
        x_unsqueeze = x.unsqueeze(-1)   # x_unsqueeze : (B, T, 1)
        
        # (embedding)
        x_embed = self.embedding_layer(x_unsqueeze)     # x_embed : (B, T, d_model)
        x_embed_pe = self.pe(x_embed)       # x_embed_pe : (B, T, d_model)
        
        # (encoder)
        if mask is not None:    # True=PAD
            src_key_padding_mask = ~mask
        else:
            src_key_padding_mask = ~torch.ones_like(x).to(torch.bool) 
        encoder_out = self.encoder(x_embed_pe, src_key_padding_mask=src_key_padding_mask)   # encoder_out : (B, T, d_model)
        
        # (information pooling)
        # pool_out = torch.mean(encoder_out, dim=-2)    # Global Average Pooling (GAP)   # pool_out : (B, d_model)
        # pool_out, _ = torch.max(encoder_out, dim=-2)    # Global Max Pooling (GMP)   # pool_out : (B, d_model)
        pool_out, _ = self.attn_pool_layer(encoder_out, mask)      # Attention Pooling (AP)   # pool_out : (B, d_model)
        
        # (classification head)
        out = self.classification_head(pool_out)   # out : (B, 1)
        return out

#--------------------------------------------------------------------------------------


class ResidualConnection(nn.Module):
    def __init__(self, block, shortcut=None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut or (lambda x: x)
    
    def forward(self, x):
        return self.block(x) + self.shortcut(x)
    
class TimeSeriesConv(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # (convolution encoder)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
            ,nn.ReLU()
            ,ResidualConnection(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
                    ,nn.BatchNorm1d(hidden_dim)
                    ,nn.ReLU()
                )
            )
            ,ResidualConnection(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
                    ,nn.BatchNorm1d(hidden_dim)
                    ,nn.ReLU()
                )
            )
        )
        
        # (information pooling)
        self.attn_pool_layer = AttentionPooling(hidden_dim)
        
        # (classification head)
        self.classification_head = nn.Linear(hidden_dim, 1)
    
    def masked_mean(self, x, mask, eps=1e-9):
        """
        x:    (B, E, T)  (float)
        mask: (B, 1, T)  (bool)
        """
        sum_x = x.sum(dim=-1)                  # (B, E)
        len_x = mask.sum(dim=-1)               # (B, 1)
        return sum_x / (len_x + eps)           # (B, E)

    def forward(self, x, mask):
        # x.shape   # (B, T)
        unsqueeze_x = x.unsqueeze(-2)   # (B, 1, T)
        
        # (convolution encoder)
        encoder_out = self.encoder(unsqueeze_x)     # (B, E, T)
        
        # (masking)
        mask_unsqueeze = mask.unsqueeze(-2).to(encoder_out.dtype)   # (B, 1, T)
        encoder_out_mask = encoder_out * mask_unsqueeze     # (B, E, T)
        
        # (information pooling)
        pool_out, _ = self.attn_pool_layer(encoder_out_mask.transpose(-2,-1), mask)    # (B, E) : Attention Pooling
        # pool_out = self.masked_mean(encoder_out_mask, mask_unsqueeze)   # (B, E) : Mean Pooling
        
        # (classification head)
        out = self.classification_head(pool_out)   # out : (B, 1)
        return out



########################################################################################







model = TimeSeriesModel(d_model=8, nhead=4)
# model = TimeSeriesConv(input_dim=1, hidden_dim=64)
f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
# model( torch.rand(10,5) )


cs_weight = compute_class_weights(train_y_tensor)
pos_weight = cs_weight[1]/ cs_weight[0]

def cross_entropy_loss(model, batch, optimizer=None):
    batch_X, mask, valid_seq_len, tail_mean, batch_y = batch
    pred_y = model(batch_X, mask)
    batch_y_float = batch_y.view(-1,1).to(torch.float32)
    # loss = nn.functional.binary_cross_entropy_with_logits(pred_y, batch_y_float)    # pos_weight : class 0 대비 class 1의 가중치
    loss = nn.functional.binary_cross_entropy_with_logits(pred_y, batch_y_float, pos_weight)    # pos_weight : class 0 대비 class 1의 가중치
    return loss

tm = TorchModeling(model, device)

tm.compile(optim.Adam(model.parameters(), lr=1e-3),
           early_stop_loss = EarlyStopping(patience=50))
tm.train_model(train_loader, valid_loader, epochs=300, loss_function=cross_entropy_loss)


from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, classification_report, roc_auc_score
threshold = 0.5
with torch.no_grad():
    model.eval()
    pred_test_y = model(tests_X_tensor_set[0].to(device), tests_X_tensor_set[1].to(device))
    
    pred_test_y_prob = torch.sigmoid(pred_test_y).to('cpu')
    
    pred_torch = (pred_test_y_prob > threshold).to(torch.int64)
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




# # visualize
# def visualize_result(x, y, mask, valid_seq_len, scaler=None):
#     x_copy = x.copy()
#     x_copy[~mask] = np.nan
    
#     if scaler is not None:
#         test_x_np = ss_X.inverse_transform(tests_x_np_norm)
  

tests_x_np_norm = tests_X_tensor_set[0].numpy().copy()
tests_mask = tests_X_tensor_set[1]
tests_x_np_norm[~tests_mask] = np.nan
test_x_np = (tests_x_np_norm * X_std) + X_mean
# test_x_np = ss_X.inverse_transform(tests_x_np_norm)
last_values = test_x_np[np.arange(len(tests_X_tensor_set[0])), tests_X_tensor_set[2]-1]



plt.figure(figsize=(7, 4))
TP, FN, TN, FP = [0]*4
for lx, ly, p, tests_x, true_y in zip(tests_X_tensor_set[2].numpy(), last_values, pred_test_y_prob.numpy().reshape(-1), test_x_np, tests_y_tensor.numpy()):
    label = None
    if true_y == 1:
        if p < 0.5:
            if FN == 0:
                label='FN'
            plt.plot(tests_x, color='red',alpha=0.5, label=label)
            plt.scatter(lx-1, ly, s=3, color='darkred')
            plt.text(lx-1, ly, f"{p:.2f}", color='red')
            FN += 1
        else:
            if TP == 0:
                label='TP'
            plt.plot(tests_x, color='orange',alpha=0.5, label=label)
            plt.scatter(lx-1, ly, s=3, color='brown')
            plt.text(lx-1, ly, f"{p:.2f}")
            TP += 1
    else:
        if p > 0.5:
            if FP == 0:
                label='FP'
            plt.plot(tests_x, color='green',alpha=0.5, label=label)
            plt.scatter(lx-1, ly, s=3, color='darkred')
            plt.text(lx-1, ly, f"{p:.2f}")
            FP +=1
        else:
            if TN == 0:
                label='TN'
            plt.plot(tests_x, color='steelblue',alpha=0.1, label=label)
            TN +=1
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=4)
plt.xticks([])
plt.yticks([])
plt.show()


































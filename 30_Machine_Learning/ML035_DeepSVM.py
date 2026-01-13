import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import torch

try:
    from DS_MachineLearning import LabelEncoder2D, DataPreprocessing
    from DS_DeepLearning import EarlyStopping, TorchDataLoader, TorchModeling, AutoML
    from DS_TorchModule import CategoricalEmbedding, EmbeddingLinear, ContinuousEmbeddingBlock
    from DS_TorchModule import PositionalEncoding, LearnablePositionalEncoding, FeatureWiseEmbeddingNorm
    from DS_TorchModule import ScaledDotProductAttention, MultiheadAttention, PreLN_TransformerEncoderLayer, AttentionPooling
    from DS_TorchModule import KwargSequential, ResidualConnection
    from DS_TorchModule import MaskedConv1d
    from DS_TimeSeries import pad_series_list, series_smoothing
    
except:
    remote_library_url = 'https://raw.githubusercontent.com/kimds929/'
    try:
        import httpimport
        with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
            from DS_MachineLearning import LabelEncoder2D, DataPreprocessing
            from DS_DeepLearning import EarlyStopping, TorchDataLoader, TorchModeling, AutoML
            from DS_TorchModule import CategoricalEmbedding, EmbeddingLinear, ContinuousEmbeddingBlock
            from DS_TorchModule import PositionalEncoding, LearnablePositionalEncoding, FeatureWiseEmbeddingNorm
            from DS_TorchModule import ScaledDotProductAttention, MultiheadAttention, PreLN_TransformerEncoderLayer, AttentionPooling
            from DS_TorchModule import KwargSequential, ResidualConnection
            from DS_TorchModule import MaskedConv1d
            from DS_TimeSeries import pad_series_list, series_smoothing
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
        



###################################################################################################
rng = np.random.RandomState(1)

# --------------------------------------------------------------------------------------------------
scale = 1/50
# n_log_mean = 1.8
# n_log_std = 1
# n_sample = 20
# n_defect = (10 ** rng.normal(loc=n_log_mean, scale=n_log_std, size=n_sample) *scale ).astype(int)

n_mean = 30
n_std = 10
n_sample = 500       # n_samples ★
n_defect = rng.normal(loc=n_mean, scale=n_std, size=n_sample).astype(int)

plt.hist(n_defect, bins=30)
plt.show()
# --------------------------------------------------------------------------------------------------

w_mean = 1250
w_std = 150
l_log_mean = 3
l_log_std = 0.2

size_w = rng.normal(loc=w_mean, scale=w_std, size=(n_sample,1)).round(1) *scale
size_l = (10**rng.normal(loc=l_log_mean, scale=l_log_std, size=(n_sample,1))).round(1) *scale
sizes = np.concatenate([size_w, size_l], axis=1)

# --------------------------------------------------------------------------------------------------

df = pd.DataFrame()
mtl_list = []
for i in range(n_sample):
    n_defect_i = n_defect[i]
    if n_defect_i > 0:
        defect_locs = (np.random.rand(n_defect_i,2) * sizes[i]).round(1)
        size_broadcast = np.ones_like(defect_locs) * sizes[i]
        mtl_mat = np.concatenate([size_broadcast, defect_locs], axis=1)
        mtl_list.append(mtl_mat)
        df_sub = pd.DataFrame(mtl_mat, columns=['W', 'L', 'loc_w', 'loc_l'])
        df_sub.insert(0, 'mtl_idx', f"mtl_{i:02d}")
        df = pd.concat([df, df_sub], axis=0)

    
df.groupby(['mtl_idx']).size()

# --------------------------------------------------------------------------------

i = 2
mtl_sample = mtl_list[i]
plt.figure(figsize=[mtl_sample[0][0]/6 , mtl_sample[0][1]/3])
plt.scatter(mtl_sample[:,-2], mtl_sample[:,-1], c="black", s=5)
plt.show()


# --------------------------------------------------------------------------------
# from DS_TimeSeries import pad_series_list
pad_series = pad_series_list(mtl_list, pad_value=-1)        # (N, Seq, features)
print(pad_series.shape)


pad_series_torch = torch.FloatTensor(pad_series)




################################################################################
import torch.nn as nn
import torch.optim as optim

# class Distance(nn.Module):
#     def __init__(self, Sigma=None, boundary=None, kernel='gaussian'):
#         super().__init__()
        
#         self.Sigma = nn.Parameter(torch.rand(1)) if Sigma is None else Sigma
#         self.boundary = nn.Parameter(torch.rand(1)) if boundary is None else boundary
#         self.kernel = kernel
    
#     def _is_mat(self, p):
#         return p is not None and getattr(p, "ndim", 0) == 2
    
#     def mahalanobis_dist(self, X, Y=None, Sigma=None):
#         """pairwise distance: Euclidean (optional /sigma) or Mahalanobis (Sigma matrix)."""
#         Y = X if Y is None else Y
#         D = X.unsqueeze(-2) - Y.unsqueeze(-3)                # (B,n,m,d)

#         if self._is_mat(Sigma):
#             Sigma_inv = torch.linalg.inv(Sigma)
#             s = (D @ Sigma_inv * D).sum(dim=-1) 
#             # s =  torch.einsum("nmd,dd,nmd->nm", D, torch.linalg.inv(Sigma), D)
#         else:
#             Sigma = 1 if Sigma is None else Sigma
#             s = (D*D).sum(dim=-1) / (Sigma**2)
#             # s = torch.einsum("nmd,nmd->nm", D, D) / (Sigma**2)       # (D*D).sum(-1)/(Sigma**2)
#         s =  torch.clamp(s, min=0.0)
#         return torch.sqrt(s)

#     def _u(self, X, Y=None, boundary=1.0):
#         """normalized distance for compact kernels."""
#         if self._is_mat(boundary):          # ellipsoid: already normalized, boundary at u<=1
#             return self.mahalanobis_dist(X, Y, Sigma=boundary)
#         return self.mahalanobis_dist(X, Y) / boundary

#     def gaussian_kernel(self, X, Y=None, Sigma=1.0):
#         d = self.mahalanobis_dist(X, Y, Sigma=Sigma)
#         return torch.exp(-0.5 * d * d)

#     def uniform_kernel(self, X, Y=None, boundary=1.0):
#         u = self._u(X, Y, boundary)
#         return (u <= 1).to(dtype=u.dtype)

#     def linear_kernel(self, X, Y=None, boundary=1.0):
#         u = self._u(X, Y, boundary)
#         return torch.clamp(1 - u, 0, 1)

#     def epanechnikov_kernel(self, X, Y=None, boundary=1.0):
#         u = self._u(X, Y, boundary)
#         return torch.clamp(1 - u*u, 0, 1)

#     def quartic_kernel(self, X, Y=None, boundary=1.0):
#         u = self._u(X, Y, boundary)
#         t = torch.clamp(1 - u*u, 0, 1)
#         return t*t

#     def forward_kernel(self, X, Y=None, kernel=None, Sigma=None, boundary=None):
#         Sigma = nn.functional.softplus(self.Sigma) if Sigma is None else Sigma
#         boundary = nn.functional.softplus(self.boundary) if boundary is None else boundary
#         kernel = self.kernel if kernel is None else kernel
        
#         if kernel == 'gaussian':
#             return self.gaussian_kernel(X, Y, Sigma)
#         elif kernel == 'uniform':
#             return self.uniform_kernel(X, Y, boundary)
#         elif kernel == 'linear':
#             return self.linear_kernel(X, Y, boundary)
#         elif kernel == 'epanechnikov':
#             return self.epanechnikov_kernel(X, Y, boundary)
#         elif kernel == 'quartic':
#             return self.quartic_kernel(X, Y, boundary)

#     def forward(self, X, Y=None, kernel=None, Sigma=None, boundary=None):
#         dist_mat = self.forward_kernel(X, Y=Y, kernel=kernel, Sigma=Sigma, boundary=boundary)
#         dist = torch.nansum(dist_mat, dim=-1, keepdim=True)
#         return dist  

################################################################################


class Distance(nn.Module):
    def __init__(self, Sigma=None, boundary=None, kernel='gaussian', valid_masking=True):
        super().__init__()
        self.Sigma = nn.Parameter(torch.rand(1)) if Sigma is None else Sigma
        self.boundary = nn.Parameter(torch.rand(1)) if boundary is None else boundary
        self.valid_masking = valid_masking
        
        self.kernel = kernel
    
    def _is_mat(self, p):
        return p is not None and getattr(p, "ndim", 0) == 2
    
    def mahalanobis_dist(self, X, Y=None, Sigma=None):
        """pairwise distance: Euclidean (optional /sigma) or Mahalanobis (Sigma matrix)."""
        Y = X if Y is None else Y
        D = X.unsqueeze(-2) - Y.unsqueeze(-3)                # (B,n,m,d)

        if self._is_mat(Sigma):
            Sigma_inv = torch.linalg.inv(Sigma)
            s = (D @ Sigma_inv * D).sum(dim=-1) 
            # s =  torch.einsum("nmd,dd,nmd->nm", D, torch.linalg.inv(Sigma), D)
        else:
            Sigma = 1 if Sigma is None else Sigma
            s = (D*D).sum(dim=-1) / (Sigma)
            # s = torch.einsum("nmd,nmd->nm", D, D) / (Sigma**2)       # (D*D).sum(-1)/(Sigma**2)
        s =  s.clamp_min(0)
        return s

    def _u(self, X, Y=None, boundary=1.0):
        """normalized distance for compact kernels."""
        if self._is_mat(boundary):          # ellipsoid: already normalized, boundary at u<=1
            return self.mahalanobis_dist(X, Y, Sigma=boundary)
        return self.mahalanobis_dist(X, Y) / boundary

    def _valid_seq_len(self, X):
        """
        X: (..., Seq, Feature)
        각 배치별로 유효한 시퀀스 길이를 반환 (list of lengths)
        """
        # 마지막 Feature 차원 기준으로 유효 여부 판단
        valid_mask = ((X > 0) & (X < torch.inf)).all(dim=-1)  # (..., Seq)
        # 각 배치별 유효한 시퀀스 길이 계산
        valid_len = valid_mask.sum(dim=-1)  # (...,)
        return valid_len
    
    def gaussian_kernel(self, X, Y=None, Sigma=1.0):
        d = self.mahalanobis_dist(X, Y, Sigma=Sigma)
        return torch.exp(-0.5 * d * d)

    def uniform_kernel(self, X, Y=None, boundary=1.0):
        u = self._u(X, Y, boundary)
        return (u <= 1).to(dtype=u.dtype)

    def linear_kernel(self, X, Y=None, boundary=1.0):
        u = self._u(X, Y, boundary)
        return torch.clamp(1 - u, 0, 1)

    def epanechnikov_kernel(self, X, Y=None, boundary=1.0):
        u = self._u(X, Y, boundary)
        return torch.clamp(1 - u*u, 0, 1)

    def quartic_kernel(self, X, Y=None, boundary=1.0):
        u = self._u(X, Y, boundary)
        t = torch.clamp(1 - u*u, 0, 1)
        return t*t

    def forward_kernel(self, X, Y=None, kernel=None, Sigma=None, boundary=None, valid_masking=None):
        Y = X if Y is None else Y
        Sigma = self.Sigma if Sigma is None else Sigma
        boundary = self.boundary if boundary is None else boundary
        valid_masking = self.valid_masking if valid_masking is None else valid_masking
        kernel = self.kernel if kernel is None else kernel
        
        kernel_result = None
        if kernel == 'gaussian':
            kernel_result = self.gaussian_kernel(X, Y, Sigma)
        elif kernel == 'uniform':
            kernel_result = self.uniform_kernel(X, Y, boundary)
        elif kernel == 'linear':
            kernel_result = self.linear_kernel(X, Y, boundary)
        elif kernel == 'epanechnikov':
            kernel_result = self.epanechnikov_kernel(X, Y, boundary)
        elif kernel == 'quartic':
            kernel_result = self.quartic_kernel(X, Y, boundary)
        
        if valid_masking is True:
            valid_len_X = self._valid_seq_len(X)    # shape: batch_dims...
            valid_len_Y = self._valid_seq_len(Y)    # shape: batch_dims...
            
            seq_x = kernel_result.shape[-2]
            seq_y = kernel_result.shape[-1]

            # 시퀀스 인덱스 생성
            idx_x = torch.arange(seq_x, device=kernel_result.device)  # (Seq_X,)
            idx_y = torch.arange(seq_y, device=kernel_result.device)  # (Seq_Y,)

            # 브로드캐스팅을 위해 차원 확장
            # valid_len_X: (..., 1, 1) → 비교 시 (..., Seq_X, 1)
            # valid_len_Y: (..., 1, 1) → 비교 시 (..., 1, Seq_Y)
            mask_x = idx_x.unsqueeze(0) < valid_len_X.unsqueeze(-1)  # (..., Seq_X)
            mask_y = idx_y.unsqueeze(0) < valid_len_Y.unsqueeze(-1)  # (..., Seq_Y)

            # 브로드캐스팅으로 최종 mask 생성
            # mask_x[..., :, None] : (..., Seq_X, 1)
            # mask_y[..., None, :] : (..., 1, Seq_Y)
            mask = mask_x[..., :, None] & mask_y[..., None, :]

            kernel_result = kernel_result.masked_fill(~mask, 0)
        return kernel_result

    def forward(self, X, Y=None, kernel=None, Sigma=None, boundary=None, valid_masking=None):
        dist_mat = self.forward_kernel(X, Y=Y, kernel=kernel, Sigma=Sigma, boundary=boundary, valid_masking=valid_masking)
        dist = torch.sum(dist_mat, dim=-1, keepdim=True)
        return dist



################################################################################

# X = pad_series_torch[:,:,-2:]
# X.shape # (N, Seq, f)

# distance
dist = Distance(Sigma=torch.tensor(2.25), kernel='gaussian')
# dist = Distance( boundary=torch.tensor(4.0), kernel='linear')

# a = torch.rand(8,2)
# a[-3:,:] = -1
# b = torch.rand(7,2)
# b[-2:,:] = -1
# dist.forward_kernel(a,b)
# dist.forward_kernel(a)
# dist.forward(a,b)

dist_mat = dist.forward_kernel(pad_series_torch[:,:,-2:])
pd.DataFrame(dist_mat[3].numpy().round(1)).to_clipboard()
intensives = dist.forward(pad_series_torch[:,:,-2:]).squeeze(-1)
max_intensives = intensives.max(dim=-1)[0].numpy()



# --------------------------------------------------------------------------------
# intensive
print(n_defect)
print(max_intensives)
# np.stack([n_defect, max_intensives.numpy()]).T

plt.hist(max_intensives, bins=30, edgecolor='gray', alpha=0.5)
plt.show()

# --------------------------------------------------------------------------------
n_defect[:10]
# Visualize
xx, yy = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
grid = np.c_[xx.ravel(), yy.ravel()]  # (40000, 2)

i = 1
mtl_sample = mtl_list[i]
x_scale, y_scale = map(lambda x: x.item(), mtl_sample.mean(0)[:2])
gird_scale = grid * np.array([x_scale, y_scale])
xx_scale = xx * x_scale
yy_scale = yy * y_scale
Z = dist.forward(torch.FloatTensor(gird_scale), torch.FloatTensor(mtl_sample[:, -2:]) ).view(xx_scale.shape).detach().numpy()
intensive = intensives[i]

plt.figure(figsize=[mtl_sample[0][0].item()/6 , mtl_sample[0][1].item()/3])
# cont = plt.contourf(xx_scale, yy_scale, Z, cmap="Reds", alpha=0.3, level=np.round(np.sqrt(np.linspace(1, 5**2, 11)), 1))
cont = plt.contourf(xx_scale, yy_scale, Z, cmap="Reds", alpha=0.3, vmin=0, vmax=5)

for (xp, yp), val in zip(mtl_sample[:,-2:], intensive):
    color='black'
    alpha=0.5
    if val >= 4:
        color='red'
        alpha= 1
    plt.scatter(xp.item(), yp.item(), color=color, s=3)
    plt.text(xp.item(), yp.item(), round(val.item(),1), color=color, alpha=alpha, fontsize=10)

plt.colorbar(cont)
plt.show()



##########################################################################################
# --------------------------------------------------------------------------------

# calculate intensive
dist = Distance(Sigma=torch.tensor(1.5), kernel='gaussian')
max_intensives = intensives.max(dim=-1)[0].numpy()

# label
threshold = 4       # true threshold
noise = rng.normal(loc=0, scale=0.5, size=max_intensives.shape)
label_true = ((max_intensives) > threshold).astype(int)
label_obs = ((max_intensives + noise) > threshold).astype(int)

label_torch = torch.LongTensor(label_obs)
print(torch.unique(label_torch, return_counts=True))


# --------------------------------------------------------------------------------
from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(pad_series_torch[:,:,-2:], label_torch)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)



##########################################################################################

import torch.nn as nn
import torch.optim as optim

class KernelModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # self.params_Sigma =  nn.Parameter( torch.rand(()) )
        self.params_Sigma_vec = nn.Parameter( torch.rand(2) )
        self.params_boundary =  nn.Parameter( torch.rand(()) )
        self.param_threhold = nn.Parameter(torch.rand(()))
        
        self.Sigma = None
        self.Sigma_vec = None
        self.boundary = None
        self.threshold = None
        self.dist_layer = Distance(kernel='gaussian')
        # self.dist_layer = Distance(kernel='linear')
    
    def forward(self, X):
        # self.Sigma = nn.functional.softplus(self.params_Sigma) 
        self.Sigma_vec = nn.functional.softplus(self.params_Sigma_vec) 
        self.Sigma = self.Sigma_vec.diag()
        self.boundary = nn.functional.softplus(self.params_boundary) 
        self.threshold = nn.functional.softplus(self.param_threhold)
        
        
        
        # intensive = self.dist_layer(X, Sigma=torch.tensor(1.5))
        intensive = self.dist_layer(X, Sigma=self.Sigma)
        # intensive = self.dist_layer(X, boundary=self.boundary)
        max_intensive, _ = intensive.squeeze(-1).max(dim=-1, keepdim=True)
        
        output = max_intensive - self.threshold
        return output 

# torch.autograd.set_detect_anomaly(False)
model = KernelModel()
# model(pad_series_torch[:,:,-2:])
f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"


# def squared_hingeloss(y_pred, y_true, margin=1):
#     return torch.clamp(margin - y_true * y_pred, min=0) ** 2

def loss_function(model, batch, optimizer=None):
    batch_X, batch_y = batch
    
    batch_y_pn = batch_y*2 - 1
    y_pred = model.forward(batch_X)
    # hinge_loss = (torch.clamp(1 - batch_y_pn * y_pred, min=0) ** 2).mean()
    
    loss_pos = (torch.clamp(1 - batch_y_pn[batch_y_pn > 0] * y_pred[batch_y_pn > 0], min=0) ** 2).mean()
    loss_neg = (torch.clamp(1 - batch_y_pn[batch_y_pn < 0] * y_pred[batch_y_pn < 0], min=0) ** 2).mean()

    hinge_loss = 0.5 * (loss_pos + loss_neg)
    return hinge_loss
    

tm = TorchModeling(model)
tm.compile(optimizer=optim.AdamW(model.parameters(), lr=3e-4)
           ,loss_function=loss_function)
tm.train_model(train_loader, epochs=300)

print(model.Sigma.detach(), model.threshold.detach())
# print(model.Sigma.detach(), model.threshold.detach())
# print(model.Precision.detach(), model.threshold.detach())
# print(model.boundary.detach(), model.threshold.detach())


# ----------------------------------------------------------------------
pred = model(pad_series_torch[:,:,-2:]).detach().to('cpu').numpy()
pred_label = ((np.sign(pred)+1)/2).ravel()

from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(pred_label, label_torch)   # pred - obs
confusion_matrix(label_true, label_torch)   # true - obs
confusion_matrix(pred_label, label_true)    # pred - true

print(classification_report(pred_label, label_torch))   # pred - obs
print(classification_report(label_torch, label_true))   # true - obs
print(classification_report(pred_label, label_true))    # pred - true


# # 예: batch_y_pn ∈ {+1, -1}
# pos_weight = N_total / (2 * N_pos)
# neg_weight = N_total / (2 * N_neg)

# weights = torch.where(batch_y_pn > 0, pos_weight, neg_weight)

# hinge_loss = (weights * (torch.clamp(1 - batch_y_pn * y_pred, min=0) ** 2)).mean()
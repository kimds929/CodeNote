import os
import sys
if 'd:' in os.getcwd().lower():
    os.chdir("D:/DataScience/")
# else:
#     os.chdir("/home/pd299370/")
sys.path.append(f"{os.getcwd()}/DataScience/00_DataAnalysis_Basic")
sys.path.append(f"{os.getcwd()}/DataScience/DS_Library")
sys.path.append(r'D:\DataScience\00_DataAnalysis_Basic')


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import torch

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f"torch_device : {device}")

try:
    from DS_MachineLearning import DS_LabelEncoder, DataPreprocessing
    from DS_DeepLearning import EarlyStopping, TorchDataLoader, TorchModeling
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
            from DS_MachineLearning import DS_LabelEncoder, DataPreprocessing
            from DS_DeepLearning import EarlyStopping, TorchDataLoader, TorchModeling
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

# visualize
plt.title('Distribution of N-Defect')
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
# visualize
i = 2
mtl_sample = mtl_list[i]
plt.figure(figsize=[mtl_sample[0][0]/6 , mtl_sample[0][1]/3])
plt.scatter(mtl_sample[:,-2], mtl_sample[:,-1], c="black", s=3)
plt.show()


# --------------------------------------------------------------------------------
# from DS_TimeSeries import pad_series_list
pad_series = pad_series_list(mtl_list, pad_value=-1)        # (N, Seq, features)
print(pad_series.shape)


pad_series_torch = torch.FloatTensor(pad_series)




################################################################################
import torch.nn as nn
import torch.optim as optim

class Distance(nn.Module):
    def __init__(self, Sigma=None, boundary=None, kernel='gaussian', valid_masking=True):
        super().__init__()
        self.Sigma = nn.Parameter(torch.rand(1)) if Sigma is None else Sigma
        self.boundary = nn.Parameter(torch.rand(1)) if boundary is None else boundary
        self.valid_masking = valid_masking
        self.epsilon = 1e-6
        
        self.kernel = kernel
    
    def _is_mat(self, p):
        return p is not None and getattr(p, "ndim", 0) == 2
    
    def square_mahalanobis_dist(self, X, Y=None, Sigma=None):
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
            return torch.sqrt( self.square_mahalanobis_dist(X, Y, Sigma=boundary**2) + self.epsilon )     # gradient explosion 방지
        return torch.sqrt( self.square_mahalanobis_dist(X, Y) + self.epsilon ) / boundary     # gradient explosion 방지

    def _valid_seq_len(self, X):
        """
        X: (..., Seq, Feature)
        각 배치별로 유효한 시퀀스 길이를 반환 (list of lengths)
        """
        # 마지막 Feature 차원 기준으로 유효 여부 판단
        valid_mask = ((X >= 0) & (X < torch.inf)).all(dim=-1)  # (..., Seq)
        # 각 배치별 유효한 시퀀스 길이 계산
        valid_len = valid_mask.sum(dim=-1)  # (...,)
        return valid_len
    
    def gaussian_kernel(self, X, Y=None, Sigma=1.0):
        d_square = self.square_mahalanobis_dist(X, Y, Sigma=Sigma)
        return torch.exp(-0.5 * d_square)

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
# dist = Distance(Sigma=torch.tensor(2.25), kernel='gaussian')
dist = Distance( boundary=torch.tensor(4), kernel='linear')


dist_mat = dist.forward_kernel(pad_series_torch[:,:,-2:])
# pd.DataFrame(dist_mat[3].numpy().round(1)).to_clipboard()
intensives = dist.forward(pad_series_torch[:,:,-2:]).squeeze(-1)
max_intensives = intensives.max(dim=-1)[0].numpy()




# --------------------------------------------------------------------------------
# intensive
print(n_defect)
print(max_intensives)
# np.stack([n_defect, max_intensives.numpy()]).T

plt.title('Distribution of Max-Intensive')
plt.hist(max_intensives, bins=30, edgecolor='gray', alpha=0.5)
plt.show()

# --------------------------------------------------------------------------------
# Visualize
xx, yy = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
grid = np.c_[xx.ravel(), yy.ravel()]  # (40000, 2)

i = 6
mtl_sample = mtl_list[i]
x_scale, y_scale = map(lambda x: x.item(), mtl_sample.mean(0)[:2])
gird_scale = grid * np.array([x_scale, y_scale])
xx_scale = xx * x_scale
yy_scale = yy * y_scale
Z = dist.forward(torch.FloatTensor(gird_scale), torch.FloatTensor(mtl_sample[:, -2:]) ).view(xx_scale.shape).detach().numpy()
intensive = intensives[i]

plt.figure(figsize=[mtl_sample[0][0].item()/6 , mtl_sample[0][1].item()/3])
# cont = plt.contourf(xx_scale, yy_scale, Z, cmap="RdYlGn_r", alpha=0.3, level=np.round(np.sqrt(np.linspace(1, 5**2, 11)), 1))
cont = plt.contourf(xx_scale, yy_scale, Z, cmap="RdYlGn_r", alpha=0.3, vmin=0, vmax=5)

for (xp, yp), val in zip(mtl_sample[:,-2:], intensive):
    color='black'
    alpha=0.5
    if val >= 4:
        color='red'
        alpha= 1
    plt.scatter(xp.item(), yp.item(), color=color, s=2)
    plt.text(xp.item(), yp.item(), round(val.item(),1), color=color, alpha=alpha, fontsize=8)

plt.colorbar(cont)
plt.show()








##########################################################################################
# --------------------------------------------------------------------------------

# calculate intensive
dist = Distance(Sigma=torch.tensor(1.5), kernel='gaussian')
intensives = dist.forward(pad_series_torch[:,:,-2:]).squeeze(-1)
max_intensives = intensives.max(dim=-1)[0].numpy()

# label
threshold = 4       # true threshold
noise = rng.normal(loc=0, scale=0.5, size=max_intensives.shape)
label_true = ((max_intensives) > threshold).astype(int)
label_obs = ((max_intensives + noise) > threshold).astype(int)

label_torch = torch.LongTensor(label_obs)
num_classes = torch.unique(label_torch, return_counts=True)[1]
print(num_classes)



# --------------------------------------------------------------------------------
from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(pad_series_torch[:,:,-2:], label_torch)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# x = pad_series_torch[[5, 11],:10,-2:]







##########################################################################################
# class imbalance weights

# x = np.linspace(0, 1, num=100)
# xp = np.stack([x, 1-x])

# np.linspace(0, 1, num=11)
# T = [0.1, 0.5, 1, 1.5, 10]


# plt.plot(xp[0], xp[0], label='N')
# for t in T:
#     xp_exp = (np.exp(t * xp) / np.exp(t * xp).sum(0))
#     plt.plot(xp[0], xp_exp[0], label=f"T:{t}")
# plt.legend(loc='upper right', bbox_to_anchor=(1,1))
# plt.show()



##########################################################################################
# 【 Deep SVM Kernel Model 】#############################################################
##########################################################################################
import torch.nn as nn
import torch.optim as optim

class KernelModel(nn.Module):
    def __init__(self, kernel='gaussian'):
        super().__init__()
        self.kernel = kernel
        
        # self.params_Sigma =  nn.Parameter( torch.rand(()) )
        self.params_Sigma_vec = nn.Parameter( torch.rand(2) )
        self.params_boundary =  nn.Parameter( torch.rand(()) )
        self.param_threhold = nn.Parameter(torch.rand(()))
        
        self.Sigma = None
        self.Sigma_vec = None
        self.boundary = None
        self.threshold = None
        self.dist_layer = Distance(kernel=kernel)
    
    def make_positive_params(self):
        # self.Sigma = nn.functional.softplus(self.params_Sigma) 
        self.Sigma_vec = nn.functional.softplus(self.params_Sigma_vec) 
        self.Sigma = self.Sigma_vec.diag()
        self.boundary = nn.functional.softplus(self.params_boundary) 
        self.threshold = nn.functional.softplus(self.param_threhold)
    
    def forward_intensive(self, X, Y=None):
        self.make_positive_params()
        
        # intensive = self.dist_layer(X, Y, Sigma=torch.tensor(1.5))
        if self.kernel == 'gaussian':
            intensive = self.dist_layer(X, Y, Sigma=self.Sigma)
        else:
            intensive_window = self.dist_layer(X, Y, boundary=self.boundary)
        return intensive
    
    def forward(self, X):
        intensive = self.forward_intensive(X)
        # intensive = self.dist_layer(X, boundary=self.boundary)
        max_intensive, _ = intensive.squeeze(-1).max(dim=-1, keepdim=True)
        
        output = max_intensive - self.threshold
        return output 

# torch.autograd.set_detect_anomaly(False)
model = KernelModel(kernel='gaussian').to(device)
# model(pad_series_torch[:,:,-2:].to(device))
print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# load_state = torch.load("D:/DataScience/Model/ConvDeepSVM_params.pth")
# model.load_state_dict(load_state)
# model.make_positive_params()


# def squared_hingeloss(y_pred, y_true, margin=1):
#     return torch.clamp(margin - (y_true*2-1) * y_pred, min=0) ** 2

class BatchSquareHingeLoss():
    def __init__(self, num_classes=None, weights_method='equal'):
        """
         . weights_method : 'equal', 'ratio', 'exp'
        """
        self.weights = None
        self.weihts_method = weights_method
        if num_classes is not None:
            weights = num_classes / num_classes.sum()
            if self.weihts_method == 'exp':
                self.weights = torch.exp(weights) / torch.exp(weights).sum()
            self.weights = weights
    
    def loss_function(self, model, batch, optiimizer=None):
        batch_X, batch_y = batch
        batch_y_pn = batch_y*2 - 1
        y_pred = model.forward(batch_X)
        
        unique_class, num_classes_batch = batch_y.unique(return_counts=True)
        if len(unique_class) > 1:
            if self.weights is not None:
                weights = self.weights
            elif self.weihts_method == 'ratio':
                weights = num_classes_batch / num_classes_batch.sum()
            elif self.weihts_method == 'exp':
                weights = num_classes_batch / num_classes_batch.sum()
                weights = torch.exp(weights) / torch.exp(weights).sum()
            else:   # 'equal;
                weights = torch.ones(2)/2
        else:
            weights = torch.zeros(2)
            weights[unique_class] = 1
        
        hinge_loss = torch.tensor(0.0)
        if len(batch_y_pn[batch_y_pn < 0]) > 0:   # negative_loss
            hinge_loss += weights[0] * (torch.clamp(1 - batch_y_pn[batch_y_pn < 0] * y_pred[batch_y_pn < 0], min=0) ** 2).mean()
            
        if len(batch_y_pn[batch_y_pn > 0]) > 0:     # positive_loss
            hinge_loss += weights[1] * (torch.clamp(1 - batch_y_pn[batch_y_pn > 0] * y_pred[batch_y_pn > 0], min=0) ** 2).mean()
        
        return hinge_loss


squared_hinge_loss = BatchSquareHingeLoss(num_classes, weights_method='exp')  # num_classes   
# squared_hinge_loss = BatchSquareHingeLoss(weights_method='exp')  # num_classes

tm = TorchModeling(model)
tm.compile(optimizer=optim.AdamW(model.parameters(), lr=1e-3)
           ,loss_function=squared_hinge_loss.loss_function)
tm.train_model(train_loader, epochs=500)

print(model.Sigma.detach(), model.threshold.detach())
# print(model.Sigma.detach(), model.threshold.detach())
# print(model.Precision.detach(), model.threshold.detach())
# print(model.boundary.detach(), model.threshold.detach())


# torch.save(model.state_dict(), "D:/DataScience/Model/DeepSVM_params_linear.pth")


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



# ----------------------------------------------------------------------
# Visualize
xx, yy = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
grid = np.c_[xx.ravel(), yy.ravel()]  # (40000, 2)

i = 6
mtl_sample = mtl_list[i]
x_scale, y_scale = map(lambda x: x.item(), mtl_sample.mean(0)[:2])
gird_scale = grid * np.array([x_scale, y_scale])
xx_scale = xx * x_scale
yy_scale = yy * y_scale

# visual dict
dist = Distance(Sigma=torch.tensor(1.5), kernel='gaussian')

Z_true = dist.forward(torch.FloatTensor(gird_scale), torch.FloatTensor(mtl_sample[:, -2:]) ).view(xx_scale.shape).detach().numpy()
intensive_true = dist.forward(torch.FloatTensor(mtl_sample[:, -2:])).squeeze()

Z_pred = model.forward_intensive(torch.FloatTensor(gird_scale), torch.FloatTensor(mtl_sample[:, -2:]) ).view(xx_scale.shape).detach().numpy()
intensive_pred = model.forward_intensive(torch.FloatTensor(mtl_sample[:, -2:])).squeeze()

params = {'true':(dist.Sigma, 4), 'pred': (model.Sigma.detach().diagonal().mean().item(), round(model.threshold.detach().item(),2))}
Z = {'true':Z_true, 'pred': Z_pred}
intensive = {'true': intensive_true, 'pred': intensive_pred}

# visualize
plt.figure(figsize=[mtl_sample[0][0].item()/2.5 , mtl_sample[0][1].item()/3])
for ei, ((Z_name, Z_value), (i_name, i_value), (param_name, param_values)) in enumerate(zip(Z.items(), intensive.items(), params.items())):
    
    plt.subplot(1,2,ei+1)
    plt.title(f"({Z_name}) intensive plot")
    cont = plt.contourf(xx_scale, yy_scale, Z_value, cmap="RdYlGn_r", alpha=0.3, vmin=0.5, vmax=5)

    for (xp, yp), val in zip(mtl_sample[:,-2:], i_value):
        color='black'
        alpha=0.5
        if val >= float(param_values[1]):
            color='red'
            alpha= 1
        plt.scatter(xp.item(), yp.item(), color=color, s=2, alpha=alpha)
        plt.text(xp.item(), yp.item(), round(val.item(),1), color=color, alpha=alpha, fontsize=8)
    plt.xlabel(f"Sigma {float(param_values[0]):.1f}, threshold: {param_values[1]:.2f}")
    plt.colorbar(cont)
plt.show()





##########################################################################################
# 【 Convolutional Deep SVM Kernel Model 】###############################################
##########################################################################################
import torch.nn as nn
import torch.optim as optim

class ConvKernelModel(nn.Module):
    def __init__(self, window_size=1000, stride=500, kernel='gaussian'):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.kernel = kernel
        
        # self.params_Sigma =  nn.Parameter( torch.rand(()) )
        self.params_Sigma_vec = nn.Parameter( torch.rand(2) )
        self.params_boundary =  nn.Parameter( torch.rand(()) )
        self.param_threhold = nn.Parameter(torch.rand(()))
        
        self.Sigma = None
        self.Sigma_vec = None
        self.boundary = None
        self.threshold = None
        self.dist_layer = Distance(kernel=kernel)
    
    def sort_sequence(self, X, padding_value=-1.0, pad_all_features=True, descending=False):
            """
             . X: (..., Seq, Feature) > 2D
             . padding_value: 패딩 값 (예: -1)
             . pad_all_features: True  -> feature 중 하나라도 padding_value면 패딩으로 간주 (네 기존 로직)
                                False -> feature 전부가 padding_value일 때만 패딩으로 간주 (보통 패딩 벡터용)
            """
            seq_dim = -2      # (..., Seq, Feature)에서 Seq 축
            feature_dim = -1     # Feature 축

            # padding mask: (..., Seq)
            if pad_all_features:
                pad_mask = (X == padding_value).any(dim=feature_dim)
            else:
                pad_mask = (X == padding_value).all(dim=feature_dim)

            # 거리: (..., Seq)
            dist = torch.norm(X, dim=feature_dim)

            # 패딩을 뒤로 보내기 위한 score 조정
            #    - descending=False(가까운 순): 패딩은 +inf
            #    - descending=True (먼 순):   패딩은 -inf
            fill_value = float('-inf') if descending else float('inf')
            dist_masked = dist.masked_fill(pad_mask, fill_value)

            # 정렬 인덱스: (..., Seq)
            sort_idx = torch.argsort(dist_masked, dim=feature_dim, descending=descending)

            # gather를 위한 인덱스 확장: (..., Seq, Feature)
            sort_idx_expanded = sort_idx.unsqueeze(feature_dim).expand(*X.shape[:-1], X.size(feature_dim))

            # Seq 축으로 gather
            X_sorted = torch.gather(X, dim=seq_dim, index=sort_idx_expanded)

            # # 역순 인덱스 생성 (원래 순서 복원용)
            inverse_idx = torch.empty_like(sort_idx)
            inverse_idx.scatter_(feature_dim, sort_idx, torch.arange(sort_idx.size(feature_dim), device=sort_idx.device).expand(*sort_idx.shape))
            # inverse_idx_expanded = inverse_idx.unsqueeze(feature_dim).expand(*X.shape[:-1], X.size(feature_dim))
    
            return X_sorted, sort_idx, inverse_idx
    
    def make_positive_params(self):
        # self.Sigma = nn.functional.softplus(self.params_Sigma) 
        self.Sigma_vec = nn.functional.softplus(self.params_Sigma_vec) 
        self.Sigma = self.Sigma_vec.diag()
        self.boundary = nn.functional.softplus(self.params_boundary) 
        self.threshold = nn.functional.softplus(self.param_threhold)
    
    def forward_intensive(self, X, Y=None):
        self.make_positive_params()
        
        seq_len = X.shape[-2]
        X_sorted, sort_idx, inverse_idx = self.sort_sequence(X)
        self.sort_idx = sort_idx
        self.inverse_idx = inverse_idx
        
        n_window = 1 if seq_len < self.window_size else (seq_len - self.window_size + self.stride-1)//self.stride + 1
        
        # windowing forward
        intensives = []
        
        # 2안)
        # max_intensives = torch.full((*X_sorted.shape[:-2], seq_len, 1), 0.0, device=X.device, dtype=X.dtype)
        for i in range(n_window):
            # 1안)
            max_intensive_window = torch.full((*X_sorted.shape[:-2], seq_len, 1), 0.0, device=X.device, dtype=X.dtype)
            
            i_start, i_end = i*self.stride, min(i*self.stride+self.window_size, seq_len)
            if i_end - i_start <= 0:
                continue
            # print(i_start, i_end)
            X_window = X_sorted[...,i_start:i_end,:]
            if self.kernel == 'gaussian':
                intensive_window = self.dist_layer(X_window, Y, Sigma=self.Sigma)
            else:
                intensive_window = self.dist_layer(X_window, Y, boundary=self.boundary)

            # intensives.append(intensive_window)
            # 1안)
            max_intensive_window[..., i_start:i_end, :] = intensive_window
            intensives.append(max_intensive_window)
            
            # 2안)
            # cur_intensives = max_intensives[..., i_start:i_end, :].clone()
            # max_intensives[..., i_start:i_end, :] = torch.maximum(cur_intensives, intensive_window)
            
            # 3안)
            # idx = torch.arange(i_start, i_end, device=X.device).view(*([1]*(intensive_window.ndim-2)), i_end - i_start , 1).expand_as(intensive_window)
            # max_intensives = max_intensives.scatter_reduce(dim=-2, index=idx, src=intensive_window, reduce="amax", include_self=True)
            
        self.intensives_cat = torch.concat(intensives, dim=-1)
        max_intensive_sorted, _ = self.intensives_cat.max(dim=-1, keepdim=True)
        max_intensive = torch.gather(max_intensive_sorted, dim=-2, index=inverse_idx.unsqueeze(-1))
        return max_intensive
    
    def forward(self, X):       
        intensives_cat = self.forward_intensive(X)
        
        max_intensive, _ = intensives_cat.squeeze(-1).max(dim=-1, keepdim=True)
        output_window = max_intensive - self.threshold
        return output_window

# model = ConvKernelModel(window_size=1000, stride=500, kernel='gaussian').to(device)
model = ConvKernelModel(window_size=1000, stride=500, kernel='linear').to(device)
# model(pad_series_torch[:,:,-2:].to(device))
# print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# load_state = torch.load("D:/DataScience/Model/ConvDeepSVM_params_gaussian.pth")
# load_state = torch.load("D:/DataScience/Model/ConvDeepSVM_params_linear.pth")
# model.load_state_dict(load_state)
# model.make_positive_params()


class BatchSquareHingeLoss():
    def __init__(self, num_classes=None, weights_method='equal'):
        """
         . weights_method : 'equal', 'ratio', 'exp'
        """
        self.weights = None
        self.weihts_method = weights_method
        if num_classes is not None:
            weights = num_classes / num_classes.sum()
            if self.weihts_method == 'exp':
                self.weights = torch.exp(weights) / torch.exp(weights).sum()
            self.weights = weights
    
    def loss_function(self, model, batch, optiimizer=None):
        batch_X, batch_y = batch
        batch_y_pn = batch_y*2 - 1
        y_pred = model.forward(batch_X)
        
        unique_class, num_classes_batch = batch_y.unique(return_counts=True)
        if len(unique_class) > 1:
            if self.weights is not None:
                weights = self.weights
            elif self.weihts_method == 'ratio':
                weights = num_classes_batch / num_classes_batch.sum()
            elif self.weihts_method == 'exp':
                weights = num_classes_batch / num_classes_batch.sum()
                weights = torch.exp(weights) / torch.exp(weights).sum()
            else:   # 'equal;
                weights = torch.ones(2)/2
        else:
            weights = torch.zeros(2)
            weights[unique_class] = 1
        
        hinge_loss = torch.tensor(0.0)
        if len(batch_y_pn[batch_y_pn < 0]) > 0:   # negative_loss
            hinge_loss += weights[0] * (torch.clamp(1 - batch_y_pn[batch_y_pn < 0] * y_pred[batch_y_pn < 0], min=0) ** 2).mean()
            
        if len(batch_y_pn[batch_y_pn > 0]) > 0:     # positive_loss
            hinge_loss += weights[1] * (torch.clamp(1 - batch_y_pn[batch_y_pn > 0] * y_pred[batch_y_pn > 0], min=0) ** 2).mean()
        
        return hinge_loss


# torch.autograd.set_detect_anomaly(True)
squared_hinge_loss = BatchSquareHingeLoss(num_classes, weights_method='exp')  # num_classes   
# squared_hinge_loss = BatchSquareHingeLoss(weights_method='exp')  # num_classes

tm = TorchModeling(model)
tm.compile(optimizer=optim.AdamW(model.parameters(), lr=1e-3)
           ,loss_function=squared_hinge_loss.loss_function)
tm.train_model(train_loader, epochs=500)

# print(model.Sigma.detach(), model.threshold.detach())
# print(model.Sigma.detach(), model.threshold.detach())
# print(model.Precision.detach(), model.threshold.detach())
print(model.boundary.detach(), model.threshold.detach())


# torch.save(model.state_dict(), "D:/DataScience/Model/ConvDeepSVM_params_linear.pth")

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


# ----------------------------------------------------------------------
# torch.where(label_torch==1)      # label_obs
# np.where(pred_label==1)          # label_pred
# np.where(label_true==1)           # label_true

# ----------------------------------------------------------------------
# Visualize
xx, yy = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
grid = np.c_[xx.ravel(), yy.ravel()]  # (40000, 2)

i = 67
mtl_sample = mtl_list[i]
x_scale, y_scale = map(lambda x: x.item(), mtl_sample.mean(0)[:2])
gird_scale = grid * np.array([x_scale, y_scale])
xx_scale = xx * x_scale
yy_scale = yy * y_scale

# visual dict
dist = Distance(Sigma=torch.tensor(1.5), kernel='gaussian')

Z_true = dist.forward(torch.FloatTensor(gird_scale), torch.FloatTensor(mtl_sample[:, -2:]) ).view(xx_scale.shape).detach().numpy()
intensive_true = dist.forward(torch.FloatTensor(mtl_sample[:, -2:])).squeeze()

Z_pred = model.forward_intensive(torch.FloatTensor(gird_scale), torch.FloatTensor(mtl_sample[:, -2:]) ).view(xx_scale.shape).detach().numpy()
intensive_pred = model.forward_intensive(torch.FloatTensor(mtl_sample[:, -2:])).squeeze()


# params = {'true':(dist.Sigma, 4), 'pred': (model.Sigma.detach().diagonal().mean().item(), round(model.threshold.detach().item(),2))}
params = {'true':(dist.Sigma, 4), 'pred': (model.boundary.detach().item(), round(model.threshold.detach().item(),2))}
Z = {'true':Z_true, 'pred': Z_pred}
intensive = {'true': intensive_true, 'pred': intensive_pred}

# visualize
plt.figure(figsize=[mtl_sample[0][0].item()/2.5 , mtl_sample[0][1].item()/3])
for ei, ((Z_name, Z_value), (i_name, i_value), (param_name, param_values)) in enumerate(zip(Z.items(), intensive.items(), params.items())):
    
    plt.subplot(1,2,ei+1)
    plt.title(f"({Z_name}) intensive plot")
    # cont = plt.contourf(xx_scale, yy_scale, Z_value, cmap="RdYlGn_r", alpha=0.3, vmin=0.5, vmax=5, levels=np.sqrt(np.linspace(0, 25, 10)) )
    cont = plt.contourf(xx_scale, yy_scale, Z_value, cmap="RdYlGn_r", alpha=0.3, vmin=0.5, vmax=5)

    for (xp, yp), val in zip(mtl_sample[:,-2:], i_value):
        color='black'
        alpha=0.33
        if val >= float(param_values[1]):
            color='red'
            alpha= 1
        plt.scatter(xp.item(), yp.item(), color=color, s=2, alpha=alpha)
        plt.text(xp.item(), yp.item(), round(val.item(),1), color=color, alpha=alpha, fontsize=8)
    plt.xlabel(f"params {float(param_values[0]):.1f}, threshold: {param_values[1]:.2f}")
    plt.colorbar(cont)
plt.show()

##########################################################################################



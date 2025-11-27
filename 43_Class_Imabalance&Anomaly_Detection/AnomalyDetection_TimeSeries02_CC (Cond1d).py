import os
import sys
if 'home' not in os.getcwd():
    base_path = 'D:'
else:
    base_path = os.getcwd()    
folder_path =f"{base_path}/DataScience"
sys.path.append(f"{folder_path}/00_DataAnalysis_Basic")
sys.path.append(f"{folder_path}/DS_Library")
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
    from DS_TimeSeries import pad_series_list_1d, pad_series_list_2d, series_smoothing
    
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
            from DS_TimeSeries import pad_series_list_1d, pad_series_list_2d, series_smoothing
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
        




############################################################################################################
df00 = pd.read_csv(f"{folder_path}/DataBase/Data_TimeSeries/20251125_Cast_2CC4MC.csv", encoding='utf-8-sig')
df00.info()

data_len = df00.groupby('SLAB_NO').apply(lambda x: len(x))

# length visualize
plt.figure()
plt.hist(data_len, bins=30, edgecolor='grey', color='skyblue')
plt.show()



# Preprocessing
valid_idx = (data_len[(data_len >= 200) & (data_len < 800)]).index

df01 = df00[df00['SLAB_NO'].isin(valid_idx)]
plt.figure()
plt.hist(df01.groupby('SLAB_NO').apply(lambda x: len(x)), bins=30, edgecolor='grey', color='skyblue')
plt.show()



############################################################################################################

X_columns = ['Mold장변Inside전열량', 'Mold장변Outside전열량', 'Mold장변Right전열량','Mold장변Left전열량']
color_dict = {'Mold장변Inside전열량': 'steelblue',
              'Mold장변Outside전열량': 'orange',
              'Mold장변Right전열량': 'mediumseagreen',
              'Mold장변Left전열량': 'purple'}

#############################################################################################################
min_values = df01.groupby('SLAB_NO')[X_columns].apply(lambda x: x.min()).min(axis=1)
valid_index2 = min_values[min_values > 0.8].index

df02 = df01[df01['SLAB_NO'].isin(valid_index2)]
#############################################################################################################

# from DS_TimeSeries import series_smoothing
df_timeseries = df02.groupby('SLAB_NO')[X_columns].apply(lambda x: series_smoothing(x, mask=x<3, window=5, center=True).to_numpy().T)

df_timeseries.apply(lambda x: x.shape)

# from DS_TimeSeries import pad_series_list_2d
df_timeseries_pad = pad_series_list_2d(df_timeseries.to_list())     # (Batch, Feature, Seq)
df_timeseries_pad_T = df_timeseries_pad.transpose(0,2,1)     # (Batch, Seq, Feature)


# visualize overall
plt.figure(figsize=(15,8))
for idx, name in enumerate(X_columns):
    plt.plot(df_timeseries_pad_T[:100,:,idx].T, color=color_dict[name], alpha=0.3, label=name)
# 중복 제거
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1,1))
plt.show()

# visualize subplot
plt.figure(figsize=(15,8))
for idx, name in enumerate(X_columns):
    plt.subplot(2,2, idx+1)
    plt.plot(df_timeseries_pad_T[:100,:,idx].T, color=color_dict[name], alpha=0.3, label=name)
    # 중복 제거
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
plt.show()

################################################################################

df_timeseries_pad_T[:,:,0]
mask = ~np.isnan(df_timeseries_pad_T[:,:,0])     # na가 아닌 부분 mask
valid_seq_len = np.where((~mask).any(axis=-1), (~mask).argmax(axis=-1), mask.shape[-1]).reshape(-1,1)


df_timeseries_fill = df_timeseries_pad_T.copy()
df_timeseries_fill[np.isnan(df_timeseries_fill)] = 0




# ------------------------------------------------------------------------------------
# (Train Test Split)
from sklearn.model_selection import train_test_split
random_state = 0
train_valid_idx, tests_idx = train_test_split(np.arange(len(df_timeseries_fill)), test_size=0.2, random_state=random_state)
train_idx, valid_idx = train_test_split(train_valid_idx, test_size=0.2, random_state=random_state)

train_idx = sorted(train_idx)
valid_idx = sorted(valid_idx)
tests_idx = sorted(tests_idx)

train_X = df_timeseries_fill[train_idx]
valid_X = df_timeseries_fill[valid_idx]
tests_X = df_timeseries_fill[tests_idx]


train_mask = mask[train_idx]
valid_mask = mask[valid_idx]
tests_mask = mask[tests_idx]

train_valid_seq_len = valid_seq_len[train_idx]
valid_valid_seq_len = valid_seq_len[valid_idx]
tests_valid_seq_len = valid_seq_len[tests_idx]

print(train_X.shape, valid_X.shape, tests_X.shape)


# ------------------------------------------------------------------------------------
# (Normalizing : StandardScaler)
from sklearn.preprocessing import StandardScaler
X_mean = np.nanmean(train_X)
X_std = np.nanstd(train_X)
train_X_norm = (train_X - X_mean) / X_std
valid_X_norm = (valid_X - X_mean) / X_std
tests_X_norm = (tests_X - X_mean) / X_std


#################################################################################################
from torch.utils.data import Dataset, TensorDataset, DataLoader
train_X_tensor = torch.FloatTensor(train_X_norm)
valid_X_tensor = torch.FloatTensor(valid_X_norm)
tests_X_tensor = torch.FloatTensor(tests_X_norm)

train_mask_tensor = torch.tensor(train_mask, dtype=bool)
valid_mask_tensor = torch.tensor(valid_mask, dtype=bool)
tests_mask_tensor = torch.tensor(tests_mask, dtype=bool)

train_valid_seq_len_tensor = torch.LongTensor(train_valid_seq_len)
valid_valid_seq_len_tensor = torch.LongTensor(valid_valid_seq_len)
tests_valid_seq_len_tensor = torch.LongTensor(tests_valid_seq_len)

train_dataset = TensorDataset(train_X_tensor, train_mask_tensor, train_valid_seq_len_tensor)
valid_dataset = TensorDataset(valid_X_tensor, valid_mask_tensor, valid_valid_seq_len_tensor)
tests_dataset = TensorDataset(tests_X_tensor, tests_mask_tensor, tests_valid_seq_len_tensor)

BATCH = 128
train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH, shuffle=True)
tests_loader = DataLoader(tests_dataset, batch_size=BATCH, shuffle=True)


# for batch in train_loader:
#     break
# len(batch)        # 5 : X, mask, valid_seq_len, tail_mean, y
# batch[0].shape
# batch[1].shape
# batch[2].shape


#################################################################################################
class AnomalyTimeSeriesConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64):
        super().__init__()
        self.out_channels = out_channels
        
        self.encoder = nn.ModuleDict({})
        self.encoder['input_conv'] = nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2)
        self.encoder['input_activation'] = nn.ReLU()
        
        self.encoder['input_conv']
        
        
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
        self.eps = 1e-5
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
        # x.shape   # (B, T, E)
        # mask.shape # (B, T)   # valid: True
        x_T = x.transpose(-2, -1)   # (B, E, T)
        
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
        return latent
    
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

model = AnomalyTimeSeriesConv1d(in_channels=4, out_channels=2, hidden_channels=64)
f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
# model( torch.rand(10,5) )

class Batch_KL_Divergece():
    def __init__(self, train_loader):
        self.len_batch = len(train_loader)
        self.count = 0
        
    def batch_counter(self):
        self.count = self.count % self.len_batch
    
    def batch_kl_divergence_loss(self, model, batch, optimizer=None):
        self.batch_counter()
        batch_X, mask, valid_seq_len = batch
        mu, cov = model.forward_gaussian(batch_X, mask, self.count)
        
        # trace term
        K = mu.size(0)
        sign, logdet = torch.linalg.slogdet(cov)
        kl_loss = 0.5 * (torch.trace(cov) + mu @ mu - K - logdet)
        self.count += 1
        return kl_loss

Batch_KL_Loss = Batch_KL_Divergece(train_loader)

tm = TorchModeling(model, device)
tm.compile(optim.Adam(model.parameters(), lr=1e-3),
           early_stop_loss = EarlyStopping(min_iter=30, patience=30))
tm.train_model(train_loader, valid_loader, epochs=300, loss_function=Batch_KL_Loss.batch_kl_divergence_loss)



##############################################################################################

train_timeseries, tain_mask, train_valid_len = train_dataset.tensors
sample_batch = train_timeseries[:]
sample_mask =  tain_mask[:]
sample_mask_unsqueeze = sample_mask.unsqueeze(-1).repeat(1,1,4)

pred_batch = model(sample_batch.to(device), sample_mask.to(device)).detach().to('cpu')
pred_batch_np = pred_batch.numpy()
# model.mu
# model.cov
maha_dists = torch.sqrt( torch.einsum("nd,nd->n", torch.einsum("nd, de->ne", pred_batch, torch.linalg.inv(model.cov)), pred_batch).reshape(-1,1) )


# Visualize 2-dim
threshold = 3
plt.scatter(pred_batch_np[:, 0], pred_batch_np[:, 1], c=maha_dists.numpy().ravel(), cmap='jet')
plt.colorbar()
plt.xlim(-4.5, 4.5)
plt.ylim(-4.5, 4.5)
circle = plt.Circle((0, 0), threshold, color='red', fill=False) 
plt.gca().add_patch(circle)
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

# X_mean, X_std
normal_idx = torch.where(maha_dists.ravel() < threshold/10)[0]
normal_samples = sample_batch[normal_idx].masked_fill(sample_mask_unsqueeze[normal_idx] == 0, float('nan'))
normal_samples = normal_samples * X_std + X_mean

abnormal_idx = torch.where(maha_dists.ravel() > threshold)[0]
abnormal_samples = sample_batch[abnormal_idx].masked_fill(sample_mask_unsqueeze[abnormal_idx] == 0, float('nan'))
abnormal_samples = abnormal_samples * X_std + X_mean

print(len(normal_idx), len(abnormal_idx))


# visualize subplot (Normal vs. Abnormal)
plt.figure(figsize=(15,8))
for idx, name in enumerate(X_columns):
    plt.subplot(2,2, idx+1)
    plt.plot(normal_samples[:,:,idx].T, color=color_dict[name], alpha=0.2, label=name)
    plt.plot(abnormal_samples[:,:,idx].T, color='red', alpha=0.2, label=name)
    # 중복 제거
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
plt.show()


# visualize subplot (Overall Plot)
plt.figure(figsize=(15,8))
for idx, name in enumerate(X_columns):
    plt.subplot(2,2, idx+1)
    plt.plot(df_timeseries_pad_T[:1000,:,idx].T, color=color_dict[name], alpha=0.03, label=name)
    plt.plot(normal_samples[:,:,idx].T, color='black', alpha=0.2, label=f"normal_{name}")
    plt.plot(abnormal_samples[:,:,idx].T, color='red', alpha=0.2, label=f"abnormal_{name}")
    # 중복 제거
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
plt.show()



##############################################################################################





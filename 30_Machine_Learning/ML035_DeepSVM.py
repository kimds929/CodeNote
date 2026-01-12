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
rng = np.random.RandomState()

# --------------------------------------------------------------------------------------------------
# scale = 1/50
# n_log_mean = 1.8
# n_log_std = 1
# n_sample = 20
# n_defect = (10 ** rng.normal(loc=n_log_mean, scale=n_log_std, size=n_sample) *scale ).astype(int)

n_mean = 25
n_std = 7
n_sample = 20
n_defect = rng.normal(loc=n_mean, scale=n_std, size=n_sample).astype(int)

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
plt.scatter(mtl_sample[:,2], mtl_sample[:,3], c="black", s=5)



# --------------------------------------------------------------------------------
# from DS_TimeSeries import pad_series_list
pad_series = pad_series_list(mtl_list, pad_value=np.inf)

print(pad_series.shape)

pad_series_torch = torch.FloatTensor(pad_series)



################################################################################



def _is_mat(p):
    return p is not None and getattr(p, "ndim", 0) == 2

def mahalanobis_dist(X, Y=None, Sigma=None):
    """pairwise distance: Euclidean (optional /sigma) or Mahalanobis (Sigma matrix)."""
    Y = X if Y is None else Y
    D = X[:, None, :] - Y[None, :, :]                # (n,m,d)

    if _is_mat(Sigma):
        s =  torch.einsum("nmd,dd,nmd->nm", D, torch.linalg.inv(Sigma), D)
    else:
        Sigma = 1 if Sigma is None else Sigma
        s = torch.einsum("nmd,nmd->nm", D, D) / (Sigma**2)       # (D*D).sum(-1)/Sigma
    s =  torch.clamp(s, min=0.0)
    return torch.sqrt(s)


def _u(X, Y=None, boundary=1.0):
    """normalized distance for compact kernels."""
    if _is_mat(boundary):          # ellipsoid: already normalized, boundary at u<=1
        return mahalanobis_dist(X, Y, Sigma=boundary)
    return mahalanobis_dist(X, Y) / boundary

def gaussian_kernel(X, Y=None, Sigma=1.0):
    d = mahalanobis_dist(X, Y, Sigma=Sigma)
    return torch.exp(-0.5 * d * d)

def uniform_kernel(X, Y=None, boundary=1.0):
    u = _u(X, Y, boundary)
    return (u <= 1).to(dtype=u.dtype)

def linear_kernel(X, Y=None, boundary=1.0):
    u = _u(X, Y, boundary)
    return torch.clamp(1 - u, 0, 1)

def epanechnikov_kernel(X, Y=None, boundary=1.0):
    u = _u(X, Y, boundary)
    return torch.clamp(1 - u*u, 0, 1)

def quartic_kernel(X, Y=None, boundary=1.0):
    u = _u(X, Y, boundary)
    t = torch.clamp(1 - u*u, 0, 1)
    return t*t


################################################################################

i = 2
plt.imshow(mahalanobis_dist(pad_series_torch[i], pad_series_torch[i]), cmap='coolwarm')

# plt.imshow(gaussian_kernel(pad_series_torch[i], pad_series_torch[i]), cmap='coolwarm')
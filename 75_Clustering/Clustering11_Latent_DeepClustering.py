import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from DS_MachineLearning import DataPreprocessing, DS_LabelEncoder, DS_StandardScaler
    from DS_DeepLearning import EarlyStopping, TorchDataLoader, TorchModeling, AutoML
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
            from DS_MachineLearning import DataPreprocessing, DS_LabelEncoder, DS_StandardScaler
            from DS_DeepLearning import EarlyStopping, TorchDataLoader, TorchModeling, AutoML
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

####################################################################################################
# from sklearn.datasets import load_iris

# iris = load_iris()
# cols = ["_".join(c.split(" ")[:-1]) for c in iris['feature_names']]
# df_iris = pd.DataFrame(iris['data'], columns=cols)
# iris_target = pd.Series(iris['target']).apply(lambda x:iris['target_names'][x])
# # df_iris.insert(0, 'target', iris_target)
# df_iris['target'] = iris_target

# df_iris.to_csv("D:/DataScience/DataBase/Data_Tabular/datasets_iris.csv", index=False, encoding='utf-8-sig')

df_iris = pd.read_csv("D:/DataScience/DataBase/Data_Tabular/datasets_iris.csv", encoding='utf-8-sig')

##############################################################################################################

# 5. 시각화 (꽃잎 길이와 너비 기준)
# plt.scatter(df_iris.iloc[:, 1], df_iris.iloc[:, 3], c=df_iris.iloc[:,0], cmap='viridis')
# plt.xlabel('Petal length (cm)')
# plt.ylabel('Petal width (cm)')
# plt.title('Iris dataset - K-means clustering')
# plt.show()


y = df_iris.iloc[:,-1]
y = pd.Categorical(y, categories=y.value_counts().index)
X = df_iris.iloc[:,:-1]


##############################################################################################################

import torch

y_encoder = DS_LabelEncoder()
y_tensor = torch.LongTensor(y_encoder.fit_transform(y.to_numpy()).reshape(-1,1))

X_encoder = DS_StandardScaler()
X_tensor = torch.FloatTensor(X_encoder.fit_transform(X))

from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(y_tensor, X_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# -----------------------------------------------------------------------------------------------------------

random_state = 1
datapreprocessing = DataPreprocessing(y, X, batch_size=32, encoder=[DS_LabelEncoder(), DS_StandardScaler()], random_state=random_state, stratify=y)
datapreprocessing.fit_tensor_dataloader()




##############################################################################################################
from sklearn.manifold import TSNE

# t-SNE
tsne = TSNE(n_components=2, random_state=random_state, perplexity=30)
# perplexity (5~50)
#   작은 perplexity → 매우 국소적인 구조를 강조 (작은 클러스터가 잘 보임, 하지만 전체 구조 왜곡 가능)
#   큰 perplexity → 더 넓은 범위의 구조를 반영 (전체적인 분포를 잘 보존, 하지만 작은 구조가 뭉개질 수 있음)
X_emb_tsne = tsne.fit_transform(X)


# 시각화
plt.figure(figsize=(8, 6))
plt.title('t-SNE')
scatter = plt.scatter(X_emb_tsne[:, 0], X_emb_tsne[:, 1], c=y.codes, cmap='viridis', label=y)
plt.legend(handles=scatter.legend_elements()[0], labels=list(y.categories))
plt.show()




##############################################################################################################
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
x_emb_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.title('PCA')
scatter = plt.scatter(x_emb_pca[:, 0], x_emb_pca[:, 1], c=y.codes, cmap='viridis', label=y)
plt.legend(handles=scatter.legend_elements()[0], labels=list(y.categories))
plt.show()




##############################################################################################################

import torch

y_encoder = DS_LabelEncoder()
y_tensor = torch.LongTensor(y_encoder.fit_transform(y.to_numpy()).reshape(-1,1))

X_encoder = DS_StandardScaler()
X_tensor = torch.FloatTensor(X_encoder.fit_transform(X))

from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(y_tensor, X_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

##############################################################################################################

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f"torch_device : {device}")

##############################################################################################################
# ===== VAE 정의 =====
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=64):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim//2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim//2, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# ===== VAE Loss =====
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


model = VAE(input_dim=4, latent_dim=2, hidden_dim=32)
print(f"parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


def loss_function(model, batch, optimizer=None):
    batch_y, batch_X = batch
    recon_batch, mu, logvar = model(batch_X)
    loss = vae_loss(recon_batch, batch_X, mu, logvar)
    return loss

tm0 = TorchModeling(model, device)
tm0.compile(optimizer=optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2),
           loss_function=loss_function
          )
tm0.train_model(train_loader, epochs=500)


# ===== Latent Vector 추출 =====
model.eval()
with torch.no_grad():
    mu, logvar = model.encode(X_tensor)
    z = model.reparameterize(mu, logvar).to('cpu').numpy()

# visualize
plt.figure(figsize=(8, 6))
plt.title('PCA')
scatter = plt.scatter(mu[:, 0], mu[:, 1], c=y.codes, cmap='viridis', label=y)
plt.legend(handles=scatter.legend_elements()[0], labels=list(y.categories))
plt.show()



# ===== GMM + BIC로 최적 클러스터 개수 찾기 =====
from sklearn.mixture import GaussianMixture
lowest_bic = np.inf
best_gmm = None
n_components_range = range(1, 10)

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(mu)
    bic = gmm.bic(mu)
    if bic < lowest_bic:
        lowest_bic = bic
        best_gmm = gmm

print(f"Best cluster number: {best_gmm.n_components}")
cluster_labels = best_gmm.predict(mu)
cluster_probs = best_gmm.predict_proba(mu)




##############################################################################################################
# ===== AutoEncoder =====
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, latent_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)
        return x

model = AutoEncoder(input_dim=4, latent_dim=2, hidden_dim=32)
print(f"parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


def loss_function(model, batch, optimizer=None):
    batch_y, batch_X = batch
    recon_X = model(batch_X)
    loss = nn.functional.mse_loss(batch_X, recon_X)
    return loss

tm1 = TorchModeling(model, device)
tm1.compile(optimizer=optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2),
           loss_function=loss_function
          )
tm1.train_model(train_loader, epochs=500)


# ===== Latent Vector 추출 =====
model.eval()
with torch.no_grad():
    latent = model.encoder(X_tensor).to('cpu').numpy()

# visualize
plt.figure(figsize=(8, 6))
plt.title('PCA')
scatter = plt.scatter(latent[:, 0], latent[:, 1], c=y.codes, cmap='viridis', label=y)
plt.legend(handles=scatter.legend_elements()[0], labels=list(y.categories))
plt.show()


# ===== GMM + BIC로 최적 클러스터 개수 찾기 =====
from sklearn.mixture import GaussianMixture
lowest_bic = np.inf
best_gmm = None
n_components_range = range(1, 10)

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(latent)
    bic = gmm.bic(latent)
    if bic < lowest_bic:
        lowest_bic = bic
        best_gmm = gmm

print(f"Best cluster number: {best_gmm.n_components}")
cluster_labels = best_gmm.predict(latent)
cluster_probs = best_gmm.predict_proba(latent)

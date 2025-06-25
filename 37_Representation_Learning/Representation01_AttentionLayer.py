
###########################################################################################################
###########################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import httpimport
remote_library_url = 'https://raw.githubusercontent.com/kimds929/'

# with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
#     from DS_DeepLearning import EarlyStopping

with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
    from DS_Torch import TorchDataLoader, TorchModeling, AutoML
    

# # loss_mse = nn.MSELoss()
# def mse_loss(model, x, y):
#     logmu = model(x)
#     mu = torch.exp(logmu)
#     loss = torch.nn.functional.mse_loss(mu, y)
#     return loss

# # loss_gaussian = nn.GaussianNLLLoss()
# def gaussian_loss(model, x, y):
#     mu, logvar = model(x)
#     # var = torch.nn.functional.softplus(logvar)
#     # std = torch.sqrt(var)
#     logvar = torch.clamp(logvar, min=-5, max=5)
#     std = torch.exp(0.5*logvar)
#     loss = torch.nn.functional.gaussian_nll_loss(mu, y, std**2)
#     # loss = loss_gaussian(mu, y, std**2)
#     return loss


# tm = TorchModeling(model=model, device=device)
# tm.compile(optimizer=optimizer
#             ,loss_function = gaussian_loss
#             # ,loss_function = weighted_gaussian_loss
#             # ,loss_function = bernoulli_loss
#             , scheduler=scheduler
#             , early_stop_loss = EarlyStopping(patience=5)
#             )

# tm.train_model(train_loader=train_loader, valid_loader=valid_loader, epochs=100, display_earlystop_result=True, early_stop=False)    
    
##############################################################################################################

def sine_curve(x, x_min, x_max, amp=1):
    return amp * np.sin(2 * np.pi * (x - x_min) / (x_max - x_min))

# x_min = -100
# x_max = 100
# y_min = -100
# y_max = 100
# change_point = 10

def complex_function(x, x_min=-1, x_max=1, y_min=-1, y_max=1, noise_std=None):
    change_points = np.random.randint(0,100, size=10)    
    mean_y = (x**2).max()/2

    y_base_curve = np.abs(x**2-mean_y)
    scale_y = (y_max - y_min)
    y_base_curve = y_base_curve / y_base_curve.max()    # 0 ~ 1
    y_base_curve = y_base_curve * scale_y - scale_y/2
    
    
    
    x_diff = (x_max - x_min)
    y_sin_curve = sine_curve(x, x_min=x_min, x_max=x_max/5, amp=scale_y/3)
    if noise_std is None:
        noise_std = scale_y*0.05
    y_noise = np.random.normal(loc=0, scale=noise_std, size=len(x))
    
    return y_base_curve + y_sin_curve + y_noise


class ComplexFunctions():
    def __init__(self, x=None, x_range=None, y_range=[-1,1], noise_std=None, random_state=None):
        if x is not None:
            self.x_min = np.min(x)
            self.x_max = np.max(x)
        elif x_range is not None:
            self.x_min = x_range[0]
            self.x_max = x_range[1]
        else:
            raise ('x input error.')
        
        self.systhetic_x = np.linspace(self.x_min, self.x_max, num=10000)
        self.y_min = y_range[0]
        self.y_max = y_range[1]        
        
        self.mean_y = (self.systhetic_x**2).max()/2
        self.scale_y = (self.y_max -self.y_min)
        self.x_diff = (self.x_max - self.x_min)
        self.noise_std = self.scale_y*0.1 if noise_std is None else noise_std
        
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        self.sine_x_rand_scale = np.abs(int(self.rng.normal(loc=5, scale=1.5)))
        self.sine_y_rand_scale = np.abs(self.rng.normal(loc=3, scale=0.7))
    
    def true_f(self, x):
        y_base_curve = np.abs(x**2-self.mean_y)
        y_base_curve = y_base_curve / y_base_curve.max()    # 0 ~ 1
        y_base_curve = y_base_curve * self.scale_y - self.scale_y/2
        
        y_sin_curve = sine_curve(x, x_min=self.x_min, x_max=self.x_max/self.sine_x_rand_scale, amp=self.scale_y/self.sine_y_rand_scale)
        return y_base_curve + y_sin_curve
    
    def obs_f(self, x):
        y_true = self.true_f(x)
        y_noise = np.random.normal(loc=0, scale=self.noise_std, size=len(x))
        
        return y_true + y_noise


# x = np.linspace(-100, 100, num=1000)
# plt.plot(x, complex_function(x, x_min=-100, x_max=100, y_min=-100, y_max=100))
# a = np.random.RandomState(1)

x_r = np.linspace(-50, 100, num=30)
f = ComplexFunctions(x_r)

x = np.random.random(1000) *150 - 50

plt.plot(x_r, f.true_f(x_r))
plt.scatter(x, f.obs_f(x), alpha=0.5)
plt.show()

######################################################################
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
######################################################################
x_tensor = torch.FloatTensor(x.reshape(-1,1))
y_tensor = torch.FloatTensor(f.obs_f(x).reshape(-1,1))

# plt.scatter(x_tensor.view(-1), y_tensor.view(-1))


dataset = TensorDataset(x_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

class SimpleFC(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.fc_layers(x)
    

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
simple_model = SimpleFC(hidden_dim=16)
print(count_parameters(simple_model))

simple_optimizer = optim.Adam(simple_model.parameters()) 

def mse_loss(model, x, y):
    pred_y = model(x)
    loss = torch.nn.functional.mse_loss(pred_y, y)
    return loss

tm_simple = TorchModeling(model=simple_model, device=device)
tm_simple.compile(optimizer=simple_optimizer
            ,loss_function = mse_loss
            # ,loss_function = weighted_gaussian_loss
            # , scheduler=scheduler
            # , early_stop_loss = EarlyStopping(patience=5)
            )

tm_simple.train_model(train_loader=loader, epochs=100)


with torch.no_grad():
    pred_y = tm_simple.model(x_tensor.to(device)).to('cpu').detach()
    

plt.plot(x_r, f.true_f(x_r))
plt.scatter(x, f.obs_f(x), alpha=0.1)
plt.scatter(x_tensor.view(-1), pred_y.view(-1))
plt.show()

############################################################################################################################################
############################################################################################################################################







############################################################################################################################################
############################################################################################################################################

# --------------------------------------------------------------------------
# Feature 자체의 고유한 특성을 학습하기 위한 embedding 생성 (Feature값에 independent)
class EmbeddingFeature_Layer(nn.Module):
    def __init__(self, embed_dim: int, max_embedding: int = 100):
        super().__init__()
        self.feature_embedding_layer = nn.Embedding(num_embeddings=max_embedding, embedding_dim=embed_dim)

    def forward(self, x, x_index=None):
        # x.shape (B, T)
        # B: batch, T: seq_len,  D:embedding_dim
        *batch_shape, T = x.shape

        # feature index: (T,)
        if x_index is None:
            x_index = torch.arange(T, device=x.device)
        else:
            x_index = x_index.to(x.device)

        # feature embedding: (T, D)
        feature_embed = self.feature_embedding_layer(x_index)

        # reshape for broadcast: (1,...,1,T,D)
        expand_shape = [1] * len(batch_shape) + [T, feature_embed.shape[-1]]
        feature_embed = feature_embed.view(*expand_shape)  # (1,...,1,T,D)

        # expand to match batch: (..., T, D)
        output_embedding_feature = feature_embed.expand(*batch_shape, T, feature_embed.shape[-1])

        return output_embedding_feature  # shape: (..., T, D)

efl = EmbeddingFeature_Layer(5)
efl(torch.rand(10,5,3,2)).shape

# Numeric Feature마다 독립적인 Linear 표현력(embedding)을 부여 (Feature값에 dependent)
class EmbeddingFC_Layer(nn.Module):
    def __init__(self, embedding_dim: int, max_embedding: int = 100):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.weight_embedding_layer = nn.Embedding(max_embedding, embedding_dim)
        self.bias_embedding_layer = nn.Embedding(max_embedding, embedding_dim)
        nn.init.xavier_uniform_(self.weight_embedding_layer.weight)
        nn.init.zeros_(self.bias_embedding_layer.weight)

    def forward(self, x, x_index=None):
        # x.shape (B, T)
        # B: batch, T: seq_len,  D:embedding_dim
        *batch_shape, T = x.shape

        # feature index: (T,)
        if x_index is None:
            x_index = torch.arange(T, device=x.device)
        else:
            x_index = x_index.to(x.device)

        # (T, D)
        weight = self.weight_embedding_layer(x_index)
        bias = self.bias_embedding_layer(x_index)

        # reshape for broadcasting: (1,...,1,T,D)
        expand_shape = [1] * len(batch_shape) + [T, self.embedding_dim]
        weight = weight.view(*expand_shape)  # (1,...,1,T,D)
        bias = bias.view(*expand_shape)      # (1,...,1,T,D)
        
        # embedding fc features
        output_embedding_fc = x.unsqueeze(-1) * weight + bias  # (..., T, D)

        return output_embedding_fc  # (..., T, D)


ef_layer = EmbeddingFC_Layer(embedding_dim=2, max_embedding=100)
ef_layer(torch.rand(5,3)).shape     # (B,T) → (B,T,D)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, batch_first: bool = True, learnable_scale=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.d_head = embed_dim // num_heads
        self.batch_first = batch_first
        self.learnable_scale = learnable_scale
        
        self.scale_layer = nn.Sequential(
            nn.Linear(self.num_heads, 1),
            nn.Softplus()
        )
        
    def forward(self, Q, K, V, mask=None):
        # Handle input shape
        if not self.batch_first:
            # Input shape: (T, B, D) → convert to (..., B, T, D)
            # T: seq_len,  B: batch,  D:embedding_dim
            Q, K, V = [x.transpose(0, -2) for x in (Q, K, V)]

        B, T, _ = Q.shape

        # Split into heads: (B, num_heads, T, d_head)
        def reshape(x):
            return x.view(B, T, self.num_heads, self.d_head).transpose(-2, -1)

        Q, K, V = map(reshape, (Q, K, V))

        # Attention scores: (B, num_heads, T, T)
        if self.learnable_scale:
            scale = self.scale_layer(Q)
            scale = scale.clamp(min=1e-4, max=10.0)
            
        else:
            scale = self.d_head ** 0.5
        scores = torch.matmul(Q, K.transpose(-2, -1))   # (B, T, T)
        scores = scores / scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (B, num_heads, T, d_head)

        # Concat heads: (B, T, D)
        attn_output = attn_output.transpose(-2, -1).contiguous().view(B, T, self.embed_dim)

        # Convert back to (T, B, D) if needed
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, attn_weights

# mha = MultiHeadAttention(8,4, learnable_scale=True)
# a,b = mha(torch.rand(6,2,8), torch.rand(6,2,8), torch.rand(6,2,8))

# -------------------------------------------------------------------------------------


class EmbeddingATT_Model(nn.Module):
    def __init__(self, hidden_dim=16, embedding_dim=1, max_embedding=100, learnable_scale=False):
        super().__init__()
        self.total_emb_dim = hidden_dim*embedding_dim
        self.emb_fc = EmbeddingFC_Layer(embedding_dim=self.total_emb_dim, max_embedding=max_embedding)
        self.mha_layer = MultiHeadAttention(embed_dim=self.total_emb_dim, num_heads=hidden_dim, learnable_scale=learnable_scale)
        self.fc_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.total_emb_dim, 1)
        )
        
    def forward(self, x):
        # x.shape : (B,T)
        emb_output = self.emb_fc(x)     # (B,T) → (B,T,D)
        emb_output = nn.functional.relu(emb_output)
        mha_output, mha_wieghts = self.mha_layer(emb_output, emb_output, emb_output)    # (B,T,D) → (B,T,D)
        mha_output = nn.functional.relu(mha_output)
        fc_output = self.fc_layers(mha_output).squeeze(-1) # (B,T,D) → (B,T,1) → (B,T)
        return fc_output.mean(dim=-1, keepdim=True) # (B,T) → (B, 1)
        
        
        

# el = EmbeddingFC_Layer(16).to(device)
# el = EmbeddingATT_Model().to(device)
# el(torch.rand(5,2).to(device)).shape



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
attntion_model = EmbeddingATT_Model(embedding_dim=1, hidden_dim=16, max_embedding=100, learnable_scale=False)
attntion_model(torch.rand(5,1))
print(count_parameters(attntion_model))

attntion_optimizer = optim.Adam(attntion_model.parameters()) 

def mse_loss(model, x, y):
    pred_y = model(x)
    loss = torch.nn.functional.mse_loss(pred_y, y)
    return loss

tm_attention = TorchModeling(model=attntion_model, device=device)
tm_attention.compile(optimizer=attntion_optimizer
            ,loss_function = mse_loss
            # ,loss_function = weighted_gaussian_loss
            # , scheduler=scheduler
            # , early_stop_loss = EarlyStopping(patience=5)
            )

tm_attention.train_model(train_loader=loader, epochs=100)


with torch.no_grad():
    pred_y = tm_attention.model(x_tensor.to(device)).to('cpu').detach()
    

plt.plot(x_r, f.true_f(x_r))
plt.scatter(x, f.obs_f(x), alpha=0.1)
plt.scatter(x_tensor.view(-1), pred_y.view(-1))
plt.show()






############################################################################################################################################
############################################################################################################################################
# -------------------------------------------------------------------------------------
# Periodic Representation
class PeriodicLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.periodic_weights = nn.Parameter(torch.randn(input_dim, output_dim))
        self.periodic_bias = nn.Parameter(torch.randn(1, output_dim))
    
    def forward(self, x):
        #(B,F) @ (F,F')
        return torch.sin(x @ self.periodic_weights + self.periodic_bias)


class StepLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.step_weights = nn.Parameter(torch.randn(input_dim, output_dim))
        self.step_bias = nn.Parameter(torch.randn(1, output_dim))
    
    def forward(self, x):
        #(B,F) @ (F,F')
        return torch.sign(torch.sin(x @ self.step_weights + self.step_bias))

############################################################################################################################################
############################################################################################################################################
# -------------------------------------------------------------------------------------

class SimplePeriodicFC(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, periodic_dim=4):
        super().__init__()
        self.periodic_dim = periodic_dim
        self.fc_layers = nn.Linear(input_dim, hidden_dim-periodic_dim)
        self.periodic_layer= PeriodicLayer(input_dim, periodic_dim)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x1 = self.fc_layers(x)
        x2 = self.periodic_layer(x)
        l1_output = nn.functional.relu(torch.cat([x1, x2], dim=-1))
        return self.regressor(l1_output)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
simple_periodic_model = SimplePeriodicFC(hidden_dim=16)
print(count_parameters(simple_periodic_model))

simple_periodic_optimizer = optim.Adam(simple_periodic_model.parameters()) 

def mse_loss(model, x, y):
    pred_y = model(x)
    loss = torch.nn.functional.mse_loss(pred_y, y)
    return loss

tm_simple_periodic = TorchModeling(model=simple_periodic_model, device=device)
tm_simple_periodic.compile(optimizer=simple_periodic_optimizer
            ,loss_function = mse_loss
            # ,loss_function = weighted_gaussian_loss
            # , scheduler=scheduler
            # , early_stop_loss = EarlyStopping(patience=5)
            )

tm_simple_periodic.train_model(train_loader=loader, epochs=100)


with torch.no_grad():
    pred_y = tm_simple_periodic.model(x_tensor.to(device)).to('cpu').detach()
    

plt.plot(x_r, f.true_f(x_r))
plt.scatter(x, f.obs_f(x), alpha=0.1)
plt.scatter(x_tensor.view(-1), pred_y.view(-1))
plt.show()





############################################################################################################################################
############################################################################################################################################
# -------------------------------------------------------------------------------------

class TemporalModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16):
        super().__init__()
        self.temporal_embedding_layer = TemporalEmbedding(input_dim=input_dim, embed_dim=hidden_dim)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        temporal_output = nn.functional.relu(self.temporal_embedding_layer(x))
        return self.regressor(temporal_output)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
temporal_model = TemporalModel(hidden_dim=16)
# temporal_model(torch.rand(5,1))
print(count_parameters(temporal_model))

temporal_optimizer = optim.Adam(temporal_model.parameters()) 

def mse_loss(model, x, y):
    pred_y = model(x)
    loss = torch.nn.functional.mse_loss(pred_y, y)
    return loss

tm_temporal = TorchModeling(model=temporal_model, device=device)
tm_temporal.compile(optimizer=temporal_optimizer
            ,loss_function = mse_loss
            # ,loss_function = weighted_gaussian_loss
            # , scheduler=scheduler
            # , early_stop_loss = EarlyStopping(patience=5)
            )

tm_temporal.train_model(train_loader=loader, epochs=100)


with torch.no_grad():
    pred_y = tm_temporal.model(x_tensor.to(device)).to('cpu').detach()
    

plt.plot(x_r, f.true_f(x_r))
plt.scatter(x, f.obs_f(x), alpha=0.1)
plt.scatter(x_tensor.view(-1), pred_y.view(-1))
plt.show()





############################################################################################################################################
############################################################################################################################################
# -------------------------------------------------------------------------------------




class PeriodicEmbeddingATT_Model(nn.Module):
    def __init__(self, hidden_dim=16, periodic_dim=4, embedding_dim=1, max_embedding=100, learnable_scale=False):
        super().__init__()
        self.total_emb_dim = hidden_dim*embedding_dim
        self.periodic_dim = periodic_dim
        self.emb_fc = EmbeddingFC_Layer(embedding_dim=self.total_emb_dim, max_embedding=max_embedding)
        self.mha_layer = MultiHeadAttention(embed_dim=self.total_emb_dim, num_heads=hidden_dim, learnable_scale=learnable_scale)
        self.fc_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.total_emb_dim, 1)
        )
        
    def forward(self, x):
        # x.shape : (B,T)
        emb_output = self.emb_fc(x)     # (B,T) → (B,T,D)
        linear_embedding_features = emb_output[..., :-self.periodic_dim]
        periodic_embedding_features = torch.sin(emb_output[...,-self.periodic_dim:])
        periodic_emb_output = torch.cat([linear_embedding_features, periodic_embedding_features], dim=-1)
        periodic_emb_output = nn.functional.relu(periodic_emb_output)
        
        mha_output, mha_wieghts = self.mha_layer(periodic_emb_output, periodic_emb_output, periodic_emb_output)    # (B,T,D) → (B,T,D)
        mha_output = nn.functional.relu(mha_output)
        # print(mha_output.shape)
        fc_output = self.fc_layers(mha_output).squeeze(-1) # (B,T,D) → (B,T,1) → (B,T)
        return fc_output.mean(dim=-1, keepdim=True) # (B,T) → (B, 1)
        
        # fc_output = self.fc_layers(mha_output)  # # (B,T,D) → (B,T,1)
        # feature_dim = fc_output.shape[-2]
        # return torch.sum((torch.softmax(fc_output @ fc_output.transpose(-1,-2), dim=-1)/feature_dim) @ fc_output, dim=-2)

# mh = MultiHeadAttention(16,8)
# mh(torch.rand(5,1,16), torch.rand(5,1,16), torch.rand(5,1,16))
# el = PeriodicEmbeddingATT_Model().to(device)
# el(torch.rand(5,1).to(device)).shape



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
periodic_attntion_model = PeriodicEmbeddingATT_Model(embedding_dim=2, hidden_dim=8, max_embedding=1, learnable_scale=False)
periodic_attntion_model(torch.rand(5,1))
print(count_parameters(periodic_attntion_model))

attntion_periodic_optimizer = optim.Adam(periodic_attntion_model.parameters()) 

def mse_loss(model, x, y):
    pred_y = model(x)
    loss = torch.nn.functional.mse_loss(pred_y, y)
    return loss

tm_periodic_attention = TorchModeling(model=periodic_attntion_model, device=device)
tm_periodic_attention.compile(optimizer=attntion_periodic_optimizer
            ,loss_function = mse_loss
            # ,loss_function = weighted_gaussian_loss
            # , scheduler=scheduler
            # , early_stop_loss = EarlyStopping(patience=5)
            )

tm_periodic_attention.train_model(train_loader=loader, epochs=100)


with torch.no_grad():
    pred_y = tm_periodic_attention.model(x_tensor.to(device)).to('cpu').detach()
    

plt.plot(x_r, f.true_f(x_r))
plt.scatter(x, f.obs_f(x), alpha=0.1)
plt.scatter(x_tensor.view(-1), pred_y.view(-1))
plt.show()




















        
    







################################################################################################
# Generate Dataset ##########################################################################
import numpy as np
import matplotlib.pyplot as plt
# matplotlib.use('Agg') 
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

example = False

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

class UnknownBernoulliFunction():
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

    def true_z(self, x):
        for i in range(self.n_polynorm+1):
            response = self.true_theta[i] * (x**i)

            if (self.normalize) and (self.y_mu is not None) and (self.y_std is not None):
                response = (response - self.y_mu)/self.y_std
        return response

    def sigmoid_f(self, x):
        return 1/(1 + torch.exp(x))

    def true_f(self, x):
        response = self.true_z(x)
        return self.sigmoid_f(-response)

    def forward_z(self, x):
        if (self.normalize) and (self.y_mu is not None) and (self.y_std is not None):
            noise_z = self.true_z(x) + self.error_scale * torch.randn((x.shape[0],1))
        else:
            noise_z = self.true_z(x) + self.true_z(x).mean()*self.error_scale * torch.randn((x.shape[0],1))
        return noise_z

    def forward(self, x):
        noise_z = self.forward_z(x)
        probs = self.sigmoid_f(-noise_z)
        bernoulli_dist = torch.distributions.Bernoulli(probs=probs)
        return bernoulli_dist.sample()

    def __call__(self, x):
        return self.forward(x)


# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

scale = 200
shift = 0
# error_scale = 0.3
# scale = torch.randint(0,100 ,size=(1,))
# shift = torch.randint(0,500 ,size=(1,))
# error_scale = torch.rand((1,))*0.3+0.1
input_dim_init=5

x_train = torch.randn(1000, input_dim_init) *scale + shift   # 1000 samples of dimension 10
x_train_add_const = torch.cat([x_train, torch.ones_like(x_train)], axis=1)
input_dim = x_train_add_const.shape[1]



# f = UnknownFuncion()
# f = UnknownFuncion(n_polynorm=2)
f = UnknownFuncion(n_polynorm=3)
# f = UnknownFuncion(n_polynorm=4)
# f = UnknownFuncion(n_polynorm=5)
# f = UnknownFuncion(n_polynorm=6)
# f = UnknownFuncion(n_polynorm=7)
# f = UnknownFuncion(n_polynorm=8)
# f = UnknownFuncion(n_polynorm=9)
# f = RewardFunctionTorch()
# f = UnknownBernoulliFunction()
f.normalize_setting(x_train)




y_true = f.true_f(x_train)
y_train = f(x_train)
error_sigma = (y_train - y_true).std()
# true_theta = torch.randn((input_dim,1))
# y_true = x_train_add_const @ true_theta
# y_train = y_true + error_scale*scale*torch.randn((x_train_add_const.shape[0],1))




# Dataset and DataLoader
batch_size = 64

train_dataset = TensorDataset(x_train_add_const, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Valid DataSet
x_valid = torch.randn(300, input_dim_init) *scale + shift   # 300 samples of validation set
x_valid_add_const = torch.cat([x_valid, torch.ones_like(x_valid)], axis=1)
y_valid = f(x_valid)
valid_dataset = TensorDataset(x_train_add_const, y_train)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

# Test DataSet
x_test = torch.randn(200, input_dim_init) *scale + shift   # 200 samples of test set
x_test_add_const = torch.cat([x_test, torch.ones_like(x_test)], axis=1)
y_test = f(x_test)
test_dataset = TensorDataset(x_train_add_const, y_train)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# visualize
x_lin = torch.linspace(x_train.min(),x_train.max(),300).reshape(-1,1)
x_lin_add_const = torch.concat([x_lin, torch.ones_like(x_lin)], axis=1)

plt.figure()
plt.scatter(x_train, y_train, label='obs')
plt.plot(x_lin, f.true_f(x_lin), color='orange', label='true')
plt.legend()
plt.show()




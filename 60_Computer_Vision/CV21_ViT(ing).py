import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor

from torchsummary import summary

from PIL import Image

import einops
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


# pip install einops --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org
# pip install torch-summary --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org





# Dataset ---------------------------------------------------------------------------
train_valid_data = torchvision.datasets.MNIST(root = '../DataSet/',
                            train=True,
                            download=True,
                            transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root = '../DataSet/',
                            train=False,
                            download=True,
                            transform=torchvision.transforms.ToTensor())

train_valid_length = len(train_valid_data)

rng = np.random.RandomState(0)
permute_indices = rng.permutation(range(train_valid_length))

valid_size = 0.2
sep_position = int(train_valid_length*(1-valid_size))

train_idx = permute_indices[:sep_position]
valid_idx =  permute_indices[sep_position:]

# n_samples = 10000
train_X = train_valid_data.data[train_idx].unsqueeze(1)/255
train_y = train_valid_data.train_labels[train_idx]
valid_X = train_valid_data.data[valid_idx].unsqueeze(1)/255
valid_y = train_valid_data.train_labels[valid_idx]
test_X = test_data.data.unsqueeze(1)/255
test_y = test_data.test_labels

train_data = torch.utils.data.TensorDataset(train_X, train_y)
valid_data = torch.utils.data.TensorDataset(valid_X, valid_y)

batch_size = 32
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size = batch_size, shuffle = True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_data,
                                           batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                           batch_size = batch_size, shuffle = True)


for batch_X, batch_y in train_loader:
    break
print(batch_X.shape,batch_y.shape)
# batch_X.max()




# VIT ()
# https://yhkim4504.tistory.com/5


# Input X --------------------------------------------------------------------------------
x = torch.randn(8, 3, 224, 224)
x2 = torch.randn(3, 1, 28, 28)

class LinearPatchEmbedding(torch.nn.Module):
    def __init__(self, patch_size=16, in_channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
    
    def forward(self, x):
        with torch.no_grad():
            x_shape = x.shape
        output = x.view(x_shape[0], -1, (self.patch_size*self.patch_size*self.in_channels))
        return output

class ConvolutionalPatchEmbedding(torch.nn.Module):
    def __init__(self, patch_size=16, in_channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.conv2d_layer = torch.nn.Conv2d(in_channels, (self.patch_size*self.patch_size*self.in_channels), 
                                            kernel_size=patch_size, stride=patch_size)  # 8, 768, 14, 14
    
    def forward(self, x):
        with torch.no_grad():
            x_shape = x.shape
        self.conv2d_output = self.conv2d_layer(x)
        # n_channel = (size - Filter) / Stride + 1
        output = self.conv2d_output.view(x_shape[0], -1, (self.patch_size*self.patch_size*self.in_channels))
        return output


# Embedding(Projection) --------------------------------------------------------------------------------
# BATCH x C x H × W →(임베딩)→ BATCH x N x (P*P*C)  
batch_size = x.shape[0] # ☆
img_size = 224      # ☆
patch_size = 16     # ☆
n_patch = img_size // patch_size    # 14 ☆



x.view(8,-1,(16*16*3)).shape

# Linear Embedding ***
x_projected = x.view(batch_size, -1, (patch_size*patch_size*3))     # (8, 196, 768)
# x_projected = einops.rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)

# Convolutional Embedding Embedding ***
# (size - Filter) / Stride + 1
in_channels = 3    # ☆
emb_size = (patch_size*patch_size*3)    # 768 ☆

projection = torch.nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            torch.nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),  # 8, 768, 14, 14
                
            Rearrange('b e (h) (w) -> b (h w) e'),      # 8, 196, 768
        )
# Convolution Output Size : O = (Size - Kernel(Filter) + 2 * Padding)/Stride + 1
x_projected = projection(x)    # (8, 196, 768)





# Cls Token과 Positional Encoding --------------------------------------------------------------------------------
# cls_token과 pos encoding Parameter 정의
cls_tokens = torch.nn.Parameter(torch.randn(1,1, emb_size)).repeat(batch_size, 1,1)  # 8, 1, 768
cls_tokens.shape

positions = torch.nn.Parameter(torch.randn( n_patch **2 + 1, emb_size))     # 14*14+1=197, 768

# cls_token과 projected_x를 concatenate
x_cat = torch.cat([cls_tokens, x_projected], dim=1)     # 8, 1+196=197, 768
x_cat_pos = x_cat + positions       # (8, 197, 768) = (8, 197, 768) + (197, 768)




# ------------------------------------------------------------------------------------------------------------------------------------------
# ViT_EmbeddingLayer 
# with httpimport.remote_repo(f"{remote_url}/DS_Library/main/"):
#     from DS_TorchModule import EmbeddingLayer, PositionalEncodingLayer, PositionwiseFeedForwardLayer, MultiHeadAttentionLayer, make_tril_mask
from DS_TorchModule import EmbeddingLayer, PositionalEncodingLayer, PositionwiseFeedForwardLayer, MultiHeadAttentionLayer, make_mask, ScaledDotProductAttention

class ViT_EmbeddingLayer(torch.nn.Module):
    def __init__(self, patch_size: int = 16, in_channels: int = 3, 
                 img_size: int = 224, pos_encoding='sinusoid',
                patch_embedding_class=ConvolutionalPatchEmbedding):
        super().__init__()
        
        self.embed_dim= patch_size*patch_size*in_channels
        self.n_patch = img_size // patch_size
        
        self.patch_embed_layer = patch_embedding_class(patch_size, in_channels)
        self.cls_tokens = torch.nn.Parameter(torch.randn(1,1, self.embed_dim))
        self.positions = torch.nn.Parameter(torch.randn( self.n_patch **2 + 1, self.embed_dim)) 
        # self.positions_layer = PositionalEncodingLayer(pos_encoding)
       
        
    def forward(self, x):
        self.img_patch = self.patch_embed_layer(x)
        self.img_patch_with_token = torch.cat([self.cls_tokens.repeat(x.shape[0],1,1), self.img_patch], dim=1)
        
        # self.positions = self.positions_layer(self.img_patch_with_token)    # positions_layer 사용시
        self.img_patch_with_positions =  self.img_patch_with_token + self.positions
        return self.img_patch_with_positions
        

ve = ViT_EmbeddingLayer()
ve(x).shape
# ve(x)

ve = ViT_EmbeddingLayer(patch_size=4, in_channels=1, img_size=28)
ve(x2).shape

# pd.DataFrame(ve.positions.detach()).to_clipboard(index=False)




# ------------------------------------------------------------------------------------------------------------------------------------------

# # Residual Connection Block
class ResidualConnectionBlock(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        
        self.layers = torch.nn.Sequential(*layers)
        
    def forward(self, x=None, **kwargs):
        res = x
        x = self.layers(x, **kwargs)
        x += res
        return x
    
# rb = ResidualConnectionBlock(
#     torch.nn.LayerNorm(16)
#     ,MultiHeadAttentionLayer(16, 2)
#     ,torch.nn.Dropout()
# )
# ve = ViT_EmbeddingLayer(patch_size=4, in_channels=1, img_size=28)
# a1 = ve(x2)
# rb(a1).shape





                

# ★ ViT_Encoder_Layer
class ViT_EncoderLayer(torch.nn.Module):
    def __init__(self, embed_dim=256, n_heads=4, posff_dim=512, dropout=0):
        super().__init__()

        self.dropout = torch.nn.Dropout(dropout)

        self.self_att_layer_norm = torch.nn.LayerNorm(embed_dim)
        self.self_att_layer = MultiHeadAttentionLayer(embed_dim, n_heads, dropout)
        
        self.posff_layer_norm = torch.nn.LayerNorm(embed_dim)
        self.posff_layer = PositionwiseFeedForwardLayer(embed_dim, posff_dim, dropout, 'GELU')
        
    def forward(self, X_emb, X_mask=None):
        # X_emb : (batch_seq, X_word, emb)
        # X_mask : (batch_seq, 1, ,1, X_word)

        # (Layer Normalization) --------------------------------------------------------------------
        self.X_layer_normed_1 = self.self_att_layer_norm(X_emb)  # layer normalization
                
        # (Self Attention Layer) ------------------------------------------------------------------
        self.X_self_att_output  = self.self_att_layer((self.X_layer_normed_1, self.X_layer_normed_1, self.X_layer_normed_1), mask=X_mask)
        self.self_attention_score = self.self_att_layer.attention_score
        
        #  (batch_seq, X_word, fc_dim=emb), (batch_seq, n_heads, X_word, key_length=X_word)
        self.X_skipconnect_1 = X_emb + self.dropout(self.X_self_att_output)   # (batch_seq, X_word, emb)
        

        # (Layer Normalization) --------------------------------------------------------------------
        self.X_layer_normed_2 = self.posff_layer_norm(self.X_skipconnect_1)
                
        # (Positional FeedForward Layer) -----------------------------------------------------------
        self.X_posff = self.posff_layer(self.X_layer_normed_2)    # (batch_seq, X_word, emb)
        self.X_skipconnect_2 = self.X_skipconnect_1 + self.dropout(self.X_posff)     # (batch_seq, X_word, emb)

        return self.X_skipconnect_2   # (batch_seq, X_word, emb), (batch_seq, n_heads, X_word, key_length)


vemb = ViT_EmbeddingLayer()
# a1 = vemb(x)
# a1.shape

# venc = ViT_EncoderLayer(embed_dim=vemb.embed_size)
# venc(a1).shape

# vemb = ViT_EmbeddingLayer(patch_size=4, in_channels=1, img_size=28)
# vemb.embed_size
# x_mask = make_mask(x2)
# x_mask.shape
# a1 = vemb(x2)
# a1

# venc = ViT_EncoderLayer(embed_dim=vemb.embed_size)
# venc(a1)



    
# , in_channels: int = 3, img_size: int = 224, 
# ★★ Encoder
class ViT_Encoder(torch.nn.Module):
    def __init__(self, X, patch_size: int = 16,
                 n_layers=1, n_heads=4, posff_dim=512, dropout=0.1, 
                 patch_embedding_class=ConvolutionalPatchEmbedding, pos_encoding='sinusoid'):
        super().__init__()
        input_shape = X.shape
        in_channels = input_shape[1]
        in_h = input_shape[3]
        in_w = input_shape[3]
        embed_dim= patch_size*patch_size*in_channels
        
        self.embed_layer = ViT_EmbeddingLayer(patch_size, in_channels, in_h, pos_encoding, patch_embedding_class)
        
        self.encoder_layers = torch.nn.ModuleList([ViT_EncoderLayer(embed_dim, n_heads, posff_dim, dropout) for _ in range(n_layers)])

    def forward(self, X, X_mask=None):
        # X : (batch_Seq, X_word)
        
        # embedding layer
        self.X_embed = self.embed_layer(X)  # (batch_seq, X_word, emb)
        
        # encoder layer
        next_input = self.X_embed
        
        for enc_layer in self.encoder_layers:
            next_input = enc_layer(next_input, X_mask)
        self.encoder_output = next_input

        return self.encoder_output  # (batch_seq, X_word, emb)


# ve = ViT_Encoder(x)
# ve(x).shape

# ve = ViT_Encoder(x2, patch_size=4)
# ve(x2).shape


# class ClassificationHead(nn.Sequential):
#     def __init__(self, emb_size: int = 768, n_classes: int = 1000):
#         super().__init__(
#             Reduce('b n e -> b e', reduction='mean'),
#             nn.LayerNorm(emb_size), 
#             nn.Linear(emb_size, n_classes))































# ☆ MultiHeadAttentionLayer
class MultiHeadAttentionLayer(torch.nn.Module):
    def __init__(self, embed_dim=256, n_heads=4, dropout=0):
        super().__init__()
        assert embed_dim % n_heads == 0, 'embed_dim은 n_head의 배수값 이어야만 합니다.'

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.query_layer = torch.nn.Linear(embed_dim, embed_dim)
        self.key_layer = torch.nn.Linear(embed_dim, embed_dim)
        self.value_layer = torch.nn.Linear(embed_dim, embed_dim)

        self.att_layer = ScaledDotProductAttention(embed_dim ** (1/2), dropout)
        self.fc_layer = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x=None, mask=None, **kwargs):
        if type(x) == tuple or type(x) == list:
            query, key, value = x
        elif len(kwargs) == 0:
            query = key = value = x
        else:
            query = x if x is not None else kwargs.pop('query')
            key = kwargs.pop('key')
            value = kwargs.pop('value')
            
        with torch.no_grad():
            batch_size = query.shape[0]

        self.query = self.query_layer(query)    # (batch_seq, query_len, emb)
        self.key   = self.key_layer(key)        # (batch_seq, key_len, emb)
        self.value = self.value_layer(value)    # (batch_seq, value_len, emb)

        self.query_multihead = self.query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)     # (batch_seq, n_heads, query_len, head_emb_dim)   ←permute←  (batch_seq, query_len, n_heads, head_emb_dim)
        self.key_multihead   =   self.key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)     # (batch_seq, n_heads, key_len, head_emb_dim)   ←permute←  (batch_seq, key_len, n_heads, head_emb_dim)
        self.value_multihead = self.value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)     # (batch_seq, n_heads, value_len, head_emb_dim)   ←permute←  (batch_seq, value_len, n_heads, head_emb_dim)

        self.weighted, self.attention_score = self.att_layer((self.query_multihead, self.key_multihead, self.value_multihead), mask=mask)
        # self.weightd          # (B, H, QL, HE)
        # self.attention_score  # (B, H, QL, QL)     ★

        self.weighted_arange = self.weighted.permute(0,2,1,3).contiguous()        # (B, QL, H, HE) ← (B, H, QL, HE)
        self.weighted_flatten = self.weighted_arange.view(batch_size, -1, self.embed_dim)   # (B, QL, E) ← (B, H, E)

        self.multihead_output = self.fc_layer(self.weighted_flatten)       # (B, QL, FC)
        return self.multihead_output       #  (batch_seq, query_length, fc_dim)



# class PatchEmbedding(nn.Module):
#     def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
#         self.patch_size = patch_size
#         super().__init__()
#         self.projection = nn.Sequential(
#             # using a conv layer instead of a linear one -> performance gains
#             nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
#             Rearrange('b e (h) (w) -> b (h w) e'),
#         )
#         self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
#         self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))
        
#     def forward(self, x: Tensor) -> Tensor:
#         b, _, _, _ = x.shape
#         x = self.projection(x)
#         cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
#         # prepend the cls token to the input
#         x = torch.cat([cls_tokens, x], dim=1)
#         # add position embedding
#         x += self.positions

#         return x

# class ResidualConnection(torch.nn.Module):
#     def __init__(self, seq_block, dropout=0):
#         super().__init__()
#         self.seq_block = seq_block
#         self.dropout = torch.nn.Dropout(dropout)
        
#     def forward(self, x, **kwargs):
#         res = x
#         self.seq_output = self.seq_block(self.seq_block, **kwargs)
#         rescon_output = res + self.dropout(self.seq_output)
#         return rescon_output



mha_layer = MultiHeadAttentionLayer(embed_dim=emb_size, n_heads=8, dropout=0)
x_mha = mha_layer(x_cat_pos, x_cat_pos, x_cat_pos)  # 8, 197, 768

# GELU (Gaussian Error Linear Unit)
# https://hongl.tistory.com/236
posff_layer = PositionwiseFeedForwardLayer(emb_size, posff_dim=512, activation=torch.nn.GELU)
x_posff = posff_layer(x_mha)    # 8, 197, 768


x_res_con = x_cat_pos + x_posff
















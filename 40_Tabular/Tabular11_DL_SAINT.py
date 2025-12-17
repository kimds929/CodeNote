import os
import sys
if 'home' not in os.getcwd():
    base_path = 'D:'
else:
    base_path = os.getcwd()    
folder_path =f"{base_path}/DataScience"
sys.path.append(f"{folder_path}/00_DataAnalysis_Basic")
sys.path.append(f"{folder_path}/DS_Library")
sys.path.append(f'D:\DataScience/★GitHub_kimds929')


from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy
import missingno as msno
from typing import Optional, Tuple

# from DS_Basic_Module import DF_Summary, ScalerEncoder

#####################################################################################################################

try:
    from DS_MachineLearning import DataPreprocessing, DS_NoneEncoder, DS_StandardScaler, DS_LabelEncoder
    from DS_DeepLearning import EarlyStopping, TorchModeling
    from DS_AutoML import BayesOptLogger, ModelPerformanceEvaluation
    from DS_TorchModule import CategoricalEmbedding, EmbeddingLinear, ContinuousEmbeddingBlock, ResidualConnection
    from DS_TorchModule import PositionalEncoding, LearnablePositionalEncoding, FeatureWiseEmbeddingNorm
    from DS_TorchModule import ScaledDotProductAttention, MultiheadAttention, FeatureWiseTransformerEncoder, PreLN_TransformerEncoderLayer, AttentionPooling
except:
    remote_library_url = 'https://raw.githubusercontent.com/kimds929/'
    try:
        import httpimport
        with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
            from DS_MachineLearning import DataPreprocessing, DS_NoneEncoder, DS_StandardScaler, DS_LabelEncoder
            from DS_DeepLearning import EarlyStopping, TorchModeling
            from DS_AutoML import BayesOptLogger, ModelPerformanceEvaluation
            from DS_TorchModule import CategoricalEmbedding, EmbeddingLinear, ContinuousEmbeddingBlock, ResidualConnection
            from DS_TorchModule import PositionalEncoding, LearnablePositionalEncoding, FeatureWiseEmbeddingNorm
            from DS_TorchModule import ScaledDotProductAttention, MultiheadAttention, FeatureWiseTransformerEncoder, PreLN_TransformerEncoderLayer, AttentionPooling
    except:
        import requests
        
        response = requests.get(f"{remote_library_url}/DS_Library/main/DS_MachineLearning.py", verify=False)
        exec(response.text)
        
        response = requests.get(f"{remote_library_url}/DS_Library/main/DS_DeepLearning.py", verify=False)
        exec(response.text)
        
        response = requests.get(f"{remote_library_url}/DS_Library/main/DS_TorchModule.py", verify=False)
        exec(response.text)
        
        response = requests.get(f"{remote_library_url}/DS_Library/main/DS_AutoML.py", verify=False)
        exec(response.text)

# device
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f"*** torch device : {device}")

####################################################################################
dataset_meta = {'datasets_Boston_house':{'y_col':['Target']
                        ,'X_cols_con': ['AGE', 'B', 'RM', 'CRIM', 'DIS', 'INDUS', 'LSTAT', 'NOX', 'PTRATIO', 'ZN', 'TAX']
                        ,'X_cols_cat':['RAD', 'CHAS']
                        ,'X_cols_cat_n_class':[25, 2]
                        ,'stratify':None
                        }
                ,'datasets_Titanic_nomissing':{'y_col': ['survived']
                                              ,'X_cols_con': ['age', 'fare']
                        ,'X_cols_cat':['pclass', 'sex', 'age_missing', 'sibsp', 'parch', 'embarked']
                        ,'X_cols_cat_n_class':[3, 2, 2, 9, 10, 3]
                        ,'stratify':None
                        }
                ,'SampleData_980DP_YS_Modeling_mini':{'y_col':['YP']
                        ,'X_cols_con': ['주문두께', '소둔_폭',
                                        'SS_POS', 'RCS_POS','OAS_POS', 'SPM_RollForce_ST1', 'SPM_RollForce_ST2',
                                        'C_실적', 'Si_실적', 'Mn_실적', 'Ti_실적']
                        ,'X_cols_cat':['소둔공장','인장_방향', '인장_호수']
                        ,'X_cols_cat_n_class':[2,3]
                        ,'stratify':['소둔공장','인장_방향', '인장_호수']
                        }     
                }


########################################################################################################
folder_path =f"{base_path}/DataScience"
# db_path = f'{folder_path}/DataBase/Data_Tabular'
db_path = f'{folder_path}/DataBase/Data_Tabular'


# db_path = f'{folder_path}/DataBase/Data_Education'

# Classification : Personal_Loan, datasets_Titanic_original
# Regression : SampleData_980DP, datasets_Boston_house, datasets_Toyota
dataset_name = 'datasets_Boston_house'
# dataset_name = 'SampleData_980DP_YS_Modeling_mini'
# dataset_name = 'datasets_Titanic_nomissing'

y_col = dataset_meta[dataset_name]['y_col']
X_cols_con = dataset_meta[dataset_name]['X_cols_con']
X_cols_cat = dataset_meta[dataset_name]['X_cols_cat']
stratify_cols = dataset_meta[dataset_name]['stratify']
X_cols_cat_n_class = dataset_meta[dataset_name]['X_cols_cat_n_class']


df_load = pd.read_csv(f"{db_path}/{dataset_name}.csv", encoding='utf-8-sig')

y = df_load[y_col]
X_con = df_load[X_cols_con]
X_cat = df_load[X_cols_cat]
X_stratify = df_load[stratify_cols] if stratify_cols is not None else None



###################################################################################################

random_state = None
encoder = [DS_StandardScaler(), DS_StandardScaler(), DS_LabelEncoder()]
# encoder = [DS_LabelEncoder(), DS_StandardScaler(), DS_LabelEncoder()]
shuffle=True
batch_size = 64

rng = np.random.RandomState(random_state)

data_prepros = DataPreprocessing(y, X_con, X_cat,
                                encoder= encoder,
                                shuffle=shuffle, stratify=X_stratify, random_state=random_state, batch_size=batch_size)
data_prepros.fit_tensor_dataloader()
n_classes = list(data_prepros.transformed_data['train'][2].max(axis=0).astype(int))



###################################################################################################
mpe = ModelPerformanceEvaluation(data_preprocessor=data_prepros, target_task='regression',
                                # ml_bayes_opt=True,
                                ml_bayes_opt=False,
                                ml_models=['RandomForest', 'GradientBoosting', 'XGB', 'LGBM'],
                                ml_kwargs = {'n_iter':30},
                                verbose=2)
mpe.make_ml_dataset()
mpe.run_ml_learning()
# mpe.make_dl_dataset()
# mpe.dl_dataset
# mpe.run_dl_learning()

print( (pd.DataFrame(mpe.results).T)['test_loss'].apply(lambda x: np.sqrt(x)) )



################################################################################################################
import torch.optim as optim
# device
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f"*** torch device : {device}")




#################################################################################################################
# [FT-Transformers] #############################################################################################
# https://towardsdatascience.com/improving-tabtransformer-part-1-linear-numerical-embeddings-dbc3be3b5bb5/
class FT_Transformeres(nn.Module):
    def __init__(self, input_n_con, input_n_cat, hidden_dim=256, embedding_dim=32,
                nhead=4, encoder_dropout=0, fc_dropout=0.1, num_layers=1):
        super().__init__()
        
        self.con_embedding = EmbeddingLinear(input_n_con, 1, embedding_dim)
        self.cat_embedding = CategoricalEmbedding(input_n_cat, 1000, embedding_dim)
        self.cls_token = nn.Parameter(torch.randn(1,embedding_dim))
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead,
                                    dim_feedforward=embedding_dim*2, dropout=encoder_dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)             
        
        self.attn_pooling = AttentionPooling(embedding_dim)
        
        self.block = nn.Sequential(
            # nn.Linear(embedding_dim, hidden_dim)
            nn.Linear((1+input_n_con+input_n_cat)*embedding_dim, hidden_dim)
            ,nn.ReLU()
            ,ResidualConnection( nn.Sequential(
                # nn.BatchNorm1d(hidden_dim)
                nn.Linear(hidden_dim, hidden_dim)
                ,nn.ReLU()
                # ,nn.Dropout(fc_dropout)
                )
            )
            ,ResidualConnection( nn.Sequential(
                # nn.BatchNorm1d(hidden_dim)
                nn.Linear(hidden_dim, hidden_dim)
                ,nn.ReLU()
                # ,nn.Dropout(fc_dropout)
                ) 
            )
            ,nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x_con, x_cat):
        batch_size = x_cat.shape[0]
        x_con_embed = self.con_embedding(x_con.unsqueeze(-1))
        x_cat_embed = self.cat_embedding(x_cat)
        cls_token_expand = self.cls_token.unsqueeze(0).repeat(batch_size,1,1)
        
        x_concat = torch.cat([cls_token_expand, x_con_embed, x_cat_embed], dim=-2)
        
        x_enc_output = self.encoder(x_concat)
        
        # x_cls = x_enc_output[..., 0, :]
        # x_cls = x_enc_output.mean(dim=-2)   # mean pooling
        # x_cls, _ = x_enc_output.max(dim=-2)   # max pooling
        # x_cls, _ = self.attn_pooling(x_enc_output)
        x_cls = x_enc_output.view(*list(x_enc_output.shape[:-2]), -1)
        
        output = self.block(x_cls)
        return output
# -----------------------------------------------------------------------------------------------------

train_loader, valid_loader, tests_loader = data_prepros.tensor_dataloader.values()


model = FT_Transformeres(11, 2, hidden_dim=256).to(device)
# model(torch.rand(10, 11).to(device), torch.ones(10,2).to(torch.int64).to(device))
f"{sum([p.numel() for p in model.parameters() if p.requires_grad]):,}"


def loss_function_dnn(model, batch, optimizer=None):
    y, X_con, X_cat = batch
    pred = model(X_con, X_cat)
    loss = nn.functional.mse_loss(pred, y)
    return loss

tm0 = TorchModeling(model=model, device=device)
tm0.compile(optimizer = optim.Adam(model.parameters(), lr=5e-5)
            ,loss_function=loss_function_dnn
            ,early_stop_loss=EarlyStopping(min_iter=30, patience=50))
tm0.train_model(train_loader, valid_loader,
                epochs=500)

# tm6.set_best_model()
N_TEST = 10
test_loss0 = []
for _ in range(N_TEST):
    with torch.no_grad():
        test_loss = np.sqrt(tm0.test_model(tests_loader)['test_loss']).item()
    test_loss0.append(test_loss)
test_loss0 = sorted(test_loss0)

print(f"test_loss0 : {np.mean(test_loss0[3:-3]):.3f}")












#################################################################################################################
# [CustomTransformers] ##########################################################################################
# https://dongsarchive.tistory.com/74
class CustomTransformeres(nn.Module):
    def __init__(self, input_n_con, input_n_cat, hidden_dim=256, embedding_dim=32,
                nhead=4, encoder_dropout=0, fc_dropout=0.1, num_layers=1):
        super().__init__()
        
        self.con_embedding = EmbeddingLinear(input_n_con, 1, embedding_dim)
        self.cat_embedding = CategoricalEmbedding(input_n_cat, 1000, embedding_dim)
        self.cls_token = nn.Parameter(torch.randn(1,embedding_dim))
        
        self.featurewise_transformer_encoder = FeatureWiseTransformerEncoder(embedding_dim, nhead, 1+input_n_con+input_n_cat,
                                                                             qkv_projection=False, dim_feedforward=hidden_dim, batch_first=True)
        
        # self.attn_pooling = AttentionPooling(embedding_dim)
        
        self.block = nn.Sequential(
            # nn.Linear(embedding_dim, hidden_dim)  # pooling
            nn.Linear((1+input_n_con+input_n_cat)*embedding_dim, hidden_dim)    # flatten
            ,nn.ReLU()
            ,ResidualConnection( nn.Sequential(
                nn.BatchNorm1d(hidden_dim)
                ,nn.Linear(hidden_dim, hidden_dim)
                ,nn.ReLU()
                ,nn.Dropout(fc_dropout)
                )
            )
            ,ResidualConnection( nn.Sequential(
                nn.BatchNorm1d(hidden_dim)
                ,nn.Linear(hidden_dim, hidden_dim)
                ,nn.ReLU()
                ,nn.Dropout(fc_dropout)
                ) 
            )
            ,nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x_con, x_cat):
        batch_size = x_cat.shape[0]
        x_con_embed = self.con_embedding(x_con.unsqueeze(-1))
        x_cat_embed = self.cat_embedding(x_cat)
        cls_token_expand = self.cls_token.unsqueeze(0).repeat(batch_size,1,1)
        
        x_concat = torch.cat([cls_token_expand, x_con_embed, x_cat_embed], dim=-2)
        x_enc_output = self.featurewise_transformer_encoder(x_concat)
        
        # x_cls = x_enc_output[..., 0, :]
        # x_cls = x_enc_output.mean(dim=-2)   # mean pooling
        # x_cls, _ = x_enc_output.max(dim=-2)   # max pooling
        # x_cls, _ = self.attn_pooling(x_enc_output)
        x_cls = x_enc_output.view(*list(x_enc_output.shape[:-2]), -1)
        
        output = self.block(x_cls)
        return output
# -----------------------------------------------------------------------------------------------------
train_loader, valid_loader, tests_loader = data_prepros.tensor_dataloader.values()


model = CustomTransformeres(11, 2, hidden_dim=196).to(device)
# model(torch.rand(10, 11).to(device), torch.ones(10,3).to(torch.int64).to(device))
f"{sum([p.numel() for p in model.parameters() if p.requires_grad]):,}"


def loss_function_dnn(model, batch, optimizer=None):
    y, X_con, X_cat = batch
    pred = model(X_con, X_cat)
    loss = nn.functional.mse_loss(pred, y)
    return loss

tm1 = TorchModeling(model=model, device=device)
tm1.compile(optimizer = optim.Adam(model.parameters(), lr=5e-5)
            ,loss_function=loss_function_dnn
            ,early_stop_loss=EarlyStopping(min_iter=30, patience=50))
tm1.train_model(train_loader, valid_loader,
                epochs=500)

# tm6.set_best_model()
N_TEST = 10
test_loss1 = []
for _ in range(N_TEST):
    with torch.no_grad():
        test_loss = np.sqrt(tm1.test_model(tests_loader)['test_loss']).item()
    test_loss1.append(test_loss)
test_loss1 = sorted(test_loss1)

print(f"test_loss1 : {np.mean(test_loss1[3:-3]):.3f}")






#################################################################################################################   
# [SAINT] #######################################################################################################
class IntersampleTransformerEncoderLayer(nn.Module):
    def __init__(self, n_tokens: int, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1, batch_first: bool = False):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        
        self.layer_norm1 = nn.LayerNorm(n_tokens*d_model)
        
        self.self_attn = MultiheadAttention(n_tokens*d_model, nhead, dropout=dropout, batch_first=batch_first, qkv_projection=True)
        self.droptout1 = nn.Dropout(dropout)
        
        self.ff_layer = nn.Sequential(
            nn.LayerNorm(n_tokens * d_model),
            nn.Linear(n_tokens * d_model, dim_feedforward),    # FF_linear1
            nn.GELU(),
            nn.Linear(dim_feedforward, n_tokens * d_model),    # FF_linear2
            nn.Dropout(dropout)
        )

    def forward(self,
            src: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            is_causal: bool = False,
            src_key_padding_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        # x: (B,F,E)
        B, F, E = src.shape
        assert F == self.n_tokens and E == self.d_model

        # 각 행(row)을 하나의 벡터로 flatten
        src_flat = src.reshape(1, B, F * E)          # (B, F*E)
        
        # layer_norm
        out = self.layer_norm1(src_flat)

        # Intersample MultiheadAttention
        mha_out, _ = self.self_attn(out, out, out)  # (1,B,E)
        
        # dropout + residual connection
        out = out + self.droptout1(mha_out)
        
        # feedforward + residual connection
        out = out + self.ff_layer(out)                    # (1,B,E)

        # 다시 (B,d)로 만들고, 각 행의 모든 토큰에 브로드캐스트
        out = out.reshape(B, F, E)  # (B,F,E)
        return out


class SAINT(nn.Module):
    def __init__(self, input_n_con, input_n_cat, hidden_dim=256, embedding_dim=32,
                nhead=4, encoder_dropout=0, fc_dropout=0.1, num_layers=1):
        super().__init__()
        self.n_token = input_n_con + input_n_cat + 1
        
        self.con_embedding = EmbeddingLinear(input_n_con, 1, embedding_dim)
        self.cat_embedding = CategoricalEmbedding(input_n_cat, 1000, embedding_dim)
        self.cls_token = nn.Parameter(torch.randn(1, embedding_dim))
        
        
        # self attention encoder
        self.self_attn_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead,
                                    dim_feedforward=embedding_dim*2, dropout=encoder_dropout, batch_first=True)
        self.self_attn_encoder = nn.TransformerEncoder(self.self_attn_encoder_layer, num_layers=num_layers)  
        
        # intersample attention encoder
        self.intersample_attn_encoder_layer = IntersampleTransformerEncoderLayer(n_tokens=self.n_token, d_model=embedding_dim, nhead=nhead,
                                    dim_feedforward=embedding_dim*2, dropout=encoder_dropout, batch_first=True)
        self.intersample_attn_encoder = nn.TransformerEncoder(self.intersample_attn_encoder_layer, num_layers=num_layers)
        
        # attention pooling
        self.attn_pooling = AttentionPooling(embedding_dim)
        
        self.block = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim)
            # nn.Linear((1+input_n_con+input_n_cat)*embedding_dim, hidden_dim)
            ,nn.ReLU()
            ,ResidualConnection( nn.Sequential(
                # nn.BatchNorm1d(hidden_dim)
                nn.Linear(hidden_dim, hidden_dim)
                ,nn.ReLU()
                # ,nn.Dropout(fc_dropout)
                )
            )
            ,ResidualConnection( nn.Sequential(
                # nn.BatchNorm1d(hidden_dim)
                nn.Linear(hidden_dim, hidden_dim)
                ,nn.ReLU()
                # ,nn.Dropout(fc_dropout)
                ) 
            )
            ,nn.Linear(hidden_dim, 1)
        )
        
    def forward_embedding(self, x_con:Optional[torch.Tensor], x_cat:Optional[torch.Tensor]):
        # embedding 
        batch_size = x_cat.shape[0]
        x_con_embed = self.con_embedding(x_con.unsqueeze(-1))
        x_cat_embed = self.cat_embedding(x_cat)
        cls_token_expand = self.cls_token.unsqueeze(0).repeat(batch_size,1,1)
        
        x_concat_embed = torch.cat([cls_token_expand, x_con_embed, x_cat_embed], dim=-2)
        return x_concat_embed
    
    def forward_encoding(self, x_embed: torch.Tensor):
        # x_selfattn_encoder
        x_self_attn_enc_output = self.self_attn_encoder(x_embed)
        
        # x_intersampleattn_encoder
        x_enc_output = self.intersample_attn_encoder(x_self_attn_enc_output)
        
        return x_enc_output
        
    def forward_cls(self, x_enc_output:torch.Tensor):
        # cls token
        x_cls = x_enc_output[..., 0, :]     # CLS token
        # x_cls = x_enc_output.mean(dim=-2)   # mean pooling
        # x_cls, _ = x_enc_output.max(dim=-2)   # max pooling
        # x_cls, _ = self.attn_pooling(x_enc_output)
        # x_cls = x_enc_output.view(*list(x_enc_output.shape[:-2]), -1)
        
        return x_cls
    
    def forward(self, x_con:Optional[torch.Tensor], x_cat:Optional[torch.Tensor]):
        # embedding
        x_concat_embed = self.forward_embedding(x_con, x_cat)
        
        # encoding
        x_enc_output = self.forward_encoding(x_concat_embed)
        
        # cls
        x_cls = self.forward_cls(x_enc_output)
        
        # target head         
        output = self.block(x_cls)
        return output

# -----------------------------------------------------------------------------------------------------
train_loader, valid_loader, tests_loader = data_prepros.tensor_dataloader.values()

# for batch in train_loader:
#     break

model = SAINT(11, 2, hidden_dim=128).to(device)
# model(torch.rand(10, 11).to(device), torch.ones(10,3).to(torch.int64).to(device))
f"{sum([p.numel() for p in model.parameters() if p.requires_grad]):,}"


def loss_function_dnn(model, batch, optimizer=None):
    y, X_con, X_cat = batch
    pred = model(X_con, X_cat)
    loss = nn.functional.mse_loss(pred, y)
    return loss

tm2 = TorchModeling(model=model, device=device)
tm2.compile(optimizer = optim.Adam(model.parameters(), lr=5e-5)
            ,loss_function=loss_function_dnn
            ,early_stop_loss=EarlyStopping(min_iter=30, patience=50))
tm2.train_model(train_loader, valid_loader,
                epochs=500)

# tm6.set_best_model()
N_TEST = 10
test_loss2 = []
for _ in range(N_TEST):
    with torch.no_grad():
        test_loss = np.sqrt(tm2.test_model(tests_loader)['test_loss']).item()
    test_loss2.append(test_loss)
test_loss2 = sorted(test_loss2)

print(f"test_loss2 : {np.mean(test_loss2[3:-3]):.3f}")

















#################################################################################################################   
# CutMix in raw space
def cutmix(x, sample, p):
    mask = (torch.rand_like(x.float()) < p).to(x.dtype)
    x_cm = torch.where(mask.bool(), x, sample)
    return x_cm

def cutmix_raw(x_cont, x_cat, p=0.3):
    """
    feature-wise CutMix:
    x' = x ⊙ m + x_perm ⊙ (1-m)
    - continuous: mask per continuous feature
    - categorical: mask per categorical feature
    """
    # 열 단위 베르누이 마스크로 다른 샘플의 값을 섞음
    B = x_cont.size(0) if x_cont is not None else x_cat.size(0)
    perm = torch.randperm(B, device=x_cont.device if x_cont is not None else x_cat.device)

    x_cont_cm = cutmix(x_cont, x_cont[perm], p) if x_cont is not None else None
    # m_cont = (torch.rand_like(x_cont) < p).float()
    # x_cont_cm = m_cont * x_cont  + (1 - m_cont) * x_cont[perm]
    
    x_cat_cm = cutmix(x_cat, x_cat[perm], p) if x_cat is not None else None
    # m_cat = (torch.rand_like(x_cat.float()) < p).long()
    # x_cat_cm = torch.where(m_cat.bool(), x_cat[perm], x_cat)
    return x_cont_cm, x_cat_cm

# ------------------------------------------------------------------------------------------------------------
def mixup_embedding(p1, p2, alpha=0.2):
    lam = alpha
    return lam * p1 + (1 - lam) * p2

# ------------------------------------------------------------------------------------------------------------
# Projection
class ProjectionHead(nn.Module):
    def __init__(self, d_model: int, d_proj: int=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_proj),
                                 nn.ReLU(),
                                 nn.Linear(d_proj, d_proj))

    def forward(self, z):  # z: (B,F,E) -> d_proj
        return self.net(z)  # (B, d_proj)

# ------------------------------------------------------------------------------------------------------------
# Contrasive : InfoNCE
def info_nce(z1, z2, tau=0.7):
    """
    자기자신의 유사도만을 1로 높이는 방식 (나머지와의 유사도는 0으로)
    z1, z2: (B,proj_dim)
    positive pairs: (i,i)
    negatives: (i,k), k!=i within batch
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    
    logits = (z1 @ z2.T) / tau  # (B,B)
    labels = torch.arange(z1.size(0), device=z1.device)
    loss = F.cross_entropy(logits, labels)  # i,i : 1 | i, j : 0
    return loss

# ------------------------------------------------------------------------------------------------------------
class DenoisingHead(nn.Module):
    def __init__(self, n_cont: int, n_cat: int, cat_cardinals: List[int], d_model: int, hidden: int = 64, reduction: str = "mean", normalize_by_feature_count: bool = True):
        super().__init__()
        assert n_cat == len(cat_cardinals), "n_cat must match len(cat_cardinals)"
        assert reduction in ["mean", "sum"], "reduction must be 'mean' or 'sum' | 'mean' 권장"
        
        self.n_cont = n_cont
        self.n_cat = n_cat
        self.cat_cardinals = cat_cardinals
        self.reduction = reduction
        self.normalize_by_feature_count = normalize_by_feature_count
        
        # feature-wise decoders
        self.mlps_cont = nn.Sequential(
                    EmbeddingLinear(n_cont, d_model, hidden),
                    nn.ReLU(),
                    EmbeddingLinear(n_cont, hidden, 1)
                    ) 
        
        self.mlps_cat = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.ReLU(),
                nn.Linear(hidden, card)
            ) for card in cat_cardinals
        ])
        
    def forward(self, noise_token, x_con, x_cat):

        # noise_token       # (B,T,E)
        # x_con             # (B,n_cont)
        # x_cat             # (B,n_cat)
        B, T, E = noise_token.shape
        expected_T = 1 + self.n_cont + self.n_cat
        assert T == expected_T, f"noise_token length mismatch: got {T}, expected {expected_T}"
        
        pred_returns = []
        # ---- continuous reconstruction ----
        if self.n_cont > 0:
            assert x_con is not None, "x_con is required when n_cont > 0"
            
            base = 1        # 0 : CLS
            noise_token_cont = noise_token[:, base:base+self.n_cont, :]     # (B,n_cont,E)
            pred_cont = self.mlps_cont(noise_token_cont).squeeze(-1)
            pred_returns.append(pred_cont)
        else:
            pred_returns.append(None)
        
        if self.n_cat > 0:
            assert x_cat is not None, "x_cat is required when n_cat > 0"
            
            base = 1 + self.n_cont
            pred_cat = []
            for j in range(self.n_cat):
                h = noise_token[:, base + j, :]                         # (B,E)
                pred_logits = self.mlps_cat[j](h)                       # (B,card)
                pred_cat.append(pred_logits)
            pred_returns.append(pred_cat)
        else:
            pred_returns.append(None)
        return tuple(pred_returns)

    def denoise_loss(self, noise_token, x_con, x_cat):
        pred_cont, pred_cat = self.forward(noise_token, x_con, x_cat)
        
        loss_cont = noise_token.new_tensor(0.0)
        loss_cat = noise_token.new_tensor(0.0)
        if pred_cont is not None:
            loss_cont += F.mse_loss(pred_cont, x_con)
        
        if pred_cat is not None:
            for j in range(self.n_cat):
                loss_cat += F.cross_entropy(pred_cat[j], x_cat[:,j])
            loss_cat /= self.n_cat
        return loss_cont + loss_cat

#################################################################################################################   

# -----------------------------------------------------------------------------------------------------------------
# SAINT : Pretraing 
# train_loader, valid_loader, tests_loader = data_prepros.tensor_dataloader.values()
# n_classes = list(data_prepros.transformed_data['train'][2].max(axis=0).astype(int))


model = SAINT(input_n_con=11, input_n_cat=2, hidden_dim=128).to(device)
prj_head_clean = ProjectionHead(d_model=32, d_proj=32).to(device)
prj_head_noise = ProjectionHead(d_model=32, d_proj=32).to(device)
denoise_head = DenoisingHead(n_cont=11, n_cat=2, cat_cardinals=n_classes+1, d_model=32, hidden=64).to(device)

pretrain_optimizer = optim.Adam(list(model.parameters()) +
                 list(prj_head_clean.parameters()) +
                 list(prj_head_noise.parameters()) +
                 list(denoise_head.parameters()), lr=1e-3)

p_cutmix = 0.3
alpha = 0.2
tau = 0.7
lambda_pt = 10.0

def loss_function_SAINT_pretrain(model, batch, optimizer=None):
    y, X_con, X_cat = batch
    B = y.size(0)
    
    # 1) CutMix
    xcm_cont, xcm_cat = cutmix_raw(X_con, X_cat, p=p_cutmix)
    
    # 2) Embedding
    x_clean = model.forward_embedding(X_con, X_cat)
    x_cutmix = model.forward_embedding(xcm_cont, xcm_cat)
    
    # 3) Mixup in embedding space
    perm = torch.randperm(B, device=device)
    x_cutmix_perm = x_cutmix[perm]
    x_noise = mixup_embedding(x_cutmix, x_cutmix_perm, alpha=alpha)
    
    # 4) Encoding
    token_clean = model.forward_encoding(x_clean)
    token_noise = model.forward_encoding(x_noise)
    
    cls_clean = model.forward_cls(token_clean)
    cls_noise = model.forward_cls(token_noise)
    
     # 5) Projection
    z_clean = prj_head_clean(cls_clean)
    z_noise = prj_head_noise(cls_noise)
    
    # 6) Contrastive loss
    loss_contrast = info_nce(z_clean, z_noise, tau=tau)

    # 7) Denoising loss
    loss_denoise = denoise_head.denoise_loss(token_noise, X_con, X_cat)
    
    # 8) Final loss
    loss = loss_contrast + lambda_pt * loss_denoise
    
    return loss

tm_pretrain = TorchModeling(model=model, device=device)
tm_pretrain.compile(optimizer = pretrain_optimizer
            ,loss_function = loss_function_SAINT_pretrain
            ,early_stop_loss=EarlyStopping(min_iter=30, patience=50))
tm_pretrain.train_model(train_loader, valid_loader,
                epochs=500)

tm_pretrain.early_stop_loss

# -----------------------------------------------------------------------------------------------------------------
# Fine Tunning
def loss_function_dnn(model, batch, optimizer=None):
    y, X_con, X_cat = batch
    pred = model(X_con, X_cat)
    loss = nn.functional.mse_loss(pred, y)
    return loss

tm_finetunning = TorchModeling(model=model, device=device)
tm_finetunning.compile(optimizer = optim.Adam(model.parameters(), lr=5e-5)
            ,loss_function=loss_function_dnn
            ,early_stop_loss=EarlyStopping(min_iter=30, patience=50))
tm_finetunning.train_model(train_loader, valid_loader,
                epochs=1000)

# tm6.set_best_model()
N_TEST = 10
test_loss_saint = []
for _ in range(N_TEST):
    with torch.no_grad():
        test_loss = np.sqrt(tm_finetunning.test_model(tests_loader)['test_loss']).item()
    test_loss_saint.append(test_loss)
test_loss_saint = sorted(test_loss_saint)

print(f"test_loss_saint : {np.mean(test_loss_saint[3:-3]):.3f}")








































#################################################################################################################   

# SAINT : Pretraing Detail
train_loader, valid_loader, tests_loader = data_prepros.tensor_dataloader.values()
n_classes = data_prepros.transformed_data['train'][2].max(axis=0)


# ----------------------------
# 6) 사전학습: CutMix + Mixup + InfoNCE + 복원
# ----------------------------
model = SAINT(11, 2, hidden_dim=128).to(device)
# device='cpu'

for batch in train_loader:
    break

batch_cont = batch[1].to(device)
batch_cat = batch[2].to(device)



p_cutmix: float = 0.3
alpha: float = 0.2
tau: float = 0.7
lambda_pt: float = 10.0


B = batch_cont.size(0) if batch_cont is not None else batch_cat.size(0)
device = batch_cont.device if batch_cont is not None else batch_cat.device

# -----------------------------------------------------------------------------------------------------------------------------
# 1) CutMix in raw space (Eq. 3: x'_i = m * x_i + (1-m) * x_a)
def cutmix(x, sample, p):
    mask = (torch.rand_like(x.float()) < p).to(x.dtype)
    x_cm = torch.where(mask.bool(), x, sample)
    return x_cm

def cutmix_raw(x_cont, x_cat, p=0.3):
    """
    feature-wise CutMix:
    x' = x ⊙ m + x_perm ⊙ (1-m)
    - continuous: mask per continuous feature
    - categorical: mask per categorical feature
    """
    # 열 단위 베르누이 마스크로 다른 샘플의 값을 섞음
    B = x_cont.size(0) if x_cont is not None else x_cat.size(0)
    perm = torch.randperm(B, device=x_cont.device if x_cont is not None else x_cat.device)

    x_cont_cm = cutmix(x_cont, x_cont[perm], p) if x_cont is not None else None
    # m_cont = (torch.rand_like(x_cont) < p).float()
    # x_cont_cm = m_cont * x_cont  + (1 - m_cont) * x_cont[perm]
    
    x_cat_cm = cutmix(x_cat, x_cat[perm], p) if x_cat is not None else None
    # m_cat = (torch.rand_like(x_cat.float()) < p).long()
    # x_cat_cm = torch.where(m_cat.bool(), x_cat[perm], x_cat)
    return x_cont_cm, x_cat_cm

def mixup_embedding(p1, p2, alpha=0.2):
    lam = alpha
    return lam * p1 + (1 - lam) * p2

xcm_cont, xcm_cat = cutmix_raw(batch_cont, batch_cat, p=p_cutmix)

# ------------------------------------------------------------------------------------------------------------------------------
# 2) Embedding clean & CutMix-ed (gradient 흐르게!)
x_clean = model.forward_embedding(batch_cont, batch_cat)      # (B,F,E)
x_cutmix = model.forward_embedding(xcm_cont, xcm_cat)          # (B,F,E) : cutmix

# ------------------------------------------------------------------------------------------------------------------------------
# 3) Mixup in embedding space (Eq. 4: p'_i = α E(x'_i) + (1-α) E(x'_b))
perm = torch.randperm(B, device=device)
x_cutmix_perm = x_cutmix[perm]                          # (B,F,E)
x_noise = mixup_embedding(x_cutmix, x_cutmix_perm, alpha=alpha)  # (B,F,E) : cutmix + mixup

# ------------------------------------------------------------------------------------------------------------------------------
# 4) SAINT backbone 통과 (clean / view 각각)
token_clean = model.forward_encoding(x_clean)       # (B,F,E) = r_i
token_noise = model.forward_encoding(x_noise)       # (B,F,E) = r'_i

cls_clean = model.forward_cls(token_clean)  # (B,E)
cls_noise  = model.forward_cls(token_noise)   # (B,E)

# ------------------------------------------------------------------------------------------------------------------------------
# 5) Projection
class ProjectionHead(nn.Module):
    def __init__(self, d_model: int, d_proj: int=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_proj),
                                 nn.ReLU(),
                                 nn.Linear(d_proj, d_proj))

    def forward(self, z):  # z: (B,F,E) -> d_proj
        return self.net(z)  # (B, d_proj)

prj_head_clean = ProjectionHead(d_model=32, d_proj=32).to(device)  # (B, latent)
prj_head_noise = ProjectionHead(d_model=32, d_proj=32).to(device)   # (B, latent)

z_clean = prj_head_clean(cls_clean)
z_noise = prj_head_noise(cls_noise)

# ------------------------------------------------------------------------------------------------------------------------------
# 6) Contrasive : InfoNCE
def info_nce(z1, z2, tau=0.7):
    """
    자기자신의 유사도만을 1로 높이는 방식 (나머지와의 유사도는 0으로)
    z1, z2: (B,proj_dim)
    positive pairs: (i,i)
    negatives: (i,k), k!=i within batch
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    
    logits = (z1 @ z2.T) / tau  # (B,B)
    labels = torch.arange(z1.size(0), device=z1.device)
    loss = F.cross_entropy(logits, labels)  # i,i : 1 | i, j : 0
    return loss

loss_contrast = info_nce(z_clean, z_noise, tau=tau)

# ------------------------------------------------------------------------------------------------------------------------------
# 7) Denoising: noisy view(r'_i)로부터 원본 feature 복원 (Eq. 5 두 번째 항)

class DenoisingHead(nn.Module):
    def __init__(self, n_cont: int, n_cat: int, cat_cardinals: List[int], d_model: int, hidden: int = 64, reduction: str = "mean", normalize_by_feature_count: bool = True):
        super().__init__()
        assert n_cat == len(cat_cardinals), "n_cat must match len(cat_cardinals)"
        assert reduction in ["mean", "sum"], "reduction must be 'mean' or 'sum' | 'mean' 권장"
        
        self.n_cont = n_cont
        self.n_cat = n_cat
        self.cat_cardinals = cat_cardinals
        self.reduction = reduction
        self.normalize_by_feature_count = normalize_by_feature_count
        
        # feature-wise decoders
        self.mlps_cont = nn.Sequential(
                    EmbeddingLinear(n_cont, d_model, hidden),
                    nn.ReLU(),
                    EmbeddingLinear(n_cont, hidden, 1)
                    ) 
        
        self.mlps_cat = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.ReLU(),
                nn.Linear(hidden, card)
            ) for card in cat_cardinals
        ])
        
    def forward(self, noise_token, x_con, x_cat):

        # noise_token       # (B,T,E)
        # x_con             # (B,n_cont)
        # x_cat             # (B,n_cat)
        B, T, E = noise_token.shape
        expected_T = 1 + self.n_cont + self.n_cat
        assert T == expected_T, f"noise_token length mismatch: got {T}, expected {expected_T}"
        
        pred_returns = []
        # ---- continuous reconstruction ----
        if self.n_cont > 0:
            assert x_con is not None, "x_con is required when n_cont > 0"
            
            base = 1        # 0 : CLS
            noise_token_cont = noise_token[:, base:base+self.n_cont, :]     # (B,n_cont,E)
            pred_cont = self.mlps_cont(noise_token_cont).squeeze(-1)
            pred_returns.append(pred_cont)
        else:
            pred_returns.append(None)
        
        if self.n_cat > 0:
            assert x_cat is not None, "x_cat is required when n_cat > 0"
            
            base = 1 + self.n_cont
            pred_cat = []
            for j in range(self.n_cat):
                h = noise_token[:, base + j, :]                         # (B,E)
                pred_logits = self.mlps_cat[j](h)                       # (B,card)
                pred_cat.append(pred_logits)
            pred_returns.append(pred_cat)
        else:
            pred_returns.append(None)
        return tuple(pred_returns)

    def denoise_loss(self, noise_token, x_con, x_cat):
        pred_cont, pred_cat = self.forward(noise_token, x_con, x_cat)
        
        loss_cont = noise_token.new_tensor(0.0)
        loss_cat = noise_token.new_tensor(0.0)
        if pred_cont is not None:
            loss_cont += F.mse_loss(pred_cont, batch_cont)
        
        if pred_cat is not None:
            for j in range(self.n_cat):
                loss_cat += F.cross_entropy(pred_cat[j], x_cat[:,j])
            loss_cat /= self.n_cat
        return loss_cont + loss_cat

denoise_head = DenoisingHead(n_cont=11, n_cat=2, cat_cardinals=n_classes+1, d_model=32, hidden=64).to(device)
loss_denoise = denoise_head.denoise_loss(token_noise, batch_cont, batch_cat)


# ------------------------------------------------------------------------------------------------------------------------------
# 8) 최종 loss
loss = loss_contrast + lambda_pt * loss_denoise





#################################################################################################################   




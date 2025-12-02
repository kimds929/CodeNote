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
from torch.utils.data import TensorDataset, DataLoader


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy
import missingno as msno
from typing import Optional, Tuple

from DS_Basic_Module import DF_Summary, ScalerEncoder


try:
    from DS_Torch import TorchDataLoader, TorchModeling, AutoML
    from DS_MachineLearning import DataPreprocessing, DS_NoneEncoder, DS_StandardEncoder, DS_LabelEncoder
    from DS_DeepLearning import EarlyStopping
    # from DS_TorchModule import CategoricalEmbedding, EmbeddingLinear, ContinuousEmbeddingBlock
    from DS_TorchModule import CategoricalEmbedding, EmbeddingLinear, ContinuousEmbeddingBlock
    from DS_TorchModule import PositionalEncoding, LearnablePositionalEncoding, FeatureWiseEmbeddingNorm
    from DS_TorchModule import ScaledDotProductAttention, MultiheadAttention, PreLN_TransformerEncoderLayer, AttentionPooling
except:
    remote_library_url = 'https://raw.githubusercontent.com/kimds929/'
    try:
        import httpimport
        with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
            from DS_Torch import TorchDataLoader, TorchModeling, AutoML
            from DS_MachineLearning import DataPreprocessing, DS_NoneEncoder, DS_StandardEncoder, DS_LabelEncoder
            from DS_DeepLearning import EarlyStopping
            from DS_TorchModule import CategoricalEmbedding, EmbeddingLinear, ContinuousEmbeddingBlock, PositionalEncoding, LearnablePositionalEncoding, ScaledDotProductAttention
    except:
        import requests
        response = requests.get(f"{remote_library_url}/DS_Library/main/DS_Torch.py", verify=False)
        exec(response.text)
        
        response = requests.get(f"{remote_library_url}/DS_Library/main/DS_MachineLearning.py", verify=False)
        exec(response.text)
        
        response = requests.get(f"{remote_library_url}/DS_Library/main/DS_DeepLearning.py", verify=False)
        exec(response.text)
        
        response = requests.get(f"{remote_library_url}/DS_Library/main/DS_TorchModule.py", verify=False)
        exec(response.text)
# device
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f"*** torch device : {device}")


########################################################################################################
execute = False
load_dataset = True
# load_dataset = False

########################################################################################################
# db_path = f'{folder_path}/DataBase/Data_Tabular'
db_path = f'{folder_path}/DataBase/Data_Education'

# Classification : Personal_Loan, datasets_Titanic_original
# Regression : SampleData_980DP, datasets_Boston_house, datasets_Toyota
dataset_name = 'SampleData_980DP_YS_Modeling_mini'      #  SampleData_980DP_YS_Modeling, SampleData_980DP_YS_Modeling_mini datasets_Boston_house, datasets_Toyota

# --------------------------------------------------------------------------------------------------------------

dataset_meta = {'datasets_Boston_house':{'y_col':['Target']
                        ,'X_cols_con': ['AGE', 'B', 'RM', 'CRIM', 'DIS', 'INDUS', 'LSTAT', 'NOX', 'PTRATIO', 'ZN', 'TAX']
                        ,'X_cols_cat':['RAD', 'CHAS']
                        ,'X_cols_cat_n_class':[25, 2]
                        ,'stratify':None
                        }
                ,'SampleData_980DP_YS_Modeling':{'y_col':['YP']
                        ,'X_cols_con': ['주문두께', '소둔_폭',
                                        'LS_POS', 'HS_POS', 'SS_POS', 'SCS_POS', 'RCS_POS','OAS_POS', 'FCS_POS', 'SPM_RollForce_ST1', 'SPM_RollForce_ST2',
                                        'SRT', 'FDT', 'CT', 
                                        'C_실적', 'Si_실적', 'Mn_실적', 'P_실적', 'S_실적', 'SolAl_실적', 'TotAl_실적', 'Cu_실적', 'Nb_실적',
                                        'B_실적', 'Ni_실적', 'Cr_실적', 'Mo_실적', 'Ti_실적', 'V_실적', 'Sn_실적', 'Ca_실적', 'Sb_실적', 'N_실적']
                        ,'X_cols_cat':['인장_방향', '인장_호수']
                        ,'X_cols_cat_n_class':[5,2,2,3]
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




y_col = dataset_meta[dataset_name]['y_col']
X_cols_con = dataset_meta[dataset_name]['X_cols_con']
X_cols_cat = dataset_meta[dataset_name]['X_cols_cat']
stratify_cols = dataset_meta[dataset_name]['stratify']
X_cols_cat_n_class = dataset_meta[dataset_name]['X_cols_cat_n_class']



########################################################################################################
BATCH_SIZE = 128
# DS_NoneEncoder, DS_StandardEncoder, DS_LabelEncoder

if load_dataset is False:
    df_load = pd.read_csv(f"{db_path}/{dataset_name}.csv", encoding='utf-8-sig')
    df_load.info()
    df_load.head()
    msno.matrix(df_load, labels=list(df_load.columns))
    
    # df_load.columns
    df_summary = DF_Summary(df_load[y_col + X_cols_con + X_cols_cat])
    
    # DataPreprocessing
    data_preprocessing = DataPreprocessing(df_load[y_col], df_load[X_cols_con], df_load[X_cols_cat],
                                        encoder=[DS_StandardEncoder(), DS_StandardEncoder(), DS_LabelEncoder()],
                                        shuffle=True, stratify=df_load[stratify_cols])
    
else:
    def data_split(dataset, *cols_args):
        return [dataset[cols] for cols in cols_args]
    
    dataset = pd.read_csv(f"{db_path}/{dataset_name}_modeling.csv", encoding='utf-8-sig')
    dataset_y, dataset_X_con, dataset_X_cat, = data_split(dataset, y_col, X_cols_con, X_cols_cat)
    
    dataset_index = dataset.groupby('Train_Valid_Tests').apply(lambda x: np.array(x.index)).to_dict()
    dataset_index
    # DataPreprocessing
    data_preprocessing = DataPreprocessing(dataset_y, dataset_X_con, dataset_X_cat,
                                           index=dataset_index,
                                           encoder=[DS_StandardEncoder(), DS_StandardEncoder(), DS_LabelEncoder()],
                                           shuffle=True, batch_size=BATCH_SIZE)
    
    print('*** load dataset.')
########################################################################################################

# data_preprocessing.encoding()
# data_preprocessing.make_tensor_dataset()
# data_preprocessing.make_tensor_dataloader(batch_size=BATCH_SIZE, shuffle=True, random_state=None)

data_preprocessing.fit_tensor_dataloader()
train_loader, valid_loader, tests_loader = data_preprocessing.tensor_dataloader.values()


# data_preprocessing.encoder[0].inverse_transform(data_preprocessing.transformed_data['train'][0])
# data_preprocessing.encoder[1].inverse_transform(data_preprocessing.transformed_data['train'][1])
# data_preprocessing.encoder[2].inverse_transform(data_preprocessing.transformed_data['train'][2])

####################################################################################################
####################################################################################################
####################################################################################################


from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

train_y_norm = data_preprocessing.transformed_data['train'][0]
valid_y_norm = data_preprocessing.transformed_data['valid'][0]
tests_y_norm = data_preprocessing.transformed_data['test'][0]

train_X_concat_norm = np.concatenate(data_preprocessing.transformed_data['train'][1:], axis=1)
valid_X_concat_norm = np.concatenate(data_preprocessing.transformed_data['valid'][1:], axis=1)
tests_X_concat_norm = np.concatenate(data_preprocessing.transformed_data['test'][1:], axis=1)

tests_y = data_preprocessing.encoder[0].inverse_transform(tests_y_norm)

# ---------------------------------------------------------------------------
# RandomForest
RF = RandomForestRegressor()
RF.fit(train_X_concat_norm, train_y_norm)

RF_pred = RF.predict(train_X_concat_norm)
np.sqrt(mean_squared_error(train_y_norm, RF_pred))


RF_pred_tests = RF.predict(tests_X_concat_norm)
RF_rmse_tests = np.sqrt(mean_squared_error(tests_y_norm, RF_pred_tests))
RF_pred_tests_origin = data_preprocessing.encoder[0].inverse_transform(RF_pred_tests.reshape(-1,1)).ravel()
RF_rmse_tests_origin = np.sqrt(mean_squared_error(np.array(tests_y).ravel(), RF_pred_tests_origin))
print(f"RF_rmse_tests : {RF_rmse_tests:.3f}")
print(f"RF_rmse_tests_origin : {RF_rmse_tests_origin:.3f}")



# ---------------------------------------------------------------------------

# GradientBoosting
GB = GradientBoostingRegressor()
GB.fit(train_X_concat_norm, train_y_norm)

GB_pred = GB.predict(train_X_concat_norm)
np.sqrt(mean_squared_error(train_y_norm, GB_pred))


GB_pred_tests = GB.predict(tests_X_concat_norm)
GB_rmse_tests = np.sqrt(mean_squared_error(tests_y_norm, GB_pred_tests))
GB_pred_tests_origin = data_preprocessing.encoder[0].inverse_transform(GB_pred_tests.reshape(-1,1)).ravel()
GB_rmse_tests_origin = np.sqrt(mean_squared_error(np.array(tests_y).ravel(), GB_pred_tests_origin))
print(f"GB_rmse_tests : {GB_rmse_tests:.3f}")
print(f"GB_rmse_tests_origin : {GB_rmse_tests_origin:.3f}")
# ---------------------------------------------------------------------------

# GradientBoosting
XGB = XGBRegressor()
XGB.fit(train_X_concat_norm, train_y_norm)

XGB_pred = XGB.predict(train_X_concat_norm)
np.sqrt(mean_squared_error(train_y_norm, XGB_pred))


XGB_pred_tests = XGB.predict(tests_X_concat_norm)
XGB_rmse_tests = np.sqrt(mean_squared_error(tests_y_norm, XGB_pred_tests))
XGB_pred_tests_origin = data_preprocessing.encoder[0].inverse_transform(XGB_pred_tests.reshape(-1,1)).ravel()
XGB_rmse_tests_origin = np.sqrt(mean_squared_error(np.array(tests_y).ravel(), XGB_pred_tests_origin))
print(f"XGB_rmse_tests : {XGB_rmse_tests:.3f}")
print(f"XGB_rmse_tests_origin : {XGB_rmse_tests_origin:.3f}")
# ---------------------------------------------------------------------------




if execute:
    from bayes_opt import BayesianOptimization
    class MinimizeFunction():
        def __init__(self, train_X, train_y, valid_X, valid_y):
            self.train_X = train_X
            self.train_y = train_y
            self.valid_X = valid_X
            self.valid_y = valid_y
        
        def minimize(self, max_depth, learning_rate, subsample, colsample_bytree, n_estimators):
            model = XGBRegressor(
                max_depth=int(max_depth),
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                n_estimators=int(n_estimators),
                enable_categorical=True,
                tree_method='hist',
            )
            model.fit(self.train_X, self.train_y)
            preds = model.predict(self.valid_X)
            rmse = np.sqrt(mean_squared_error(self.valid_y, preds))
            return -rmse  # BayesianOptimization은 최대화를 하므로 음수로 반환
        
        def __call__(self, max_depth, learning_rate, subsample, colsample_bytree, n_estimators):
            return self.minimize(max_depth, learning_rate, subsample, colsample_bytree, n_estimators)

    min_f = MinimizeFunction(train_X_concat_norm, train_y_norm, valid_X_concat_norm, valid_y_norm)

    # 파라미터 범위 설정
    pbounds = {
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0),
        'n_estimators': (50, 300)
    }

    # Bayesian Optimization 실행
    optimizer = BayesianOptimization(
        f=min_f,
        pbounds=pbounds
    )

    optimizer.maximize(init_points=5, n_iter=20)
    best_params = optimizer.max['params']
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['n_estimators'] = int(best_params['n_estimators'])

    XGB_best = XGBRegressor(**best_params)
    XGB_best.fit(train_X_concat_norm, train_y_norm)
    XGB_best_pred = XGB_best.predict(train_X_concat_norm)
    np.sqrt(mean_squared_error(train_y_norm, XGB_best_pred))

    XGB_best_pred_tests = XGB_best.predict(tests_X_concat_norm)
    XGB_best_rmse_tests = np.sqrt(mean_squared_error(tests_y_norm, XGB_best_pred_tests))
    XGB_best_pred_tests_origin = data_preprocessing.encoder[0].inverse_transform(XGB_best_pred_tests.reshape(-1,1)).ravel()
    XGB_best_rmse_tests_origin = np.sqrt(mean_squared_error(np.array(tests_y).ravel(), XGB_best_pred_tests_origin))
    print(f"XGB_best_rmse_tests : {XGB_best_rmse_tests:.3f}")
    print(f"XGB_best_rmse_tests_origin : {XGB_best_rmse_tests_origin:.3f}")



####################################################################################################



for batch in train_loader:
    print(f"dataloader_label : {[b.shape for b in batch]}")
    break


# print(f"RF_rmse_tests : {RF_rmse_tests:.3f}")   # 0.525
# print(f"GB_rmse_tests : {GB_rmse_tests:.3f}")   # 0.527
# print(f"XGB_rmse_tests : {XGB_rmse_tests:.3f}") # 0.528
# print(f"XGB_best_rmse_tests : {XGB_best_rmse_tests:.3f}") # 0.507

####################################################################################################
N_TEST = 30

####################################################################################################
class ResidualConnection(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    
    def forward(self, x):
        return self.layer(x) + x






####################################################################################################
# 0.509
class DNN(nn.Module):
    def __init__(self, input_n_con, input_n_cat, hidden_dim=256, dropout=0.1):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Linear(input_n_con + input_n_cat, hidden_dim)
            ,nn.ReLU()
            ,ResidualConnection( nn.Sequential(
                nn.BatchNorm1d(hidden_dim)
                ,nn.Linear(hidden_dim, hidden_dim)
                ,nn.ReLU()
                ,nn.Dropout(dropout)
                )
            )
            ,ResidualConnection( nn.Sequential(
                nn.BatchNorm1d(hidden_dim)
                ,nn.Linear(hidden_dim, hidden_dim)
                ,nn.ReLU()
                ,nn.Dropout(dropout)
                ) 
            )
            ,nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x_con, x_cat):
        x_cat = torch.cat([x_con, x_cat], dim=-1)
        return self.block(x_cat)
# -----------------------------------------------------------------------------------------------------
data_preprocessing.index
dataset_index['train']

model = DNN(11, 3, hidden_dim=256).to(device)
# model(torch.rand(10,11).to(device), torch.rand(10,3).to(device))
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

# [x.shape for x in next(iter(train_loader))]
# [x.shape for x in next(iter(valid_loader))]
# tm1.set_best_model()
test_loss1 = []
for _ in range(N_TEST):
    test_loss = np.sqrt(tm1.test_model(tests_loader)['test_loss']).item()
    test_loss1.append(test_loss)
test_loss1 = sorted(test_loss1)

print(f"test_loss1 : {np.mean(test_loss1[3:-3]):.3f}")


####################################################################################################








####################################################################################################
# 0.504
class CatEmbedDNN(nn.Module):
    def __init__(self, input_n_con, input_n_cat, hidden_dim=256, embedding_dim=2, dropout=0.1):
        super().__init__()
        
        self.cat_embedding = CategoricalEmbedding(input_n_cat, 1000, embedding_dim)
        
        self.block = nn.Sequential(
            nn.Linear(input_n_con + (input_n_cat*embedding_dim), hidden_dim)
            ,nn.ReLU()
            ,ResidualConnection( nn.Sequential(
                nn.BatchNorm1d(hidden_dim)
                ,nn.Linear(hidden_dim, hidden_dim)
                ,nn.ReLU()
                ,nn.Dropout(dropout)
                
                )
            )
            ,ResidualConnection( nn.Sequential(
                nn.BatchNorm1d(hidden_dim)
                ,nn.Linear(hidden_dim, hidden_dim)
                ,nn.ReLU()
                ,nn.Dropout(dropout)
                ) 
            )
            ,nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x_con, x_cat):
        batch_size = x_cat.shape[0]
        x_cat_embed = self.cat_embedding(x_cat).view(batch_size, -1)
        x_cat = torch.cat([x_con, x_cat_embed], dim=-1)
        return self.block(x_cat)
# -----------------------------------------------------------------------------------------------------

model = CatEmbedDNN(11, 3, hidden_dim=256, embedding_dim=4).to(device)
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

# tm2.set_best_model()
test_loss2 = []
for _ in range(N_TEST):
    test_loss = np.sqrt(tm2.test_model(tests_loader)['test_loss']).item()
    test_loss2.append(test_loss)
test_loss2 = sorted(test_loss2)

print(f"test_loss2 : {np.mean(test_loss2[3:-3]):.3f}")





####################################################################################################


class TabTransformers(nn.Module):
    def __init__(self, input_n_con, input_n_cat, hidden_dim=256, embedding_dim=32,
                nhead=4, encoder_dropout=0, fc_dropout=0.1, num_layers=1):
        super().__init__()
        
        self.cat_embedding = CategoricalEmbedding(input_n_cat, 1000, embedding_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead,
                                    dim_feedforward=embedding_dim*2, dropout=encoder_dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)             
        
        self.block = nn.Sequential(
            nn.Linear(input_n_con + (input_n_cat*embedding_dim), hidden_dim)
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
        x_cat_embed = self.cat_embedding(x_cat)
        
        x_cat_encoder = self.encoder(x_cat_embed)
        x_cat = torch.cat([x_con, x_cat_encoder.view(batch_size, -1)], dim=-1)
        return self.block(x_cat)
# -----------------------------------------------------------------------------------------------------


model = TabTransformers(11, 3).to(device)
# model(torch.rand(10, 11).to(device), torch.ones(10,3).to(torch.int64).to(device))
f"{sum([p.numel() for p in model.parameters() if p.requires_grad]):,}"   

 
def loss_function_dnn(model, batch, optimizer=None):
    y, X_con, X_cat = batch
    pred = model(X_con, X_cat)
    loss = nn.functional.mse_loss(pred, y)
    return loss

tm3 = TorchModeling(model=model, device=device)
tm3.compile(optimizer = optim.Adam(model.parameters(), lr=5e-5)
            ,loss_function=loss_function_dnn
            ,early_stop_loss=EarlyStopping(min_iter=30, patience=50))
tm3.train_model(train_loader, valid_loader,
                epochs=500)

# tm3.set_best_model()
test_loss3 = []
for _ in range(N_TEST):
    test_loss = np.sqrt(tm3.test_model(tests_loader)['test_loss']).item()
    test_loss3.append(test_loss)
test_loss3 = sorted(test_loss3)

print(f"test_loss3 : {np.mean(test_loss3[3:-3]):.3f}")







####################################################################################################
class CatConEmbedDNN(nn.Module):
    def __init__(self, input_n_con, input_n_cat, hidden_dim=256, embedding_dim=2, dropout=0.1):
        super().__init__()
        
        self.con_embedding = EmbeddingLinear(input_n_con, 1, embedding_dim)
        self.cat_embedding = CategoricalEmbedding(input_n_cat, 1000, embedding_dim)
        
        self.block = nn.Sequential(
            nn.Linear((input_n_con + input_n_cat)*embedding_dim, hidden_dim)
            ,nn.ReLU()
            ,ResidualConnection( nn.Sequential(
                nn.BatchNorm1d(hidden_dim)
                ,nn.Linear(hidden_dim, hidden_dim)
                ,nn.ReLU()
                ,nn.Dropout(dropout)
                )
            )
            ,ResidualConnection( nn.Sequential(
                nn.BatchNorm1d(hidden_dim)
                ,nn.Linear(hidden_dim, hidden_dim)
                ,nn.ReLU()
                ,nn.Dropout(dropout)
                ) 
            )
            ,nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x_con, x_cat):
        batch_size = x_cat.shape[0]
        x_con_embed = self.con_embedding(x_con.unsqueeze(-1)).view(batch_size, -1)
        x_cat_embed = self.cat_embedding(x_cat).view(batch_size, -1)
        x_concat = torch.cat([x_con_embed, x_cat_embed], dim=-1)
        return self.block(x_concat)
# -----------------------------------------------------------------------------------------------------

model = CatConEmbedDNN(11, 3, hidden_dim=256, embedding_dim=4).to(device)
# model(torch.rand(10, 11).to(device), torch.ones(10,3).to(torch.int64).to(device))
f"{sum([p.numel() for p in model.parameters() if p.requires_grad]):,}"
 
def loss_function_dnn(model, batch, optimizer=None):
    y, X_con, X_cat = batch
    pred = model(X_con, X_cat)
    loss = nn.functional.mse_loss(pred, y)
    return loss

tm4 = TorchModeling(model=model, device=device)
tm4.compile(optimizer = optim.Adam(model.parameters(), lr=5e-5)
            ,loss_function=loss_function_dnn
            ,early_stop_loss=EarlyStopping(min_iter=30, patience=50))
tm4.train_model(train_loader, valid_loader,
                epochs=500)

# tm4.set_best_model()
test_loss4 = []
for _ in range(N_TEST):
    test_loss = np.sqrt(tm4.test_model(tests_loader)['test_loss']).item()
    test_loss4.append(test_loss)
test_loss4 = sorted(test_loss4)

print(f"test_loss4 : {np.mean(test_loss4[3:-3]):.3f}")









####################################################################################################
class CatConEmbedClsDNN(nn.Module):
    def __init__(self, input_n_con, input_n_cat, hidden_dim=256, embedding_dim=2, dropout=0.1):
        super().__init__()
        
        self.con_embedding = EmbeddingLinear(input_n_con, 1, embedding_dim)
        self.cat_embedding = CategoricalEmbedding(input_n_cat, 1000, embedding_dim)
        self.cls_token = nn.Parameter(torch.randn(1,embedding_dim))
        
        self.block = nn.Sequential(
            nn.Linear( (1 + input_n_con + input_n_cat)*embedding_dim, hidden_dim)
            ,nn.ReLU()
            ,ResidualConnection( nn.Sequential(
                nn.BatchNorm1d(hidden_dim)
                ,nn.Linear(hidden_dim, hidden_dim)
                ,nn.ReLU()
                ,nn.Dropout(dropout)
                )
            )
            ,ResidualConnection( nn.Sequential(
                nn.BatchNorm1d(hidden_dim)
                ,nn.Linear(hidden_dim, hidden_dim)
                ,nn.ReLU()
                ,nn.Dropout(dropout)
                ) 
            )
            ,nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x_con, x_cat):
        batch_size = x_cat.shape[0]
        x_con_embed = self.con_embedding(x_con.unsqueeze(-1)).view(batch_size, -1)
        x_cat_embed = self.cat_embedding(x_cat).view(batch_size, -1)
        cls_token_expand = self.cls_token.unsqueeze(0).repeat(batch_size,1,1).view(batch_size, -1)
        
        x_concat = torch.cat([cls_token_expand, x_con_embed, x_cat_embed], dim=-1)
        return self.block(x_concat)
# -----------------------------------------------------------------------------------------------------


model = CatConEmbedClsDNN(11, 3, hidden_dim=256, embedding_dim=4).to(device)
# model(torch.rand(10, 11).to(device), torch.ones(10,3).to(torch.int64).to(device))
f"{sum([p.numel() for p in model.parameters() if p.requires_grad]):,}"
 
def loss_function_dnn(model, batch, optimizer=None):
    y, X_con, X_cat = batch
    pred = model(X_con, X_cat)
    loss = nn.functional.mse_loss(pred, y)
    return loss

tm5 = TorchModeling(model=model, device=device)
tm5.compile(optimizer = optim.Adam(model.parameters(), lr=5e-5)
            ,loss_function=loss_function_dnn
            ,early_stop_loss=EarlyStopping(min_iter=30, patience=50))
tm5.train_model(train_loader, valid_loader,
                epochs=500)

# tm5.set_best_model()
test_loss5 = []
for _ in range(N_TEST):
    test_loss = np.sqrt(tm5.test_model(tests_loader)['test_loss']).item()
    test_loss5.append(test_loss)
test_loss5 = sorted(test_loss5)

print(f"test_loss5 : {np.mean(test_loss5[3:-3]):.3f}")












####################################################################################################
# https://towardsdatascience.com/improving-tabtransformer-part-1-linear-numerical-embeddings-dbc3be3b5bb5/
class FT_Transformeres(nn.Module):
    def __init__(self, input_n_con, input_n_cat, hidden_dim=256, embedding_dim=32,
                nhead=4, encoder_dropout=0, fc_dropout=0.1, num_layers=1):
        super().__init__()
        
        self.con_embedding = EmbeddingLinear(input_n_con, 1, embedding_dim)
        self.cat_embedding = CategoricalEmbedding(input_n_cat, 1000, embedding_dim)
        self.cls_token = nn.Parameter(torch.randn(1,embedding_dim))
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead,
                                    dim_feedforward=embedding_dim*2, dropout=encoder_dropout)
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



model = FT_Transformeres(11, 3, hidden_dim=256).to(device)
# model(torch.rand(10, 11).to(device), torch.ones(10,3).to(torch.int64).to(device))
f"{sum([p.numel() for p in model.parameters() if p.requires_grad]):,}"


def loss_function_dnn(model, batch, optimizer=None):
    y, X_con, X_cat = batch
    pred = model(X_con, X_cat)
    loss = nn.functional.mse_loss(pred, y)
    return loss

tm6 = TorchModeling(model=model, device=device)
tm6.compile(optimizer = optim.Adam(model.parameters(), lr=5e-5)
            ,loss_function=loss_function_dnn
            ,early_stop_loss=EarlyStopping(min_iter=30, patience=50))
tm6.train_model(train_loader, valid_loader,
                epochs=500)

# tm6.set_best_model()
test_loss6 = []
for _ in range(N_TEST):
    test_loss = np.sqrt(tm6.test_model(tests_loader)['test_loss']).item()
    test_loss6.append(test_loss)
test_loss6 = sorted(test_loss6)

print(f"test_loss6 : {np.mean(test_loss6[3:-3]):.3f}")















####################################################################################################

class FeatureWiseTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, feature_dim, dim_feedforward=2048, dropout=0.1, 
                batch_first=True, qkv_projection=True, ind_qkv_projection=True):
        super().__init__()
        
        self.layer_norm1 = nn.LayerNorm(d_model)      # layer_norm1
        self.self_attn = MultiheadAttention(d_model, nhead, batch_first=True, 
                                    qkv_projection=qkv_projection, ind_qkv_projection=True, feature_dim=feature_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ff_layer = nn.Sequential(
            nn.LayerNorm(d_model),      # layer_norm2
            EmbeddingLinear(feature_dim, d_model, dim_feedforward),     # FF_linear1
            nn.ReLU(),
            nn.Dropout(dropout),
            EmbeddingLinear(feature_dim, dim_feedforward, d_model),     # FF_linear2
        )
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        src_key_padding_mask: Optional[torch.Tensor] = None
        ):
        
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
# --------------------------------------------------------------------------------------------------
####################################################################################################




# https://dongsarchive.tistory.com/74
class CustomTransformeres(nn.Module):
    def __init__(self, input_n_con, input_n_cat, hidden_dim=256, embedding_dim=32,
                nhead=4, encoder_dropout=0, fc_dropout=0.1, num_layers=1):
        super().__init__()
        
        self.con_embedding = EmbeddingLinear(input_n_con, 1, embedding_dim)
        self.cat_embedding = CategoricalEmbedding(input_n_cat, 1000, embedding_dim)
        self.cls_token = nn.Parameter(torch.randn(1,embedding_dim))
        
        self.featurewise_transformer_encoder = FeatureWiseTransformerEncoder(embedding_dim, nhead, 1+input_n_con+input_n_cat)
        
        self.attn_pooling = AttentionPooling(embedding_dim)
        
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



model = CustomTransformeres(11, 3, hidden_dim=256).to(device)
# model(torch.rand(10, 11).to(device), torch.ones(10,3).to(torch.int64).to(device))
f"{sum([p.numel() for p in model.parameters() if p.requires_grad]):,}"


def loss_function_dnn(model, batch, optimizer=None):
    y, X_con, X_cat = batch
    pred = model(X_con, X_cat)
    loss = nn.functional.mse_loss(pred, y)
    return loss

tm7 = TorchModeling(model=model, device=device)
tm7.compile(optimizer = optim.Adam(model.parameters(), lr=5e-5)
            ,loss_function=loss_function_dnn
            ,early_stop_loss=EarlyStopping(min_iter=30, patience=50))
tm7.train_model(train_loader, valid_loader,
                epochs=500)

# tm7.set_best_model()
test_loss7 = []
for _ in range(N_TEST):
    test_loss = np.sqrt(tm7.test_model(tests_loader)['test_loss']).item()
    test_loss7.append(test_loss)
test_loss7 = sorted(test_loss7)

print(f"test_loss7 : {np.mean(test_loss7[3:-3]):.3f}")




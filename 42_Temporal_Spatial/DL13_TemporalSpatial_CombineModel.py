example = False

###########################################################################################################

# import requests
# response_files = requests.get("https://raw.githubusercontent.com/kimds929/CodeNote/main/60_Graph_Neural_Network/GNN01_GenerateGraph.py")
# exec(response_files.text)
# response_files.text

# import importlib
# importlib.reload(httpimport)


import httpimport
remote_url = "https://raw.githubusercontent.com/kimds929/"
with httpimport.remote_repo(f"{remote_url}/CodeNote/main/42_Temporal_Spatial/"):
    from DL13_Temporal_12_TemporalEmbedding import PeriodicEmbedding, TemporalEmbedding

with httpimport.remote_repo(f"{remote_url}/CodeNote/main/42_Temporal_Spatial/"):
    from DL13_Spatial_11_SpatialEmbedding import SpatialEmbedding

if example:
    with httpimport.remote_repo(f"{remote_url}/DS_Library/main/"):
        from DS_DeepLearning import EarlyStopping

    with httpimport.remote_repo(f"{remote_url}/DS_Library/main/"):
        from DS_Torch import TorchModeling



###########################################################################################################
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Basic Block of DirectEnsemble
class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.ReLU(),
                batchNorm=True,  dropout=0.2):
        super().__init__()
        ff_block = [nn.Linear(input_dim, output_dim)]
        if activation:
            ff_block.append(activation)
        if batchNorm:
            ff_block.append(nn.BatchNorm1d(output_dim))
        if dropout > 0:
            ff_block.append(nn.Dropout(dropout))
        self.ff_block = nn.Sequential(*ff_block)
    
    def forward(self, x):
        return self.ff_block(x)


class CombinedEmbedding(nn.Module):
    def __init__(self, input_dim, t_input_dim,  output_dim=None, t_emb_dim=8, t_hidden_dim=None, s_emb_dim=None, **spatial_kwargs):
        super().__init__()

        # temporal embedding layer (t_input_dim)
        self.t_input_dim = t_input_dim
        self.temporal_embedding = TemporalEmbedding(input_dim=t_input_dim, embed_dim=t_emb_dim, hidden_dim=t_hidden_dim)

        # spatial embedding layer (4)
        self.spatial_embedding = SpatialEmbedding(embed_dim=s_emb_dim, **spatial_kwargs)
        
        # other feature dimension (input_dim - t_input_dim - 4)
        self.other_feature_dim = input_dim - t_input_dim - 4

        # embed_dim
        self.output_dim = output_dim
        self.embed_dim = self.temporal_embedding.embed_dim + self.spatial_embedding.embed_dim + self.other_feature_dim

        # fc block
        if output_dim is not None:
            self.fc_layer = nn.Linear(self.embed_dim, output_dim)
            self.embed_dim = output_dim
        
    def forward(self, x):
        temporal_features = self.temporal_embedding(x[:,:self.t_input_dim])
        spatial_features = self.spatial_embedding(x[:,self.t_input_dim:self.t_input_dim+2], x[:,self.t_input_dim+2:self.t_input_dim+4])
        other_features = x[:,self.t_input_dim+4:]
        outputs = torch.cat([temporal_features, spatial_features, other_features], dim=1)
        
        if self.output_dim is not None:
            outputs = self.fc_layer(outputs)
        return outputs

class EnsembleCombinedModel(nn.Module):
    def __init__(self, input_dim, output_dim, t_input_dim, hidden_dim,  n_layers=3, n_models = 10, n_output=1,
                embed_output_dim=None, t_emb_dim=8, t_hidden_dim=None, s_emb_dim=None, **spatial_kwargs):
        super().__init__()

        # combined embedding layer
        self.combined_embedding = CombinedEmbedding(input_dim=input_dim, t_input_dim=t_input_dim,  output_dim=embed_output_dim,
                                                    t_emb_dim=t_emb_dim, t_hidden_dim=t_hidden_dim, s_emb_dim=s_emb_dim, **spatial_kwargs)
        self.embed_output_dim = self.combined_embedding.embed_dim

        # fc block
        self.fc_block = nn.ModuleDict({'in_layer':FeedForwardBlock(self.embed_output_dim, hidden_dim, batchNorm=False, dropout=0)})

        out_dim = output_dim*n_models if n_output == 1 else output_dim*n_output*n_models
        for h_idx in range(n_layers):
            if h_idx < n_layers-1:
               self.fc_block[f'hidden_layer{h_idx+1}'] = FeedForwardBlock(hidden_dim, hidden_dim, batchNorm=False, dropout=0)
            else:
                self.fc_block['out_layer'] = FeedForwardBlock(hidden_dim, out_dim, activation=False, batchNorm=False, dropout=0)
        
        self.output_dim = output_dim
        self.n_output = n_output
        self.n_layers = n_layers
        self.n_models = n_models

    def train_forward(self, x):
        x = self.combined_embedding(x)

        for layer_name, layer in self.fc_block.items():
            if layer_name == 'in_layer' or layer_name == 'out_layer':
                x = layer(x)
            else:
                x = layer(x) + x    # residual connection
        
        if self.n_output == 1:
            return x
        else:
            return torch.split(x, self.output_dim*self.n_models, dim=1)

    def predict(self, x, idx=None):
        if self.n_output == 1:
            if idx is None:
                return self.train_forward(x).mean(dim=1, keepdims=True)
            else:
                return self.train_forward(x)[:, idx].mean(dim=1, keepdims=True)
        else:
            if idx is None:
                return tuple([output.mean(dim=1, keepdims=True) for output in self.train_forward(x)])
            else:
                return tuple([output[:, idx].mean(dim=1, keepdims=True) for output in self.train_forward(x)])

    def forward(self, x, idx=None):
        if self.training:
            return self.train_forward(x)
        else:
            return self.predict(x, idx)

def make_feature_set_embedding(context_df, temporal_cols, spatial_cols, other_cols, fillna=None):
    # temproal features preprocessing
    temproal_arr = context_df[temporal_cols].applymap(lambda x: format_str_to_time(x) if type(x) == str else x).fillna(0).to_numpy().astype(np.float32)

    # spatial features preprocessing
    spatial_cols_transform = list(np.stack([[f"{cols}_x", f"{cols}_y"] for cols in spatial_cols]).ravel())

    spatial_arr_stack = np.stack(list(context_df[spatial_cols].applymap(lambda x: np.array(x)).to_dict('list').values())).astype(np.float32)
    spatial_arr = spatial_arr_stack.transpose(1,0,2).reshape(-1,4)

    # other features
    other_arr = context_df[other_cols].to_numpy().astype(np.float32)

    # # combine and transform to dataframe
    df_columns = temporal_cols + spatial_cols_transform + other_cols
    df_transform = pd.DataFrame(np.hstack([temproal_arr, spatial_arr, other_arr]),
                             columns=df_columns, index=context_df.index)
    if fillna is not None:
        df_transform = df_transform.fillna(fillna)
    return df_transform

###########################################################################################################



if example:
    # device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # make data set -----------------------------------------------------------------------------------

    # class SimpleModel(nn.Module):
    #     def __init__(self, input_dim, output_dim, hidden_dim=32):
    #         super().__init__()
    #         self.fc_block = nn.Sequential(
    #             nn.Linear(input_dim, hidden_dim)
    #             ,nn.ReLU()
    #             ,nn.Linear(hidden_dim, hidden_dim)
    #             ,nn.ReLU()
    #             ,nn.Linear(hidden_dim, hidden_dim)
    #             ,nn.ReLU()
    #             ,nn.Linear(hidden_dim, output_dim)
    #             )

    #     def forward(self, x):
    #         return self.fc_block(x)
    
    n_data = 1000
    temporal_data = torch.rand(n_data, 5)
    spatial_data = torch.rand(n_data, 4)
    other_data = torch.rand(n_data, 6)

    train_x = torch.cat([temporal_data, spatial_data, other_data], dim=-1)

    # sm = SimpleModel(15, 1)
    # train_y = sm(train_x).detach()
    train_y = torch.rand(n_data, 1)

    from torch.utils.data import DataLoader, TensorDataset
    batch_size = 64
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # train model --------------------------------------------------------------------------------------
    # temporal input dimension : n
    # spatial input dimension : 4
    # other input dimension : input-dim - n - 4
    ecm = EnsembleCombinedModel(input_dim=15, output_dim=1, t_input_dim=5, hidden_dim=32, n_models=1, n_output=1)
    ecm(torch.rand(5,15))

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    def mse_loss(model, x, y):
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        return loss

    tm = TorchModeling(model=ecm, device=device)
    tm.compile(optimizer=optimizer
                ,loss_function = mse_loss
                , early_stop_loss = EarlyStopping(patience=5)
                )
    tm.train_model(train_loader=train_loader, epochs=100, display_earlystop_result=True, early_stop=False)



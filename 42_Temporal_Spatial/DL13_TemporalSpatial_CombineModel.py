example = False

###########################################################################################################

# import requests
# response_files = requests.get("https://raw.githubusercontent.com/kimds929/CodeNote/main/60_Graph_Neural_Network/GNN01_GenerateGraph.py")
# exec(response_files.text)
# response_files.text

# import importlib
# importlib.reload(httpimport)


if example:
    import httpimport

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


###########################################################################################################
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


###########################################################################################################
# Temporal Embedding
class PeriodicEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        # Linear Component
        self.linear_layer = nn.Linear(input_dim , 1)
        if embed_dim % 2 == 0:
            self.linear_layer2 = nn.Linear(input_dim , 1)
        else:
            self.linear_layer2 = None
        
        # Periodic Components
        self.periodic_weights = nn.Parameter(torch.randn(input_dim, (embed_dim - 1)//2 ))
        self.periodic_bias = nn.Parameter(torch.randn(1, (embed_dim - 1)//2 ))

        # NonLinear Purse Periodic Component
        self.nonlinear_weights = nn.Parameter(torch.randn(input_dim, (embed_dim - 1)//2 ))
        self.nonlinear_bias = nn.Parameter(torch.randn(1, (embed_dim - 1)//2 ))

    def forward(self, x):
        # Linear Component
        linear_term = self.linear_layer(x)
        
        # Periodic Component
        periodic_term = torch.sin(x @ self.periodic_weights + self.periodic_bias)

        # NonLinear Purse Periodic Component
        nonlinear_term = torch.sign(torch.sin(x @ self.nonlinear_weights + self.nonlinear_bias))
        
        # Combine All Components
        if self.linear_layer2 is None:
            return torch.cat([linear_term, periodic_term, nonlinear_term], dim=-1)
        else:
            linear_term2 = self.linear_layer2(x)
            return torch.cat([linear_term, linear_term2, periodic_term, nonlinear_term], dim=-1)


# -------------------------------------------------------------------------------------------
# ★ Main Embedding
class TemporalEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = input_dim * embed_dim

        if hidden_dim is None:
            self.temporal_embed_layers = nn.ModuleList([PeriodicEmbedding(input_dim=1, embed_dim=embed_dim) for _ in range(input_dim)])
        else:
            self.temporal_embed_layers = nn.ModuleList([PeriodicEmbedding(input_dim=1, embed_dim=hidden_dim) for _ in range(input_dim)])
            self.hidden_layer = nn.Linear(input_dim*hidden_dim, embed_dim)
            self.embed_dim = embed_dim
    
    def forward(self, x):
        emb_outputs = [layer(x[:,i:i+1]) for i, layer in enumerate(self.temporal_embed_layers)]
        output = torch.cat(emb_outputs, dim=1)
        if self.hidden_dim is not None:
            output = self.hidden_layer(output)

        return output



###########################################################################################################
# Spatial Embedding
class CoordinateEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, depth=1):
        super().__init__()
        self.embedding_block = nn.ModuleDict({'in_layer':FeedForwardBlock(input_dim, hidden_dim, batchNorm=False, dropout=0)})

        for h_idx in range(depth):
            if h_idx < depth-1:
               self.embedding_block[f'hidden_layer{h_idx+1}'] = FeedForwardBlock(hidden_dim, hidden_dim, batchNorm=False, dropout=0)
            else:
                self.embedding_block['out_layer'] = FeedForwardBlock(hidden_dim, embed_dim, activation=False, batchNorm=False, dropout=0)

    def forward(self, x):
        for layer_name, layer in self.embedding_block.items():
            if layer_name == 'in_layer' or layer_name == 'out_layer':
                x = layer(x)
            else:
                x = layer(x) + x
        return x

# -------------------------------------------------------------------------------------------
class GridEmbedding(nn.Module):
    def __init__(self, grid_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(grid_size**2, embed_dim)
        self.grid_size = grid_size

    def forward(self, x):
        # 좌표를 그리드로 매핑
        x_grid = (x * self.grid_size).long()  # 좌표를 그리드 인덱스로 변환
        x_index = x_grid[:, 0] * self.grid_size + x_grid[:, 1]  # 인덱스화
        # print(x_grid, x_index)
        return self.embedding(x_index)

# -------------------------------------------------------------------------------------------
def positional_encoding(coords, d_model):
    # coords: [N, 2]
    N, dim = coords.shape
    pe = []
    for i in range(d_model // 4):
        freq = 10000 ** (2 * i / d_model)
        pe.append(np.sin(coords * freq))
        pe.append(np.cos(coords * freq))
    pe = np.concatenate(pe, axis=1)  # [N, 2*d_model//2]
    return torch.tensor(pe, dtype=torch.float32)


# -------------------------------------------------------------------------------------------
# ★ Main Embedding Block
class SpatialEmbedding(nn.Module):
    def __init__(self, embed_dim=None, coord_hidden_dim=32, coord_embed_dim=8, coord_depth=2,
                grid_size=10, grid_embed_dim=8, periodic_embed_dim=5, 
                relative=True, euclidean_dist=True, angle=True):
        """
        embed_dim : (None) end with combined result
        coord_embed_dim : (None) not use coordinate embedding
        grid_embed_dim : (None) not use grid embedding
        periodic_embed_dim : (None) not use periodic embedding
        relative : (False) not use relative coordinate, (True) use relative coordinate
        euclidean_dist : (False) not use euclidean distance, (True) use euclidean distance
        angle : (False) not use angle, (True) use angle

        """
        super().__init__()
        self.coord_hidden_dim = coord_hidden_dim        ## 32
        self.coord_embed_dim = coord_embed_dim          ## 4
        self.grid_size = grid_size                      ## 10
        self.grid_embed_dim = grid_embed_dim            ## 4
        self.periodic_embed_dim = periodic_embed_dim    ## 3

        self.relative = relative                    ## True: 2
        self.euclidean_dist = euclidean_dist        ## True : 1
        self.angle = angle                          ## True : 1
        
        self.embed_dim = 0

        if self.coord_embed_dim is not None:
            self.coord_embedding = CoordinateEmbedding(input_dim=2, hidden_dim=coord_hidden_dim, embed_dim=coord_embed_dim, depth=coord_depth)
            self.embed_dim += self.coord_embed_dim * 2

        if self.grid_embed_dim is not None:
            self.grid_embedding = GridEmbedding(grid_size=grid_size, embed_dim=grid_embed_dim)
            self.embed_dim += self.grid_embed_dim * 2

        if self.periodic_embed_dim is not None:
            self.periodic_embedding = PeriodicEmbedding(input_dim=2, embed_dim=periodic_embed_dim)
            self.embed_dim += self.periodic_embed_dim * 2
        
        if self.relative:
            self.embed_dim += 2
        
        if self.euclidean_dist:
            self.embed_dim += 1

        if self.angle:
            self.embed_dim += 1

        if embed_dim is not None:
            self.hidden_dim = self.embed_dim
            self.embed_dim = embed_dim
            self.hidden_layer = nn.Linear(self.hidden_dim, self.embed_dim)
        else:
            self.hidden_dim = None

    def forward(self, coord1, coord2):
        spatial_embeddings = []
        # [embed_1, embed_2, grid_1, grid_2, relative, euclidean_dist, angle, period_1, period_2]

        # embed
        if self.coord_embed_dim is not None:
            embed_1 = self.coord_embedding(coord1)
            embed_2 = self.coord_embedding(coord2)
            spatial_embeddings.append(embed_1)
            spatial_embeddings.append(embed_2)

        # grid
        if self.grid_embed_dim is not None:
            grid_1 = self.grid_embedding(coord1)
            grid_2 = self.grid_embedding(coord1)
            spatial_embeddings.append(grid_1)
            spatial_embeddings.append(grid_2)
        
        # periodic
        if self.periodic_embed_dim is not None:
            period_1 = self.periodic_embedding(coord1)
            period_2 = self.periodic_embedding(coord2)
            spatial_embeddings.append(period_1)
            spatial_embeddings.append(period_2)

        # norm
        if self.relative:
            relative = coord2 - coord1 
            spatial_embeddings.append(relative)

        if self.euclidean_dist:
            euclidean_dist = torch.norm(coord2 - coord1, p=2, dim=1, keepdim=True)
            spatial_embeddings.append(euclidean_dist)

        # angle
        if self.angle:
            relative = coord2 - coord1
            angle = torch.atan2(relative[:,1], relative[:,0]).unsqueeze(1)  
            spatial_embeddings.append(angle)

        # combine
        output = torch.cat(spatial_embeddings, dim=1)
        # embed_dim = coord_embed_dim * 2 + grid_embed_dim * 2 + periodic_embed_dim * 2 + 2(relative) + 1(euclidean_dist) + 1(angle)

        if self.hidden_dim is not None:
            output = self.hidden_layer(output)
        return output














###########################################################################################################
# Combined Embedding
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



###########################################################################################################
# EnsembleCombinedModel

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


###########################################################################################################
# (Version 4.0 Update)
# 시간을 "요일. 시:분" 형식으로 변환하는 함수
def format_time_to_str(time, return_type='str'):
    """
    return_type : 'str', 'dict'
    """
    if time is None:
        return None
    else:
        int_time = int(time)
        week_code = ["Mon.", "Tue.", "Wed.", "Thu.", "Fri.", "Sat.", "Sun."]
        week = int_time // (24*60)
        hour = (int_time % (24*60)) // 60
        min = (int_time % (24*60)) % 60

        # (Version 4.0 Update) -----------------------------------
        if return_type == 'str':
            return f"{week_code[week % 7]} {hour:02d}:{min:02d}"
        elif return_type == 'dict':
            return {"week": week_code[week % 7], "hour": hour, "min":min}
            # return {"week": week, "hour": hour, "min":min}
        # ---------------------------------------------------------

# (Version 4.0 Update)
# "요일. 시:분" 형식을 시간형식으로 변환하는 함수
def format_str_to_time(time_str):
    if time_str is None:
        return None
    else:
        week_dict = {"Mon.":0, "Tue.":1, "Wed.":2, "Thu.":3, "Fri.":4, "Sat.":5, "Sun.":6}

        week_str, hour_min_str = time_str.split(" ")
        hour, min = hour_min_str.split(":")
        return week_dict[week_str]*24*60 + 60*int(hour) + int(min)


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



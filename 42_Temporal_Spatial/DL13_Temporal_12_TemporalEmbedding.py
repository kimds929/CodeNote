import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

example = False

# -------------------------------------------------------------------------------------------
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
# TemporalEmbedding     ★ Main Embedding
class TemporalEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = input_dim * embed_dim

        if hidden_dim is None:
            self.temporal_embed_layers = nn.ModuleList([PeriodicEmbedding(input_dim=1, embed_dim=embed_dim) for _ in range(input_dim)])
        else:
            self.temporal_embed_layers = nn.ModuleList([PeriodicEmbedding(input_dim=1, embed_dim=hidden_dim) for _ in range(input_dim)])
            self.hidden_layer = nn.Linear(input_dim*hidden_dim, embed_dim)
            self.embed_dim = embed_dim
    
    def forward(self, x):
        if x.shape[-1] != self.input_dim:
            raise Exception(f"input shape does not match.")
        emb_outputs = [layer(x[...,i:i+1]) for i, layer in enumerate(self.temporal_embed_layers)]
        output = torch.cat(emb_outputs, dim=1)
        if self.hidden_dim is not None:
            output = self.hidden_layer(output)

        return output
# te = TemporalEmbedding(1,5,32)
# te(torch.rand(5,1))

if example:
    te = TemporalEmbedding(input_dim=1, embed_dim=6, hidden_dim=64)
    x = torch.linspace(0,10,100).view(-1,1)

    for i in range(6):
        plt.plot(x.ravel().numpy(), te(x)[:,i].ravel().detach().numpy(), label=i)
    plt.legend()

# -------------------------------------------------------------------------------------------
class TimePredictModel(nn.Module):
    def __init__(self, input_dim=1, embed_dim=5, hidden_dim=32, output_dim=1):
        super().__init__()
        self.periodic_embedding = TemporalEmbedding(input_dim, embed_dim, hidden_dim)

        self.fc_block = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        periodic_embed = self.periodic_embedding(x)
        output = self.fc_block(periodic_embed)
        return output


# -------------------------------------------------------------------------------------------
# FullyConnected Base Model
class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.ReLU(),
                batchNorm=True,  dropout=0.5):
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

class FullyConnectedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=3):
        super().__init__()
        
        self.EnsembleBlock = nn.ModuleDict({'in_layer':FeedForwardBlock(input_dim, hidden_dim)})

        for h_idx in range(n_layers):
            if h_idx < n_layers-1:
               self.EnsembleBlock[f'hidden_layer{h_idx+1}'] = FeedForwardBlock(hidden_dim, hidden_dim)
            else:
                self.EnsembleBlock['out_layer'] = FeedForwardBlock(hidden_dim, output_dim, activation=False, batchNorm=False, dropout=0)
        self.n_layers = n_layers
        self.output_dim = output_dim

    def forward(self, x):
        for layer_name, layer in self.EnsembleBlock.items():
            if layer_name == 'in_layer' or layer_name == 'out_layer':
                x = layer(x)
            else:
                x = layer(x) + x    # residual connection

        output = x
        return output

# -------------------------------------------------------------------------------------------

if example:
    # PeriodicEmbedding 모델 초기화
    in_dim = 1
    em_dim = 7
    periodic_embed = PeriodicEmbedding(input_dim=in_dim, embed_dim=em_dim)

    # 월요일(0)부터 일요일(6)까지 7일을 24*60분 단위로 나눈 시간 데이터 생성
    time_data = np.linspace(0, 7, 7 * 60*24, endpoint=False).reshape(-1, 1)
    time_data_tensor = torch.tensor(time_data, dtype=torch.float32)

    # Time2Vec 모델을 통해 시간 데이터를 임베딩
    embedding = periodic_embed(time_data_tensor)
    print(embedding.shape)

    plt.figure(figsize=(14,6))
    for ei in range(em_dim):
        plt.plot(time_data.ravel(), embedding[:,ei].ravel().detach().numpy(), label=ei)
    plt.legend(loc='upper right')
    plt.show()

 

# 데이터 생성: 주기 함수를 예시로 사용 ##################################################
def sine_wave(x):
    return np.sin(2 * np.pi * x)

def square_wave(x, period=1.0):
    return np.sign(np.sin(2 * np.pi * x / period))

def triangle_wave(x, freq=1):
    return 2 * np.abs(np.arcsin(np.sin(freq * x))) / np.pi

# plt.plot(periodic_improve(time_data*24*60))

if example:
    # time_data_x = np.linspace(0, 7, 7 * 60*24, endpoint=False)
    time_data_x = np.linspace(0, 7, 7 * 24*6, endpoint=False)

    period_y = sine_wave(time_data_x)
    # period_y = square_wave(time_data_x)
    # period_y = triangle_wave(time_data_x, freq=2)
    # period_y = periodic_improve(time_data_x*60*24)

    train_x = torch.tensor(time_data_x.astype(np.float32).reshape(-1, 1))
    train_y = torch.tensor(period_y.astype(np.float32).reshape(-1,1))

    plt.figure(figsize=(14,6))
    plt.plot(time_data_x, period_y)
    plt.show()

    from torch.utils.data import DataLoader, TensorDataset
    # Dataset and DataLoader
    batch_size = 64
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

##################################################################################

if example:
    import httpimport
    remote_url = 'https://raw.githubusercontent.com/kimds929/'

    with httpimport.remote_repo(f"{remote_url}/DS_Library/main/"):
        from DS_DeepLearning import EarlyStopping

    with httpimport.remote_repo(f"{remote_url}/DS_Library/main/"):
        from DS_Torch import TorchDataLoader, TorchModeling, AutoML


    # device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 모델 초기화
    model = TimePredictModel(input_dim=1, embed_dim=5, hidden_dim=32, output_dim=1) 
    # model = FullyConnectedModel(input_dim=1, hidden_dim=32, output_dim=1, n_layers=3)
    sum(p.numel() for p in model.parameters())    # the number of parameters in model

    # loss_mse = nn.MSELoss()
    def mse_loss(model, x, y):
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        return loss

    optimizer = optim.Adam(model.parameters(), lr=1e-2)


    tm = TorchModeling(model=model, device=device)
    tm.compile(optimizer=optimizer
                , loss_function = mse_loss
                , early_stop_loss = EarlyStopping(patience=5)
                )
    tm.train_model(train_loader=train_loader, epochs=900, display_earlystop_result=True, early_stop=False)
    # tm.test_model(test_loader=test_loader)
    # tm.recompile(optimizer=optim.Adam(model.parameters(), lr=1e-3))


    with torch.no_grad():
        model.eval()
        pred = model(train_x.to(device))
        pred_arr = pred.to('cpu').numpy()

    plt.figure(figsize=(15,3))
    plt.plot(train_x.ravel(), pred_arr.ravel(), alpha=0.5, label='pred')
    plt.plot(train_x.ravel(), train_y.ravel(), alpha=0.5, label='true')
    plt.legend(loc='upper right')
    plt.show()





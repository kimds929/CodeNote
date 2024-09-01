
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

example = False
# -------------------------------------------------------------------------------------------

class CoordinateEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, depth=1):
        super().__init__()
        coord_embed = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth-1):
            coord_embed.append(nn.Linear(hidden_dim, hidden_dim))
            coord_embed.append(nn.ReLU())
        coord_embed.append(nn.Linear(hidden_dim, embed_dim))

        self.embedding = nn.Sequential(*coord_embed)

    def forward(self, x):
        return self.embedding(x)

if example:
    model = CoordinateEmbedding(input_dim=2, hidden_dim=32, embed_dim=16, depth=3)
    model
    coordinates = torch.tensor([[0.5, 0.5], [0.7, 0.2]], dtype=torch.float32)
    embedding = model(coordinates)
    print(embedding.shape)


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

if example:
    grid_size = 10
    model = GridEmbedding(grid_size=grid_size, embed_dim=16)
    coordinates = torch.tensor([[0.5, 0.5], [0.7, 0.2]], dtype=torch.float32)
    embedding = model(coordinates)
    print(embedding.shape)

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
class PeriodicEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        # Linear Component
        self.linear_weights = nn.Parameter(torch.randn(input_dim, 1))
        self.linear_bias = nn.Parameter(torch.randn(1, 1))
        
        # Periodic Components
        self.periodic_weights = nn.Parameter(torch.randn(input_dim, (embed_dim - 1)//2 ))
        self.periodic_bias = nn.Parameter(torch.randn(1, (embed_dim - 1)//2 ))

        # NonLinear Purse Periodic Component
        self.nonlinear_weights = nn.Parameter(torch.randn(input_dim, (embed_dim - 1)//2 ))
        self.nonlinear_bias = nn.Parameter(torch.randn(1, (embed_dim - 1)//2 ))

    def forward(self, x):
        # Linear Component
        linear_term = x @ self.linear_weights + self.linear_bias
        
        # Periodic Component
        periodic_term = torch.sin(x @ self.periodic_weights + self.periodic_bias)

        # NonLinear Purse Periodic Component
        nonlinear_term = torch.sign(torch.sin(x @ self.nonlinear_weights + self.nonlinear_bias))
        
        # Combine All Components
        return torch.cat([linear_term, periodic_term, nonlinear_term], dim=-1)

# pe = PeriodicEmbedding(input_dim=2, embed_dim=5)
# pe(torch.rand(5,2))




# -------------------------------------------------------------------------------------------
class SpatialEmbedding(nn.Module):
    def __init__(self, coord_hidden_dim=32, coord_embed_dim=None, coord_depth=1,
                grid_size=10, grid_embed_dim=None, periodic_embed_dim=None, 
                relative=False, euclidean_dist=False, angle=False):
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
            angle = torch.atan2(relative[:,1], relative[:,0]).unsqueeze(1)  
            spatial_embeddings.append(angle)

        # combine
        combined = torch.cat(spatial_embeddings, dim=1)
        # embed_dim = coord_embed_dim * 2 + grid_embed_dim * 2 + periodic_embed_dim * 2 + 2(relative) + 1(euclidean_dist) + 1(angle)
        return combined

# se = SpatialEmbedding(coord_hidden_dim=32, coord_embed_dim=4, coord_depth=1, grid_embed_dim=4, grid_size=10, periodic_embed_dim=3, relative=True, euclidean_dist=True, angle=True)
# se(torch.rand(5,2), torch.rand(5,2)).shape


# -------------------------------------------------------------------------------------------

class SpatialPredictModel(nn.Module):
    def __init__(self, hidden_dim, output_dim, coord_hidden_dim=32, coord_embed_dim=None, coord_depth=1, grid_size=10, grid_embed_dim=None, periodic_embed_dim=None, 
                    relative=False, euclidean_dist=False, angle=False):
        super().__init__()
        self.spatial_embedding = SpatialEmbedding(coord_hidden_dim=coord_hidden_dim, coord_embed_dim=coord_embed_dim, coord_depth=coord_depth,
                                                grid_size=grid_size, grid_embed_dim=grid_embed_dim, periodic_embed_dim=periodic_embed_dim,
                                                relative=relative, euclidean_dist=euclidean_dist, angle=angle)

        self.fc_block = nn.Sequential(
            nn.Linear(self.spatial_embedding.embed_dim, hidden_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_dim),
            # nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_dim),
            # nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, coord1, coord2):
        spatial_embed = self.spatial_embedding(coord1, coord2)
        output = self.fc_block(spatial_embed)
        return output

# model = SpatialPredictModel(hidden_dim=64, output_dim=1,
#                             coord_hidden_dim=32, coord_embed_dim=4, coord_depth=1, grid_size=10, grid_embed_dim=4, periodic_embed_dim=3, 
#                             relative=True, euclidean_dist=True, angle=True)
# model(torch.rand(5,2), torch.rand(5,2))









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

class FullyConnectedEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, n_layers=2, batchNorm=False, dropout=0.5):
        super().__init__()
        
        self.EnsembleBlock = nn.ModuleDict({'in_layer':FeedForwardBlock(input_dim, hidden_dim, batchNorm=batchNorm, dropout=dropout)})

        for h_idx in range(n_layers):
            if h_idx < n_layers-1:
               self.EnsembleBlock[f'hidden_layer{h_idx+1}'] = FeedForwardBlock(hidden_dim, hidden_dim, batchNorm=batchNorm, dropout=dropout)
            else:
                self.EnsembleBlock['out_layer'] = FeedForwardBlock(hidden_dim, embed_dim, activation=False, batchNorm=False, dropout=0)

    def forward(self, x):
        for layer_name, layer in self.EnsembleBlock.items():
            if layer_name == 'in_layer' or layer_name == 'out_layer':
                x = layer(x)
            else:
                x = layer(x) + x    # residual connection
        output = x
        return output

class FullyConnectedModel(nn.Module):
    def __init__(self, hidden_dim, output_dim, fc_hidden_dim=32, fc_embed_dim=4, n_layers=2, batchNorm=False, dropout=0.5):
        super().__init__()
        self.fc_embed_coord1 = FullyConnectedEmbedding(input_dim=2, hidden_dim=fc_hidden_dim, embed_dim=fc_embed_dim, n_layers=n_layers, batchNorm=batchNorm, dropout=dropout)
        self.fc_embed_coord2 = FullyConnectedEmbedding(input_dim=2, hidden_dim=fc_hidden_dim, embed_dim=fc_embed_dim, n_layers=n_layers, batchNorm=batchNorm, dropout=dropout)

        self.fc_block = nn.Sequential(
            nn.Linear(2*fc_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, coord1, coord2):
        fc_embed_1 = self.fc_embed_coord1(coord1)
        fc_embed_2 = self.fc_embed_coord1(coord2)
        fc_embed = torch.cat([fc_embed_1, fc_embed_2], dim=1)

        output = self.fc_block(fc_embed)
        return output

# model = FullyConnectedModel(hidden_dim=64, output_dim=1, fc_hidden_dim=32, fc_embed_dim=4,
#                          n_layers=3, batchNorm=False, dropout=0.5)
# model(torch.rand(5,2), torch.rand(5,2))




############################################################################################################################################
if example:
    import httpimport
    remote_url = 'https://raw.githubusercontent.com/kimds929/'

    with httpimport.remote_repo(f"{remote_url}/CodeNote/main/60_Graph_Neural_Network/"):
        from GNN01_GenerateGraph import GenerateNodeMap, visualize_graph, Dijkstra


    # --------------------------------------------------------------------------------------------------------------------------------
    n_nodes = 50
    random_state = 1

    # (create base graph) 
    node_map = GenerateNodeMap(n_nodes, random_state=random_state)
    node_map.create_node(node_scale=50, cov_knn=3)           # create node
    node_map.create_connect(connect_scale=0)             # create connection

    visualize_graph(centers=node_map.centers, adjacent_matrix=node_map.adj_matrix)

    features = []
    dists = []
    for _ in range(1000):
        start_node, end_node = np.random.choice(np.arange(n_nodes), size=2, replace=False)
        dijkstra = Dijkstra(node_map.adj_matrix)
        shortest_distance, path = dijkstra.dijkstra(start_node, end_node)

        # visualize_graph(centers=node_map.centers, adjacent_matrix=node_map.adj_matrix,
        #                 path=path, distance=shortest_distance)

        feature = np.append(node_map.centers[start_node], node_map.centers[end_node])
        dist = shortest_distance

        features.append(feature)
        dists.append(dist)

    x_train_arr = np.stack(features).astype(np.float32)
    y_train_arr = np.stack(dists).astype(np.float32).reshape(-1,1)

    x_train = torch.tensor(x_train_arr)
    y_train = torch.tensor(y_train_arr)


    from torch.utils.data import DataLoader, TensorDataset
    # Dataset and DataLoader
    batch_size=64
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


############################################################################################################################################
if example:
    import matplotlib.pyplot as plt
    import httpimport
    remote_url = 'https://raw.githubusercontent.com/kimds929/'

    with httpimport.remote_repo(f"{remote_url}/DS_Library/main/"):
        from DS_DeepLearning import EarlyStopping

    with httpimport.remote_repo(f"{remote_url}/DS_Library/main/"):
        from DS_Torch import TorchDataLoader, TorchModeling, AutoML



    # device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    model = SpatialPredictModel(hidden_dim=128, output_dim=1,
                                coord_hidden_dim=64, coord_embed_dim=16, coord_depth=3, grid_size=30, grid_embed_dim=16, periodic_embed_dim=9, 
                                relative=True, euclidean_dist=True, angle=True)
    # model = FullyConnectedModel(hidden_dim=128, output_dim=1, fc_hidden_dim=64, fc_embed_dim=16,
    #                           n_layers=5, batchNorm=False, dropout=0.5)
    sum(p.numel() for p in model.parameters())    # the number of parameters in model

    # loss_mse = nn.MSELoss()
    def mse_loss(model, x, y):
        pred = model(x[:,:2], x[:,2:])
        loss = torch.nn.functional.mse_loss(pred, y)
        return loss

    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    tm = TorchModeling(model=model, device=device)
    tm.compile(optimizer=optimizer
                , loss_function = mse_loss
                , early_stop_loss = EarlyStopping(patience=5)
                )
    tm.train_model(train_loader=train_loader, epochs=100, display_earlystop_result=True, early_stop=False)
    # tm.test_model(test_loader=test_loader)
    tm.recompile(optimizer=optim.Adam(model.parameters(), lr=1e-4))


# --------------------------------------------------------------------------------------------------------------------------------
if example:
    with torch.no_grad():
        model.eval()
        pred = model(x_train[:,:2].to(device), x_train[:,2:].to(device))
        pred_arr = pred.to('cpu').numpy()


    plt.figure()
    plt.scatter(pred_arr.ravel(), y_train.numpy().ravel(), alpha=0.5)
    plt.legend(loc='upper right')
    plt.show()


# --------------------------------------------------------------------------------------------------------------------------------
if example:
    start_node, end_node = np.random.choice(np.arange(n_nodes), size=2, replace=False)
    # start_node, end_node = 12, 39
    # start_node, end_node = 16, 38
    # start_node, end_node = 48, 12
    start_node, end_node = 8, 44
    # start_node, end_node = 32, 47
    dijkstra = Dijkstra(node_map.adj_matrix)
    shortest_distance, path = dijkstra.dijkstra(start_node, end_node)
    feature = np.append(node_map.centers[start_node], node_map.centers[end_node])
    dist = shortest_distance

    x = torch.tensor(feature[np.newaxis,...].astype(np.float32)).to(device)
    with torch.no_grad():
        model.eval()
        pred = model(x[:,:2].to(device), x[:,2:].to(device))
        pred_arr = pred.to('cpu').numpy()

    print(pred.item(), dist)
    # start_node, end_node
    # visualize_graph(centers=node_map.centers, adjacent_matrix=node_map.adj_matrix,
    #                     path=path, distance=shortest_distance)


# --------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import torch
from torch import nn

# !pip install torch-geometric
from torch_geometric.nn import GCNConv

####################################################################################

# adjacent_list → edge_index
def adj_list_to_edge_index(adj_list):
    # edge 개수만큼 source node 반복
    source_nodes = [src for src, neighbors in adj_list.items() for _ in neighbors]
    # target node 전개
    target_nodes = [tgt for neighbors in adj_list.values() for tgt in neighbors]
    return np.array([source_nodes, target_nodes], dtype=np.int64)

# adjacent_matrix → edge_index
def adj_matrix_to_edge_index(adj_matrix):
    src, tgt = np.nonzero(adj_matrix)  # 간선 있는 위치
    edge_index = np.array([src, tgt])
    return edge_index


########################################################################################################
# (GCN Cov Layer with Scalar weigted edge) 
X = torch.rand((4, 3))  # 4 nodes, 3-dim features

edge_index = torch.tensor([
    [0, 1, 2, 3],  # source
    [1, 2, 3, 0]   # target
], dtype=torch.long)

edge_weight = torch.tensor([0.1, 0.5, 1.0, 0.3], dtype=torch.float)  # 4개 간선의 가중치

conv = GCNConv(in_channels=3, out_channels=2)
conv(X, edge_index, edge_weight=edge_weight)

# (GCN Cov Network) 
class GCN_Weighted(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, bias=True):
        super(GCN_Weighted, self).__init__()
        self.gcn_layers = nn.ModuleList([
            GCNConv(in_channels, hidden_channels, bias=bias)
            ,GCNConv(hidden_channels, hidden_channels, bias=bias)
            ,GCNConv(hidden_channels, out_channels, bias=bias)
        ])
        self.relu = nn.ReLU()
        self.edge_index = None
        self.edge_weight = None

    def forward(self, X, edge_index=None, edge_weight=None):
        edge_index = self.edge_index if edge_index is None else torch.IntTensor(edge_index).to(X.device)
        edge_weight = self.edge_weight if edge_weight is None else torch.FloatTensor(edge_weight).to(X.device)
        for i, layer in enumerate(self.gcn_layers):
            X = layer(X, edge_index, edge_weight)
            if i < len(self.gcn_layers) - 1:
                X = self.relu(X)
        return X
    
# Executing Full Model
X = torch.randn(4, 8)
A = np.array([
    [0, 1, 0, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 0]
    ])

edge_index = adj_matrix_to_edge_index(A)
edge_weight = torch.FloatTensor([0.46, 0.07, 0.83, 0.74, 0.37, 0.35, 0.79, 0.67])  # 4개 간선의 가중치
gcn = GCN_Weighted(in_channels=8, hidden_channels=16, out_channels=1)
gcn(X, edge_index, edge_weight)







########################################################################
########################################################################
# (GCN Cov Layer with Vector weigted edge) 
# from torch_geometric.nn import MessagePassing
# (Concept)
# class GCNConvEdge(MessagePassing):
#     def __init__(self, in_channels, edge_channels, out_channels, bias=True, **kwargs):
#         super().__init__(aggr='add')  # 'add', 'mean', 'max'
#         self.lin_node = nn.Linear(in_channels, out_channels, bias=bias, **kwargs)
#         self.lin_edge = nn.Linear(edge_channels, out_channels, bias=bias, **kwargs)
#         self.lin_final = nn.Linear(out_channels, out_channels, bias=bias, **kwargs)

#     def forward(self, x, edge_index, edge_attr):
#         x = self.lin_node(x)
#         edge_emb = self.lin_edge(edge_attr)
#         return self.propagate(edge_index, x=x, edge_attr=edge_emb)

#     def message(self, x_j, edge_attr):
#         # 메시지를 노드 j의 임베딩과 edge feature로부터 생성
#         return x_j + edge_attr  # 또는 concat 후 MLP 등 자유롭게

#     def update(self, aggr_out):
#         return self.lin_final(aggr_out)
    
# self.propagate(edge_index, x, edge_attr, norm) : message() → aggregate() → update() 를 자동으로 호출
#   [1] 메시지 생성       (message)
#   [2] 메시지 집계       (aggregate)
#   [3] 노드 상태 업데이트 (update)
#
# def message(self, x_j, edge_attr, norm) : 메시지를 생성하는 함수, 각 edge에 대해 호출
#
# def update(self, aggr_out)  : 메시지를 집계(aggregate)한 뒤 노드의 상태를 업데이트하는 함수로 node마다 한번씩 호출된다.


# (Full implemented GCN Layer with Weights)
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops
from torch_geometric.typing import OptTensor

class GCNConvLayer(MessagePassing):
    """
    Optimized GCNConv with optional vector-valued edge features.

    Args:
        - in_channels (int): Input node feature dimensionality.
        - edge_channels (int): Input edge feature dimensionality.
        - out_channels (int): Output feature dimensionality.
        - improved (bool, optional): Whether to use the improved GCN normalization
            (adds self‐loops with weight=2.0). (default: False)
        - cached (bool, optional): If True, caches the normalized adjacency
            (only use if the graph is static). (default: False)
        - add_self_loops (bool, optional): If True, adds self‐loops before
            normalization. (default: True)
        - normalize (bool, optional): If True, applies symmetric normalization
            D^{-1/2} A D^{-1/2}. (default: True)
        - bias (bool, optional): If True, adds a learnable bias term. (default: True)

    Features:
        - Optional vector edge features (edge_channels > 0).
        - Self-loop addition, symmetric normalization (D^-1/2 A D^-1/2).
        - Improved normalization (self-loop weight=2.0 when improved=True).
        - Single-pass caching of normalized adjacency + loop-padded edge_attr.
        - Final bias transform.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 edge_channels: int = 1,
                 improved: bool = False,
                 cached: bool = False,
                 add_self_loops: bool = True,
                 normalize: bool = True,
                 bias: bool = True):
        super().__init__(aggr='add')
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.edge_channels  = edge_channels
        self.improved       = improved
        self.cached         = cached
        self.add_self_loops = add_self_loops
        self.normalize      = normalize

        # Node and (optional) edge transforms
        self.lin_node  = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_edge  = nn.Linear(edge_channels, out_channels, bias=False) if edge_channels > 0 else None
        # self.lin_final = nn.Linear(out_channels, out_channels, bias=bias)

        # Buffers for caching static graph preprocessing
        self.register_buffer('_cached_edge_index', None)
        self.register_buffer('_cached_norm', None)
        if edge_channels > 0:
            self.register_buffer('_cached_edge_attr', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_node.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        # self.lin_final.reset_parameters()
        # clear caches
        self._cached_edge_index = None
        self._cached_norm       = None
        if self.edge_channels > 0:
            self._cached_edge_attr = None

    def forward(self,
                node: Tensor,
                edge_index: Tensor,
                edge_attr: OptTensor = None,
                size: tuple = None) -> Tensor:
        use_edge_attr = (edge_attr is not None and self.edge_channels > 0)
        N = node.size(self.node_dim)

        # Precompute and cache static processing once
        if self.cached:
            if self._cached_edge_index is None:
                # 1) Self-loop
                ei_loop, _ = add_self_loops(edge_index, num_nodes=N)
                # 2) Edge_attr padding
                if use_edge_attr:
                    loop_attr = edge_attr.new_zeros(N, self.edge_channels)
                    self._cached_edge_attr = torch.cat([edge_attr, loop_attr], dim=0)
                # 3) Normalization
                ei_norm, norm = gcn_norm(
                    ei_loop,
                    edge_weight=None,
                    num_nodes=N,
                    improved=self.improved,
                    add_self_loops=False,
                    dtype=node.dtype)
                self._cached_edge_index = ei_norm
                self._cached_norm       = norm
            # reuse
            edge_index = self._cached_edge_index
            norm       = self._cached_norm
            edge_attr  = self._cached_edge_attr if self.edge_channels > 0 else None
        else:
            # dynamic processing
            if self.add_self_loops:
                edge_index, _ = add_self_loops(edge_index, num_nodes=N)
                if use_edge_attr:
                    loop_attr = edge_attr.new_zeros(N, self.edge_channels)
                    edge_attr  = torch.cat([edge_attr, loop_attr], dim=0)      # self‐loop zero-padding
            if self.normalize:
                edge_index, norm = gcn_norm(
                    edge_index,
                    edge_weight=None,
                    num_nodes=N,
                    improved=self.improved,
                    add_self_loops=False,
                    dtype=node.dtype)
            else:
                norm = None

        # Node embedding
        node_emb = self.lin_node(node)
        edge_emb = self.lin_edge(edge_attr) if use_edge_attr else None
        norm = norm if norm is not None else node.new_ones(edge_index.size(1))
        
        # Message passing
        out = self.propagate(edge_index, x=node_emb, edge_attr=edge_emb, norm=norm, size=size)
        return out
        # return self.lin_final(out)

    def message(self, x_j: Tensor, edge_attr: OptTensor, norm: Tensor) -> Tensor:
        if edge_attr is not None:
            return norm.view(-1, 1) * (x_j + edge_attr)
        else:
            return norm.view(-1, 1) * x_j 

    def update(self, aggr_out: Tensor) -> Tensor:
        return aggr_out


X = torch.randn(4, 8)   # node_feature
edge_index = torch.randint(0,4, size=(2,10))

# CGN without EdgeWeights
gcn_noweight = GCNConvLayer(in_channels=8, out_channels=4)
gcn_noweight(X, edge_index)

# CGN with 1-dim EdgeWeights
edge_weight = torch.FloatTensor([0.46, 0.07, 0.83, 0.74, 0.37, 0.35, 0.79, 0.67, 0.17, 0.92]).unsqueeze(-1)  # 4개 간선의 가중치
gcn_weight1 = GCNConvLayer(in_channels=8, out_channels=4, edge_channels=1)
gcn_weight1(X, edge_index, edge_weight)
gcn_weight1(X, edge_index)

# CGN with N-dim EdgeWeights
edge_weight = torch.rand(edge_index.shape[-1], 5)
gcn_weight2 = GCNConvLayer(in_channels=8, out_channels=4, edge_channels=5)
gcn_weight2(X, edge_index, edge_weight)
gcn_weight2(X, edge_index)




################################################
# (graph_training_example)
# 1 Graph : (한 시점에서의) 1 Observations 
#   ㄴ node : 각 node (ex. 도시)
#   ㄴ node_feature : 각 node의 정보 (ex. 도시에 대한 정보: 인구, 인구유동량, ... )
#   ㄴ edge : 각 edge (ex. 도시와 도시사이의 도로)
#   ㄴ edge_feature : 각 edge의 정보 (ex. 도로의 정보: 너비, 차량 유동량)
#   ㄴ target : (1) Graph-level prediction : (Regression) ex. 전체 유동인구는 몇명이나 되나?
#                                            (Classification) ex. 해당 graph는 새벽/아침/점심/저녁/밤 시간대 언제인가?
#              (2) Node-level prediction : (Regression) ex. 도시(node)마다(혹은 특정 node의) 1시간 뒤의 인구 유입량은 얼마나 될 것 같나?
#                                          (Classification) ex. 도시(node)마다(혹은 특정 node의) 유형은 소도시/시단위도시/광역도시 중에 어떤 것인가?
#              (3) Edge-level prediction : (Regression) ex. 도로(edge)마다 (혹은 특정 edge) 1시간 뒤의 도로 정체 정도는 얼마나 되나? 0~1
#                                          (Classification) ex. 도로(edge)마다 (혹은 특정 edge) 어떤 유형의 도로인가? 골목길/국도/고속도로
#       => Target task의 경우 위의 3가지 경우가 혼합되어 task를 수행할 수도 있다. (Graph + Node)

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops, to_networkx



# def generate_random_graphs(num_graphs=5, nodes=(30,8), edges=(None, 0) **kwargs)



# Random Graph Data Generator
def generate_random_graphs(num_graphs=5,
                           num_nodes=30,
                           node_feature_dim=8,
                           edge_feature_dim=0,
                           
                           target_graph_classes=None,
                           target_node_classes=None,
                           target_edge_classes=None,
                           fixed_relation: bool = True,
                           p_edge=None,
                           seed=None):
    gen = torch.Generator()
    if seed is None:
        seed = torch.randint(low=0, high=int(1e+6), size=(1,)).item()
    gen.manual_seed(seed)
    
    dataset = []
    # 고정 관계(fixed_relation=True)인 경우 그래프 구조를 하나만 생성
    if fixed_relation:
        if p_edge is None:
            p_edge = 1/torch.sqrt(torch.FloatTensor([num_nodes])).item()
        mask = (torch.rand(num_nodes, num_nodes, generator=gen) < p_edge)
        mask = torch.triu(mask, diagonal=1) | torch.tril(mask.t(), -1)
        edge_index_fixed = mask.nonzero(as_tuple=False).t().contiguous()
        edge_index_fixed, _ = remove_self_loops(edge_index_fixed)
    
    # num_graphs개의 Graph생성
    for i in range(num_graphs):
        data = {}
        # Node features
        x = torch.randn(num_nodes, node_feature_dim)
        data.update(dict(x=x))
        
        # edge_index 결정
        if fixed_relation:
            edge_index = edge_index_fixed
        else:
            mask = (torch.rand(num_nodes, num_nodes, generator=gen) < p_edge)
            mask = torch.triu(mask, diagonal=1) | torch.tril(mask.t(), -1)
            edge_index = mask.nonzero(as_tuple=False).t().contiguous()
            edge_index, _ = remove_self_loops(edge_index)
        data.update(dict(edge_index=edge_index))
        
        # Edge features
        num_edges = edge_index.size(1)
        if edge_feature_dim > 0:
            edge_attr = torch.randn(num_edges, edge_feature_dim)
            data.update(dict(edge_attr=edge_attr))
        
        # Graph-Level Target
        if target_graph_classes is not None:
            if target_graph_classes > 0:    # Classification Task
                y_graph = torch.randint(target_graph_classes, (1,), generator=gen)
            else:                   # Regression Task
                y_graph = torch.rand((1, ), generator=gen)
            data.update(dict(y_graph=y_graph))
        
        # Node-Level Target
        if target_node_classes is not None:
            if target_node_classes > 0:    # Classification Task
                y_node = torch.randint(target_node_classes, (num_nodes,), generator=gen)
            else:                   # Regression Task
                y_node = torch.rand((num_nodes, ), generator=gen)
            data.update(dict(y_node=y_node))
        
        # Edge-Level Target
        if target_edge_classes is not None:
            if target_edge_classes > 0:    # Classification Task
                y_edge = torch.randint(target_edge_classes, (num_edges,), generator=gen)
            else:                   # Regression Task
                y_edge = torch.rand((num_edges, ), generator=gen)
            data.update(dict(y_edge=y_edge))
        
        # Data구성
        data = Data(**data)
        dataset.append(data)
    return dataset

################################################################################
# Data-Set & Data-Loader (By Graph Generator)
# 하이퍼파라미터
num_graphs          = 100
num_nodes           = 10
node_feature_dim    = 8
hidden_channels     = 16
edge_feature_dim    = 4

graph_classes       = 5
node_classes        = 3
edge_classes        = 0     # regression

batch_size      = 4

# 데이터 준비
dataset = generate_random_graphs(
    num_graphs=num_graphs,
    num_nodes=num_nodes,
    node_feature_dim=node_feature_dim,
    edge_feature_dim=edge_feature_dim,
    target_graph_classes=graph_classes,
    target_node_classes=node_classes,
    target_edge_classes=edge_classes,
    fixed_relation=True
)

# GraphDataLoader
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(dataset[0])

# 첫 그래프 한 개를 NetworkX로 변환 후 시각화
first = dataset[0]
G = to_networkx(first, to_undirected=True)
plt.figure(figsize=(5,5))
nx.draw(G, with_labels=True, node_size=200, font_size=8)
plt.title('Sample Graph Structure')
plt.show()


loader_iter = iter(loader)
batch_data = next(loader_iter)
# DataBatch(x=[40, 8], edge_index=[2, 144], edge_attr=[144, 4], 
#           y_graph=[4], y_node=[40], y_edge=[144], 
#           batch=[40],     # batch_index
#           ptr=[5])        # data_index
#
# batch_data.num_nodes
# batch_data.num_graphs
# batch_data.num_edges
################################################################################




# (Trainning Settings) ########################################################
# 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train Loop Function
def train_loop(model, dataloader, epochs, optimizer, loss_function, target_name):
    device = next(model.parameters()).device
    train_losses = []

    model.train()
    for epoch in range(1, epochs+1):
        total_loss = 0
        num_batches   = 0
        
        for data in dataloader:
            batch_size = len(data.ptr)-1
            data = data.to(device)
            optimizer.zero_grad()
            pred_y = model(data)
            target_y = eval(f'data.{target_name}') 
            # print(pred_y.shape, target_y.shape)
            loss = loss_function(pred_y, target_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()   # 한 배치 평균 손실을 더하고
            num_batches += 1        # batch counter 하나 올리고
        
        # 에포크 당 배치 평균 손실
        avg_loss = total_loss / num_batches  
        train_losses.append(avg_loss)
        print(f"Epoch {epoch:02d}, Loss: {avg_loss:.4f}")
    return train_losses

# 학습 곡선 시각화
def visualize_train_result(train_losses, title=None):
    plt.figure()
    if title is not None:
        plt.title(title)
    else:
        plt.title('GCN Training Curve')
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.grid(True)
    plt.show()
#############################################################################




#############################################################################
# (graph-level prediction) ##################################################
from torch_geometric.nn import global_mean_pool  # 또는 sum/max_pool
class GraphNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_channels,
                edge_pooling=False, cached=True):
        super().__init__()
        self.conv1 = GCNConvLayer(in_channels=in_channels, out_channels=hidden_channels, edge_channels=edge_channels, cached=cached)
        self.relu = nn.ReLU()
        self.conv2 = GCNConvLayer(in_channels=hidden_channels, out_channels=hidden_channels, edge_channels=edge_channels, cached=cached)
        self.global_mean_pool = global_mean_pool        # batch size로 mean_pooling
        self.node_linear  = nn.Linear(hidden_channels, out_channels)
        
        # (Optional adding edge pooling)
        self.edge_pooling = edge_pooling
        self.edge_linear = nn.Linear(edge_channels, hidden_channels)
        self.concat_linear = nn.Linear(hidden_channels*2, out_channels)
        
    def forward(self, data):
        x, edge_index, edge_attr, batch_index = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.relu(x)
        
        # global node pooling
        x_pool = self.global_mean_pool(x, batch_index)        # batch size로 mean_pooling
            
        if self.edge_pooling is False:  # Only Node-pooling
            return self.node_linear(x_pool)
        
        else:       # Node-pooling + Edge pooling
            e = self.edge_linear(edge_attr)
            src, _ = edge_index     # 어느 그래프에 속한 엣지인지 표시 (source 노드 batch 정보 활용)
            edge_batch = batch_index[src]
            e_pool = self.global_mean_pool(e, edge_batch)
            h = torch.cat([x_pool, e_pool], dim=-1) 
            return self.concat_linear(h) 
        

epochs          = 50
lr              = 0.01

# model setting
graph_model = GraphNet(in_channels=node_feature_dim, hidden_channels=hidden_channels,
            out_channels=graph_classes, edge_channels=edge_feature_dim, edge_pooling=False).to(device)
# graph_model(batch_data.to(device))
graph_optimizer = optim.Adam(graph_model.parameters(), lr=lr)
graph_loss_function = nn.CrossEntropyLoss()

# train_loop
graph_train_result = train_loop(model=graph_model, dataloader=loader, epochs=epochs, 
                                optimizer=graph_optimizer, loss_function=graph_loss_function, 
                                target_name='y_graph')
# vis_result
visualize_train_result(graph_train_result, title='GCN graph-level training')



#############################################################################
# (node-level prediction) ###################################################
class NodeNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_channels, cached=True):
        super().__init__()
        self.conv1 = GCNConvLayer(in_channels=in_channels, out_channels=hidden_channels, edge_channels=edge_channels, cached=cached)
        self.relu = nn.ReLU()
        self.conv2 = GCNConvLayer(in_channels=hidden_channels, out_channels=hidden_channels, edge_channels=edge_channels, cached=cached)
        self.linear  = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, data):
        x, edge_index, edge_attr, batch_index = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.relu(x)
        
        return x
        # return torch.log_softmax(x, dim=1)

epochs          = 50
lr              = 0.01

# model setting
node_model = NodeNet(in_channels=node_feature_dim, hidden_channels=hidden_channels,
            out_channels=node_classes, edge_channels=edge_feature_dim).to(device)
# node_model(batch_data.to(device))
node_optimizer = optim.Adam(node_model.parameters(), lr=lr)
node_loss_function = nn.CrossEntropyLoss()

# train_loop
node_train_result = train_loop(model=node_model, dataloader=loader, epochs=epochs, 
                               optimizer=node_optimizer, loss_function=node_loss_function, 
                               target_name='y_node')
# vis_result
visualize_train_result(node_train_result, title='GCN node-level training')




#############################################################################
# (edge-level prediction) ###################################################
class EdgeNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_channels, cached=True):
        super().__init__()
        self.conv1 = GCNConvLayer(in_channels=in_channels, out_channels=hidden_channels, edge_channels=edge_channels, cached=cached)
        self.relu = nn.ReLU()
        self.conv2 = GCNConvLayer(in_channels=hidden_channels, out_channels=hidden_channels, edge_channels=edge_channels, cached=cached)
        
        # Edge MLP 헤드: [h_i, h_j, edge_attr] → 예측
        mlp_in_dim = hidden_channels * 2 + edge_channels    # node x, y 좌표 + edge정보
        self.edge_mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        
    def forward(self, data):
        x, edge_index, edge_attr, batch_index = data.x, data.edge_index, data.edge_attr, data.batch
        
        # node_embedding
        x = self.conv1(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.relu(x)
        
        # edeg embedding
        row, col = edge_index
        h_i = x[row]               # [E, hidden]
        h_j = x[col]               # [E, hidden]
        
        # edge_attr 없으면 zero‐vector로 채우기
        if edge_attr is None:
            edge_feat = x.new_zeros(edge_index.size(1), 0)
        else:
            edge_feat = edge_attr   # [E, edge_channels]
        
        # [h_i || h_j || e_ij]
        edge_repr = torch.cat([h_i, h_j, edge_feat], dim=1)  # [E, 2*hidden + edge_channels]

        # edge 단위 prediction        
        return self.edge_mlp(edge_repr) 
        # return torch.log_softmax(x, dim=1)



epochs          = 50
lr              = 0.01

# model setting
edge_model = EdgeNet(in_channels=node_feature_dim, hidden_channels=hidden_channels,
            out_channels=1, edge_channels=edge_feature_dim).to(device)
# edge_model(batch_data.to(device))
edge_optimizer = optim.Adam(node_model.parameters(), lr=lr)
edge_loss_function = nn.MSELoss()

# train_loop
edge_train_result = train_loop(model=edge_model, dataloader=loader, epochs=epochs, 
                               optimizer=edge_optimizer, loss_function=edge_loss_function, 
                               target_name='y_edge')
# vis_result
visualize_train_result(edge_train_result)





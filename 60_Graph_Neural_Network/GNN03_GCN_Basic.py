import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import torch
from torch import nn


##################################################################################
# 【 Conceptional GCN 】 #########################################################
# adj_mat_normalization_random_walk
def adj_mat_normalization_random_walk(adj_mat, add_self_loop=True):
    adj_mat_np = np.asarray(adj_mat).astype(float)
    if add_self_loop:
        adj_mat_np += np.eye(adj_mat.shape[0])
    deg = adj_mat_np.sum(axis=1, keepdims=True)  # degree
    normalized_adj = adj_mat_np / deg  # mean aggregation
    return normalized_adj

# GCNLayer_RandomWalk
class GCNLayer_RandomWalk(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.U = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, X, A_hat=None):
        A_hat_tensor = self.A_hat_tensor if A_hat is None else torch.FloatTensor(A_hat).to(X.device)
        neighbor_agg = A_hat_tensor @ X
        return self.W(neighbor_agg) + self.U(X)

X = torch.randn(4, 8)   # node embedding feature 
A = np.array([
    [0, 1, 0, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 0]
    ])

# Executing Single Layer
adj_mat_normalized = adj_mat_normalization_random_walk(A)
gcn = GCNLayer_RandomWalk(in_dim=8, out_dim=8)
gcn(X, adj_mat_normalized)

# GCN_RandomWalk Model
class GCN_RandomWalk(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, bias=False):
        super().__init__()
        self.gcn_layers_rw = nn.ModuleList([
            GCNLayer_RandomWalk(in_dim, hidden_dim, bias=bias)
            ,GCNLayer_RandomWalk(hidden_dim, hidden_dim, bias=bias)
            ,GCNLayer_RandomWalk(hidden_dim, out_dim, bias=bias)
        ])
        self.relu = nn.ReLU()
    
    def forward(self, X, A_hat=None):
        A_hat_tensor = self.A_hat_tensor if A_hat is None else torch.FloatTensor(A_hat).to(X.device)
        for i, layer in enumerate(self.gcn_layers_rw):
            X = layer(X, A_hat_tensor)
            if i < len(self.gcn_layers_rw) - 1:
                X = self.relu(X)
        return X

# Executing Full Model
adj_mat_normalized = adj_mat_normalization_random_walk(A)
gcn_rw = GCN_RandomWalk(in_dim=8, hidden_dim=16, out_dim=1)
gcn_rw(X, adj_mat_normalized)




###################################################################################
# 【 GCN [Kipf & Welling, 2017] 】 ################################################
# adj_mat_normalization_symmetric_normalization
def adj_mat_normalization(adj_mat, add_self_loop=True):
    adj_mat_np = np.asarray(adj_mat).astype(float)
    if add_self_loop:
        adj_mat_np += np.eye(adj_mat.shape[0])
    deg = adj_mat_np.sum(axis=1)
    deg[deg == 0] = 1  # divide-by-zero 방지
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    return deg_inv_sqrt @ adj_mat_np @ deg_inv_sqrt

# GCN Layer (symmetric normalization)
class GCNLayer_symmetric(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.A_hat_tensor = None

    def forward(self, X, A_hat=None):
        """
        A_hat: normalized adjacency matrix (symmetric normalized + self-loop)
        """
        A_hat_tensor = self.A_hat_tensor if A_hat is None else torch.FloatTensor(A_hat).to(X.device)
        return self.linear(A_hat_tensor @ X)
    
X = torch.randn(4, 8)   # node embedding feature 
A = np.array([
    [0, 1, 0, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 0]
    ])

# Executing Single Layer
adj_mat_normalized = adj_mat_normalization(A)
gcn = GCNLayer_symmetric(in_dim=8, out_dim=8)
gcn(X, adj_mat_normalized)


# GCN_Symmetric Model
class GCN_symmetric(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, bias=True):
        super().__init__()
        self.gcn_layers = nn.ModuleList([
            GCNLayer_symmetric(in_dim, hidden_dim, bias=bias)
            ,GCNLayer_symmetric(hidden_dim, hidden_dim, bias=bias)
            ,GCNLayer_symmetric(hidden_dim, out_dim, bias=bias)
        ])
        self.relu = nn.ReLU()
        self.A_hat_tensor = None
        
    def forward(self, X, A_hat=None):
        A_hat_tensor = self.A_hat_tensor if A_hat is None else torch.FloatTensor(A_hat).to(X.device)
        
        for i, layer in enumerate(self.gcn_layers):
            X = layer(X, A_hat_tensor)
            if i < len(self.gcn_layers) - 1:
                X = self.relu(X)
        return X

# Executing Full Model
adj_mat_normalized = adj_mat_normalization(A)
gcn = GCN_symmetric(in_dim=8, hidden_dim=16, out_dim=1)
gcn(X, adj_mat_normalized)

###################################################################################












###################################################################################
###################################################################################
# !pip install torch-geometric
from torch_geometric.nn import GCNConv

# Edge Index
edge_index = np.array([
    [[0, 0, 1, 2, 2, 3, 3, 3],      # source
     [1, 3, 3, 0, 3, 0, 1, 2]]   # target
    ]).astype(float)
# column-wise direction : 0->1, 0->3, 1->3, 2->0, ...

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

A = np.array([
    [0, 1, 0, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 0]
    ])

adj_matrix_to_edge_index(A)
# array([[0, 0, 1, 2, 2, 3, 3, 3],
#        [1, 3, 3, 0, 3, 0, 1, 2]], dtype=int64)

########################################################################


########################################################################
# (GCN Cov Layer) 
X = torch.randn(4, 8)   # node embedding feature 
A = np.array([
    [0, 1, 0, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 0]
    ])

edge_index_torch = torch.IntTensor(adj_matrix_to_edge_index(A))
gcn_layer = GCNConv(in_channels=8, out_channels=8, add_self_loops=True, normalize=True)
#       add_self_loops: A_tilde = A + I (default: True)
#       normalize: A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt (default: True)
#       bias : bias term (default: True) 
gcn_layer(X, edge_index_torch)


# (GCN Cov Network) 
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, bias=True):
        super(GCN, self).__init__()
        self.gcn_layers = nn.ModuleList([
            GCNConv(in_channels, hidden_channels, bias=bias)
            ,GCNConv(hidden_channels, hidden_channels, bias=bias)
            ,GCNConv(hidden_channels, out_channels, bias=bias)
        ])
        self.relu = nn.ReLU()
        self.edge_index = None

    def forward(self, X, edge_index=None):
        edge_index = self.edge_index if edge_index is None else torch.IntTensor(edge_index).to(X.device)
        for i, layer in enumerate(self.gcn_layers):
            X = layer(X, edge_index)
            if i < len(self.gcn_layers) - 1:
                X = self.relu(X)
        return X
    
# Executing Full Model
edge_index = adj_matrix_to_edge_index(A)
gcn = GCN(in_channels=8, hidden_channels=16, out_channels=1)
gcn(X, edge_index)


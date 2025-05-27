import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops
from torch_geometric.typing import OptTensor


###########################################################################
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
    

###########################################################################    
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
    
    
###########################################################################
# GCNCovLayer (with/without edge vector weight)
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
                x: Tensor,
                edge_index: Tensor,
                edge_attr: OptTensor = None,
                size: tuple = None) -> Tensor:
        use_edge_attr = (edge_attr is not None and self.edge_channels > 0)
        N = x.size(self.node_dim)

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
                    dtype=x.dtype)
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
                    dtype=x.dtype)
            else:
                norm = None

        # Node embedding
        node_emb = self.lin_node(x)
        edge_emb = self.lin_edge(edge_attr) if use_edge_attr else None
        norm = norm if norm is not None else x.new_ones(edge_index.size(1))
        
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
# edge_index = torch.randint(0,4, size=(2,10))

# # CGN without EdgeWeights
# gcn_noweight = GCNConvLayer(in_channels=8, out_channels=4)
# gcn_noweight(X, edge_index)

# # CGN with 1-dim EdgeWeights
# edge_weight = torch.FloatTensor([0.46, 0.07, 0.83, 0.74, 0.37, 0.35, 0.79, 0.67, 0.17, 0.92]).unsqueeze(-1)  # 4개 간선의 가중치
# gcn_weight1 = GCNConvLayer(in_channels=8, out_channels=4, edge_channels=1)
# gcn_weight1(X, edge_index, edge_weight)
# gcn_weight1(X, edge_index)

# # CGN with N-dim EdgeWeights
# edge_weight = torch.rand(edge_index.shape[-1], 5)
# gcn_weight2 = GCNConvLayer(in_channels=8, out_channels=4, edge_channels=5)
# gcn_weight2(X, edge_index, edge_weight)
# gcn_weight2(X, edge_index)

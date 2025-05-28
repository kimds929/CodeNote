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


# X = torch.randn(4, 8)   # node_feature
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





###########################################################################
# EGCN Layer (with/without edge vector weight)

def scatter(src: torch.Tensor,
            index: torch.LongTensor,
            dim: int = 0,
            dim_size: int = None,
            reduce: str = 'sum') -> torch.Tensor:
    """
    Pure-PyTorch scatter function supporting sum, mean, amax, amin, prod.
    - Requires PyTorch >= 2.0 for full reduce support via scatter_reduce_.
    - Falls back to sum and mean for older versions.
    """
    # 1) dim_size 결정
    if dim_size is None:
        dim_size = int(index.max().item()) + 1

    # 2) 출력 텐서 준비
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = torch.zeros(*out_shape, dtype=src.dtype, device=src.device)

    # 3) index 확장
    if src.dim() == 1:
        idx_exp = index
    else:
        view_shape = [1] * src.dim()
        view_shape[dim] = src.size(dim)
        idx_exp = index.view(view_shape).expand_as(src)

    # 4) PyTorch 2.0+ scatter_reduce_ 사용
    if hasattr(out, "scatter_reduce_"):
        out.scatter_reduce_(dim, idx_exp, src, reduce=reduce, include_self=False)
        return out

    # 5) fallback for older PyTorch
    if reduce == 'sum':
        return out.scatter_add_(dim, idx_exp, src)
    elif reduce == 'mean':
        summed = out.scatter_add_(dim, idx_exp, src)
        # count groups
        ones = torch.ones_like(src)
        count = torch.zeros(*out_shape, dtype=src.dtype, device=src.device)
        count = count.scatter_add_(dim, idx_exp, ones)
        return summed / count.clamp(min=1)
    else:
        raise NotImplementedError(f"Fallback for reduce='{reduce}' is not implemented.")
    

class EGConvLayer(nn.Module):
    """
    Equivariant Graph Convolution Layer (EGCL) as defined in #EGNN.
    Implements formulas (3)~(6):
      (3) m_{ij} = \phi_e(h_i, h_j, ||x_i - x_j||^2, a_{ij})
      (4) x_i^{l+1} = x_i^l + C \sum_{j \neq i} (x_i - x_j) \phi_x(m_{ij})
      (5) m_i = \sum_{j \neq i} m_{ij}
      (6) h_i^{l+1} = \phi_h(h_i, m_i)
      
    Args:
        in_features: Dimension of input node features
        out_features: Dimension of output node features
        hidden_features: Hidden dimension for MLPs
        edge_attr_dim: Dimensionality of optional edge attributes
        aggr: Aggregation method ('add', 'mean', etc.) for feature messages
    """
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 hidden_features: int = None,
                 edge_features: int = 0,
                 aggr='sum'
                 ):
        super(EGConvLayer, self).__init__()
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features if hidden_features is not None else max(in_features, out_features)
        self.edge_features = edge_features
        self.aggr = aggr
        
        # Edge MLP φ_e
        self.concat_dim = 2*in_features + 1 + self.edge_features
        self.phi_e = nn.Sequential(
            nn.Linear(self.concat_dim, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.hidden_features)
        )
        # Scalar MLP φ_x (maps m_{ij} to a scalar weight)
        self.phi_x = nn.Linear(self.hidden_features, 1)
        
        # Node MLP φ_h
        self.phi_h = nn.Sequential(
            nn.Linear(in_features + self.hidden_features, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.out_features)
        )

    def forward(self, node, coordinate, edge_index, edge_attr=None):
        """
        Args:
            node (Tensor): [N, F] node features
            coordinate (Tensor): [N, D] node coordinates
            edge_index (LongTensor): [2, E] edge indices (source, target)
            edge_attr (Tensor): [E, edge_features] edge attributes a_{ij}
        Returns:
            node (Tensor): [N, out_features] updated node features
            coordinate (Tensor): [N, D] updated node coordinates
        """
        src, dst = edge_index  # j -> i
        h_j, h_i = node[src], node[dst]
        x_j, x_i = coordinate[src], coordinate[dst]

        # (3) Compute distance feature ||x_i - x_j||²
        dist = (x_i - x_j)
        sqdist = dist.pow(2).sum(dim=1, keepdim=True)

        concat_inputs = [h_i, h_j, sqdist]
        if self.edge_features > 0:
            concat_inputs.append(edge_attr)
            
        # Edge message m_ij = φ_e(...)
        m_ij = self.phi_e(torch.cat(concat_inputs, dim=-1))

        # (4) coordinate update φ_x(m_ij)
        w_ij = self.phi_x(m_ij)  # scalar weights [E,1]
        vec = dist * w_ij  # [E, D]

        # normalization constant C = 1/(N-1)
        N = node.size(0)       # node_size
        C = 1.0 / (N - 1)

        # aggregate coordinate updates per node i
        delta_x = scatter(vec, dst, dim=0, reduce='sum') * C
        x_new = coordinate + delta_x

        # (5) message aggregation
        m_i = scatter(m_ij, dst, dim=0, reduce=self.aggr)

        # (6) feature update
        h_new = self.phi_h(torch.cat([node, m_i], dim=1))

        return h_new, x_new

# X = torch.randn(4, 8)   # node_feature
# edge_index = torch.randint(0,4, size=(2,10))
# coordinates = torch.randn(4,2)

# # CGN without EdgeWeights
# gcn_noweight = EGConvLayer(in_features=8, out_features=4)
# gcn_noweight(X, coordinates, edge_index)

# # CGN with 1-dim EdgeWeights
# edge_weight = torch.FloatTensor([0.46, 0.07, 0.83, 0.74, 0.37, 0.35, 0.79, 0.67, 0.17, 0.92]).unsqueeze(-1)  # 4개 간선의 가중치
# gcn_weight1 = EGConvLayer(in_features=8, out_features=4, edge_features=1)
# gcn_weight1(X, coordinates, edge_index, edge_weight)

# # CGN with N-dim EdgeWeights
# edge_weight = torch.rand(edge_index.shape[-1], 5)
# gcn_weight2 = EGConvLayer(in_features=8, out_features=4, edge_features=5)
# gcn_weight2(X, coordinates, edge_index, edge_weight)


################################################################################################
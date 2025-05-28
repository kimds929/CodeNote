# !pip install torch-scatter
import torch
import torch.nn as nn
torch.cuda.is_available()

# from torch_geometric.nn import GCNConv
# from torch_scatter import scatter
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

# # 사용 예시
# src = torch.tensor([1,2,3,4,5], dtype=torch.float)
# idx = torch.tensor([0,1,0,1,2])
# print(scatter(src, idx, dim=0, reduce='sum'))   # tensor([4., 6., 5.])
# print(scatter(src, idx, dim=0, reduce='mean'))  # tensor([2., 3., 5.])
# print(scatter(src, idx, dim=0, reduce='max'))   # tensor([3., 4., 5.])
# print(scatter(src, idx, dim=0, reduce='min'))   # tensor([1., 2., 5.])


################################################################################################
# (torch_scatter) ##############################################################################
src = torch.tensor([1, 2, 3, 4, 5])
index = torch.tensor([0, 0, 1, 1, 1])

# index 0 그룹: 1 + 2 = 3
# index 1 그룹: 3 + 4 + 5 = 12
out = scatter(src, index, dim=0, reduce='sum')
print(out)  # tensor([ 3, 12])


################################################################################################
# (EGNN) #######################################################################################
# from torch_scatter import scatter
import torch
import torch.nn as nn

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



X = torch.randn(4, 8)   # node_feature
edge_index = torch.randint(0,4, size=(2,10))
coordinates = torch.randn(4,2)

# CGN without EdgeWeights
gcn_noweight = EGConvLayer(in_features=8, out_features=4)
gcn_noweight(X, coordinates, edge_index)

# CGN with 1-dim EdgeWeights
edge_weight = torch.FloatTensor([0.46, 0.07, 0.83, 0.74, 0.37, 0.35, 0.79, 0.67, 0.17, 0.92]).unsqueeze(-1)  # 4개 간선의 가중치
gcn_weight1 = EGConvLayer(in_features=8, out_features=4, edge_features=1)
gcn_weight1(X, coordinates, edge_index, edge_weight)

# CGN with N-dim EdgeWeights
edge_weight = torch.rand(edge_index.shape[-1], 5)
gcn_weight2 = EGConvLayer(in_features=8, out_features=4, edge_features=5)
gcn_weight2(X, coordinates, edge_index, edge_weight)







##########################################################################################
# (graph_training_example)
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import httpimport
remote_url = 'https://raw.githubusercontent.com/kimds929/'

import torch.optim as optim

with httpimport.remote_repo(f"{remote_url}/CodeNote/main/60_Graph_Neural_Network/"):
    from GNN_utils import scatter, generate_random_graphs, visualize_train_result, train_loop


from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops, to_networkx

# from torch.utils.data import Dataset, DataLoader
# from torch_geometric.loader import DataLoader as GeoDataLoader



class EGNNNet(nn.Module):
    """Simple EGNN for node-level prediction."""
    def __init__(self, in_features, out_features, hidden_features, edge_features):
        super(EGNNNet, self).__init__()
        self.layer1 = EGConvLayer(in_features, hidden_features, hidden_features, edge_features)
        self.layer2 = EGConvLayer(hidden_features, hidden_features, hidden_features, edge_features)
        # Final MLP classifier for node-level prediction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, data):
        node, coordinate, edge_index, edge_attr = data.x, data.coordinate, data.edge_index, data.edge_attr
        node, coordinate = self.layer1(node, coordinate, edge_index, edge_attr)
        node, coordinate = self.layer2(node, coordinate, edge_index, edge_attr)
        # Node-level logith
        out = self.classifier(node)
        return out
    
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

for data in dataset:
    # weight 텐서 생성 (25×4)
    coordinate = torch.randn(10, 2)
    # Data 객체에 새 속성으로 추가
    data.coordinate = coordinate


# GraphDataLoader
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



# 첫 그래프 한 개를 NetworkX로 변환 후 시각화
first = dataset[0]
G = to_networkx(first, to_undirected=True)
plt.figure(figsize=(5,5))
nx.draw(G, with_labels=True, node_size=200, font_size=8)
plt.title('Sample Graph Structure')
plt.show()

loader_iter = iter(loader)
batch_data = next(loader_iter)
# print(np.__version__)


#################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


epochs          = 50
lr              = 0.01

# model setting
node_model = EGNNNet(in_features=node_feature_dim, hidden_features=hidden_channels,
            out_features=graph_classes, edge_features=edge_feature_dim).to(device)
# node_model(batch_data.to(device))
node_optimizer = optim.Adam(node_model.parameters(), lr=lr)
node_loss_function = nn.CrossEntropyLoss()

# train_loop
graph_train_result = train_loop(model=node_model, dataloader=loader, epochs=epochs, 
                                optimizer=node_optimizer, loss_function=node_loss_function, 
                                target_name='y_node')
# vis_result
visualize_train_result(graph_train_result, title='GCN node-level training')


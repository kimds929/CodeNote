import numpy as np
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops, to_networkx


###########################################################################
def adj_list_to_matrix(adj_list, weighted=False):
    """
    인접 리스트 (list of lists)를 인접 행렬로 변환 (가중치 optional)
    """
    num_nodes = len(adj_list)
    dtype = float if weighted else int
    adj_mat = np.zeros((num_nodes, num_nodes), dtype=dtype)

    if weighted:
        # 가중치 있는 경우
        for src, neighbors in enumerate(adj_list):
            if neighbors:  # skip empty rows
                rows, cols = zip(*neighbors)  # 빠르게 분해
                adj_mat[src, list(rows)] = cols
    else:
        # 가중치 없는 경우
        for src, neighbors in enumerate(adj_list):
            adj_mat[src, neighbors] = 1  # numpy broadcasting 사용

    return adj_mat


###########################################################################
# adj_mat_normalization_random_walk
def adj_mat_normalization_random_walk(adj_mat, add_self_loop=True):
    adj_mat_np = np.asarray(adj_mat).astype(float)
    if add_self_loop:
        adj_mat_np += np.eye(adj_mat.shape[0])
    deg = adj_mat_np.sum(axis=1, keepdims=True)  # degree
    normalized_adj = adj_mat_np / deg  # mean aggregation
    return normalized_adj


###########################################################################
# adj_mat_normalization_symmetric_normalization
def adj_mat_normalization(adj_mat, add_self_loop=True):
    adj_mat_np = np.asarray(adj_mat).astype(float)
    if add_self_loop:
        adj_mat_np += np.eye(adj_mat.shape[0])
    deg = adj_mat_np.sum(axis=1)
    deg[deg == 0] = 1  # divide-by-zero 방지
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    return deg_inv_sqrt @ adj_mat_np @ deg_inv_sqrt


###########################################################################
# adjacent_list → edge_index
def adj_list_to_edge_index(adj_list):
    # edge 개수만큼 source node 반복
    source_nodes = [src for src, neighbors in adj_list.items() for _ in neighbors]
    # target node 전개
    target_nodes = [tgt for neighbors in adj_list.values() for tgt in neighbors]
    return np.array([source_nodes, target_nodes], dtype=np.int64)


###########################################################################
# adjacent_matrix → edge_index
def adj_matrix_to_edge_index(adj_matrix):
    src, tgt = np.nonzero(adj_matrix)  # 간선 있는 위치
    edge_index = np.array([src, tgt])
    return edge_index



###########################################################################
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


###########################################################################
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

###########################################################################
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
    
    
###########################################################################
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

import numpy as np
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

# 비가중치 그래프
adj_list = [
    [0,1],      # node 0 → 0, 1
    [1,3],      # node 1 → 1, 3
    [0, 3],     # node 2 → 0, 3
    [0,1,2,3]   # node 3 → 0, 1, 2, 3
]

# 가중치 그래프
adj_list_w = [
    [(1, 1.5), (2, 2.0)],
    [(0, 1.0)],
    [(0, 2.0), (3, 3.5)],
    []
]

adj_mat1 = adj_list_to_matrix(adj_list)
adj_mat2 = adj_list_to_matrix(adj_list_w, weighted=True)


# (visualize) #####################################################
import networkx as nx
import matplotlib.pyplot as plt
G = nx.from_numpy_array(adj_mat1)  # 방향 없는 경우는 이걸 사용
# G = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph())  # 방향 그래프 원하면 DiGraph, 아니면 Graph

pos = nx.spring_layout(G, seed=42)  # 노드 위치 자동 조정
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray', arrows=True)
plt.title("Graph from Adjacency Matrix")
plt.show()



# (Neighborhood Normalization) #####################################################
# adjacent matrix
A = np.array([
    [0, 1, 0, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 0]
])

# Random‐Walk Normalization -------------------------------------------
tilde_A = A + np.eye(A.shape[0]) 
tilde_D = np.diag(tilde_A.sum(1))  # degree matrix
hat_A = np.linalg.inv(tilde_D) @ tilde_A   # normalized

# adj_mat_normalization_random_walk
def adj_mat_normalization_random_walk(adj_mat, add_self_loop=True):
    adj_mat_np = np.asarray(adj_mat).astype(float)
    if add_self_loop:
        adj_mat_np += np.eye(adj_mat.shape[0])
    deg = adj_mat_np.sum(axis=1, keepdims=True)  # degree
    normalized_adj = adj_mat_np / deg  # mean aggregation
    return normalized_adj

adj_mat_normalization_random_walk(A)


# Symmetric Normalization ------------------------------------------
tilde_A = A + np.eye(A.shape[0]) 
D_inv_sqrt = np.diag(1/np.sqrt(tilde_A.sum(1)))
hat_A = D_inv_sqrt @ tilde_A @ D_inv_sqrt
np.round(hat_A,3)

# adj_mat_normalization_symmetric_normalization
def adj_mat_normalization(adj_mat, add_self_loop=True):
    adj_mat_np = np.asarray(adj_mat).astype(float)
    if add_self_loop:
        adj_mat_np += np.eye(adj_mat.shape[0])
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(adj_mat_np.sum(axis=1)))
    return deg_inv_sqrt @ adj_mat_np @ deg_inv_sqrt

adj_mat_normalization(A)




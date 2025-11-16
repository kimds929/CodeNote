import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt


# rng = np.random.default_rng(0)
rng = np.random.RandomState()

def normalize(v, eps=1e-9):
    n = np.linalg.norm(v) + eps
    return v / n

bianry_to_decimal = lambda binary_list: int(''.join(map(str, binary_list)), 2)

user_dim = 5
user_vec = (rng.rand(user_dim) > 0.5).astype(np.int64)




class AgentTS:
    id = -1
    true_compatibility_dim = rng.randint(20,50)
    
    def __init__(self, user_dim=4, compatibility_dim=16, lambda0=1, noise_sigma=0.1, forgetting_decay=1):
        # uhknown true compatibility vector
        self.user_dim = user_dim
        
        self.true_compatibility = normalize(rng.uniform(size=AgentTS.true_compatibility_dim))
        
        self.user_vec = (rng.rand(user_dim) > 0.5).astype(np.int64)
        self.user_vec_group = int(''.join(map(str, self.user_vec)), 2)
        
        AgentTS.id += 1
        self.id = AgentTS.id
        
        # info
        self.compatibility_dim = compatibility_dim
        self.noise_sigma = noise_sigma
        self.lambda0 = lambda0      # λ_0·I : Ridge Reguarization parameters (얼마나 prior를 신뢰할지?)
        self.forgetting_decay = forgetting_decay
        
        self.A = lambda0 * np.eye(compatibility_dim)
        self.b = np.zeros(compatibility_dim)
        
        self.Sigma = np.linalg.inv(self.A)      # 현재 추정된 compatibility Covariance
        self.mu = rng.uniform(size=compatibility_dim)*0.1
    
    def update(self, compatibility_other, r, Sigma_other=None, inplace=True):
        x_compatibility = compatibility_other.copy()
        A = self.forgetting_decay * self.A
        b = self.forgetting_decay * self.b
        
        E_xx = np.outer(x_compatibility, x_compatibility)
        if Sigma_other is not None:
            E_xx += Sigma_other
        A += (1/ self.noise_sigma) * E_xx    # += 1/(σ^2) * (X_I^T X_I)
        b += (1/ self.noise_sigma) * x_compatibility * r        # +=  1/(σ^2) * (X_I r)
        
        Sigma = np.linalg.inv(A)
        mu = Sigma @ b
        
        if inplace is True:
            self.A = A
            self.b = b
            self.Sigma = Sigma
            self.mu = mu
        
        return mu, Sigma
    
    def sampling(self):
        return rng.multivariate_normal(self.mu, self.Sigma)
    
    def __repr__(self):
        return f"UserAgent_{self.id}"

# COMPATIBILITY_DIM = 16
# users = [AgentTS(user_dim=4, compatibility_dim=COMPATIBILITY_DIM) for _ in range(5)]


####################################################################################################
class TrueCompatibilityNet(nn.Module):
    def __init__(self, compatibility_dim):
        super().__init__()
        self.compatibility_dim = compatibility_dim
        self.block = nn.Sequential(
            nn.Linear(compatibility_dim, compatibility_dim*2)
            ,nn.ReLU()
            ,nn.Linear(compatibility_dim*2, compatibility_dim*4)
            ,nn.ReLU()
            ,nn.Linear(compatibility_dim*4, compatibility_dim*2)
            ,nn.ReLU()
            ,nn.Linear(compatibility_dim*2, compatibility_dim)
        )
        
    def forward(self, x):
        return self.block(torch.FloatTensor(x)).detach().to('cpu').numpy()

# true_net = TrueCompatibilityNet(AgentTS.true_compatibility_dim)
# true_net(users[0].true_compatibility)
# true_net( np.stack([user.true_compatibility for user in users]) ).shape



####################################################################################################
# ---------- Helpers: upper-tri vectorization ----------
def triu_indices(k, device=None, dtype=torch.long):
    return torch.triu_indices(k, k, offset=0, device=device, dtype=dtype)

def vec_to_triu(vec, k, device=None):
    """
    vec: (..., k*(k+1)//2)
    return: (..., k, k) upper-triangular matrix with zeros elsewhere
    """
    idx = triu_indices(k, device=vec.device)
    out = vec.new_zeros(*vec.shape[:-1], k, k)
    out[..., idx[0], idx[1]] = vec
    return out

def triu_to_vec(U):
    """
    U: (..., k, k) upper-triangular
    return: (..., k*(k+1)//2)
    """
    k = U.shape[-1]
    idx = triu_indices(k, device=U.device)
    return U[..., idx[0], idx[1]]

####################################################################################################

class CategorialEmbedding(nn.Module):
    def __init__(self, n_features, num_embeddings, embedding_dim):
        super().__init__()
        self.nf, self.ne, self.ed = n_features, num_embeddings, embedding_dim
        self.embedding = nn.Embedding(n_features * num_embeddings, embedding_dim)

    def forward(self, x):
        # x: (..., n_features), long
        *batch, F = x.shape
        device = x.device
        feature_idx = torch.arange(F, device=device).view(*([1]*len(batch)), F).expand(*x.shape)
        flat_idx = feature_idx * self.ne + x                        # (..., F)
        out = self.embedding(flat_idx)                                      # (..., F, ed)
        return out

class ResidualConnection(nn.Module):
    def __init__(self, block, shortcut=None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut or (lambda x: x)
    
    def forward(self, x):
        return self.block(x) + self.shortcut(x)


######################################################################################################
class ThompsonSamplingFeatureMap(nn.Module):
    def __init__(self, output_dim, user_dim, user_embed_dim, preference_embed_dim, 
                hidden_dim=64, max_users=1000, dropout=0.1, lambda0=1e-5):
        super().__init__()
        self.user_info_embedding = CategorialEmbedding(user_dim, max_users, embedding_dim=user_embed_dim)
        self.preference_embedding = nn.Embedding(max_users, embedding_dim=preference_embed_dim)
        
        concat_dim = (user_dim*user_embed_dim) + preference_embed_dim
        self.backbone = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim)
            ,nn.ReLU()
            ,ResidualConnection(nn.Sequential(
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim)
                ,nn.ReLU()
                ,nn.Dropout(dropout)
                )
            )
            ,ResidualConnection(nn.Sequential(
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim)
                ,nn.ReLU()
                ,nn.Dropout(dropout)
                )
            )
        )
        
        # head
        self.output_dim = output_dim
        self.tdim = output_dim * (output_dim+1)//2
        
        self.softplus = nn.Softplus(beta=1.0)
        self.mu_head = nn.Linear(hidden_dim, output_dim)
        self.triu_head = nn.Linear(hidden_dim, self.tdim)
        
        # cache diagonal indices in the triu-vector
        self.lambda0 = lambda0
        di_rows, di_cols = torch.triu_indices(output_dim, output_dim, offset=0)
        diag_mask = (di_rows == di_cols)
        self.register_buffer("diag_mask", diag_mask)  # shape [tdim] : (register_buffer : 역전파로 업데이트되지 않지만 모델과 함께 저장/로드(move to cuda 등)되는 값)

    def vec_to_triu(self, vec, k):
        """
        vec: (..., k*(k+1)//2)
        return: (..., k, k) upper-triangular matrix (zeros elsewhere)
        """
        idx = torch.triu_indices(k, k, device=vec.device)
        out = vec.new_zeros(*vec.shape[:-1], k, k)
        out[..., idx[0], idx[1]] = vec
        return out

    def forward(self, user_vec:torch.LongTensor, user_id:torch.LongTensor):
        batch_size = user_vec.shape[:-1]
        
        user_emb = self.user_info_embedding(user_vec).view(*batch_size, -1)
        pref_emb = self.preference_embedding(user_id).view(*batch_size, -1)
        x_concat = torch.cat([user_emb, pref_emb], dim=-1)
        
        # backbone
        x_latent = self.backbone(x_concat)
        
        # mu
        mu = self.mu_head(x_latent)
        
        # U
        raw_triu = self.triu_head(x_latent)  # unconstrained
        diag_part = raw_triu[..., self.diag_mask]
        off_part  = raw_triu[..., ~self.diag_mask]
        
        # constrain diagonal: positive with softplus + epsilon
        diag_part = self.softplus(diag_part) + self.lambda0
        
        U_triu = raw_triu.clone()       # .clone()도 gradient 전파 가능 # 파이토치에서 inplace 연산은 그래프 추적 중에 이전 연산에 영향을 줄 수 있기 때문에 종종 에러를 발생시킨다.
        U_triu[..., self.diag_mask] = diag_part
        U_triu[..., ~self.diag_mask] = off_part
        
        U = self.vec_to_triu(U_triu, self.output_dim)   # (..., k, k)
        Lambda = U.transpose(-2, -1) @ U 
        
        return mu, U, Lambda

    # @torch.no_grad()
    def U_to_L(self, U:torch.Tensor):
        """
        Given U (upper of precision), compute Sigma = (U^T U)^{-1} without explicit inverse.
        Solve (U^T) Y = I  ->  L = U^{-T}
        """
        k = U.shape[-1]
        I = torch.eye(k, device=U.device, dtype=U.dtype).expand(U.shape[:-2] + (k, k))
        # U^T is lower-triangular
        L = torch.linalg.solve_triangular(U.transpose(-2, -1), I, upper=False, left=True)
        return L
    
    def gaussian(self, mu, U, requires_grad=False):
        ctx = torch.enable_grad() if requires_grad else torch.no_grad()
        with ctx:
            L = self.U_to_L(U)
            Sigma = L @ L.transpose(-2,-1)
        return mu, Sigma
    
    def forward_gaussian(self, user_vec:torch.LongTensor, user_id:torch.LongTensor):
        mu, U, Lambda = self.forward(user_vec, user_id)
        return self.gaussian(mu, U)
    
    def sampling(self, mu, U, n_samples=1, requires_grad=False):
        ctx = torch.enable_grad() if requires_grad else torch.no_grad()
        
        with ctx:
            L = self.U_to_L(U)
            multivariate_gaussian_dist = torch.distributions.MultivariateNormal(loc=mu, scale_tril=L)
            # multivariate_gaussian_dist = torch.distributions.MultivariateNormal(loc=mu, precision_matrix=Lambda)
            samples = multivariate_gaussian_dist.rsample((n_samples,))  # (n_sample, batch , dim) rsample: 미분가능, sample: 미분불가
            samples = torch.movedim(samples, 0, -2) # (batch, n_sample, dim)
        return samples
    
    # @torch.no_grad()
    def forward_sampling(self, user_vec:torch.LongTensor, user_id:torch.LongTensor, n_samples=1):
        """
        Thompson-style sampling: c = mu + U^{-T} z,  z ~ N(0,I)
        returns (..., n_samples, k)
        """
        mu, U, Lambda = self.forward(user_vec, user_id)
        return self.sampling(mu, U)
        
    

####################################################################################################
# users[0].id
# users[0].user_vec
# users[0].user_vec_group
# model = ThompsonSamplingFeatureMap(COMPATIBILITY_DIM, users[0].user_dim,
#                                    user_embed_dim=3, preference_embed_dim=8)
# a = torch.LongTensor([users[0].user_vec]).repeat(5,1)
# b = torch.LongTensor([[users[0].id]]).repeat(5,1)
# a = torch.LongTensor([users[0].user_vec])
# b = torch.LongTensor([[users[0].id]])

# model(a,b)[0].shape
# model(a,b)[1].shape
# model(a,b)[2].shape
# model.forward_gaussian(a,b)
# model.forward_sample(a,b, n_samples=1).shape






# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Global Optima Matching
# 1) hungarian matching
from scipy.optimize import linear_sum_assignment
def hungarian(similarity_matrix):
    S_max = similarity_matrix.max()         # 행렬 최대값
    cost_matrix = S_max - similarity_matrix     # 비용행렬
    np.fill_diagonal(cost_matrix, np.inf)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    optimal_matching_values = similarity_matrix[row_ind, col_ind]
    matching_result = np.concatenate([row_ind[np.newaxis,...].T, col_ind[np.newaxis,...].T], axis=-1).T
    return matching_result, optimal_matching_values


# 2) 일반 그래프 최대 가중 매칭(Blossom)
import networkx as nx
def blossom_max_weight_matching(similarity_matrix):
    n = similarity_matrix.shape[0]
    G = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            w = similarity_matrix[i, j]
            if np.isfinite(w):
                G.add_edge(i, j, weight=float(w))
    # 최대 가중 + 최대 카디널리티(가능한 많이 페어링)
    matching = nx.algorithms.matching.max_weight_matching(
        G, maxcardinality=True, weight='weight'
    )

    # matching result processing
    pair_half = list(matching)  # (i, j) 튜플들의 집합
    pair_half_np =  np.stack(pair_half)
    pair_np = np.concatenate([pair_half_np, pair_half_np[:, ::-1]], axis=0)
    sort_idx = np.argsort(pair_np[:,0])
    matching_result = pair_np[sort_idx].T

    # optimal_matching_values
    row_ind = matching_result[0]
    col_ind = matching_result[1]
    optimal_matching_values = similarity_matrix[row_ind, col_ind]
    return matching_result, optimal_matching_values

# -----------------------------------------------------------------------------------------------

# Local Optima Matching
import numpy as np
def greedy_matching(similarity_matrix):
    """
    유사도 행렬을 한 번 정렬한 후 탐욕적으로 매칭을 수행합니다.
    O(N^2 log N) 복잡도로, O(N^3)에 가까운 반복적 argmax 방식보다 빠릅니다.

    Args:
        similarity_matrix (np.ndarray): N x N 유사도 행렬.

    Returns:
        tuple: (matching_result, optimal_matching_values)
    """
    N = similarity_matrix.shape[0]

    # 1. 유사도 행렬에서 매칭 후보 쌍(i, j)만 추출합니다. (i < j 조건)
    # 삼각 행렬 인덱스 추출 (자기 자신과의 매칭 방지)
    iu = np.triu_indices(N, k=1)

    # 2. 모든 유사도 값을 추출하고 내림차순으로 정렬합니다.
    flat_similarities = similarity_matrix[iu]

    # 유사도 값이 높은 순서(내림차순)로 정렬하기 위한 인덱스
    sort_idx = np.argsort(flat_similarities)[::-1]

    # 정렬된 유사도 값과 해당 쌍의 인덱스
    sorted_values = flat_similarities[sort_idx]
    sorted_pairs = np.column_stack(iu)[sort_idx] # (row, col) 쌍

    # 매칭 결과를 저장할 리스트와 매칭된 유저 집합
    matched_pairs = []
    matched_values = []
    matched_indices = set()

    # 3. 정렬된 쌍을 순회하며 매칭합니다. (O(N^2) 순회)
    for value, (i, j) in zip(sorted_values, sorted_pairs):
        # 이미 i 또는 j가 매칭되었는지 확인 (O(1) set lookup)
        if i not in matched_indices and j not in matched_indices:
            # 매칭 확정
            matched_pairs.append((i, j))
            matched_values.append(value)

            # 유저 제거 (논리적 제거)
            matched_indices.add(i)
            matched_indices.add(j)

            # 매칭이 절반 이상 완료되면 종료
            if len(matched_indices) >= N - 1:
                break

    if not matched_pairs:
        return np.array([[], []]), np.array([])

    # 4. 결과 포맷팅 (이 부분은 이전 코드와 동일)
    matched_pairs_np = np.array(matched_pairs)

    # (i, j)뿐만 아니라 (j, i)도 포함
    all_pairs = np.concatenate([matched_pairs_np, matched_pairs_np[:, ::-1]], axis=0)
    all_values = np.concatenate([np.array(matched_values), np.array(matched_values)], axis=0)

    # 행 인덱스(user_i) 기준으로 정렬
    sort_idx = np.argsort(all_pairs[:, 0])
    matching_result = all_pairs[sort_idx].T
    optimal_matching_values = all_values[sort_idx]

    return matching_result, optimal_matching_values



class KNN_GreedyMatching():
    def __init__(self, k=None):
        self.k = k

    def knn_greedy_matching(self, similarity_matrix):
        """
        유사도 행렬을 기반으로, 각 유저의 k명의 가장 유사한 후보군 내에서만
        탐욕적으로 매칭을 수행하는 근사 매칭 함수입니다.

        Args:
            similarity_matrix (np.ndarray): N x N 유사도 행렬.
            k (int): 각 유저가 고려할 이웃의 최대 수.

        Returns:
            tuple: (matching_result, optimal_matching_values)
                matching_result (np.ndarray): 매칭된 쌍의 인덱스 (2, M).
                optimal_matching_values (np.ndarray): 매칭된 쌍의 유사도 값 (M,).
        """
        N = similarity_matrix.shape[0]
        k = self.k
        if k is None or (k >= N - 1):
            # k가 N보다 크면 그냥 일반적인 탐욕적 매칭과 동일
            return greedy_matching(similarity_matrix)

        # 1. k-NN 기반 후보 쌍만 추출하여 정렬합니다.

        # 1-1. 각 유저별 k명의 이웃 인덱스를 찾습니다.
        # np.argsort는 기본적으로 오름차순. [::-1]로 내림차순 정렬 후, k개 선택.
        # 자기 자신(가장 유사도 1.0)을 제외하고, 인접한 k개의 인덱스를 찾습니다.

        # 유사도 행렬의 각 행(유저 i)에서 k+1번째로 유사한 인덱스까지
        # 즉, 자기 자신(0순위) 포함 k+1개
        k_plus_1_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1][:, :k+1]

        # 1-2. 매칭 후보 (쌍) 리스트 생성
        # 모든 유저 i에 대해, 그들의 k-NN 후보 j만을 추출합니다. (i < j 조건 만족)
        # 이는 중복 쌍을 피하고 대칭성을 깨지 않기 위함입니다.

        candidate_pairs = set()

        for i in range(N):
            # 유저 i의 k-NN 후보 목록
            neighbors = k_plus_1_indices[i]

            for j in neighbors:
                # 1. 자기 자신과의 매칭 제외
                if i == j:
                    continue
                # 2. 이미 (j, i)로 처리된 쌍을 피하기 위해 정규화된 쌍 (min, max)만 추가
                pair = tuple(sorted((i, j)))
                candidate_pairs.add(pair)

        # 2. 후보 쌍의 유사도 값을 추출하고 내림차순으로 정렬합니다.

        # 후보 쌍 (i, j)와 해당 유사도 값 추출
        pairs_list = np.array(list(candidate_pairs))
        flat_similarities = similarity_matrix[pairs_list[:, 0], pairs_list[:, 1]]

        # 유사도 값이 높은 순서(내림차순)로 정렬하기 위한 인덱스
        sort_idx = np.argsort(flat_similarities)[::-1]

        sorted_values = flat_similarities[sort_idx]
        sorted_pairs = pairs_list[sort_idx]

        # 3. 탐욕적으로 매칭합니다. (나머지 과정은 optimized_greedy_matching과 동일)

        matched_pairs = []
        matched_values = []
        matched_indices = set()

        for value, (i, j) in zip(sorted_values, sorted_pairs):
            if i not in matched_indices and j not in matched_indices:
                # 매칭 확정
                matched_pairs.append((i, j))
                matched_values.append(value)

                # 유저 제거
                matched_indices.add(i)
                matched_indices.add(j)

                if len(matched_indices) >= N - 1:
                    break

        if not matched_pairs:
            return np.array([[], []]), np.array([])

        # 4. 결과 포맷팅
        matched_pairs_np = np.array(matched_pairs)

        # (i, j)뿐만 아니라 (j, i)도 포함하여 정렬
        all_pairs = np.concatenate([matched_pairs_np, matched_pairs_np[:, ::-1]], axis=0)
        all_values = np.concatenate([np.array(matched_values), np.array(matched_values)], axis=0)

        # 행 인덱스(user_i) 기준으로 정렬
        sort_idx = np.argsort(all_pairs[:, 0])
        matching_result = all_pairs[sort_idx].T
        optimal_matching_values = all_values[sort_idx]

        return matching_result, optimal_matching_values

# ----------------------------------------------------------------------------------------------




# ########################################################################################
# USER_DIM = 4
# COMPATIBILITY_DIM = 16
# USER_EMBEDDING_DIM = 3
# PREFERENCE_EMBEDDING_DIM = 8
# users = [AgentTS(user_dim=USER_DIM, compatibility_dim=COMPATIBILITY_DIM) for _ in range(6)]

# # (True matching) -------------------------------------------------------------------
# users_true_vec = np.array([user.true_compatibility for user in users])
# users_true_vec

# # # true net
# # true_net = TrueCompatibilityNet(AgentTS.true_compatibility_dim)
# # users_true_vec = true_net(users_true_vec)
# SM_true = users_true_vec @ users_true_vec.T

# # true_matching, true_matching_values = hungarian(SM_true)  # True matching
# true_matching, true_matching_values = blossom_max_weight_matching(SM_true)  # True matching
# print('< true_matching >')
# print(true_matching)
# print(true_matching_values)
# # ------------------------------------------------------------------------------------

# # sample matching
# TS_model = ThompsonSamplingFeatureMap(COMPATIBILITY_DIM, USER_DIM, USER_EMBEDDING_DIM, PREFERENCE_EMBEDDING_DIM)

# users_vec = torch.LongTensor( np.stack([user.user_vec for user in users]) )
# users_id = torch.LongTensor( np.stack([[user.id] for user in users]) )

# compatibility_samples = TS_model.forward_sample(users_vec, users_id).squeeze(-2)
# compatibility_samples_np = compatibility_samples.detach().to('cpu').numpy()

# SM_samples = compatibility_samples_np @ compatibility_samples_np.T

# # sample_matching, sample_matching_values = hungarian(SM_samples)
# sample_matching, sample_matching_values = blossom_max_weight_matching(SM_samples)
# print('< sampling_matching >')
# print(sample_matching)
# print(sample_matching_values)

########################################################################################
# After Maching, Revealed Rewards
def revealed_reward(true_similarity_matrix, sample_matching, noise_std=0.1):
    mu_rewards = true_similarity_matrix.copy()
    # true_similarity_matrix_copy = true_similarity_matrix.copy()
    # np.fill_diagonal(true_similarity_matrix_copy, -np.inf)
    # np.exp(true_similarity_matrix_copy) / np.exp(true_similarity_matrix_copy).sum(axis=1)
    # mu_rewards = 1/2*(np.exp(true_similarity_matrix_copy) / np.exp(true_similarity_matrix_copy).sum(axis=0)) + 1/2*(np.exp(true_similarity_matrix_copy) / np.exp(true_similarity_matrix_copy).sum(axis=0)).T

    reward_noise_gen = rng.random(size=mu_rewards.shape)*noise_std-noise_std/2
    reward_noise_gen = (reward_noise_gen + reward_noise_gen.T)/2
    np.fill_diagonal(reward_noise_gen, 0)

    revealed_rewards = mu_rewards+reward_noise_gen
    # revealed_rewards = np.clip(mu_rewards+reward_noise_gen, a_min=0, a_max=1)
    return revealed_rewards[sample_matching[0], sample_matching[1]]

# rewards_obs = revealed_reward(SM_true, sample_matching)
# print(f"revealed rewards : {rewards_obs}")
########################################################################################



# rewards = revealed_reward(SM_true, sample_matching)
# mu_target, Sigma_target = users[0].update(compatibility_samples_np[3], rewards[0], users[3].Sigma, inplace=True)



def calculate_target_pair(idx_i:float, idx_j:int, 
                    reward:float, users:AgentTS, compatibility_samples:np.array, inplace:bool=True):
    mu_target_i, Sigma_target_i = users[idx_i].update(compatibility_samples[idx_j], reward, users[idx_j].Sigma, inplace=inplace)
    mu_target_j, Sigma_target_j = users[idx_j].update(compatibility_samples[idx_i], reward, users[idx_i].Sigma, inplace=inplace)
    return (mu_target_i, Sigma_target_i), (mu_target_j, Sigma_target_j)
    
def calculate_targets(rewards, users, compatibility_samples, sample_matching, inplace=True):
    mu_targets = [np.array([])]*len(users)
    Sigma_targets = [np.array([])]*len(users)
    for reward, match in zip(rewards, sample_matching.T):
        idx_i = match[0]
        idx_j = match[1]
        
        if len(mu_targets[idx_i]) ==0 and len(mu_targets[idx_j]) ==0:
            (mu_target_i, Sigma_target_i), (mu_target_j, Sigma_target_j) = calculate_target_pair(idx_i, idx_j, reward, users, compatibility_samples, inplace)
            mu_targets[idx_i] = mu_target_i
            mu_targets[idx_j] = mu_target_j
            Sigma_targets[idx_i] = Sigma_target_i
            Sigma_targets[idx_j] = Sigma_target_j
    return mu_targets, Sigma_targets


##########################################################################################

def symmetrize(S): 
    return 0.5*(S + S.transpose(-2,-1))

@torch.no_grad()
def project_to_spd(S, min_eig=1e-6, use_float64=True):
    orig_dtype = S.dtype
    S = symmetrize(torch.nan_to_num(S))             # 비대칭/NaN/Inf 정리
    if use_float64 and S.dtype != torch.float64:
        S = S.to(torch.float64)

    # 1) cholesky_ex 시도 + 작은 지터
    I = torch.eye(S.shape[-1], device=S.device, dtype=S.dtype).expand_as(S)
    jitter_list = [0.0, 1e-10, 3e-10, 1e-9, 3e-9, 1e-8, 1e-7, 1e-6]
    ok = None; L = None
    for eps in jitter_list:
        try:
            L, info = torch.linalg.cholesky_ex(S + eps*I)
            if torch.any(info>0):
                continue
            S = S + eps*I
            ok = True
            break
        except RuntimeError:
            continue

    # 2) 그래도 실패하면 SVD로 SPD 재구성
    if ok is None:
        U, s, _ = torch.linalg.svd(S, full_matrices=False)
        s = torch.clamp(s, min=min_eig)
        S = (U * s.unsqueeze(-2)) @ U.transpose(-2,-1)
        S = symmetrize(S)
        L  = torch.linalg.cholesky(S)

    if S.dtype != orig_dtype:
        S = S.to(orig_dtype); L = L.to(orig_dtype)
    return S, L   # SPD, lower-Chol

def kl_target_to_pred_from_U(mu_tgt, Sigma_tgt, mu_hat, U_hat, reduction="mean", min_eig=1e-6):
    assert mu_tgt.shape == mu_hat.shape
    *_, k = mu_tgt.shape

    # --- 타깃 SPD 투영 ---
    Sigma_tgt_spd, L_tgt = project_to_spd(Sigma_tgt, min_eig=min_eig)
    logdet_Sigma_tgt = 2.0 * torch.log(torch.diagonal(L_tgt, dim1=-2, dim2=-1)).sum(-1)

    # --- 예측: 정밀도 촐레스키 ---
    diagU = torch.diagonal(U_hat, dim1=-2, dim2=-1)
    logdet_Sigma_hat = -2.0 * torch.log(diagU).sum(-1)

    delta = (mu_hat - mu_tgt).unsqueeze(-1)   # (..., k, 1)
    y = U_hat @ delta
    quad = (y.transpose(-2,-1) @ y).squeeze(-1).squeeze(-1)

    Lambda_hat = U_hat.transpose(-2,-1) @ U_hat
    tr_term = torch.einsum('...ij,...ij->...', Lambda_hat, Sigma_tgt_spd)

    kl = 0.5 * (quad + tr_term - logdet_Sigma_hat + logdet_Sigma_tgt - k)
    if reduction=="mean": return kl.mean()
    if reduction=="sum":  return kl.sum()
    return kl

# -----------------------------------------------------------------------



# def kl_target_to_pred_from_U(mu_tgt: torch.Tensor,
#                              Sigma_tgt: torch.Tensor,
#                              mu_hat: torch.Tensor,
#                              U_hat: torch.Tensor,
#                              reduction: str = "mean"):
#     """
#     KL[ N(mu_tgt, Sigma_tgt) || N(mu_hat, Sigma_hat) ]
#     with predicted precision via U_hat (upper-Cholesky of precision).
#     Uses:
#       Lambda_hat = U_hat^T U_hat
#       logdet(Sigma_hat) = - 2 * sum(log diag(U_hat))
#       quad = (mu_hat - mu_tgt)^T Lambda_hat (mu_hat - mu_tgt) = || U_hat (mu_hat - mu_tgt) ||^2
#       trace = tr(Lambda_hat * Sigma_tgt) = sum_ij Lambda_ij * Sigma_tgt_ij
#     """
#     assert mu_tgt.shape == mu_hat.shape
#     *batch, k = mu_tgt.shape

#     # log det(Sigma_tgt) via cholesky_ex (더 안정적)
#     L_tgt, info = torch.linalg.cholesky_ex(Sigma_tgt)  # lower
#     if torch.any(info > 0):
#         # 수치적으로 불안하면 약간의 jitter를 주고 재시도
#         eps = 1e-6
#         Sigma_tgt = Sigma_tgt + eps * torch.eye(k, device=Sigma_tgt.device, dtype=Sigma_tgt.dtype)
#         L_tgt = torch.linalg.cholesky(Sigma_tgt)
#     logdet_Sigma_tgt = 2.0 * torch.log(torch.diagonal(L_tgt, dim1=-2, dim2=-1)).sum(dim=-1)  # (...,)

#     # 예측 쪽: 정밀도와 로그행렬식
#     # log det(Sigma_hat) = - 2 * sum log diag(U_hat)
#     logdet_Sigma_hat = -2.0 * torch.log(torch.diagonal(U_hat, dim1=-2, dim2=-1)).sum(dim=-1)  # (...,)

#     # quad term: (mu_hat - mu_tgt)^T Lambda_hat (mu_hat - mu_tgt) = || U_hat * delta ||^2
#     delta = (mu_hat - mu_tgt).unsqueeze(-1)                # (..., k, 1)
#     y = U_hat @ delta                                      # (..., k, 1)
#     quad = (y.transpose(-2, -1) @ y).squeeze(-1).squeeze(-1)  # (...,)

#     # trace term: tr(Lambda_hat * Sigma_tgt)
#     Lambda_hat = U_hat.transpose(-2, -1) @ U_hat           # (..., k, k)
#     tr_term = torch.einsum('...ij,...ij->...', Lambda_hat, Sigma_tgt)  # (...,)

#     kl = 0.5 * (quad + tr_term - logdet_Sigma_hat + logdet_Sigma_tgt - k)  # (...,)

#     if reduction == "mean":
#         return kl.mean()
#     elif reduction == "sum":
#         return kl.sum()
#     return kl

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################

N_USERS = 8
USER_DIM = 4
COMPATIBILITY_DIM = 8
USER_EMBEDDING_DIM = 3
PREFERENCE_EMBEDDING_DIM = 8

users = [AgentTS(user_dim=USER_DIM, compatibility_dim=COMPATIBILITY_DIM, forgetting_decay=1) for _ in range(N_USERS)]

# (True matching) -------------------------------------------------------------------
users_true_vec = np.array([user.true_compatibility for user in users])
users_true_vec

# # true net
# true_net = TrueCompatibilityNet(AgentTS.true_compatibility_dim)
# users_true_vec = true_net(users_true_vec)
SM_true = users_true_vec @ users_true_vec.T

# true_matching, true_matching_values = hungarian(SM_true)  # True matching
true_matching, true_matching_values = blossom_max_weight_matching(SM_true)  # True matching
print('< true_matching >')
print(true_matching)
print(true_matching_values)
# ------------------------------------------------------------------------------------
##########################################################################################

# # matching
# rewards_obs_list = []

# # matching
# matching_function = greedy_matching
# for _ in tqdm(range(300)):
#     # # user vector / id
#     # users_vec = np.stack([user.user_vec for user in users])
#     # users_id = np.stack([[user.id] for user in users])
    
#     # sampling 
#     users_samples_vec = np.stack([user.sampling() for user in users])
    
#     # calculate similarity matrix
#     SM_samples = users_samples_vec @ users_samples_vec.T
#     # np.round(SM_samples,2)
       
#     # users matching
#     sample_matching, sample_matching_values = matching_function(SM_samples)
    
#     # obseve rewards
#     rewards_obs = revealed_reward(SM_true, sample_matching, noise_std=0.2)
#     rewards_obs
#     rewards_obs_list.append(np.sum(rewards_obs))
    
#     # Users parameter Update
#     calculate_targets(rewards_obs, users, users_samples_vec, sample_matching, inplace=True)

# plt.plot(rewards_obs_list)
# plt.axhline(np.sum(true_matching_values), color='red')
# plt.ylim(0, None)
# plt.show()
# # [user.theta_true for user in users]
# # [user.mu for user in users]

# # sample matching
# users_samples_vec = np.stack([user.sampling() for user in users])
# SM_samples = users_samples_vec @ users_samples_vec.T


# # sample_matching, sample_matching_values = hungarian(SM_samples)
# sample_matching, sample_matching_values = blossom_max_weight_matching(SM_samples)
# print('< sampling_matching >')
# print(sample_matching)
# print(sample_matching_values)







##########################################################################################
##########################################################################################
# sample matching
TS_model = ThompsonSamplingFeatureMap(COMPATIBILITY_DIM, USER_DIM, USER_EMBEDDING_DIM, PREFERENCE_EMBEDDING_DIM)

optimizer = optim.Adam(TS_model.parameters(), lr=1e-5)
# users_vec = torch.LongTensor( np.stack([user.user_vec for user in users]) )
# users_id = torch.LongTensor( np.stack([[user.id] for user in users]) )

# mu_Hat, U_Hat, Lambda_Hat = TS_model(users_vec,users_id)
# mu_Hat
# U_Hat




# matching
rewards_obs_list = []

# matching
# matching_function = KNN_GreedyMatching(k=int(np.sqrt(SM_true.shape[0]))).knn_greedy_matching   # hungarian, blossom_max_weight_matching, greedy_matching, KNN_GreedyMatching(k=10).knn_greedy_matching

N_UPDATES = 300
matching_function = greedy_matching
for i in range(50):
    # user vector / id
    users_vec = torch.LongTensor( np.stack([user.user_vec for user in users]) )
    users_id = torch.LongTensor( np.stack([[user.id] for user in users]) )
    
    # forward : mu, U, Lambda
    mu_Hat, U_Hat, Lambda_Hat = TS_model(users_vec, users_id) # Gradient
    
    # sampling 
    compatibility_samples = TS_model.sampling(mu_Hat, U_Hat).squeeze(-2)
    compatibility_samples_np = compatibility_samples.to('cpu').numpy()
    
    # calculate similarity matrix
    SM_samples = compatibility_samples_np @ compatibility_samples_np.T

    # users matching
    sample_matching, sample_matching_values = matching_function(SM_samples)
    
    # obseve rewards
    rewards_obs = revealed_reward(SM_true, sample_matching, noise_std=0.2)
    rewards_obs_list.append(np.sum(rewards_obs))
    
    # calculate_targets    
    mu_targets, Sigma_targets = calculate_targets(rewards_obs, users, compatibility_samples_np, sample_matching, inplace=True)
    mu_targets_tensor = torch.FloatTensor(np.stack(mu_targets))
    Sigma_targets_tensor = torch.FloatTensor(np.stack(Sigma_targets))
    
    # TS-Model Update (KL-divergence Loss)
    loss_mu = ((mu_targets_tensor - mu_Hat)**2).mean()
    loss_Sigma = ((Sigma_targets_tensor - torch.linalg.inv(Lambda_Hat))**2).mean()
    loss = loss_mu + loss_Sigma
    # loss_kl = kl_target_to_pred_from_U(mu_targets_tensor, Sigma_targets_tensor,
    #                         mu_Hat, U_Hat)
    # loss = loss_mu + loss_kl
    
    # model update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"(ITER: {i}) loss : {loss.detach().to('cpu').numpy():.3f}")
    
    for j in range(N_UPDATES-1):
        mu_Hat, U_Hat, Lambda_Hat = TS_model(users_vec, users_id) # Gradient
        loss = kl_target_to_pred_from_U(mu_targets_tensor, Sigma_targets_tensor,
                            mu_Hat, U_Hat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

plt.plot(rewards_obs_list)
plt.axhline(np.sum(true_matching_values), color='red')
plt.show()
# [user.theta_true for user in users]
# [user.mu for user in users]




# sample matching
users_samples_vec = np.array([np.concatenate([user.u, user.sample_preference()], axis=-1) for user in users])
SM_samples = users_samples_vec @ users_samples_vec.T

# sample_matching, sample_matching_values = hungarian(SM_samples)
sample_matching, sample_matching_values = blossom_max_weight_matching(SM_samples)
print('< sampling_matching >')
print(sample_matching)
print(sample_matching_values)




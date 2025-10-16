import numpy as np
import matplotlib.pyplot as plt

import operator
import torch

################################################################################################################

def frozenlake_visualize_grid_probs(prob_map, env, ax=None, z_base=5, return_plot=False):
    n_rows = env.unwrapped.nrow
    n_cols = env.unwrapped.ncol
    grid = env.unwrapped.desc.astype(str)
    
    
    import matplotlib.cm as cm
    # 방향: ← ↓ → ↑
    dirs = [(-0.3, 0), (0, -0.3), (0.3, 0), (0, 0.3)]
    color_dict = {'S': 'mediumseagreen', 'H':'blue', 'G':'red'}

    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        created_ax = True
    
    norm = plt.Normalize(vmin=0.2, vmax=0.3)
    cmap = cm.get_cmap('jet')

    for i in range(n_rows):
        for j in range(n_cols):
            cx, cy = j, (n_rows-1) -i  # 좌측 상단이 index 0
            cell_probs = prob_map[i, j]
            max_dir = np.argmax(cell_probs)
            
            label = grid[i, j]
            if label != 'F':
                ax.text(cx, cy, label, fontsize=12, color=color_dict[label], 
                        ha='center', va='center', weight='bold', zorder=z_base+3)
            for d, (dx, dy) in enumerate(dirs):
                prob = cell_probs[d]
                color = cmap(norm(prob))
                ax.arrow(cx, cy, dx * prob * 2, dy * prob * 2,
                        head_width=0.05, head_length=0.05,
                        fc=color, ec=color, alpha=0.5, zorder=z_base+2)

                # 색상 조건: max 확률이면 빨간색, 아니면 검정
                text_color = 'red' if d == max_dir else 'black'
                alpha = 1 if d == max_dir else 0.5
                # 확률 수치 annotation
                offset_x, offset_y = dx * 0.9, dy * 0.9
                ax.text(cx + offset_x, cy + offset_y, f"{prob:.3f}",
                        fontsize=9, ha='center', va='center', color=text_color, alpha=alpha, zorder=z_base+3)
    # 컬러바 추가
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Action Probability')

    # 축 및 스타일
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(np.arange((n_rows-1),-1,-1))
    ax.xaxis.set_ticks_position('top')     # x축 눈금을 위쪽으로
    ax.xaxis.set_label_position('top')     # x축 라벨도 위쪽으로
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=z_base+1)
    ax.set_xlim(-0.5, (n_cols-1) + 0.5)
    ax.set_ylim(-0.5, (n_rows-1) + 0.5)
    ax.set_aspect('equal')
    ax.set_title("Policy Action Probabilities (← ↓ → ↑) with Values")
    plt.tight_layout()

    if created_ax:
        return ax
    elif return_plot:
        plt.close()
        return fig
    else:
        plt.show()



################################################################################################################
def pendulum_actor_evaluator(actor, device='cpu'):
    # θ : -np.pi ~ -np.pi/2 : 4사분면 / -np.pi/2 ~ 0 : 1사분면 / 0 ~ np.pi/2 : 2사분면 / np.pi/2 ~ np.pi : 3사분면
    # ang_velocity : -8(시계방향) ~ 8 (반시계방향)
    # action : -2 (시계방향 힘) ~ 2 (반시계방향 힘)

    radians = np.linspace(np.pi, -np.pi, num=9)
    # degrees = np.degrees(radians)
    ang_velocity = np.linspace(-8, 8, num=9)
    grid = np.stack(np.meshgrid(radians, ang_velocity)).reshape(2,-1).T
    grid_obs = np.concatenate([np.sin(grid[:, [0]]), np.cos(grid[:, [0]]),grid[:, [1]]], axis=1)

    grid_obs_tensor = torch.FloatTensor(grid_obs).to(device)
    pred_actions, _ = actor.predict(grid_obs_tensor)
    if 'torch' in str(type(pred_actions)):
        pred_actions = pred_actions.detach().to('cpu').numpy()
    grid_obs_pred = np.concatenate([np.degrees(grid[:,[0]]), grid[:,[0]], grid[:, [1]], pred_actions], axis=1)

    df_grid_obs_pred = pd.DataFrame(grid_obs_pred, columns=['degrees', 'radians','ang_velocity','pred_action'])

    df_predict_tb = df_grid_obs_pred.groupby(['radians', 'ang_velocity'])['pred_action'].mean().unstack('radians')
    df_predict_tb = df_grid_obs_pred.groupby(['degrees', 'ang_velocity'])['pred_action'].mean().unstack('degrees')
    df_predict_tb = np.round(df_predict_tb,2)
    df_predict_tb = df_predict_tb.sort_values('degrees', axis=1, ascending=False)
    return df_predict_tb

################################################################################################################
class ReplayMemory:
    def __init__(self, max_size=8192, batch_size=None, method='sequential', alpha=0.6, beta=0.4, random_state=None):
        """

        Args:
            max_size (int, optional): maximum saving experience data. Defaults to 8192.
            batch_size (int, optional): batch_size. If None, all data is drawn, Defaults to None.
            method (str, optional): sampling method. Defaults to 'sequential'. 
                        (sequential: sequential sampling / random: random sampling / priority: priority sampling) 
            alpha (float, optional): priority alpha. Defaults to 0.6.
            beta (float, optional): priority beta_. Defaults to 0.4.
            random_state (int, optional): random state. Defaults to None.
        """
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0
        self.batch_size = batch_size
        
        # priority sampling structures
        self.max_priority = 1.0
        self.priorities = np.zeros(self.max_size, dtype=np.float32)
        self.epsilon = 1e-10
        self._cached_probs = None
        
        # sampling configuration
        self.method = 'sequential' if method is None else method # None, 'random', or 'priority'
        if self.method == 'priority':
            if alpha is None or beta is None:
                raise ValueError("alpha, beta must be provided for priority sampling")
            self.alpha = alpha
            self.beta = beta
        
        # pointer for sequential or epoch-based sampling
        self.sample_pointer = 0
        self._iter_sample_pointer = 0   # iteration pointer
        self.shuffled_indices = None
        
        # random number generator
        self.random_state = random_state
        self.rng = np.random.RandomState(self.random_state)

    # experience push
    def push(self, obj, td_error=None):
        # assign priority
        if td_error is not None:
            priority = abs(td_error) + self.epsilon
        else:
            priority = self.max_priority if self.size else 1.0

        # insert into buffer
        self.buffer[self.index] = obj
        self.priorities[self.index] = priority

        # update position and size
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size
        
        self.shuffled_indices = None

    # index permutation
    def reset(self, method=None, alpha=0.6, beta=0.4):
        if method in ['sequential', 'random', 'priority']:
            self.method = method
            if method == 'priority':
                if alpha is None or beta is None:
                    raise ValueError("alpha, beta must be provided for priority sampling")
                self.alpha = alpha
                self.beta = beta
        
        if self.method == 'priority':
            probs = self.priorities[:self.size] ** self.alpha
            # probs /= np.sum(probs)
            self._cached_probs = probs / np.sum(probs)
            self.shuffled_indices = self.rng.choice(np.arange(self.size), size=self.size, 
                                                    replace=False, p=self._cached_probs)
            
        elif self.method == 'random':
            self.shuffled_indices = self.rng.permutation(self.size)
        else:  # 'sequential' or None
            self.shuffled_indices = np.arange(self.size)
        
        # initialize sample_pointer
        self.sample_pointer = 0
        self._iter_sample_pointer = 0
        # print(f'reset buffer : {self.method}')

    def _get_batch(self, pointer, batch_size):
        if self.size == 0:
            return None, None, None  # 비어 있을 경우만 None 반환

        batch_size = min(batch_size, self.size - pointer) if batch_size is not None else self.size - pointer
        if batch_size <= 0:
            return [], [], np.array([])  # 빈 인덱스 방어 처리

        indices = self.shuffled_indices[pointer:pointer + batch_size]
        samples = list(operator.itemgetter(*indices)(self.buffer)) if len(indices) != 0 else []

        if self.method == 'priority':
            probs = self._cached_probs
            if len(indices) > 0:
                IS_weights = (self.size * probs[indices]) ** (-self.beta)
                IS_weights /= IS_weights.max()
            else:
                IS_weights = np.array([])
        else:
            IS_weights = np.ones(len(indices))

        return samples, indices, IS_weights

    # sampling
    def sample(self, batch_size=None):
        """
        Sample a batch of experiences according to the configured method:
        - 'sequential': sequential order batches
        - 'random': shuffle once per epoch and return sequential chunks
        - 'priority': prioritized sampling with importance weights
        Returns (samples, indices, is_weights)
        """
        batch_size = self.batch_size if batch_size is None else batch_size
        if self.sample_pointer >= self.size or self.shuffled_indices is None:
            self.reset()

        result = self._get_batch(self.sample_pointer, batch_size)
        if result is None:
            return None

        _, indices, _ = result
        self.sample_pointer += len(indices)
        return result
    
    # iteration : __iter__
    def __iter__(self):
        self.reset()
        return self

    # iteration : __next__
    def __next__(self):
        if self._iter_sample_pointer >= self.size:
            raise StopIteration

        result = self._get_batch(self._iter_sample_pointer, self.batch_size or self.size)
        if result is None:
            raise StopIteration

        _, indices, _ = result
        self._iter_sample_pointer += len(indices)
        return result

    # update priority
    def update_priorities(self, indices, td_errors):
        td_errors = np.abs(np.asarray(td_errors)) + self.epsilon
        self.priorities[indices] = td_errors
        self.max_priority = max(self.max_priority, td_errors.max())

    def __len__(self):
        return self.size

# class ReplayMemory:
#     """
#     Args:
#         max_size (int, optional): maximum saving experience data. Defaults to 8192.
#         batch_size (int, optional): batch_size. If None, all data is drawn, Defaults to None.
#         method (str, optional): sampling method. Defaults to 'sequential'. 
#                     (sequential: sequential sampling / random: random sampling / priority: priority sampling) 
#         alpha (float, optional): priority alpha. Defaults to 0.6.
#         beta (float, optional): priority beta_. Defaults to 0.4.
#         random_state (int, optional): random state. Defaults to None.
#     """
#     def __init__(self, max_size=8192, batch_size=None, method='sequential',
#                  alpha=0.6, beta_start=0.4, beta_frames=100000, random_state=None):
#         self.max_size = max_size
#         self.batch_size = batch_size
#         self.buffer = [None] * max_size
#         self.priorities = np.zeros(max_size, dtype=np.float32)
#         self.index = 0
#         self.size = 0

#         self.alpha = alpha
#         self.beta_start = beta_start
#         self.beta_frames = beta_frames
#         self.frame = 1
#         self.epsilon = 1e-6
#         self.max_priority = 1.0

#         self.method = method if method is not None else 'sequential'
#         self.random_state = random_state
#         self.rng = np.random.RandomState(random_state)

#         self.sample_pointer = 0
#         self._iter_sample_pointer = 0
#         self.shuffled_indices = None

#     def push(self, experience):
#         self.buffer[self.index] = experience
#         self.priorities[self.index] = self.max_priority
#         self.index = (self.index + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)

#     def sample(self, batch_size=None):
#         batch_size = self.batch_size if batch_size is None else batch_size

#         if self.method == 'priority':
#             scaled = self.priorities[:self.size] ** self.alpha
#             probs = scaled / scaled.sum()
#             indices = self.rng.choice(self.size, batch_size, replace=True, p=probs)
            
#             beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
#             self.frame += 1
            
#             weights = (self.size * probs[indices]) ** (-beta)
#             weights /= weights.max()
#             weights = np.array(weights).astype(np.float32).reshape(-1,1)

#             samples = [self.buffer[i] for i in indices]
#             return samples, indices, weights

#         else:
#             if self.shuffled_indices is not None and (self.sample_pointer + self.batch_size >= len(self.shuffled_indices)):
#                 self.shuffled_indices = None
            
#             if self.method == 'random':
#                 if self.shuffled_indices is None or self.sample_pointer >= self.size:
#                     self.shuffled_indices = self.rng.permutation(self.size)
#                     self.sample_pointer = 0
#                 indices = self.shuffled_indices[self.sample_pointer:self.sample_pointer + batch_size]
#             else:  # sequential
#                 indices = np.arange(self.sample_pointer, min(self.sample_pointer + batch_size, self.size))

#             self.sample_pointer += len(indices)
#             samples = list(operator.itemgetter(*indices)(self.buffer))
#             weights = np.ones(len(indices)).astype(np.float32).reshape(-1,1)
#             return samples, indices, weights

#     def update_priorities(self, indices, td_errors):
#         td_errors = np.abs(np.asarray(td_errors)) + self.epsilon
#         self.priorities[indices] = td_errors
#         self.max_priority = max(self.max_priority, td_errors.max())

#     def reset(self, method=None):
#         if method:
#             self.method = method
#         self.sample_pointer = 0
#         self._iter_sample_pointer = 0
#         self.shuffled_indices = None

#     def __len__(self):
#         return self.size

#     def __iter__(self):
#         self.reset()
#         return self

#     def __next__(self):
#         if self._iter_sample_pointer >= self.size:
#             raise StopIteration

#         batch_size = self.batch_size or self.size
#         indices = np.arange(self._iter_sample_pointer, min(self._iter_sample_pointer + batch_size, self.size))
#         self._iter_sample_pointer += len(indices)
        
#         if len(indices) > 1:
#             samples = list(operator.itemgetter(*indices)(self.buffer))
#         else:
#             samples = list(tuple([operator.itemgetter(*indices)(self.buffer)]))
        
#         weights = np.ones(len(indices)).astype(np.float32).reshape(-1,1)
#         return samples, indices, weights

################################################################################################################

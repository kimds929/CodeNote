import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
import operator

import time
from IPython.display import clear_output
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

print(f"CUDA available: {torch.cuda.is_available()}")


###########################################################################################################
# ( Util Functions ) ######################################################################################

try:
    from RL00_Env01_CustomGridWorld import CustomGridWorld
except:
    import httpimport
    remote_url = 'https://raw.githubusercontent.com/kimds929/'

    with httpimport.remote_repo(f"{remote_url}/CodeNote/main/85_Reinforcement_Learning/"):
        from RL00_Env01_CustomGridWorld import CustomGridWorld




# from stable_baselines3.common.buffers import ReplayBuffer

# rb = ReplayBuffer(buffer_size=1000, observation_space=env.observation_space, action_space=env.action_space)
# rb.add(obs=np.array([1]), next_obs=np.array([2]), action=np.array([3]), reward=np.array([1.0]), done=np.array([True]), infos=[{}])
# sample = rb.sample(batch_size=1)
# ?rb.sample
# sample.observations
# sample.next_observations
# sample.actions
# sample.rewards
# sample.dones





################################################################################################################

# policy-network 정의 (Actor Network)
class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(state_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.fc_block(x)
    
    def predict_prob(self, x):
        return nn.functional.softmax(self.forward(x), dim=-1)

    def greedy_action(self, x, possible_actions=None):
        with torch.no_grad():
            action_prob = self.predict_prob(x)
            result_tensor = torch.full_like(action_prob, float('-inf'))
            if possible_actions is not None:
                result_tensor[..., possible_actions] = action_prob[..., possible_actions]
            else:
                result_tensor = action_prob
            return torch.argmax(result_tensor, dim=-1).item()

    def explore_action(self, x, possible_actions=None):
        with torch.no_grad():
            action_logit = self.forward(x)
            result_tensor = torch.full_like(action_logit, 0)
            if possible_actions is not None:
                result_tensor[..., possible_actions] = action_logit[..., possible_actions]
            else:
                result_tensor = action_logit
            action_dist = Categorical(logits=result_tensor)
            return action_dist.sample().item()

    def get_action(self, x, possible_actions=None):
        if self.training:
            return self.explore_action(x, possible_actions=possible_actions)
        else:
            return self.greedy_action(x, possible_actions=possible_actions)

# StateValueNetwork 정의 (Critic Network)
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(state_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.fc_block(x)


from stable_baselines3.common.buffers import ReplayBuffer


################################################################################################################
class ReplayMemory:
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
    def __init__(self, max_size=8192, batch_size=None, method='sequential',
                 alpha=0.6, beta_start=0.4, beta_frames=100000, random_state=None):
        self.max_size = max_size
        self.batch_size = batch_size
        self.buffer = [None] * max_size
        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.index = 0
        self.size = 0

        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.epsilon = 1e-6
        self.max_priority = 1.0

        self.method = method if method is not None else 'sequential'
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.sample_pointer = 0
        self._iter_sample_pointer = 0
        self.shuffled_indices = None

    def push(self, experience):
        self.buffer[self.index] = experience
        self.priorities[self.index] = self.max_priority
        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size

        if self.method == 'priority':
            scaled = self.priorities[:self.size] ** self.alpha
            probs = scaled / scaled.sum()
            indices = self.rng.choice(self.size, batch_size, replace=True, p=probs)
            
            beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
            self.frame += 1
            
            weights = (self.size * probs[indices]) ** (-beta)
            weights /= weights.max()
            weights = np.array(weights).astype(np.float32).reshape(-1,1)

            samples = [self.buffer[i] for i in indices]
            return samples, indices, weights

        else:
            if self.shuffled_indices is not None and (self.sample_pointer + self.batch_size >= len(self.shuffled_indices)):
                self.shuffled_indices = None
            
            if self.method == 'random':
                if self.shuffled_indices is None or self.sample_pointer >= self.size:
                    self.shuffled_indices = self.rng.permutation(self.size)
                    self.sample_pointer = 0
                indices = self.shuffled_indices[self.sample_pointer:self.sample_pointer + batch_size]
            else:  # sequential
                indices = np.arange(self.sample_pointer, min(self.sample_pointer + batch_size, self.size))

            self.sample_pointer += len(indices)
            samples = list(operator.itemgetter(*indices)(self.buffer))
            weights = np.ones(len(indices)).astype(np.float32).reshape(-1,1)
            return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        td_errors = np.abs(np.asarray(td_errors)) + self.epsilon
        self.priorities[indices] = td_errors
        self.max_priority = max(self.max_priority, td_errors.max())

    def reset(self, method=None):
        if method:
            self.method = method
        self.sample_pointer = 0
        self._iter_sample_pointer = 0
        self.shuffled_indices = None

    def __len__(self):
        return self.size

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self._iter_sample_pointer >= self.size:
            raise StopIteration

        batch_size = self.batch_size or self.size
        indices = np.arange(self._iter_sample_pointer, min(self._iter_sample_pointer + batch_size, self.size))
        self._iter_sample_pointer += len(indices)

        samples = list(operator.itemgetter(*indices)(self.buffer))
        weights = np.ones(len(indices)).astype(np.float32).reshape(-1,1)
        return samples, indices, weights







# class ReplayMemory:
#     def __init__(self, max_size=8192, batch_size=None, method='sequential', alpha=0.6, beta=0.4, random_state=None):
#         """

#         Args:
#             max_size (int, optional): maximum saving experience data. Defaults to 8192.
#             batch_size (int, optional): batch_size. If None, all data is drawn, Defaults to None.
#             method (str, optional): sampling method. Defaults to 'sequential'. 
#                         (sequential: sequential sampling / random: random sampling / priority: priority sampling) 
#             alpha (float, optional): priority alpha. Defaults to 0.6.
#             beta (float, optional): priority beta_. Defaults to 0.4.
#             random_state (int, optional): random state. Defaults to None.
#         """
#         self.buffer = [None] * max_size
#         self.max_size = max_size
#         self.index = 0
#         self.size = 0
#         self.batch_size = batch_size
        
#         # priority sampling structures
#         self.max_priority = 1.0
#         self.priorities = np.zeros(self.max_size, dtype=np.float32)
#         self.epsilon = 1e-10
#         self._cached_probs = None
        
#         # sampling configuration
#         self.method = 'sequential' if method is None else method # None, 'random', or 'priority'
#         if self.method == 'priority':
#             if alpha is None or beta is None:
#                 raise ValueError("alpha, beta must be provided for priority sampling")
#             self.alpha = alpha
#             self.beta = beta
        
#         # pointer for sequential or epoch-based sampling
#         self.sample_pointer = 0
#         self._iter_sample_pointer = 0   # iteration pointer
#         self.shuffled_indices = None
        
#         # random number generator
#         self.random_state = random_state
#         self.rng = np.random.RandomState(self.random_state)

#     # experience push
#     def push(self, obj, td_error=None):
#         # assign priority
#         if td_error is not None:
#             priority = abs(td_error) + self.epsilon
#         else:
#             priority = self.max_priority if self.size else 1.0

#         # insert into buffer
#         self.buffer[self.index] = obj
#         self.priorities[self.index] = priority

#         # update position and size
#         self.size = min(self.size + 1, self.max_size)
#         self.index = (self.index + 1) % self.max_size
        
#         self.shuffled_indices = None

#     # index permutation
#     def reset(self, method=None, alpha=0.6, beta=0.4):
#         if method in ['sequential', 'random', 'priority']:
#             self.method = method
#             if method == 'priority':
#                 if alpha is None or beta is None:
#                     raise ValueError("alpha, beta must be provided for priority sampling")
#                 self.alpha = alpha
#                 self.beta = beta
        
#         if self.method == 'priority':
#             probs = self.priorities[:self.size] ** self.alpha
#             # probs /= np.sum(probs)
#             self._cached_probs = probs / np.sum(probs)
#             self.shuffled_indices = self.rng.choice(np.arange(self.size), size=self.size, 
#                                                     replace=False, p=self._cached_probs)
            
#         elif self.method == 'random':
#             self.shuffled_indices = self.rng.permutation(self.size)
#         else:  # 'sequential' or None
#             self.shuffled_indices = np.arange(self.size)
        
#         # initialize sample_pointer
#         self.sample_pointer = 0
#         self._iter_sample_pointer = 0
#         # print(f'reset buffer : {self.method}')

#     def _get_batch(self, pointer, batch_size):
#         if self.size == 0:
#             return None, None, None  # 비어 있을 경우만 None 반환

#         batch_size = min(batch_size, self.size - pointer) if batch_size is not None else self.size - pointer
#         if batch_size <= 0:
#             return [], [], np.array([])  # 빈 인덱스 방어 처리

#         indices = self.shuffled_indices[pointer:pointer + batch_size]
#         samples = list(operator.itemgetter(*indices)(self.buffer)) if len(indices) != 0 else []

#         if self.method == 'priority':
#             probs = self._cached_probs
#             if len(indices) > 0:
#                 IS_weights = (self.size * probs[indices]) ** (-self.beta)
#                 IS_weights /= IS_weights.max()
#             else:
#                 IS_weights = np.array([])
#         else:
#             IS_weights = np.ones(len(indices))

#         return samples, indices, IS_weights

#     # sampling
#     def sample(self, batch_size=None):
#         """
#         Sample a batch of experiences according to the configured method:
#         - 'sequential': sequential order batches
#         - 'random': shuffle once per epoch and return sequential chunks
#         - 'priority': prioritized sampling with importance weights
#         Returns (samples, indices, is_weights)
#         """
#         batch_size = self.batch_size if batch_size is None else batch_size
#         if self.sample_pointer >= self.size or self.shuffled_indices is None:
#             self.reset()

#         result = self._get_batch(self.sample_pointer, batch_size)
#         if result is None:
#             return None

#         _, indices, _ = result
#         self.sample_pointer += len(indices)
#         return result
    
#     # iteration : __iter__
#     def __iter__(self):
#         self.reset()
#         return self

#     # iteration : __next__
#     def __next__(self):
#         if self._iter_sample_pointer >= self.size:
#             raise StopIteration

#         result = self._get_batch(self._iter_sample_pointer, self.batch_size or self.size)
#         if result is None:
#             raise StopIteration

#         _, indices, _ = result
#         self._iter_sample_pointer += len(indices)
#         return result

#     # update priority
#     def update_priorities(self, indices, td_errors):
#         td_errors = np.abs(np.asarray(td_errors)) + self.epsilon
#         self.priorities[indices] = td_errors
#         self.max_priority = max(self.max_priority, td_errors.max())

#     def __len__(self):
#         return self.size
################################################################################################################


# (Step 단위 Update) --------------------------------------------------------------------------------------------

# env = CustomGridWorld()
# env = CustomGridWorld(grid_size=5)
# env = CustomGridWorld(grid_size=5, traps=[], obstacles=[], treasures=[])
# env = CustomGridWorld(grid_size=5, obstacles=obstacles)
# env = CustomGridWorld(grid_size=5, obstacles=obstacles, traps=traps)
# env = CustomGridWorld(grid_size=5, obstacles=obstacles, traps=traps, treasures=treasures)
env = CustomGridWorld(grid_size=4, reward_step=-1)
env.reset()
env.render()

gamma = 0.9

policy_network = Actor(state_dim=2, hidden_dim=16, action_dim=4)
policy_optimizer = optim.Adam(policy_network.parameters(), lr=1e-4)

value_network = Critic(state_dim=2, hidden_dim=16)
value_optimizer = optim.Adam(value_network.parameters(), lr=1e-4)



# (Collect Rollout) #######################
memory = ReplayMemory(batch_size=64)
# memory = ReplayMemory()
# memory = ReplayMemory(batch_size=64, method='random')
# memory = ReplayMemory(batch_size=64, method='priority')

# episode save
num_episodes = 2
for epoch in tqdm(range(num_episodes)):
        # run episode
    env.reset()
    done = False

    i = 0
    while (done is not True):
        state_tensor = torch.tensor(env.cur_state).type(torch.float32)
        action_probs = policy_network.predict_prob(state_tensor)
        
        # select action
        action = policy_network.explore_action(state_tensor)
        log_prob = torch.log(action_probs[action]).item()
        
        # estimate value
        value = value_network(state_tensor).item()
        
        from_state, next_state, reward, done = env.step(action)
        experience = (from_state, action, log_prob, next_state, reward, value, done)
        
        # buffer에 experience 저장
        memory.push(experience)
        i += 1
        if i >=100:
            break


# --------------------------------------------------------------------
# memory.reset('sequential')
# memory.reset('random')
# memory.reset('priority')
batch, indices, weights = memory.sample()
indices
# # memory.priorities[:memory.index] = np.arange(1, memory.index+1)**5
# memory.method
memory.index
memory.size
memory.sample_pointer

for samples, indices, _ in memory:
    print(indices)
# --------------------------------------------------------------------


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
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.categorical import Categorical

from IPython.display import clear_output

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
################################################################################################################

def compute_gae(rewards, dones, values, gamma=0.99, lmbda=0.95):
    """
    rewards:  [r0, r1, ..., rN-1] (length N)
    dones:    [0, 0, ..., 1, 0, ..., 1] (length N)   # 1: terminal(episode done)
    values:   [v0, v1, ..., vN-1] (length N)        
    returns:  (advantage, returns)
    """
    N = len(rewards)
    advantages = np.zeros(N, dtype=np.float32)
    gae = 0.0
    next_value = 0.0  # 에피소드 종료 이후 값은 0
    values_array = np.asarray(values).ravel()

    for t in reversed(range(N)):
        mask = 1.0 - dones[t]  # done이면 0, 아니면 1
        delta = rewards[t] + gamma * next_value * mask - values_array[t]
        gae = delta + gamma * lmbda * mask * gae
        advantages[t] = gae
        next_value = values_array[t]

    returns = advantages + values_array
    return advantages, returns
################################################################################################################




# (Step 단위 Update) --------------------------------------------------------------------------------------------
# device = 'cpu'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# env = CustomGridWorld()
# env = CustomGridWorld(grid_size=5)
# env = CustomGridWorld(grid_size=5, traps=[], obstacles=[], treasures=[])
# env = CustomGridWorld(grid_size=5, obstacles=obstacles)
# env = CustomGridWorld(grid_size=5, obstacles=obstacles, traps=traps)
# env = CustomGridWorld(grid_size=5, obstacles=obstacles, traps=traps, treasures=treasures)
env = CustomGridWorld(grid_size=5, reward_step=-1)
env.reset()
env.render()

clip_eps = 0.1
gamma = 0.99
lmbda = 0.95
c_1 = 0.5
c_2 = 0.01
batch_size = 64

policy_network = Actor(state_dim=2, hidden_dim=16, action_dim=4).to(device)
policy_optimizer = optim.Adam(policy_network.parameters(), lr=1e-4)

value_network = Critic(state_dim=2, hidden_dim=16).to(device)
value_optimizer = optim.Adam(value_network.parameters(), lr=1e-4)


N_ITER = 50
N_REPLAY = 10
N_EPOCHS = 10

for iter in range(N_ITER):
    # (Collect Rollout) ############################################################################
    memory = ReplayMemory()
    print(f"\r({iter+1}/{N_ITER} ITER) ", end='')
    # episode save
    for replay_idx in range(N_REPLAY):
        print(f"\r({iter+1}/{N_ITER} ITER) collect_rollout: {replay_idx+1}/{N_REPLAY}, ", end='')
        env.reset()
        done = False

        i = 0
        while (done is not True):
            state_tensor = torch.FloatTensor(env.cur_state).to(device)
            action_probs = policy_network.predict_prob(state_tensor)
            
            # select action
            action = policy_network.explore_action(state_tensor)
            log_prob = torch.log(action_probs[action]).item()
            
            value = value_network(state_tensor).item()
            
            from_state, next_state, reward, done = env.step(action)
            experience = (from_state, action, log_prob, next_state, reward, done, value)
            
            # buffer에 experience 저장
            memory.push(experience)
            i += 1
            if i >=1000:
                break
    ################################################################################################
    # print(f"  2) dataset & learning")
    # Sampling from buffer
    batch, indices, weights = memory.sample()

    # TorchDataset & DataLoader
    from_states, actions, old_log_probs, next_states, rewards, dones, values = (np.array(batch, dtype='object').T).tolist()

    # compute advantage (GAE, return)
    advantages, returns = compute_gae(rewards, dones, values)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)       # advantage normalize

    # dataset
    states_tensor = torch.FloatTensor(from_states).to(device)
    actions_tensor = torch.LongTensor(actions).to(device)
    log_probs_tensor = torch.FloatTensor(old_log_probs).to(device)
    advantages_tensor = torch.FloatTensor(advantages).to(device)
    returns_tensor = torch.FloatTensor(returns).to(device)

    dataset = TensorDataset(states_tensor, actions_tensor, log_probs_tensor, advantages_tensor, returns_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    ################################################################################################
    # for epoch in range(N_EPOCHS*5):
    #     for batch_data in data_loader:
    #         states, actions, log_probs, advantages, returns = batch_data
    #         values = value_network(states)
    #         value_loss = ((returns.ravel() - values.ravel()) ** 2).mean()
    #         value_optimizer.zero_grad()
    #         value_loss.backward()
    #         value_optimizer.step()
    #     print(f"\r({iter+1}/{N_ITER} ITER) collect_rollout: {replay_idx+1}/{N_REPLAY}, learning_epoch: {epoch+1}/{N_EPOCHS} (CriticLoss: {value_loss.item():.3f})", end='')
    
    for epoch in range(N_EPOCHS):
        for batch_data in data_loader:
            states, actions, log_probs, advantages, returns = batch_data
            logits = policy_network(states)
            values = value_network(states)

            dist = Categorical(logits=logits)
            entropy_mean = dist.entropy().mean()
            new_log_probs = dist.log_prob(actions)
            
            # ratio for clipping
            ratio = (new_log_probs - log_probs).exp()     # Gradient : new_log_porbs

            # PPO objective 
            surr_1 = ratio * advantages
            surr_2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps ) * advantages
            policy_loss = -torch.min(surr_1, surr_2).mean()
            
            value_loss = ((returns.ravel() - values.ravel()) ** 2).mean()
            loss = policy_loss + c_1 * value_loss - c_2 * entropy_mean
            
            # backpropagation
            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()
            loss.backward()
            policy_optimizer.step()
            value_optimizer.step()
        print(f"\r({iter+1}/{N_ITER} ITER) collect_rollout: {replay_idx+1}/{N_REPLAY}, learning_epoch: {epoch+1}/{N_EPOCHS} (CriticLoss: {value_loss.item():.3f}, ActorLoss: {policy_loss.item():.3f})", end='')
    print()
    # clear_output(wait=True)
    ################################################################################################






# Simulation Test ---------------------------------------------------------------------------------
env.reset()
# env.render()
i = 0
done = False
while (done is not True):
    
    with torch.no_grad():
        cur_state_tensor = torch.tensor(env.cur_state).type(torch.float32)
        action = policy_network.greedy_action(cur_state_tensor)
        # action_outputs = policy_network(cur_state_tensor)
        # action_prob = torch.softmax(action_outputs, dim=-1)
        # action_possible_prob = torch.full_like(action_prob, float('-inf'))
        # action_possible_prob[..., env.get_possible_actions()] = action_prob[..., env.get_possible_actions()]
        # action = torch.argmax(action_possible_prob, dim=-1).item()

        from_state, next_state, reward, done = env.step(action)
        env.render()
        time.sleep(0.2)
        clear_output(wait=True)
    i += 1
    if i >=30:
        break
################################################################################################################
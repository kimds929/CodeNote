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

import gymnasium as gym
from IPython.display import clear_output


from stable_baselines3 import PPO

# actor_network = Actor(state_dim=54, hidden_dim=64, action_dim=5)
# sum(p.numel() for p in actor_network.parameters())

print(f"CUDA available: {torch.cuda.is_available()}")

################################################################################################################
################################################################################################################
# [ FrozenLake ] ###############################################################################################

import httpimport


try:
    try:
        from Environments.RL00_Env01_FrozenLake_v1 import generate_frozenlake_map
        from DS_RL import frozenlake_visualize_grid_probs, ReplayMemory
        custom_map = generate_frozenlake_map(5,5, hole=0.17)
    except:
        remote_url = 'https://raw.githubusercontent.com/kimds929/'

        with httpimport.remote_repo(f"{remote_url}/CodeNote/refs/heads/main/85_Reinforcement_Learning/Environments"):
            from RL00_Env01_FrozenLake_v1 import generate_frozenlake_map
        custom_map = generate_frozenlake_map(5,5, hole=0.17)
        
        with httpimport.remote_repo(f"{remote_url}/CodeNote/refs/heads/main/85_Reinforcement_Learning/utils"):
            from DS_RL import frozenlake_visualize_grid_probs, ReplayMemory
except:
    custom_map=None




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
            random_state (int, optional): random obs. Defaults to None.
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

################################################################################################################
# policy-network 정의 (Actor Network)
class Actor(nn.Module):
    def __init__(self, action_dim, hidden_dim, max_states=100, embed_dim=1):
        super().__init__()
        self.state_embedding = nn.Embedding(num_embeddings=max_states, embedding_dim=embed_dim)
        self.actor_network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim)
            ,nn.Tanh()
            ,nn.Linear(hidden_dim, hidden_dim)
            ,nn.Tanh()
            ,nn.Linear(hidden_dim, action_dim)
        )
    
    def forward_logits(self, obs):
        embed_x = self.state_embedding(obs).squeeze(-2)
        logits = self.actor_network(embed_x)
        return logits
    
    def execute_model(self, obs, actions=None, temperature=1, deterministic=False):
        logits = self.forward_logits(obs)
        action_dist = Categorical(logits=logits)
        entropy = action_dist.entropy()
        
        if actions is None:
            if deterministic is True:
                action = torch.argmax(logits, dim=-1)
            elif temperature is None:
                action = torch.argmax(logits, dim=-1)
            else:
                explore_dist = Categorical(logits=logits/temperature)
                action = explore_dist.sample()
            log_prob = action_dist.log_prob(action)
            return action, log_prob, entropy
        
        else:
            log_prob = action_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
            return log_prob, entropy
    
    def forward(self, obs, temperature=1, deterministic=False):
        action, log_prob, entropy = self.execute_model(obs, temperature=temperature, deterministic=deterministic)
        return action, log_prob, entropy
    
    def evaluate_actions(self, obs, actions, temperature=1, deterministic=False):
        log_prob, entropy = self.execute_model(obs, actions=actions, temperature=temperature, deterministic=deterministic)
        return log_prob, entropy
    
    def predict(self, obs, temperature=1, deterministic=False):
        action, log_prob, entropy = self.execute_model(obs, temperature=temperature, deterministic=deterministic)
        return action

# StateValueNetwork 정의 (Critic Network)
class Critic(nn.Module):
    def __init__(self, hidden_dim, max_states=100, embed_dim=1):
        super().__init__()
        self.state_embedding = nn.Embedding(num_embeddings=max_states, embedding_dim=embed_dim)
        self.fc_block = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim)
            ,nn.Tanh()
            ,nn.Linear(hidden_dim, hidden_dim)
            ,nn.Tanh()
            ,nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.state_embedding(x).squeeze(-2)
        return self.fc_block(x)

################################################################################################################
# compute_gae
def compute_gae(rewards, values, next_values, gamma=0.99, lmbda=0.95):
    """
    ReplayMemory에 저장된 sequential rollout에서 GAE 계산

    Args:
        rewards:      list or array of shape (N,)    ← [r0, r1, ..., r_{N-1}]
        values:       list or array of shape (N,)    ← [v0, v1, ..., v_{N-1}]
        next_value: : list or array of shape (N,)    ← [None, ...]  (terminated or truncated시 next_value )
        gamma:      할인률 (default=0.99)
        lmbda:      GAE λ 파라미터 (default=0.95)

    Returns:
        advantages: numpy.ndarray of shape (N,)   ← GAE(λ) 기반 advantage
        returns:    numpy.ndarray of shape (N,)   ← advantage + values
    """
    N = len(rewards)
    advantages = np.zeros(N, dtype=np.float32)
    returns = np.zeros(N, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(N)):
        if next_values[t] is None:
            next_value = values[t + 1]
        else:       # terminated or truncated 
            next_value = next_values[t]
            gae = 0.0

        # δ_t = r_t + γ * V(s_{t+1}) * mask - V(s_t)
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lmbda * gae
        advantages[t] = gae

    # return = advantage + V(s_t)
    for i in range(N):
        returns[i] = advantages[i] + values[i]

    return advantages, returns

################################################################################################################


# device = torch.device('cpu')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

###########################################################################################################
# ( CustomPPO with FrozenLake_v1 ) ########################################################################
# import gymnasium as gym
# from stable_baselines3 import PPO

env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode="rgb_array") 
obs, info = env.reset()
plt.imshow(env.render())


clip_eps = 0.1
gamma = 0.99
lmbda = 0.95

clip_range_pi = 0.2
clip_range_vf = None
c_1_vf_coef = 0.5
c_2_ent_coef = 0.0         # Atari : 0.01, MujoCo : 0.0
max_grad_norm = 0.5
batch_size = 64

actor_network = Actor(hidden_dim=16, action_dim=env.action_space.n, embed_dim=1).to(device)
optimizer_actor = optim.Adam(actor_network.parameters(), lr=3e-4)
# optimizer_actor = optim.AdamW(actor_network.parameters(), lr=1e-4, weight_decay=1e-2)

critic_network = Critic(hidden_dim=16, max_states=env.observation_space.n, embed_dim=1).to(device)
optimizer_critic = optim.Adam(critic_network.parameters(), lr=3e-4)
# optimizer_critic = optim.AdamW(critic_network.parameters(), lr=1e-3, weight_decay=1e-2)

memory = ReplayMemory()

N_ITER = 10
TOTAL_TIMESTEPS = 500        # Rollout Timstep
N_REPLAY_STEPS = 100         # Rollout 횟수
N_EPOCHS = 5               # Model Learning 횟수

# iter : N_ITER
#   ㄴ loop : num_loop = (TOTAL_TIMESTEPS-1)//N_REPLAY_STEPS+1     # 전체 loop 횟수
#       ㄴ rolout : N_REPLAY_STEPS (loop내 rollout 횟수)
#       ㄴ learning : N_EPOCHS (loop내 model backprop 횟수)

for iter in range(N_ITER):
    print(f"\r({iter+1}/{N_ITER} ITER) ", end='')
    replay_time_step = 0
    
    total_loop = (TOTAL_TIMESTEPS-1)//N_REPLAY_STEPS+1
    loop_count = 1
    
    while (replay_time_step < TOTAL_TIMESTEPS):
        # (Collect Rollout) ##################################################################################
        memory.reset()
        obs, info = env.reset()
        env_states = env.observation_space.n
        
        for _ in range(N_REPLAY_STEPS):
            print(f"\r[{iter+1}/{N_ITER}ITER](Loop {loop_count}/{total_loop}) Rollout: {replay_time_step+1}/{TOTAL_TIMESTEPS}, ", end='')

            # estimate policy, value
            obs_tensor = torch.LongTensor([obs]).to(device)
            action, log_prob, entropy = actor_network.forward(obs_tensor, deterministic=False)
            action = action.item()
            log_prob = log_prob.item()
            value = critic_network(obs_tensor)
            value = value.item()
            
            # env_step
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                next_value = 0.0
            elif truncated:
                next_obs_tensor = torch.LongTensor([next_obs]).to(device)
                next_value = critic_network(next_obs_tensor).item()
            else:
                next_value = None
                
            
            # buffer push
            experience = (obs, action, log_prob, reward, value, next_value)
            memory.push(experience)
            
            # counting
            if terminated or truncated:
                obs, info = env.reset()
            else:
                obs = next_obs
            replay_time_step += 1
        ######################################################################################################
        
        
        # (Compute GAE & Dataset) ############################################################################    
        # RolloutData
        batch, indices, weights = memory.sample()
        obss, actions, log_probs, rewards, values, next_values  = (np.array(batch, dtype='object').T).tolist()
        torch.LongTensor(obss).to(device)

        # 마지막 스텝에서의 next_value 확보 (부트스트랩용)
        next_obs_tensor = torch.LongTensor([next_obs]).to(device)
        with torch.no_grad():
            next_values[-1] = critic_network(next_obs_tensor).item()
        
        # ComputeGAE
        advantages, returns = compute_gae(rewards=rewards, values=values, next_values=next_values, gamma=gamma, lmbda=lmbda)
         # advantage 정규화 (선택적)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        
        # dataset & dataloader
        obss_tensor = torch.LongTensor(obss).view(-1,1).to(device)
        actions_tensor = torch.LongTensor(actions).view(-1,1).to(device)
        log_probs_tensor = torch.FloatTensor(log_probs).view(-1,1).to(device)
        values_tensor = torch.FloatTensor(values).view(-1,1).to(device)
        advantages_tensor = torch.FloatTensor(advantages).view(-1,1).to(device)
        returns_tensor = torch.FloatTensor(returns).view(-1,1).to(device)

        dataset = TensorDataset(obss_tensor, actions_tensor, log_probs_tensor, values_tensor, advantages_tensor, returns_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        ######################################################################################################

        # (Critic Learning) -----------------------------------------------------------------------------------
        # CRITIC_N_EPOCHS = N_EPOCHS*3
        # for epoch in range(CRITIC_N_EPOCHS):
        #     for batch_data in data_loader:
        #         obss, actions, log_probs, advantages, returns = batch_data
        #         values = critic_network(obss)
        #         value_loss = ((returns.ravel() - values.ravel()) ** 2).mean()
        #         optimizer_critic.zero_grad()
        #         value_loss.backward()
        #         optimizer_critic.step()
        #     print(f"\r[{iter+1}/{N_ITER}ITER](Loop {loop_count}/{total_loop}) Rollout: {replay_time_step+1}/{TOTAL_TIMESTEPS}, learning_epoch: {epoch+1}/{CRITIC_N_EPOCHS} (CriticLoss: {value_loss.item():.3f})", end='')
        
        # (PPO Learning) ---------------------------------------------------------------------------------------
        
        for epoch in range(N_EPOCHS):
            for batch_data in data_loader:
                # break
                # 현재 policy, value 평가: evaluate value,log_prob, entropy
                obss, actions, old_log_probs, values, advantages, returns = batch_data
                new_log_probs, entropy = actor_network.evaluate_actions(obss, actions, deterministic=False)
                
                values_pred = critic_network(obss)
                
                # (log_prob ratio)
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # (Clipping policy loss)
                policy_surr_loss_1 = ratio * advantages
                policy_surr_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range_pi, 1 + clip_range_pi)
                policy_loss = -torch.min(policy_surr_loss_1, policy_surr_loss_2).mean()                
                
                # (Entrophy Loss)
                entropy_loss = -entropy.mean()
                
                # (Clipping value)
                if clip_range_vf is not None:
                    values_pred = values + torch.clamp(values_pred - values, -clip_range_vf, +clip_range_vf)
                
                # (Value loss)
                value_loss = ((returns.ravel() - values_pred.ravel()) ** 2).mean()

                # (Final loss)
                # loss = policy_loss + c_2_ent_coef * entropy_loss + c_1_vf_coef * value_loss
                actor_loss = policy_loss + c_2_ent_coef * entropy_loss 
                critic_loss = c_1_vf_coef * value_loss
                
                # (backpropagation)
                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()
                
                # loss.backward()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor_network.parameters(), max_grad_norm)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic_network.parameters(), max_grad_norm)
                
                optimizer_actor.step()
                optimizer_critic.step()
            print(f"\r[{iter+1}/{N_ITER}ITER](Loop {loop_count}/{total_loop}) Rollout: {replay_time_step+1}/{TOTAL_TIMESTEPS}, learning_epoch: {epoch+1}/{N_EPOCHS} (CriticLoss: {value_loss.item():.3f}, ActorLoss: {policy_loss.item():.3f})", end='')
        
        loop_count += 1
    ######################################################################################################





###########################################################################
obs, info = env.reset()
env_states = env.observation_space.n
for i in range(50):
    # Animation
    if i % 1 == 0:
        plt.imshow(env.render())
        plt.show()
        time.sleep(0.05)
        clear_output(wait=True)
    
    
    obs_tensor = torch.LongTensor([obs]).to(device)
    action, log_prob, entropy = actor_network.forward(obs_tensor, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action.item())
        
    if terminated:
        print("Goal reached!", "Reward:", reward)
        break
    elif truncated:
        print("Fail to find goal", "Reward:", reward)
        break

plt.imshow(env.render())
plt.show()

# env.close()
############################################################



################################################################################################################
################################################################################################################

# ← ↓ → ↑
actor_network.forward_logits(torch.LongTensor([0]).to(device))
# env.step(3)

allgrid_logits = actor_network.forward_logits(torch.arange(env.observation_space.n).to(device))
allgrid_logits
allgrid_probs = (torch.exp(allgrid_logits) / torch.exp(allgrid_logits).sum(dim=-1, keepdim=True)).to('cpu').detach().numpy()


actor_probs = allgrid_probs.reshape(env.unwrapped.nrow,env.unwrapped.ncol,env.action_space.n).copy()
actor_probs.shape
frozenlake_visualize_grid_probs(actor_probs, env)



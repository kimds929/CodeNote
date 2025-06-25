# import gym
# import numpy as np
# import matplotlib.pyplot as plt
# from IPython import display
# # get_ipython().run_line_magic('matplotlib', 'inline')

# from time import sleep
# from IPython.display import clear_output
# from tqdm import tqdm_notebook

# import sys
# RLpath = r'D:\DataScience\SNU_DataScience\SNU_OhLab\Reinforce_Learning_Code\[POSTECH] ReinforcementLearning'
# sys.path.append(RLpath) # add project root to the python path

# import httpimport
# remote_url = 'https://raw.githubusercontent.com/kimds929/'

# with httpimport.remote_repo(f"{remote_url}/CodeNote/main/85_Reinforce_Learning/"):
#     from RL00_Env01_CustomGridWorld import CustomGridWorld


import sys
sys.path.append(r'D:\DataScience\★Git_CodeNote\85_Reinforcement_Learning')

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict

import time
from IPython.display import clear_output
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# import httpimport
# remote_url = 'https://raw.githubusercontent.com/kimds929/'

# with httpimport.remote_repo(f"{remote_url}/CodeNote/main/85_Reinforce_Learning/"):
#     from RL00_Env01_CustomGridWorld import CustomGridWorld



# Replay Buffer
class ReplayMemory:
    def __init__(self, max_size, batch_size=64):
        # deque object that we've used for 'episodic_memory' is not suitable for random sampling
        # here, we instead use a fix-size array to implement 'buffer'
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0  # 어디까지 채워져있는지 check
        self.size = 0
        self.batch_size = batch_size

    def push(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size

        if self.size >= batch_size:
            # indices = sample(range(self.size), batch_size)
            indices = np.random.choice(range(self.size), size=batch_size, replace=False)
            return [self.buffer[index] for index in indices]

    def reset(self):
        self.__init__(max_size=self.max_size, batch_size=self.batch_size)
    
    def __len__(self):
        return self.size

    def __iter__(self):
        self._iter_indices = np.random.permutation(self.size)  # 셔플된 인덱스
        self._iter_pos = 0
        return self

    def __next__(self):
        if self._iter_pos + self.batch_size > self.size:
            raise StopIteration
        batch_indices = self._iter_indices[self._iter_pos:self._iter_pos + self.batch_size]
        self._iter_pos += self.batch_size
        return [self.buffer[idx] for idx in batch_indices]
###########################################################################################################

























###########################################################################################################
# ( Util Functions ) ######################################################################################

try:
    from Environments.RL00_Env01_CustomGridWorld import CustomGridWorld
except:
    import httpimport
    remote_url = 'https://raw.githubusercontent.com/kimds929/'

    with httpimport.remote_repo(f"{remote_url}/CodeNote/main/85_Reinforce_Learning/"):
        from RL00_Env01_CustomGridWorld import CustomGridWorld

# ★ argmax with duplicated max number (max 값이 여러개일 때, max값 중 랜덤하게 sample해주는 함수)
def rand_argmax(a, axis=None, return_max=False, random_state=None):
    rng = np.random.RandomState(random_state)
    if axis is None:
        mask = (a == a.max())
        idx = rng.choice(np.flatnonzero(mask))
        if return_max:
            return idx, a.flatten()[idx]
        else:
            return idx
    else:
        mask = (a == a.max(axis=axis, keepdims=True))
        idx = np.apply_along_axis(lambda x: rng.choice(np.flatnonzero(x)), axis=axis, arr=mask)
        expanded_idx = np.expand_dims(idx, axis=axis)
        if return_max:
            return idx, np.take_along_axis(a, expanded_idx, axis=axis)
        else:
            return idx

def epsilon_greedy(policy, epsilon=1e-1, random_state=None):
    rng = np.random.RandomState(random_state)
    greedy_action = rand_argmax(policy)

    if rng.rand() < epsilon:
        return rng.choice(np.where(policy != greedy_action)[0])
    else:
        return greedy_action




################################################################################################################
# DeepNetwork TD - Q-Learning #############################################################################



import httpimport
remote_url = 'https://raw.githubusercontent.com/kimds929/'


import requests
response = requests.get("https://raw.githubusercontent.com/kimds929/DS_Library/main/DS_Torch.py")
exec(response.text)

import importlib
importlib.reload(httpimport)

with httpimport.remote_repo(f"{remote_url}/DS_Library/main/"):
    from DS_DeepLearning import EarlyStopping

with httpimport.remote_repo(f"{remote_url}/DS_Library/main/"):
    from DS_Torch import TorchDataLoader, TorchModeling, AutoML




# Q-network 정의
class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc_block(x)




################################################################################################################

def epsilon_greedy_torch(state, model, epsilon=1e-1, device='cpu'):
    with torch.no_grad():
        model.eval()
        pred = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
        greedy_action = torch.argmax(pred).item()

        if torch.rand(1).item() > epsilon:
            action = greedy_action
        else:
            filtered_tensor = torch.arange(pred.shape[-1]).to(device)[~(pred==torch.max(pred)).squeeze()]
            action = filtered_tensor[torch.randint(0, len(filtered_tensor), (1,))].item()
    return action, pred


################################################################################################################
# ( Q-Learning Control ) ######################################################################################

# env = CustomGridWorld()
# env = CustomGridWorld(grid_size=5)
# env = CustomGridWorld(grid_size=5, traps=[], obstacles=[], treasures=[])
# env = CustomGridWorld(grid_size=5, obstacles=obstacles)
# env = CustomGridWorld(grid_size=5, obstacles=obstacles, traps=traps)
# env = CustomGridWorld(grid_size=5, obstacles=obstacles, traps=traps, treasures=treasures)
# env = CustomGridWorld(grid_size=5)
env = CustomGridWorld()
env.reset()
env.render()

memory_size = 1024
batch_size = 64
sample_only_until = 500
n_epochs = 1
target_update_interval = 100

gamma = 0.9

# main network
main_Qnetwork = Critic(2, 16, 4).to(device)
main_Qnetwork.to(device)

# target network
target_Qnetwork = Critic(2, 16, 4).to(device)
target_Qnetwork.to(device)
target_Qnetwork.load_state_dict(main_Qnetwork.state_dict())

# memory
memory = ReplayMemory(max_size=memory_size, batch_size=batch_size)

optimizer = optim.Adam(main_Qnetwork.parameters(), lr=1e-3)
# loss_function = nn.MSELoss()
loss_function = nn.SmoothL1Loss()
# |x_i - y_i| < 1 : z_i = 0.5(x_i - y_i)^2
# |x_i - y_i| >= 1 : z_i = |x_i - y_i| - 0.5



num_episodes = 100
total_step = 0
with tqdm(total=num_episodes) as pbar:
    for episode in range(num_episodes):
            # run episode
        state, info = env.reset()
        done = False

        i = 0
        while (done is not True):
            eps = 1- min(episode*2, num_episodes)/ num_episodes
            action, q_pred = epsilon_greedy_torch(state=env.cur_state, model=main_Qnetwork, epsilon=eps, device=device)
            from_state, next_state, reward, done = env.step(action)
            experience = (action, from_state, next_state, reward, done)

            # buffer에 experience 저장
            memory.push(experience)
            i += 1
            if i >=100:
                break

            # 최소 요구 sample수가 만족 했을경우, Trainning 진행
            avg_loss = 0
            if len(memory) >= sample_only_until:
                for epoch in range(n_epochs):
                    sampled_exps = memory.sample()
                    actions = torch.tensor(np.stack([sample[0] for sample in sampled_exps])).type(torch.int64).view(-1,1).to(device)
                    states = torch.tensor(np.stack([sample[1] for sample in sampled_exps])).type(torch.float32).to(device)
                    next_states = torch.tensor(np.stack([sample[2] for sample in sampled_exps])).type(torch.float32).to(device)
                    rewards = torch.tensor(np.stack([sample[3] for sample in sampled_exps])).type(torch.float32).view(-1,1).to(device)
                    dones = torch.tensor(np.stack([sample[4] for sample in sampled_exps])).type(torch.float32).view(-1,1).to(device)
                    
                    with torch.no_grad():
                        q_max, q_max_idx = target_Qnetwork(next_states).max(dim=-1, keepdims=True)
                        q_target = rewards + gamma*q_max * (1-dones)

                    q_value = main_Qnetwork(states).gather(1, actions)        # gather : 1번째 axis에서 action위치의 값을 choose)
                    loss = loss_function(q_value, q_target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    avg_loss += loss.to('cpu')

            if total_step % target_update_interval == 0:
                target_Qnetwork.load_state_dict(main_Qnetwork.state_dict())
                # print('target_network update')
            total_step += 1
     
        if episode % 1 == 0:
            pbar.set_postfix(Q_loss=f"{avg_loss/n_epochs:.3f}", Len_episodes=f"{i}")
        pbar.update(1)




# Simulation Test ---------------------------------------------------------------------------------
env.reset()
# env.render()
i = 0
done = False
while (done is not True):
    
    with torch.no_grad():
        main_Qnetwork.eval()
        q_pred = main_Qnetwork(torch.tensor(env.cur_state, dtype=torch.float32).unsqueeze(0).to(device))
        action = torch.argmax(q_pred).item()
        from_state, next_state, reward, done = env.step(action)
        env.render()
        time.sleep(0.2)
        clear_output(wait=True)
    i += 1
    if i >=30:
        break

################################################################################################################






















################################################################################################################
################################################################################################################
# [ FrozenLake ] ###############################################################################################

import gymnasium as gym

try:
    from Environments.RL00_Env01_FrozenLake_v1 import generate_frozenlake_map
    # from utils.DS_RL import ReplayMemory
    custom_map = generate_frozenlake_map(5,5, hole=0.2)
except:
    custom_map=None

# Q-network 정의
class Critic(nn.Module):
    def __init__(self, action_dim, hidden_dim, max_states=100, embed_dim=1):
        super().__init__()
        self.state_embedding = nn.Embedding(num_embeddings=max_states, embedding_dim=embed_dim)
        self.fc_block = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim)
            ,nn.SELU()
            ,nn.Linear(hidden_dim, hidden_dim)
            ,nn.SELU()
            ,nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        x = self.state_embedding(x).squeeze(-2)
        return self.fc_block(x)
##############################################################################################################

env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode="rgb_array")  # deterministic
obs, info = env.reset()
plt.imshow(env.render())
# env.observation_space.n

memory_size = 1024
batch_size = 64
sample_only_until = 500
n_epochs = 3
target_update_interval = 100

gamma = 0.9


# main network
main_Qnetwork = Critic(action_dim=env.action_space.n, hidden_dim=16,
                       max_states=env.observation_space.n, embed_dim=2).to(device)
main_Qnetwork.to(device)

# target network
target_Qnetwork = Critic(action_dim=env.action_space.n, hidden_dim=16,
                       max_states=env.observation_space.n, embed_dim=2).to(device)
target_Qnetwork.to(device)
target_Qnetwork.load_state_dict(main_Qnetwork.state_dict())

# memory
memory = ReplayMemory(max_size=memory_size, batch_size=batch_size)

optimizer = optim.AdamW(main_Qnetwork.parameters(), lr=1e-3, weight_decay=1e-2)
# loss_function = nn.MSELoss()
loss_function = nn.SmoothL1Loss()

num_episodes = 300
total_step = 0


with tqdm(total=num_episodes) as pbar:
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        i = 0
        
        while(not done):
            # # (epsilon greedy) -------------------------------------------
            # eps = 1- min(episode, num_episodes)/ num_episodes
            # if np.random.rand() < eps:
            #     action = np.random.choice(np.arange(env.action_space.n))
            # else:
            #     q_logits = main_Qnetwork(torch.LongTensor([obs]).to(device))
            #     action = torch.argmax(q_logits).item()
            # # ------------------------------------------------------------
            
            # (exp temperature) -------------------------------------------
            q_logits = main_Qnetwork(torch.LongTensor([obs]).to(device))
            T = np.logspace(-2, 2, num=num_episodes)[episode]
            dist = Categorical(logits=T*q_logits)
            action = dist.sample().item()
            # ------------------------------------------------------------

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            experience = (action, obs, next_obs, reward, done)
            # buffer에 experience 저장
            memory.push(experience)
            
            obs = next_obs
            i += 1
            if i >=100:
                break
        
            avg_loss = 0
            if len(memory) >= sample_only_until:
                for epoch in range(n_epochs):
                    sampled_exps = memory.sample()
                    batch_actions = torch.LongTensor(np.stack([sample[0] for sample in sampled_exps])).view(-1,1).to(device)
                    batch_obs = torch.LongTensor(np.stack([sample[1] for sample in sampled_exps])).view(-1,1).to(device)
                    batch_next_obs = torch.LongTensor(np.stack([sample[2] for sample in sampled_exps])).view(-1,1).to(device)
                    batch_rewards = torch.FloatTensor(np.stack([sample[3] for sample in sampled_exps])).view(-1,1).to(device)
                    batch_dones = torch.FloatTensor(np.stack([sample[4] for sample in sampled_exps])).view(-1,1).to(device)
                    
                    with torch.no_grad():
                        q_max, q_max_idx = target_Qnetwork(batch_next_obs).max(dim=-1, keepdims=True)
                        q_target = batch_rewards + gamma*q_max * (1-batch_dones)

                    q_value = main_Qnetwork(batch_obs).gather(1, batch_actions)        # gather : 1번째 axis에서 action위치의 값을 choose)
                    loss = loss_function(q_value, q_target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    avg_loss += loss.to('cpu')

            if total_step % target_update_interval == 0:
                target_Qnetwork.load_state_dict(main_Qnetwork.state_dict())
                # print('target_network update')
            total_step += 1
            
        if episode % 1 == 0:
            pbar.set_postfix(Q_loss=f"{avg_loss/n_epochs:.3f}", Len_episodes=f"{i}")
        pbar.update(1)



# Simulation Test ---------------------------------------------------------------------------------
obs, info = env.reset()
# env.render()
i = 0
done = False
while (done is not True):
    
    with torch.no_grad():
        main_Qnetwork.eval()
        
        q_pred = main_Qnetwork(torch.LongTensor([obs]).to(device))
        action = torch.argmax(q_pred).item()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        plt.imshow(env.render())
        plt.show()
        time.sleep(0.1)
        clear_output(wait=True)
        obs = next_obs
    i += 1
    if i >=30:
        break
    
################################################################################################################
################################################################################################################
################################################################################################################
































# ################################################################################################################
# ################################################################################################################
# # [ PPO Style REPLAY ] #########################################################################################
# import gymnasium as gym

# try:
#     from Environments.RL00_Env01_FrozenLake_v1 import generate_frozenlake_map
#     # from utils.DS_RL import ReplayMemory
#     custom_map = generate_frozenlake_map(5,5, hole=0.2)
# except:
#     custom_map=None

# # Q-network 정의
# class Critic(nn.Module):
#     def __init__(self, action_dim, hidden_dim, max_states=100, embed_dim=1):
#         super().__init__()
#         self.state_embedding = nn.Embedding(num_embeddings=max_states, embedding_dim=embed_dim)
#         self.fc_block = nn.Sequential(
#             nn.Linear(embed_dim, hidden_dim)
#             ,nn.SELU()
#             ,nn.Linear(hidden_dim, hidden_dim)
#             ,nn.SELU()
#             ,nn.Linear(hidden_dim, action_dim)
#         )

#     def forward(self, x):
#         x = self.state_embedding(x).squeeze(-2)
#         return self.fc_block(x)
# ##############################################################################################################

# env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode="rgb_array")  # deterministic
# obs, info = env.reset()
# plt.imshow(env.render())
# # env.observation_space.n

# memory_size = 1024
# batch_size = 64
# sample_only_until = 500
# n_epochs = 3
# gamma = 0.9


# # main network
# main_Qnetwork = Critic(action_dim=env.action_space.n, hidden_dim=16,
#                        max_states=env.observation_space.n, embed_dim=2).to(device)
# main_Qnetwork.to(device)

# # target network
# target_Qnetwork = Critic(action_dim=env.action_space.n, hidden_dim=16,
#                        max_states=env.observation_space.n, embed_dim=2).to(device)
# target_Qnetwork.to(device)
# target_Qnetwork.load_state_dict(main_Qnetwork.state_dict())

# # memory
# memory = ReplayMemory(max_size=memory_size, batch_size=batch_size)

# optimizer = optim.AdamW(main_Qnetwork.parameters(), lr=1e-3, weight_decay=1e-2)
# # loss_function = nn.MSELoss()
# loss_function = nn.SmoothL1Loss()


# truncated_step = 100
# N_ITER = 10
# TOTAL_TIMESTEPS = 1000        # Rollout Timstep
# N_REPLAY_STEPS = 100         # Rollout 횟수
# N_EPOCHS = 5               # Model Learning 횟수
# target_update_interval = 3   # target network 주기


# for iter in range(N_ITER):
#     print(f"\r({iter+1}/{N_ITER} ITER) ", end='')
#     replay_time_step = 0
    
#     total_loop = (TOTAL_TIMESTEPS-1)//N_REPLAY_STEPS+1
#     loop_count = 1
    
#     while (replay_time_step < TOTAL_TIMESTEPS):
#         # (Collect Rollout) ##################################################################################
#         memory.reset()
#         obs, info = env.reset()
#         i = 0
        
        
#         # (RollOut) ------------------------------------------------------------------------------------------------
#         len_episodes = []
#         for _ in range(N_REPLAY_STEPS):
#             print(f"\r[{iter+1}/{N_ITER}ITER](Loop {loop_count}/{total_loop}) Rollout: {replay_time_step+1}/{TOTAL_TIMESTEPS}, ", end='')
            
#             # (exp temperature) -------------------------------------------
#             q_logits = main_Qnetwork(torch.LongTensor([obs]).to(device))
#             T = np.logspace(-2, 2, num=TOTAL_TIMESTEPS)[replay_time_step]
#             dist = Categorical(logits=T*q_logits)
#             action = dist.sample().item()
#             # ------------------------------------------------------------

#             next_obs, reward, terminated, truncated, info = env.step(action)
#             if (i >= truncated_step) or (_ == (N_REPLAY_STEPS-1)):
#                 truncated = True
#             done = terminated or truncated
            
#             experience = (action, obs, next_obs, reward, done)
#             # buffer에 experience 저장
#             memory.push(experience)
            
#             if done:
#                 obs, info = env.reset()
#                 len_episodes.append(i)
#                 i = 0
#             else:
#                 obs = next_obs
#                 i += 1
#             replay_time_step += 1
#         # -------------------------------------------------------------------------------------------------------

        
#         avg_loss = 0
#         for epoch in range(N_EPOCHS):
#             for batch in memory:
#                 sampled_exps = batch
#                 batch_actions = torch.LongTensor(np.stack([sample[0] for sample in sampled_exps])).view(-1,1).to(device)
#                 batch_obs = torch.LongTensor(np.stack([sample[1] for sample in sampled_exps])).view(-1,1).to(device)
#                 batch_next_obs = torch.LongTensor(np.stack([sample[2] for sample in sampled_exps])).view(-1,1).to(device)
#                 batch_rewards = torch.FloatTensor(np.stack([sample[3] for sample in sampled_exps])).view(-1,1).to(device)
#                 batch_dones = torch.FloatTensor(np.stack([sample[4] for sample in sampled_exps])).view(-1,1).to(device)
                
#                 with torch.no_grad():
#                     q_max, q_max_idx = target_Qnetwork(batch_next_obs).max(dim=-1, keepdims=True)
#                     q_target = batch_rewards + gamma*q_max * (1-batch_dones)

#                 q_value = main_Qnetwork(batch_obs).gather(1, batch_actions)        # gather : 1번째 axis에서 action위치의 값을 choose)
#                 loss = loss_function(q_value, q_target)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             avg_loss += loss.to('cpu') * len(sampled_exps)
#             print(f"\r[{iter+1}/{N_ITER}ITER](Loop {loop_count}/{total_loop}) Rollout: {replay_time_step+1}/{TOTAL_TIMESTEPS}, Avg_len_episode: {np.mean(len_episodes):.1f} learning_epoch: {epoch+1}/{N_EPOCHS} (CriticLoss: {avg_loss/(n_epochs * len(memory)):.3f})", end='')

#         if (loop_count % target_update_interval == 0) or (replay_time_step >= TOTAL_TIMESTEPS):
#             target_Qnetwork.load_state_dict(main_Qnetwork.state_dict())
        
#         loop_count += 1



# # Simulation Test ---------------------------------------------------------------------------------
# obs, info = env.reset()
# # env.render()
# i = 0
# done = False
# while (done is not True):
    
#     with torch.no_grad():
#         main_Qnetwork.eval()
        
#         q_pred = main_Qnetwork(torch.LongTensor([obs]).to(device))
#         action = torch.argmax(q_pred).item()
#         next_obs, reward, terminated, truncated, info = env.step(action)
#         done = terminated or truncated
        
#         plt.imshow(env.render())
#         plt.show()
#         time.sleep(0.1)
#         clear_output(wait=True)
#         obs = next_obs
#     i += 1
#     if i >=30:
#         break
    
    
    
    
    
    
    
    



















    
    
    

# ################################################################################################################
# # ( Actor-Critic Control ) ######################################################################################
# # policy-network 정의 (Actor Network)
# class Actor(nn.Module):
#     def __init__(self, state_dim, hidden_dim, action_dim):
#         super().__init__()
#         self.fc_block = nn.Sequential(
#             nn.Linear(state_dim, hidden_dim)
#             ,nn.ReLU()
#             ,nn.Linear(hidden_dim, hidden_dim)
#             ,nn.ReLU()
#             ,nn.Linear(hidden_dim, action_dim)
#         )

#     def forward(self, x):
#         return self.fc_block(x)
    
#     def predict_prob(self, x):
#         return nn.functional.softmax(self.forward(x), dim=-1)

#     def greedy_action(self, x, possible_actions=None):
#         with torch.no_grad():
#             action_prob = self.predict_prob(x)
#             result_tensor = torch.full_like(action_prob, float('-inf'))
#             if possible_actions is not None:
#                 result_tensor[..., possible_actions] = action_prob[..., possible_actions]
#             else:
#                 result_tensor = action_prob
#             return torch.argmax(result_tensor, dim=-1).item()

#     def explore_action(self, x, possible_actions=None):
#         with torch.no_grad():
#             action_logit = self.forward(x)
#             result_tensor = torch.full_like(action_logit, 0)
#             if possible_actions is not None:
#                 result_tensor[..., possible_actions] = action_logit[..., possible_actions]
#             else:
#                 result_tensor = action_logit
#             action_dist = Categorical(logits=result_tensor)
#             return action_dist.sample().item()

#     def get_action(self, x, possible_actions=None):
#         if self.training:
#             return self.explore_action(x, possible_actions=possible_actions)
#         else:
#             return self.greedy_action(x, possible_actions=possible_actions)


# # Q-network 정의
# class Critic(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.fc_block = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim)
#             ,nn.ReLU()
#             ,nn.Linear(hidden_dim, hidden_dim)
#             ,nn.ReLU()
#             ,nn.Linear(hidden_dim, output_dim)
#         )

#     def forward(self, x):
#         return self.fc_block(x)



# # env = CustomGridWorld()
# # env = CustomGridWorld(grid_size=5)
# # env = CustomGridWorld(grid_size=5, traps=[], obstacles=[], treasures=[])
# # env = CustomGridWorld(grid_size=5, obstacles=obstacles)
# # env = CustomGridWorld(grid_size=5, obstacles=obstacles, traps=traps)
# # env = CustomGridWorld(grid_size=5, obstacles=obstacles, traps=traps, treasures=treasures)
# # env = CustomGridWorld(grid_size=5)
# env = CustomGridWorld()

# env.reset()
# env.render()

# memory_size = 1024
# batch_size = 64
# sample_only_until = 1000
# target_update_interval = 30
# gamma = 0.9

# # actor network
# actor_network = Actor(state_dim=2, hidden_dim=16, action_dim=4)
# actor_optimizer = optim.Adam(actor_network.parameters(), lr=1e-5)

# # main network
# main_critic_network = Critic(2, 16, 4)
# main_critic_network.to(device)
# critic_optimizer = optim.Adam(main_critic_network.parameters(), lr=1e-4)

# # target network
# target_critic_network = Critic(2, 16, 4)
# target_critic_network.to(device)

# # memory
# memory = ReplayMemory(max_size=memory_size, batch_size=batch_size)



# # loss_function = nn.MSELoss()
# loss_function = nn.SmoothL1Loss()
# # |x_i - y_i| < 1 : z_i = 0.5(x_i - y_i)^2
# # |x_i - y_i| >= 1 : z_i = |x_i - y_i| - 0.5



# num_episodes = 300
# with tqdm(total=num_episodes) as pbar:
#     for epoch in range(num_episodes):
#         # run episode
#         env.reset()
#         done = False

#         i = 0
#         while (done is not True):
#             # select action from policy
#             cur_state_tensor = torch.tensor(env.cur_state).type(torch.float32)
            
#             action_probs = actor_network.predict_prob(cur_state_tensor)
#             action = torch.multinomial(action_probs, 1).item()
            
#             from_state, next_state, reward, done = env.step(action)
#             experience = (action, from_state, next_state, reward, done)

#             # buffer에 experience 저장
#             memory.push(experience)
#             cumulative_reward += reward
#             i += 1
#             if i >=50:
#                 break

#         # 최소 요구 sample수가 만족 했을경우, Trainning 진행
#         if len(memory) >= sample_only_until:
#             sampled_exps = memory.sample()
#             actions = torch.tensor(np.stack([sample[0] for sample in sampled_exps])).type(torch.int64).view(-1,1)
#             states = torch.tensor(np.stack([sample[1] for sample in sampled_exps])).type(torch.float32)
#             next_states = torch.tensor(np.stack([sample[2] for sample in sampled_exps])).type(torch.float32)
#             rewards = torch.tensor(np.stack([sample[3]/100 for sample in sampled_exps])).type(torch.float32).view(-1,1)
#             dones = torch.tensor(np.stack([sample[4] for sample in sampled_exps])).type(torch.float32).view(-1,1)
            
#             with torch.no_grad():
#                 next_q_max, next_q_max_idx = target_critic_network(next_states).max(dim=-1, keepdims=True)
#                 td_target = rewards + gamma*next_q_max.detach() * (1-dones)

#             q_value = main_critic_network(states).gather(1, actions)        # gather : 1번째 axis에서 action위치의 값을 choose)
#             td_error = (td_target - q_value).detach()

#             # actor loss
#             actions_probs = actor_network.predict_prob(states)
#             log_probs = torch.log(actions_probs.gather(1, actions) + 1e-8)
#             # actor_loss = (-log_probs * td_target).mean()
#             actor_loss = (-log_probs * td_error).mean()      # base_line

#             # actor loss update
#             actor_optimizer.zero_grad()
#             actor_loss.backward()
#             actor_optimizer.step()

#              # critic loss
#             critic_loss = loss_function(q_value, td_target)

#             # critic loss update
#             critic_optimizer.zero_grad()
#             critic_loss.backward()
#             critic_optimizer.step()

#             if epoch % 1 == 0:
#                 pbar.set_postfix(Actor_Loss=f"{actor_loss.to('cpu'):.3f}", Critic_Loss=f"{critic_loss.to('cpu'):.3f}", Len_episodes=f"{i}")
            

#         if epoch % target_update_interval == 0:
#             target_critic_network.load_state_dict(main_critic_network.state_dict())
#             print('target_network update')
#         pbar.update(1)




# # Simulation Test ---------------------------------------------------------------------------------
# env.reset()
# # env.render()
# i = 0
# done = False
# while (done is not True):
    
#     with torch.no_grad():
#         cur_state_tensor = torch.tensor(env.cur_state).type(torch.float32)
#         action = actor_network.greedy_action(cur_state_tensor)
#         from_state, next_state, reward, done = env.step(action)
#         env.render()
#         time.sleep(0.2)
#         clear_output(wait=True)
#     i += 1
#     if i >=30:
#         break

# ################################################################################################################



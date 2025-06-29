
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



################################################################################################################
################################################################################################################
# [ FrozenLake ] ###############################################################################################

import gymnasium as gym
import httpimport

try:
    try:
        from Environments.RL00_Env01_FrozenLake_v1 import generate_frozenlake_map
        # from utils.DS_RL import ReplayMemory
        custom_map = generate_frozenlake_map(5,5, hole=0.2)
    except:
        remote_url = 'https://raw.githubusercontent.com/kimds929/'

        with httpimport.remote_repo(f"{remote_url}/CodeNote/main/85_Reinforce_Learning/"):
            from Environments.RL00_Env01_FrozenLake_v1 import generate_frozenlake_map
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
start_training_after  = 500
n_epochs = 3
target_update_interval = 100

gamma = 0.9


# main network
main_Q1 = Critic(action_dim=env.action_space.n, hidden_dim=16,
                 max_states=env.observation_space.n, embed_dim=2).to(device)
main_Q2 = Critic(action_dim=env.action_space.n, hidden_dim=16,
                 max_states=env.observation_space.n, embed_dim=2).to(device)

# target network
target_Q1 = Critic(action_dim=env.action_space.n, hidden_dim=16,
                   max_states=env.observation_space.n, embed_dim=2).to(device)
target_Q2 = Critic(action_dim=env.action_space.n, hidden_dim=16,
                   max_states=env.observation_space.n, embed_dim=2).to(device)
target_Q1.load_state_dict(main_Q1.state_dict())
target_Q2.load_state_dict(main_Q2.state_dict())

# memory
memory = ReplayMemory(max_size=memory_size, batch_size=batch_size)

optimizer_Q1 = optim.AdamW(main_Q1.parameters(), lr=1e-3, weight_decay=1e-2)
optimizer_Q2 = optim.AdamW(main_Q2.parameters(), lr=1e-3, weight_decay=1e-2)
# loss_function = nn.MSELoss()
loss_function = nn.SmoothL1Loss()

num_episodes = 200
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
            #     q_logits = main_Q1(torch.LongTensor([obs]).to(device))
            #     action = torch.argmax(q_logits).item()
            # # ------------------------------------------------------------
            
            # (exp temperature) -------------------------------------------
            # q_logits = main_Q1(torch.LongTensor([obs]).to(device))
            q1_logits = main_Q1(torch.LongTensor([obs]).to(device))
            q2_logits = main_Q2(torch.LongTensor([obs]).to(device))
            q_logits = (q1_logits + q2_logits)/2
            
            T = np.logspace(2, -1, num=num_episodes)[episode]
            dist = Categorical(logits=q_logits/T)
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
            if len(memory) >= start_training_after :
                for epoch in range(n_epochs):
                    sampled_exps = memory.sample()
                    batch_actions = torch.LongTensor(np.stack([sample[0] for sample in sampled_exps])).view(-1,1).to(device)
                    batch_obs = torch.LongTensor(np.stack([sample[1] for sample in sampled_exps])).view(-1,1).to(device)
                    batch_next_obs = torch.LongTensor(np.stack([sample[2] for sample in sampled_exps])).view(-1,1).to(device)
                    batch_rewards = torch.FloatTensor(np.stack([sample[3] for sample in sampled_exps])).view(-1,1).to(device)
                    batch_dones = torch.FloatTensor(np.stack([sample[4] for sample in sampled_exps])).view(-1,1).to(device)
                    
                    with torch.no_grad():
                        q1_next = target_Q1(batch_next_obs)
                        q2_next = target_Q2(batch_next_obs)
                        q_min_next, _ = torch.min(torch.stack([q1_next, q2_next], dim=0), dim=0)
                        q_max, _ = q_min_next.max(dim=1, keepdim=True)
                        q_target = batch_rewards + gamma * q_max * (1 - batch_dones)
                    
                    q1_pred = main_Q1(batch_obs).gather(1, batch_actions)
                    q2_pred = main_Q2(batch_obs).gather(1, batch_actions)
                    
                    loss_q1 = loss_function(q1_pred, q_target)
                    loss_q2 = loss_function(q2_pred, q_target)
                    
                    optimizer_Q1.zero_grad()
                    loss_q1.backward()
                    optimizer_Q1.step()

                    optimizer_Q2.zero_grad()
                    loss_q2.backward()
                    optimizer_Q2.step()
                    
                    avg_loss += (loss_q1 + loss_q2).to('cpu')

            # ----------------------------------------------------------------------------------------
            # # Hard Update
            # if total_step % target_update_interval == 0:
            #     # Hard Update
            #     target_Q1.load_state_dict(main_Q1.state_dict())
            #     target_Q2.load_state_dict(main_Q2.state_dict())
                
            # Soft Update
            tau = 0.1
            for target_param, source_param in zip(target_Q1.parameters(), main_Q1.parameters()):
                target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
            for target_param, source_param in zip(target_Q2.parameters(), main_Q2.parameters()):
                target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
            # ----------------------------------------------------------------------------------------
            
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
        main_Q1.eval()
        main_Q2.eval()
        q1_logits = main_Q1(torch.LongTensor([obs]).to(device))
        q2_logits = main_Q2(torch.LongTensor([obs]).to(device))
        q_logits = (q1_logits + q2_logits)/2
        action = torch.argmax(q_logits).item()
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





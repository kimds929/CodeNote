
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



################################################################################################################
import gymnasium as gym


try:
    from Environments.RL00_Env01_FrozenLake_v1 import generate_frozenlake_map
    from utils.DS_RL import ReplayMemory
    custom_map = generate_frozenlake_map(5,5, hole=0.2)
except:
    custom_map=None



# Q-network 정의 # ★ (DuelingNetwork)
class DuelingCritic(nn.Module):
    def __init__(self, action_dim, hidden_dim, max_states=100, embed_dim=1):
        super().__init__()
        self.state_embedding = nn.Embedding(num_embeddings=max_states, embedding_dim=embed_dim)
        self.encoder_network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim)
            ,nn.ReLU()
        )
        
        self.value_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, 1)
        )
        
        self.advantage_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, action_dim)
        )
        

    def forward(self, x):
        embed_x = self.state_embedding(x).squeeze(-2)
        encoder_x = self.encoder_network(embed_x)
        
        value = self.value_network(encoder_x)
        advantage = self.advantage_network(encoder_x)
        q_logits = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q_logits
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
main_Qnetwork = DuelingCritic(action_dim=env.action_space.n, hidden_dim=16,
                       max_states=env.observation_space.n, embed_dim=2).to(device)
main_Qnetwork.to(device)

# target network
target_Qnetwork = DuelingCritic(action_dim=env.action_space.n, hidden_dim=16,
                       max_states=env.observation_space.n, embed_dim=2).to(device)
target_Qnetwork.to(device)
target_Qnetwork.load_state_dict(main_Qnetwork.state_dict())

# ★ (Prioritize Replay Buffer)
memory = ReplayMemory(max_size=memory_size, batch_size=batch_size, method='priority')

optimizer = optim.AdamW(main_Qnetwork.parameters(), lr=1e-3, weight_decay=1e-2)
# loss_function = nn.MSELoss()
loss_function = nn.SmoothL1Loss(reduction='none')

num_episodes = 150
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
            T = np.logspace(2, -2, num=num_episodes)[episode]
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
            if len(memory) >= sample_only_until:
                for epoch in range(n_epochs):
                    sampled_exps, indices, weights = memory.sample()

                    weights = torch.FloatTensor(weights)
                    batch_actions = torch.LongTensor(np.stack([sample[0] for sample in sampled_exps])).view(-1,1).to(device)
                    batch_obs = torch.LongTensor(np.stack([sample[1] for sample in sampled_exps])).view(-1,1).to(device)
                    batch_next_obs = torch.LongTensor(np.stack([sample[2] for sample in sampled_exps])).view(-1,1).to(device)
                    batch_rewards = torch.FloatTensor(np.stack([sample[3] for sample in sampled_exps])).view(-1,1).to(device)
                    batch_dones = torch.FloatTensor(np.stack([sample[4] for sample in sampled_exps])).view(-1,1).to(device)
                    
                    with torch.no_grad():
                        q_max, q_max_idx = target_Qnetwork(batch_next_obs).max(dim=-1, keepdims=True)
                        q_target = batch_rewards + gamma*q_max * (1-batch_dones)

                    q_value = main_Qnetwork(batch_obs).gather(1, batch_actions)        # gather : 1번째 axis에서 action위치의 값을 choose)
                    loss_unreduced = loss_function(q_value, q_target)
                    loss = (loss_unreduced * weights.to(device)).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        td_errors = loss_unreduced.detach().cpu().numpy().reshape(-1)
                        memory.update_priorities(indices, td_errors)
                    
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



import sys
sys.path.append(r'D:\DataScience\★GitHub_kimds929\CodeNote\85_Reinforcement_Learning')
# sys.path.append(r'D:\DataScience\★GitHub_kimds929\CodeNote\85_Reinforcement_Learning\Environments')
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
    from Environments.RL00_Env01_FrozenLake_v1 import generate_frozenlake_map, SightMask, frozenlake_visualize_grid_probs
    from utils.DS_RL import ReplayMemory
    custom_map = generate_frozenlake_map(5,5, hole=0.1)
except:
    custom_map=None


################################################################################################################


# import httpimport
# remote_url = 'https://raw.githubusercontent.com/kimds929/'


# import requests
# response = requests.get("https://raw.githubusercontent.com/kimds929/DS_Library/main/DS_Torch.py")
# exec(response.text)

# import importlib
# importlib.reload(httpimport)

# with httpimport.remote_repo(f"{remote_url}/DS_Library/main/"):
#     from DS_DeepLearning import EarlyStopping

# with httpimport.remote_repo(f"{remote_url}/DS_Library/main/"):
#     from DS_Torch import TorchDataLoader, TorchModeling, AutoML
    

############################################################################################################
# Random Policy ############################################################################################

custom_map = generate_frozenlake_map(5,5, hole=0.15)
env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode="rgb_array")
obs, info = env.reset()

sight_mask = SightMask(env)
plt.imshow(env.render())
plt.imshow(sight_mask.mask())
plt.show()





# Episode
sight_mask.reset()
obs, info = env.reset()

i=0
while True:
    # Animation
    if i % 1 == 0:
        plt.imshow(env.render())
        plt.imshow(sight_mask.mask())
        plt.show()
        time.sleep(0.05)
        clear_output(wait=True)
        
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    sight_mask.append(obs, terminated=terminated, truncated=truncated)
    
    if terminated:
        if obs == env.observation_space.n:
            print("Goal reached!", "Reward:", reward)
        else:
            print("Fail.", "Reward:", reward)
        break
    elif truncated or i > 100:
        print("Fail to find goal", "Reward:", reward)
        break
    i+=1
    
plt.imshow(env.render())
plt.imshow(sight_mask.mask())
plt.show()






############################################################################################################
# DQN Policy ############################################################################################
custom_map = generate_frozenlake_map(5,5, hole=0.15)
env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode="rgb_array")
obs, info = env.reset()

sight_mask = SightMask(env)
plt.imshow(env.render())
# plt.imshow(sight_mask.mask())
plt.show()

class Critic(nn.Module):
    def __init__(self, embed_dim=2, hidden_dim=8, action_dim=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding_layer = nn.Embedding(embedding_dim=embed_dim, num_embeddings=100)
        self.block = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, action_dim)
        )
        
    def forward(self, x):
        x_shape = x.shape
        x = self.embedding_layer(x)
        x = x.view(x.shape[0], self.embed_dim)
        x = self.block(x)
        return x

# Episode
sight_mask.reset()
gamma = 0.9
alpha = 100   # entropy parameter

critic_main = Critic(embed_dim=4)
critic_target = Critic(embed_dim=4)


optimizer = optim.Adam(critic_main.parameters(), lr=1e-3)

T = 100

N_EPISODES = 500
for episode_num in range(N_EPISODES):
    obs, info = env.reset()
    i=0
    while True:
        # current state
        obs_tensor = torch.LongTensor([obs])
        pred_values = critic_main(obs_tensor)
        dist = torch.distributions.Categorical(logits=pred_values/T)
        action = dist.sample()
        next_obs, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        sight_mask.append(next_obs, done=done)
        
        if terminated:
            if next_obs == env.observation_space.n-1:
                reward = 100
            else:
                reward = -10
        else:
            reward = -1
        
        # next state
        with torch.no_grad():
            next_obs_tensor = torch.LongTensor([next_obs])
            pred_next_values_main = critic_main(next_obs_tensor)
            pred_next_max_action = torch.argmax(pred_next_values_main).item()
            
            pred_next_values = critic_target(next_obs_tensor)
            # alpha*(pred_next_values/alpha).exp().sum().log()
            pred_next_max_value = pred_next_values[0, pred_next_max_action]
            
        td_target = reward + gamma * pred_next_max_value.detach() * (1-done)
        
        # loss
        # td_loss = nn.SmoothL1Loss()(pred_values[0, action], td_target)
        td_loss = nn.MSELoss()(pred_values[0, action], td_target)
        
        # learning
        optimizer.zero_grad()
        td_loss.backward()
        # nn.utils.clip_grad_norm_(critic_main.parameters(), 1.0)
        optimizer.step()
        
        print(f"\r (Episode {episode_num}, Temp: {T:.2f}) TD-Loss : {td_loss:.2f}", end=' / ')
        if terminated:
            if next_obs == env.observation_space.n-1:
                print("Goal reached!", "Reward:", reward)
            else:
                print("Fail.", "Reward:", reward)
            break
        elif truncated or i > 100:
            print("Fail to find goal")
            break
            
        # obs_update
        obs = next_obs
        
        i+=1
    
    # # (Hard Update)
    # if episode_num % 5 == 0:
    #     critic_target.load_state_dict(critic_main.state_dict())
    
    # (Soft Update)
    tau = 0.01
    for target_param, source_param in zip(critic_target.parameters(), critic_main.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
    
    # Temerature Decay
    T = 1 if T < 1 else T * 0.99

# print(f" → Exploration Temperature :{T:.2f}")
    

# Visualize Policy
observations_tensor = torch.arange(env.observation_space.n).view(-1,1)
pred_values = critic_main(observations_tensor)

state_values = pred_values.mean(dim=-1).to('cpu').detach().numpy().reshape(env.unwrapped.nrow,env.unwrapped.ncol)
observations_probs = nn.functional.softmax(pred_values,dim=-1).to('cpu').detach().numpy().reshape(env.unwrapped.nrow,env.unwrapped.ncol,env.action_space.n)

obs, info = env.reset()
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(env.render(), alpha=0.5, extent=[-0.5, 5-0.5, -0.5, 5-0.5])
ax.imshow(sight_mask.mask(), extent=[-0.5, 5-0.5, -0.5, 5-0.5], zorder=1)
frozenlake_visualize_grid_probs(observations_probs, env, ax=ax)
plt.show()



# Learning Result
obs, info = env.reset()
while True:
    # Animation
    if i % 1 == 0:
        plt.imshow(env.render())
        plt.imshow(sight_mask.mask())
        plt.show()
        time.sleep(0.05)
        clear_output(wait=True)
    
    obs_tensor = torch.LongTensor([obs])
    with torch.no_grad():
        critic_main.eval()
        pred_values = critic_main(obs_tensor)
    action = torch.argmax(pred_values).item()
    obs, reward, terminated, truncated, info = env.step(action)
    sight_mask.append(obs, terminated=terminated, truncated=truncated)
    
    if terminated:
        if obs == env.observation_space.n-1:
            print("Goal reached!", "Reward: 100")
        else:
            print("Fail.", "Reward: -10")
        break
    elif truncated or i > 100:
        print("Fail to find goal")
        break
    i+=1

# plt.imshow(state_values, cmap='coolwarm')
# plt.colorbar()

# plt.imshow(sight_mask.mask())

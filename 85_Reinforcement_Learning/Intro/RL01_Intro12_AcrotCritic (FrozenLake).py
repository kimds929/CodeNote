
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
    custom_map = generate_frozenlake_map(5,5, hole=0.1)
except:
    custom_map=None



################################################################################################################
class Actor(nn.Module):
    def __init__(self, action_dim, hidden_dim, max_states=100, embed_dim=1):
        super().__init__()
        self.state_embedding = nn.Embedding(num_embeddings=max_states, embedding_dim=embed_dim)
        self.policy_network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, action_dim)
        )
    
    def execute_model(self, obs, actions=None, temperature=None):
        embed_x = self.state_embedding(obs).squeeze(-2)
        logits = self.policy_network(embed_x)
        action_dist = Categorical(logits=logits)
        entropy = action_dist.entropy()
        
        if actions is None:
            if temperature is None:
                action = torch.argmax(logits, dim=-1)
            else:
                explore_dist = Categorical(logits=logits/temperature)
                action = explore_dist.sample()
            log_prob = action_dist.log_prob(action)
            return action, log_prob, entropy
        
        else:
            log_prob = action_dist.log_prob(actions)
            return log_prob, entropy
    
    def forward(self, obs, temperature=None):
        action, log_prob, entropy = self.execute_model(obs, temperature=temperature)
        return action, log_prob, entropy
    
    def evaluate_actions(self, obs, actions, temperature=None):
        log_prob, entropy = self.execute_model(obs, actions=actions, temperature=temperature)
        return log_prob, entropy
    
    def predict(self, obs, temperature=None):
        action, log_prob, entropy = self.execute_model(obs, temperature=temperature)
        return action
        
# Q-network 정의(StateValue)
class Critic(nn.Module):
    def __init__(self, hidden_dim, max_states=100, embed_dim=1):
        super().__init__()
        self.state_embedding = nn.Embedding(num_embeddings=max_states, embedding_dim=embed_dim)
        
        self.value_network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs):
        embed_obs = self.state_embedding(obs).squeeze(-2)
        value = self.value_network(embed_obs)
        return value


##############################################################################################################
env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode="rgb_array")  # deterministic
obs, info = env.reset()
plt.imshow(env.render())
# env.observation_space.n

memory_size = 1024
batch_size = 64
sample_only_until = 500
n_epochs = 1
target_update_interval = 100

gamma = 0.9

# Actor Network
actor_network = Actor(action_dim=env.action_space.n, hidden_dim=32,
                       max_states=env.observation_space.n, embed_dim=2).to(device)
actor_optimizer = optim.AdamW(actor_network.parameters(), lr=1e-4, weight_decay=1e-2)

# main network
main_critic_network = Critic(hidden_dim=32,
                       max_states=env.observation_space.n, embed_dim=2).to(device)
critic_optimizer = optim.AdamW(main_critic_network.parameters(), lr=5e-4, weight_decay=1e-2)
critic_loss_function = nn.SmoothL1Loss(reduction='none')


# target network
target_critic_network = Critic(hidden_dim=32,
                       max_states=env.observation_space.n, embed_dim=2).to(device)
target_critic_network.load_state_dict(main_critic_network.state_dict())

truncated_step = 100
num_episodes = 300


episode = 0
total_step = 0
with tqdm(total=num_episodes) as pbar:
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        
        episode_step=0
        cumulative_reward = 0
        
        cumulative_critic_loss = 0
        cumulative_actor_loss = 0
        while(not done):
            obs_tensor = torch.LongTensor([obs]).to(device)
            # T = np.logspace(2, 0, num=num_episodes)[episode]
            T = 1
            action, log_prob, entropy  = actor_network(obs_tensor, temperature=T)
            
            action = action.item()
            
            value = main_critic_network(obs_tensor)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            if (episode_step >= truncated_step):
                truncated = True
            reward -= 0.01  # step-penalty
            done = terminated or truncated
            
            
            cumulative_reward += reward
            
            with torch.no_grad():
                next_value = target_critic_network(torch.LongTensor([next_obs]).to(device))
            
            td_target = reward + gamma * next_value.detach() * (1-done)
            advantage = (td_target - value).detach()        # advantage = td_error
            
            # (critic_loss)
            critic_loss = nn.functional.smooth_l1_loss(value, td_target)
            
            # (actor_loss)
            actor_loss =  -(log_prob * advantage).mean()

            # critic update
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            
            # actor update
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            obs = next_obs
            
            # verbose
            cumulative_critic_loss += critic_loss.to('cpu')
            cumulative_actor_loss += actor_loss.to('cpu')

            if total_step % target_update_interval == 0:
                target_critic_network.load_state_dict(main_critic_network.state_dict())
                # print('target_network update')
            
            episode_step += 1
            total_step += 1
        
        if episode % 1 == 0:
            pbar.set_postfix(critic_loss=f"{cumulative_critic_loss/(episode_step):.3f}", 
                            actor_loss=f"{cumulative_actor_loss/(episode_step):.3f}",
                            Len_episodes=f"{episode_step}",
                            total_reward = f"{cumulative_reward:.2f}"
                            )
        pbar.update(1)             


# Simulation Test ---------------------------------------------------------------------------------
obs, info = env.reset()
# env.render()
i = 0
done = False
while (done is not True):
    
    with torch.no_grad():
        actor_network.eval()
        action, _, _ = actor_network(torch.LongTensor([obs]).to(device))
        action = action.item()  
        
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


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
    


##############################################################################################################
# policy-network 정의 (Actor Network)
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
    
    def forward_logits(self, obs):
        embed_x = self.state_embedding(obs).squeeze(-2)
        logits = self.policy_network(embed_x)
        return logits
    
    def execute_model(self, obs, actions=None, temperature=None):
        logits = self.forward_logits(obs)
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
actor_update_interval = 100

gamma = 0.9
policy_noise = 0.2


# policy_network
actor_main_network = Actor(hidden_dim=64, action_dim=env.action_space.n, embed_dim=4).to(device)
optimizer_main_A = optim.Adam(actor_main_network.parameters(), lr=3e-4)

actor_target_network = Actor(hidden_dim=64, action_dim=env.action_space.n, embed_dim=4).to(device)
actor_target_network.load_state_dict(actor_main_network.state_dict())


# main network
main_Q1 = Critic(action_dim=env.action_space.n, hidden_dim=16,
                 max_states=env.observation_space.n, embed_dim=2).to(device)
main_Q2 = Critic(action_dim=env.action_space.n, hidden_dim=16,
                 max_states=env.observation_space.n, embed_dim=2).to(device)
optimizer_main_Q1 = optim.AdamW(main_Q1.parameters(), lr=1e-3, weight_decay=1e-2)
optimizer_main_Q2 = optim.AdamW(main_Q2.parameters(), lr=1e-3, weight_decay=1e-2)

# target network
target_Q1 = Critic(action_dim=env.action_space.n, hidden_dim=16,
                   max_states=env.observation_space.n, embed_dim=2).to(device)
target_Q2 = Critic(action_dim=env.action_space.n, hidden_dim=16,
                   max_states=env.observation_space.n, embed_dim=2).to(device)
target_Q1.load_state_dict(main_Q1.state_dict())
target_Q2.load_state_dict(main_Q2.state_dict())

# memory
memory = ReplayMemory(max_size=memory_size, batch_size=batch_size)


loss_function = nn.MSELoss()
# loss_function = nn.SmoothL1Loss()

num_episodes = 100
total_step = 0


with tqdm(total=num_episodes) as pbar:
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        i = 0
        
        while(not done):
            obs_tensor = torch.LongTensor([obs]).to(device)
            # T = np.logspace(2, 0, num=num_episodes)[episode]
            T = 1
            action, log_prob, entropy  = actor_main_network(obs_tensor, temperature=T)
            action = action.item()
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            # if env.unwrapped.desc.ravel()[next_obs] == b'H':
            #     reward -= 10
            # elif env.unwrapped.desc.ravel()[next_obs] == b'G':
            #     reward += 100
            # elif next_obs == obs:
            #     reward -= 1
            # else:
            #     reward -= 0.2  # step-penalty
            
            if env.unwrapped.desc.ravel()[next_obs] == b'G':
                print('goal')
            done = terminated or truncated
            experience = (action, obs, next_obs, reward, done)
            # buffer에 experience 저장
            memory.push(experience)
            
            obs = next_obs
            i += 1
            if i >=100:
                break
        
            cumulative_critic_loss = 0
            cumulative_actor_loss = 0
            if len(memory) >= start_training_after :
                for epoch in range(n_epochs):
                    sampled_exps = memory.sample()
                    batch_actions = torch.LongTensor(np.stack([sample[0] for sample in sampled_exps])).view(-1,1).to(device)
                    batch_obs = torch.LongTensor(np.stack([sample[1] for sample in sampled_exps])).view(-1,1).to(device)
                    batch_next_obs = torch.LongTensor(np.stack([sample[2] for sample in sampled_exps])).view(-1,1).to(device)
                    batch_rewards = torch.FloatTensor(np.stack([sample[3] for sample in sampled_exps])).view(-1,1).to(device)
                    batch_dones = torch.FloatTensor(np.stack([sample[4] for sample in sampled_exps])).view(-1,1).to(device)
                    
                    with torch.no_grad():
                        next_logits = actor_target_network.forward_logits(batch_next_obs)  # 추가: target Actor 사용
                        next_probs = torch.exp(next_logits/T) / torch.exp(next_logits/T).sum(dim=-1, keepdim=True)
                        noisy_probs = next_probs + torch.randn_like(next_probs) * policy_noise
                        next_actions = noisy_probs.argmax(dim=-1, keepdim=True)
                        
                        q1_next = target_Q1(batch_next_obs).gather(1, next_actions)
                        q2_next = target_Q2(batch_next_obs).gather(1, next_actions)
                        q_min_next = torch.min(q1_next, q2_next)
                        q_target = batch_rewards + gamma * q_min_next * (1 - batch_dones)
                    
                    q1_pred = main_Q1(batch_obs).gather(1, batch_actions)
                    q2_pred = main_Q2(batch_obs).gather(1, batch_actions)
                    
                    loss_q1 = loss_function(q1_pred, q_target)
                    loss_q2 = loss_function(q2_pred, q_target)
                    
                    # update
                    optimizer_main_Q1.zero_grad()
                    loss_q1.backward()
                    optimizer_main_Q1.step()

                    optimizer_main_Q2.zero_grad()
                    loss_q2.backward()
                    optimizer_main_Q2.step()
                    
                    cumulative_critic_loss += (loss_q1 + loss_q2).to('cpu')
                    
                    if total_step % actor_update_interval == 0:
                        action, log_prob, entropy  = actor_main_network(batch_obs, temperature=T)
                        action = action.view(-1,1)
                        log_prob = log_prob.view(-1,1)
                        # q_val = main_Q1(batch_obs).gather(1, action)
                        advantage = main_Q1(batch_obs).gather(1, action) - main_Q1(batch_obs).mean(dim=-1, keepdim=True)
                        advantage = (advantage - advantage.mean())/(advantage.std()+1e-8)
                        
                        # beta = np.logspace(1, -2, num=num_episodes)[episode].item()
                        # beta = np.linspace(10, 0.01, num=num_episodes)[episode].item()
                        beta = 0.1
                        actor_loss = -(log_prob * advantage.detach() + beta*entropy).mean()
                        
                        optimizer_main_A.zero_grad()
                        actor_loss.backward()
                        optimizer_main_A.step()
                        # ----------------------------------------------------------------------------------------
                        cumulative_actor_loss += (actor_loss).to('cpu')
                        
                        # ----------------------------------------------------------------------------------------
                        # Soft Update
                        tau = 0.01
                        for target_param, source_param in zip(actor_target_network.parameters(), actor_main_network.parameters()):
                            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
                    
                    tau = 0.01
                    for target_param, source_param in zip(target_Q1.parameters(), main_Q1.parameters()):
                        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
                    for target_param, source_param in zip(target_Q2.parameters(), main_Q2.parameters()):
                        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
                    # ----------------------------------------------------------------------------------------
            # ----------------------------------------------------------------------------------------
            
            
            total_step += 1
            
        if episode % 1 == 0:
            if cumulative_actor_loss != 0:
                pbar.set_postfix(A_loss=f"{cumulative_actor_loss/n_epochs:.3f}" , Q_loss=f"{cumulative_critic_loss/n_epochs:.3f}", Len_episodes=f"{i}")
            else:
                pbar.set_postfix(Q_loss=f"{cumulative_critic_loss/n_epochs:.3f}", Len_episodes=f"{i}")
        pbar.update(1)


# q_val = main_Q1(batch_obs).gather(1, action)
# actor_loss = -(log_prob.unsqueeze(1) * q_val.detach()).mean()


# advantage = main_Q1(batch_obs).gather(1, action) - main_Q1(batch_obs).mean(dim=-1, keepdim=True)
# actor_loss = -(log_prob.unsqueeze(1) * advantage.detach()).mean()

# advantage = main_Q1(batch_obs).gather(1, action) - main_Q1(batch_obs).mean(dim=-1, keepdim=True)
# advantage = (advantage - advantage.mean())/(advantage.std()+1e-8)
# actor_loss = -(log_prob.unsqueeze(1) * advantage.detach()).mean()

# Simulation Test ---------------------------------------------------------------------------------
obs, info = env.reset()
# env.render()
i = 0
done = False
while (done is not True):
    
    with torch.no_grad():
        # main_Q1.eval()
        # main_Q2.eval()
        # q1_logits = main_Q1(torch.LongTensor([obs]).to(device))
        # q2_logits = main_Q2(torch.LongTensor([obs]).to(device))
        # q_logits = (q1_logits + q2_logits)/2
        # action = torch.argmax(q_logits).item()
        # next_obs, reward, terminated, truncated, info = env.step(action)
        # done = terminated or truncated
        
        obs_tensor = torch.LongTensor([obs]).to(device)
        action, log_prob, entropy  = actor_main_network(obs_tensor)
        action = action.item()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        plt.imshow(env.render())
        plt.show()
        time.sleep(0.2)
        clear_output(wait=True)
        obs = next_obs
    i += 1
    if i >=30:
        break

################################################################################################################
################################################################################################################
################################################################################################################

# ← ↓ → ↑
actor_main_network(torch.LongTensor([0]).to(device))
env.step(3)


allgrid_logits = actor_main_network.forward_logits(torch.arange(env.observation_space.n).to(device))
allgrid_probs = (torch.exp(allgrid_logits) / torch.exp(allgrid_logits).sum(dim=-1, keepdim=True)).to('cpu').detach().numpy()
actor_probs = allgrid_probs.reshape(env.unwrapped.nrow,env.unwrapped.ncol,env.action_space.n).copy()


q1_probs = torch.softmax(main_Q1(torch.arange(env.observation_space.n)), dim=-1).detach().to('cpu').numpy().reshape(env.unwrapped.nrow,env.unwrapped.ncol,env.action_space.n)
q2_probs = torch.softmax(main_Q2(torch.arange(env.observation_space.n)), dim=-1).detach().to('cpu').numpy().reshape(env.unwrapped.nrow,env.unwrapped.ncol,env.action_space.n)


# visualize_grid_probs(actor_probs, env)
# visualize_grid_probs(q1_probs, env)
# visualize_grid_probs(q2_probs, env)
# visualize_grid_probs((q1_probs + q2_probs)/2, env)





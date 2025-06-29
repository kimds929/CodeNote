
import sys
sys.path.append(r'D:\DataScience\★Git_CodeNote\85_Reinforcement_Learning')

import os
from six.moves import cPickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict

import time
from IPython.display import clear_output
from tqdm.auto import tqdm



######################################################################################################
def visualize_grid_probs(prob_map, env):
    n_rows = env.unwrapped.nrow
    n_cols = env.unwrapped.ncol
    grid = env.unwrapped.desc.astype(str)
    
    
    import matplotlib.cm as cm
    # 방향: ← ↓ → ↑
    dirs = [(-0.3, 0), (0, -0.3), (0.3, 0), (0, 0.3)]
    color_dict = {'S': 'mediumseagreen', 'H':'blue', 'G':'red'}

    fig, ax = plt.subplots(figsize=(8, 8))
    norm = plt.Normalize(vmin=0.2, vmax=0.3)
    cmap = cm.get_cmap('jet')

    for i in range(n_rows):
        for j in range(n_cols):
            cx, cy = j, (n_rows-1) -i  # 좌측 상단이 index 0
            cell_probs = prob_map[i, j]
            max_dir = np.argmax(cell_probs)
            
            label = grid[i, j]
            if label != 'F':
                ax.text(cx, cy, label, fontsize=12, color=color_dict[label], ha='center', va='center', weight='bold')
            for d, (dx, dy) in enumerate(dirs):
                prob = cell_probs[d]
                color = cmap(norm(prob))
                ax.arrow(cx, cy, dx * prob * 2, dy * prob * 2,
                        head_width=0.05, head_length=0.05,
                        fc=color, ec=color, alpha=0.5)

                # 색상 조건: max 확률이면 빨간색, 아니면 검정
                text_color = 'red' if d == max_dir else 'black'
                alpha = 1 if d == max_dir else 0.5
                # 확률 수치 annotation
                offset_x, offset_y = dx * 0.9, dy * 0.9
                ax.text(cx + offset_x, cy + offset_y, f"{prob:.3f}",
                        fontsize=9, ha='center', va='center', color=text_color, alpha=alpha)
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
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(-0.5, (n_cols-1) + 0.5)
    ax.set_ylim(-0.5, (n_rows-1) + 0.5)
    ax.set_aspect('equal')
    ax.set_title("Policy Action Probabilities (← ↓ → ↑) with Values")
    plt.tight_layout()
    # plt.show()
    plt.close()
    return fig


######################################################################################################
import httpimport
import gymnasium as gym

try:
    try:
        from Environments.RL00_Env01_FrozenLake_v1 import generate_frozenlake_map
        # from utils.DS_RL import ReplayMemory
        custom_map = generate_frozenlake_map(5,5, hole=0.2)
    except:
        remote_url = 'https://raw.githubusercontent.com/kimds929/'

        with httpimport.remote_repo(f"{remote_url}/CodeNote/refs/heads/main/85_Reinforcement_Learning/Environments"):
            from RL00_Env01_FrozenLake_v1 import generate_frozenlake_map
        custom_map = generate_frozenlake_map(5,5, hole=0.2)
except:
    custom_map=None


# action(0,1,2,3): ←, ↓, →, ↑
# next_obs, reward, terminated, truncated, info = env.step(action)
env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode="rgb_array")  # deterministic
obs, info = env.reset()
plt.imshow(env.render())

#########################################################################################################
# MDP
env_mdp = env.unwrapped.P
policy = np.ones((25,4)) / np.ones((25,4)).sum(axis=-1, keepdims=True)
V = np.zeros(25).astype(np.float32)
gamma = 0.9
threshold = 0.01

while True:
    # Policy Evaluation
    while True:
        Delta = 0
        for state in range(env.observation_space.n):
            v = V[state]
            vs = 0
            for action, next_info in env_mdp[state].items():
                for prob, next_state, reward, done in next_info:
                    vs += policy[state][action] * (reward + gamma * V[next_state]) * prob
            V[state] = vs
            Delta = max(Delta, abs(v - V[state]))
        if Delta < threshold:
            break

    # Policy Improvement
    policy_stable = True
    for state in range(env.observation_space.n):
        old_action = np.argmax(policy[state])
        q = np.zeros(env.action_space.n)
        for action, next_info in env_mdp[state].items():
            for prob, next_state, reward, done in next_info:
                q[action] += prob * (reward + gamma * V[next_state])
        new_action = np.argmax(q)
        policy[state] = np.eye(env.action_space.n)[new_action]
        if old_action != new_action:
            policy_stable = False

    if policy_stable:
        break
    

visualize_grid_probs(policy.reshape(5,5,4), env)
V.reshape(5,5)


# Simulation Test ---------------------------------------------------------------------------------
obs, info = env.reset()
# env.render()
i = 0
done = False
while (done is not True):
    action = np.argmax(policy[obs]).item()
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






#########################################################################################


import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#####################################################################################################
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
            log_prob = action_dist.log_prob(actions)
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
    
######################################################################################################
# action(0,1,2,3): ←, ↓, →, ↑
# V.reshape(5,5)

# actor_network.forward_logits(torch.LongTensor([0]).to(device)).to('cpu').detach()
# tensor([-0.0668, -0.0182,  0.0692, -0.0111])

env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode="rgb_array")  # deterministic
obs, info = env.reset()
plt.imshow(env.render())


gamma = 0.95

# policy_network
actor_network = Actor(hidden_dim=64, action_dim=env.action_space.n, embed_dim=4).to(device)
actor_optimizer = optim.AdamW(actor_network.parameters(), lr=1e-3, weight_decay=1e-2)

max_truncated = 50
num_episodes = 100
for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    
    i = 0
    cumulative_rewards = 0
    cumulative_actor_loss = 0
    while (done is not True):
        obs_tensor = torch.LongTensor([obs]).to(device)
        # T = np.logspace(10, 0, num=num_episodes)[episode]
        T = 1
        action, log_prob, entropy = actor_network(obs_tensor, temperature=T)
        logits = actor_network.forward_logits(obs_tensor).to('cpu').detach().numpy()
        logits = [round(logit.item(),3) for logit in logits]
        value = torch.FloatTensor([V[obs]])

        next_obs, reward, terminated, truncated, info = env.step(action.item())
        if i >= max_truncated:
            truncated = True
        done = terminated or truncated
        if env.unwrapped.desc.ravel()[next_obs] == b'G':
            print('goal')
        
        next_obs_tensor = torch.LongTensor([next_obs]).to(device)
        td_target = torch.FloatTensor([reward + gamma * V[next_obs]])
        
        # critic_loss = nn.functional.smooth_l1_loss(value,)
        advantage = (td_target - value).detach()   # td_error
        actor_loss = -(log_prob * advantage + 0.1*entropy).mean()
        # actor_loss = -(log_prob * advantage).mean()
        
        
        # print(f"obs: {obs}, action: {action}, next_obs: {next_obs}, log_prob: {log_prob.item():.3f}, prob: {torch.exp(log_prob).item():.3f}, logit:{logits}")
        # print(f"\t td_target:{td_target.item():.3f}, value: {value.item():.3f}, advantage: {advantage.item():.3f}, actor_loss: {actor_loss.item():.3f}")
        # print()
        # actor update
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        obs = next_obs
        
        cumulative_actor_loss += actor_loss.item()
        cumulative_rewards += reward
        i +=1
    print(f"\r[{episode+1}/{num_episodes}ITER] Episode_len: {i}, Total_Reward:{cumulative_rewards:.1f},  ActorLoss: {cumulative_actor_loss/i:.3f})", end='')


# # Simulation Test ---------------------------------------------------------------------------------
# obs, info = env.reset()
# # env.render()
# i = 0
# done = False
# while (done is not True):
    
#     with torch.no_grad():        
#         actor_network.eval()
#         action = actor_network.predict(torch.LongTensor([obs]).to(device), deterministic=True)
#         action = action.item()  
        
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


##############################################################################################################

allgrid_logits = actor_network.forward_logits(torch.arange(env.observation_space.n).to(device))
allgrid_probs = (torch.exp(allgrid_logits) / torch.exp(allgrid_logits).sum(dim=-1, keepdim=True)).to('cpu').detach().numpy()
actor_probs = allgrid_probs.reshape(env.unwrapped.nrow,env.unwrapped.ncol,env.action_space.n).copy()

visualize_grid_probs(actor_probs, env)

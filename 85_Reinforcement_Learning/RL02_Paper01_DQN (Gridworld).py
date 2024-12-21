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



import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict

import time
from IPython.display import clear_output
from tqdm.auto import tqdm

# import httpimport
# remote_url = 'https://raw.githubusercontent.com/kimds929/'

# with httpimport.remote_repo(f"{remote_url}/CodeNote/main/85_Reinforce_Learning/"):
#     from RL00_Env01_CustomGridWorld import CustomGridWorld





###########################################################################################################
# ( Util Functions ) ######################################################################################

try:
    from RL00_Env01_CustomGridWorld import CustomGridWorld
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

import torch
import torch.nn as nn
import torch.optim as optim

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


# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# Replay Buffer
from random import sample
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

    def __len__(self):
        return self.size


# Q-network 정의
class QNetwork(nn.Module):
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
env = CustomGridWorld(grid_size=5)
env.reset()
env.render()

memory_size = 1024
batch_size = 64
sample_only_until = 500
target_update_interval = 10

gamma = 0.9

# main network
main_Qnetwork = QNetwork(2, 16, 4)
main_Qnetwork.to(device)

# target network
target_Qnetwork = QNetwork(2, 16, 4)
target_Qnetwork.to(device)

# memory
memory = ReplayMemory(max_size=memory_size, batch_size=batch_size)

optimizer = optim.Adam(main_Qnetwork.parameters(), lr=1e-3)
# loss_function = nn.MSELoss()
loss_function = nn.SmoothL1Loss()
# |x_i - y_i| < 1 : z_i = 0.5(x_i - y_i)^2
# |x_i - y_i| >= 1 : z_i = |x_i - y_i| - 0.5



num_episodes = 300
for epoch in tqdm(range(num_episodes)):
        # run episode
    env.reset()
    done = False

    cumulative_reward = 0
    i = 0
    while (done is not True):
        action, q_pred = epsilon_greedy_torch(state=env.cur_state, model=main_Qnetwork, epsilon=2e-1, device=device)
        from_state, next_state, reward, done = env.step(action)
        experience = (action, from_state, next_state, reward, done)

        # buffer에 experience 저장
        memory.push(experience)
        cumulative_reward += reward
        i += 1
        if i >=100:
            break

    # 최소 요구 sample수가 만족 했을경우, Trainning 진행
    if len(memory) >= sample_only_until:
        sampled_exps = memory.sample()
        actions = torch.tensor(np.stack([sample[0] for sample in sampled_exps])).type(torch.int64).view(-1,1)
        states = torch.tensor(np.stack([sample[1] for sample in sampled_exps])).type(torch.float32)
        next_states = torch.tensor(np.stack([sample[2] for sample in sampled_exps])).type(torch.float32)
        rewards = torch.tensor(np.stack([sample[3]/100 for sample in sampled_exps])).type(torch.float32).view(-1,1)
        dones = torch.tensor(np.stack([sample[4]/100 for sample in sampled_exps])).type(torch.float32).view(-1,1)
        
        with torch.no_grad():
            q_max, q_max_idx = target_Qnetwork(next_states).max(dim=-1, keepdims=True)
            q_target = rewards + gamma*q_max * (1-dones)

        q_value = main_Qnetwork(states).gather(1, actions)        # gather : 1번째 axis에서 action위치의 값을 choose)
        loss = loss_function(q_value, q_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            print(f"\r (epoch {epoch}) loss :{loss.to('cpu'):.3f} | Cumulative_Reward: {cumulative_reward:.4f}")
    
    if epoch % target_update_interval == 0:
        target_Qnetwork.load_state_dict(main_Qnetwork.state_dict())
        print('target_network update')




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
        time.sleep(0.5)
        clear_output(wait=True)
    i += 1
    if i >=30:
        break

################################################################################################################
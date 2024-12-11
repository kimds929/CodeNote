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
# ( Temporal Difference Control (TD0) : Off-policy(Q-learning) ) ###############################################

# env = CustomGridWorld()
# env = CustomGridWorld(grid_size=5)
# env = CustomGridWorld(grid_size=5, traps=[], obstacles=[], treasures=[])
# env = CustomGridWorld(grid_size=5, obstacles=obstacles)
# env = CustomGridWorld(grid_size=5, obstacles=obstacles, traps=traps)
# env = CustomGridWorld(grid_size=5, obstacles=obstacles, traps=traps, treasures=treasures)
env = CustomGridWorld(grid_size=4)
env.reset()
env.render()

# env.step(1)

# Q-learning Update
def q_learning_update(Q, state, next_state, reward, action, alpha, gamma):
    best_next_action = np.argmax(Q[next_state])     # epsilon_greedy로 하면 안됨! : exploration으로 suboptimal action이 선택되어 수렴성을 방해 할 수 있음.
    return (1-alpha)*Q[state][action] +  alpha * (reward + gamma * Q[next_state][best_next_action])

alpha = 0.3
gamma = 0.9
# Q = np.ones([env.grid_size, env.grid_size, env.nA]) / env.nA
Q = defaultdict(lambda: np.ones(env.nA)/env.nA)


num_episodes = 100
for _ in tqdm(range(num_episodes)):
    # run episode
    env.reset()
    episode_info = []
    actions = []
    done = False

    i = 0
    while (done is not True):
        # select action from policy
        action = epsilon_greedy(Q[env.cur_state], epsilon=2e-1)
        from_state, next_state, reward, done = env.step(action)
        cur_info = (from_state, next_state, reward, done)

        # Q-learning Update
        Q[from_state][action] = q_learning_update(Q, from_state, next_state, reward, action, alpha, gamma)
            
        # save episode info
        episode_info.append(cur_info)
        actions.append(action)

        i += 1
        if i >=500:
            break

# Simulation Test ---------------------------------------------------------------------------------
env.reset()
# env.render()

i = 0
done = False
while (done is not True):
    # select action from policy
    action = rand_argmax(Q[env.cur_state])
    from_state, next_state, reward, done = env.step(action)
    env.render()
    time.sleep(1)
    clear_output(wait=True)

    i += 1
    if i >=100:
        break

################################################################################################################
















###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
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
env = CustomGridWorld(grid_size=4)
env.reset()
env.render()

gamma = 0.9
model = QNetwork(2, 16, 4)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_function = nn.MSELoss()

num_episodes = 100
for epoch in tqdm(range(num_episodes)):
    # run episode
    env.reset()
    done = False

    i = 0
    while (done is not True):
        # select action from policy
        action, q_pred = epsilon_greedy_torch(state=env.cur_state, model=model, epsilon=2e-1, device=device)
        
        from_state, next_state, reward, done = env.step(action)

        # select next action from policy
        with torch.no_grad():
            model.eval()
            next_q_pred = model(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device))
            best_next_action = torch.argmax(next_q_pred).item()
            
            # Q-learning Target : Gradient Descent가 이미 네트워크의 학습률을 고려하여 가중치를 업데이트하기 때문에 Network학습에서는 Q-Learning의 완전한 Update식을 따르지 않음
            # q_values[action] = (1-alpha)*q_pred.squeeze()[action] +  alpha * (reward + gamma * next_q_pred.squeeze()[best_next_action])
            q_values = q_pred.squeeze().clone()
            q_values[action] = reward + gamma * next_q_pred.squeeze()[best_next_action]
        
        # training
        model.train()
        optimizer.zero_grad()
        q_pred = model(torch.tensor(from_state, dtype=torch.float32).unsqueeze(0).to(device))
        loss = loss_function(q_pred.squeeze(), q_values)
        loss.backward()
        optimizer.step()

        i += 1
        if i >=500:
            break

    with torch.no_grad():
        print(f"\r (epoch {epoch}) loss :{loss.to('cpu'):.3f}")



# Simulation Test ---------------------------------------------------------------------------------
env.reset()
# env.render()
i = 0
done = False
while (done is not True):
    
    with torch.no_grad():
        model.eval()
        q_pred = model(torch.tensor(env.cur_state, dtype=torch.float32).unsqueeze(0).to(device))
        action = torch.argmax(q_pred).item()
        from_state, next_state, reward, done = env.step(action)
        env.render()
        time.sleep(0.5)
        clear_output(wait=True)
    i += 1
    if i >=50:
        break

################################################################################################################

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
# ( Monte Carlo Control ) ######################################################################################

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

        # save episode info
        episode_info.append(cur_info)
        actions.append(action)

        i += 1
        if i >=500:
            break

    # states in episode
    states_in_episode = [x[0] for x in episode_info]
    unique_states = np.unique(np.array(states_in_episode), axis=0)
    first_visit = [np.where((us == np.array(states_in_episode)).all(axis=1))[0][0] for us in unique_states]
    rev_first_visit = len(episode_info) - 1 - np.array(first_visit)

    # policy update
    G = 0
    for ei, ((state, next_state, reward, done), action) in enumerate(zip(reversed(episode_info), reversed(actions))):
        
        G = reward + gamma * G
        if ei in rev_first_visit:
            idx = np.where(rev_first_visit == ei)
            if state == tuple(list(unique_states[idx].squeeze())):
                Q[state][action] = G
                # Q[(*state, action)] = G




# plt.imshow(Q.reshape(-1,4), cmap='coolwarm')
# np.argmax(Q[(0,0)])
# pd.DataFrame(episode_info).to_clipboard()

np.array(dir(Q))
np.array(list(Q.values())).shape
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
    if i >=50:
        break

################################################################################################################







###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
# DeepNetwork MonteCarlo ##################################################################################

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
# ( Monte Carlo Control ) ######################################################################################

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
    episode_info = []
    actions = []
    q_preds = []
    torch_traindata = []
    done = False

    i = 0
    while (done is not True):
        # select action from policy
        action, q_pred = epsilon_greedy_torch(state=env.cur_state, model=model, epsilon=2e-1, device=device)
        from_state, next_state, reward, done = env.step(action)
        cur_info = (from_state, next_state, reward, done)

        # save episode info
        episode_info.append(cur_info)
        actions.append(action)
        q_preds.append(q_pred.squeeze().to('cpu').numpy())
        i += 1
        if i >=500:
            break

    # (first visit) ------------------------------------------------------------------------------------------
    # for each state in episode 
    states_in_episode = [x[0] for x in episode_info]
    unique_states = np.unique(np.array(states_in_episode), axis=0)
    first_visit = [np.where((us == np.array(states_in_episode)).all(axis=1))[0][0] for us in unique_states]
    rev_first_visit = len(episode_info) - 1 - np.array(first_visit)
    actions_first_visit = np.array(actions)[first_visit]
    q_preds_first_visit = np.array(q_preds)[first_visit]
    
    # calculate cumulative reward
    G = 0
    for ei, ((state, next_state, reward, done), action) in enumerate(zip(reversed(episode_info), reversed(actions))):
        G = reward + gamma * G
        if ei in rev_first_visit:
            idx = np.where(rev_first_visit == ei)
            if state == tuple(list(unique_states[idx].squeeze())):
                q_pred = q_preds_first_visit[idx].squeeze()
                q_pred[actions_first_visit[idx].item()] = G
                torch_traindata.append((state, q_pred))
    
    # # (every visit)  ------------------------------------------------------------------------------------------
    # # for each state in episode
    # states_in_episode = [x[0] for x in episode_info]

    # # calculate cumulative reward
    # G = 0
    # for ei, (state, action, q_pred) in enumerate(zip(reversed(states_in_episode), reversed(actions), reversed(q_preds))):
    #     print(q_pred)
    #     q_pred[action] = G
    #     torch_traindata.append((state, q_pred))
    
    # ---------------------------------------------------------------------------------------------------------
    # training dataset
    states_tensor = torch.tensor([data[0] for data in torch_traindata], dtype=torch.float32)
    q_pred_tensor = torch.tensor([data[1] for data in torch_traindata], dtype=torch.float32)

    # training
    model.train()
    optimizer.zero_grad()
    q_pred = model(states_tensor.to(device))
    loss = loss_function(q_pred, q_pred_tensor.to(device))
    loss.backward()
    optimizer.step()

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
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
from torch.utils.data import TensorDataset, DataLoader

# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

################################################################################################################

try:
    try:
        from Environments.RL00_Env01_FrozenLake_v1 import generate_frozenlake_map
        from DS_RL import visualize_grid_probs, ReplayMemory
        custom_map = generate_frozenlake_map(5,5, hole=0.17)
    except:
        remote_url = 'https://raw.githubusercontent.com/kimds929/'

        with httpimport.remote_repo(f"{remote_url}/CodeNote/refs/heads/main/85_Reinforcement_Learning/Environments"):
            from RL00_Env01_FrozenLake_v1 import generate_frozenlake_map
        custom_map = generate_frozenlake_map(5,5, hole=0.17)
        
        with httpimport.remote_repo(f"{remote_url}/CodeNote/refs/heads/main/85_Reinforcement_Learning/utils"):
            from DS_RL import visualize_grid_probs, ReplayMemory
except:
    custom_map=None


################################################################################################################


# pip install gymnasium[classic-control]
# pip install pygame

import gymnasium as gym


# 환경 생성
env = gym.make("CartPole-v1", render_mode="rgb_array")  # render_mode='human'은 화면에 직접 그리기용
obs, info = env.reset()
plt.imshow(env.render())

print(env.observation_space)
# Cart_Position, Cart_Velocity, Pole_Angle, Pole_Angular_Velocity
print(env.action_space)
# {0: 왼쪽 힘, 1: 오른쪽 힘}

obs, info = env.reset()
done=False
i=0
for _ in range(500):
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    done = terminated or truncated
    
    print(reward)
    plt.figure()
    plt.imshow(env.render())
    plt.show()
    time.sleep(0.1)
    clear_output(wait=True)
    
    obs = next_obs
    i += 1
    if done or i >=50:
        break

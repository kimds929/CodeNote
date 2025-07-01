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
env = gym.make("Pendulum-v1", render_mode="rgb_array")  # render_mode='human'은 화면에 직접 그리기용
obs, info = env.reset()
plt.imshow(env.render())
env.step([2])

print(env.observation_space)
# Box([-1. -1. -8.], [1. 1. 8.], (3,), float32)
# θ : -np.pi ~ -np.pi/2 : 4사분면 / -np.pi/2 ~ 0 : 1사분면 / 0 ~ np.pi/2 : 2사분면 / np.pi/2 ~ np.pi : 3사분면
#   (0-dim) cos(θ): 진자의 각도에 대한 코사인
#   (1-dim) sin(θ): 진자의 각도에 대한 사인
#   (2-dim) θ_dot: 진자의 각속도        # -8(시계방향) ~ 8 (반시계방향)
# θ를 직접 주지 않고 cos(θ), sin(θ)로 표현 → 주기성을 반영하기 위함


print(env.action_space)
# Box(low=-2.0, high=2.0, shape=(1,), dtype=float32)
# 범위: -2.0 ~ +2.0 (좌/우로 회전시키는 힘) # -2 ~ 0 : 시계방향 힘, 0 ~ 2 반시계방향 힘

# 보상구조
# reward = -(θ^2 + 0.1 * θ̇^2 + 0.001 * torque^2)
# 보상은 항상 음수이며, 0에 가까울수록 좋은 것
# 보상을 줄이기 위해:
#     진자가 위로 가까이 가야 하고
#     각속도가 작아야 하며
#     힘도 적게 써야 함

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





def actor_evaluator(actor):
    # θ : -np.pi ~ -np.pi/2 : 4사분면 / -np.pi/2 ~ 0 : 1사분면 / 0 ~ np.pi/2 : 2사분면 / np.pi/2 ~ np.pi : 3사분면
    # ang_velocity : -8(시계방향) ~ 8 (반시계방향)
    # action : -2 (시계방향 힘) ~ 2 (반시계방향 힘)

    radians = np.linspace(np.pi, -np.pi, num=9)
    # degrees = np.degrees(radians)
    ang_velocity = np.linspace(-8, 8, num=9)
    grid = np.stack(np.meshgrid(radians, ang_velocity)).reshape(2,-1).T
    grid_obs = np.concatenate([np.sin(grid[:, [0]]), np.cos(grid[:, [0]]),grid[:, [1]]], axis=1)

    grid_obs_tensor = torch.FloatTensor(grid_obs)
    pred_actions, _ = actor.predict(grid_obs_tensor)
    if 'torch' in str(type(pred_actions)):
        pred_actions = pred_actions.detach().to('cpu').numpy()
    grid_obs_pred = np.concatenate([np.degrees(grid[:,[0]]), grid[:,[0]], grid[:, [1]], pred_actions], axis=1)

    df_grid_obs_pred = pd.DataFrame(grid_obs_pred, columns=['degrees', 'radians','ang_velocity','pred_action'])

    df_predict_tb = df_grid_obs_pred.groupby(['radians', 'ang_velocity'])['pred_action'].mean().unstack('radians')
    df_predict_tb = df_grid_obs_pred.groupby(['degrees', 'ang_velocity'])['pred_action'].mean().unstack('degrees')
    df_predict_tb = np.round(df_predict_tb,2)
    df_predict_tb = df_predict_tb.sort_values('degrees', axis=1, ascending=False)
    return df_predict_tb



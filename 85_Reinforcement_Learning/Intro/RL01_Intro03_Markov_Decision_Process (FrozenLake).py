
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

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


def get_policy_from_state_value(env, V):
    policy = np.zeros(env.nA)
    for action in [0,1,2,3]:
        next_state, _ = env.get_nearby_state(action)
        if next_state is not None:
            policy[action] = V[next_state]
    return policy

def epsilon_greedy(policy, epsilon=1e-1, random_state=None):
    rng = np.random.RandomState(random_state)
    greedy_action = rand_argmax(policy)
    
    if rng.rand() < epsilon:
        not_greedy_action = np.where(policy != greedy_action)[0]
        if len(not_greedy_action) == 0:
            return greedy_action
        else:
            return rng.choice(not_greedy_action)
    else:
        return greedy_action




################################################################################################################
# ( Temporal Difference Control (TD0) : On-policy(SARSA) ) #####################################################



# env = CustomGridWorld()
# env = CustomGridWorld(grid_size=5)
# env = CustomGridWorld(grid_size=5, traps=[], obstacles=[], treasures=[])
# env = CustomGridWorld(grid_size=5, obstacles=obstacles)
# env = CustomGridWorld(grid_size=5, obstacles=obstacles, traps=traps)
# env = CustomGridWorld(grid_size=5, obstacles=obstacles, traps=traps, treasures=treasures)
env = CustomGridWorld(grid_size=4)
env.reset()
env.render()

env.grid_size
# env.step(1)


def TD_update(V, state, next_state, reward, alpha, gamma):
    td_target = reward + gamma * V[next_state]
    return V[state] + alpha * (td_target - V[state])
    # return (1-alpha) * V[state] + alpha * td_target

alpha = 0.3
gamma = 0.9
V = defaultdict(lambda : 0)

num_episodes = 100
for _ in tqdm(range(num_episodes)):
    # run episode
    env.reset()
    episode_info = []
    actions = []
    done = False
    
    i = 0
    while (done is not True):
        # get current policy
        action_value = get_policy_from_state_value(env, V)
        action = epsilon_greedy(action_value, epsilon=2e-1)

        # select action from policy
        from_state, next_state, reward, done = env.step(action)
        cur_info = (from_state, next_state, reward, done)

        V[from_state] = TD_update(V, from_state, next_state, reward, alpha, gamma)

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
    action_value = get_policy_from_state_value(env, V)
    action = rand_argmax(action_value)
    from_state, next_state, reward, done = env.step(action)
    env.render()
    time.sleep(1)
    clear_output(wait=True)

    i += 1
    if i >=30:
        break

################################################################################################################




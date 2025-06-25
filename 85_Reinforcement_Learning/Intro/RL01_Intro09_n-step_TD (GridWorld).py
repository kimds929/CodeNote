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
# ( Temporal Difference Control (TD0) : On-policy(SARSA) ) #####################################################



# env = CustomGridWorld()
# env = CustomGridWorld(grid_size=5)
# env = CustomGridWorld(grid_size=5, traps=[], obstacles=[], treasures=[])
# env = CustomGridWorld(grid_size=5, obstacles=obstacles)
# env = CustomGridWorld(grid_size=5, obstacles=obstacles, traps=traps)
# env = CustomGridWorld(grid_size=5, obstacles=obstacles, traps=traps, treasures=treasures)

env = CustomGridWorld(grid_size=4, traps=[])
# env = CustomGridWorld(grid_size=4)
env.reset()
env.render()


alpha=0.3
gamma = 0.9
num_episodes = 1
n = 5

Q = defaultdict(lambda: np.ones(env.nA)*1e-10)

num_episodes = 100
for _ in tqdm(range(num_episodes)):
    # run episode
    env.reset()
    done = False

    T = 150
    states = []
    actions = []
    rewards = []
    # while(True):
    for t in range(T):
        # print(t)

        if t < T:      
            action = epsilon_greedy(Q[env.cur_state], epsilon=2e-1)
            from_state, next_state, reward, done = env.step(action)

            states.append(from_state)
            actions.append(action)
            rewards.append(reward)

            if done:
                T = t + 1
                # break
        tau = t - n + 1

        if tau >= 0:
            G = 0
            for i_, r in enumerate(rewards[tau+1: min(tau+n, T)]):
                i = i_ + tau + 1
                G += gamma**(i-tau-1) * r

            if tau + n < T:
                G += gamma**n * Q[env.cur_state][actions[-1]]
            
            Q[states[tau]][actions[tau]] += alpha * (G - Q[states[tau]][actions[tau]])
            Q[states[tau]] = np.exp(Q[states[tau]]) / np.exp(Q[states[tau]]).sum()

        if tau == T-1:
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
    time.sleep(0.3)
    clear_output(wait=True)

    i += 1
    if i >=30:
        break

################################################################################################################



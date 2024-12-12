
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
from torch.distributions.categorical import Categorical




###########################################################################################################
# ( Util Functions ) ######################################################################################

try:
    from RL00_Env01_CustomGridWorld import CustomGridWorld
except:
    import httpimport
    remote_url = 'https://raw.githubusercontent.com/kimds929/'

    with httpimport.remote_repo(f"{remote_url}/CodeNote/main/85_Reinforcement_Learning/"):
        from RL00_Env01_CustomGridWorld import CustomGridWorld


try:
    from RL00_Env02_CartPole import CustomCartPole
except:
    import httpimport
    remote_url = 'https://raw.githubusercontent.com/kimds929/'

    with httpimport.remote_repo(f"{remote_url}/CodeNote/main/85_Reinforcement_Learning/"):
        from RL00_Env02_CartPole import CustomCartPole




################################################################################################################

# policy-network 정의
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(state_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.fc_block(x)
    
    def predict_prob(self, x):
        return nn.functional.softmax(self.forward(x), dim=-1)

    def greedy_action(self, x, possible_actions=None):
        with torch.no_grad():
            action_prob = self.predict_prob(x)
            result_tensor = torch.full_like(action_prob, float('-inf'))
            if possible_actions is not None:
                result_tensor[..., possible_actions] = action_prob[..., possible_actions]
            else:
                result_tensor = action_prob
            return torch.argmax(result_tensor, dim=-1).item()

    def explore_action(self, x, possible_actions=None):
        with torch.no_grad():
            action_logit = self.forward(x)
            result_tensor = torch.full_like(action_logit, 0)
            if possible_actions is not None:
                result_tensor[..., possible_actions] = action_logit[..., possible_actions]
            else:
                result_tensor = action_logit
            action_dist = Categorical(logits=result_tensor)
            return action_dist.sample().item()

    def get_action(self, x, possible_actions=None):
        if self.training:
            return self.explore_action(x, possible_actions=possible_actions)
        else:
            return self.greedy_action(x, possible_actions=possible_actions)

# Q-network 정의
class QNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(state_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.fc_block(x)



################################################################################################################

# (Step 단위 Update) --------------------------------------------------------------------------------------------

# env = CustomGridWorld()
# env = CustomGridWorld(grid_size=5)
# env = CustomGridWorld(grid_size=5, traps=[], obstacles=[], treasures=[])
# env = CustomGridWorld(grid_size=5, obstacles=obstacles)
# env = CustomGridWorld(grid_size=5, obstacles=obstacles, traps=traps)
# env = CustomGridWorld(grid_size=5, obstacles=obstacles, traps=traps, treasures=treasures)
# env = CustomGridWorld(grid_size=4)
env = CustomGridWorld(grid_size=5, traps=[], obstacles=[], treasures=[])
env.reset()
env.render()

gamma = 0.9

policy_network = PolicyNetwork(state_dim=2, hidden_dim=16, action_dim=4)
policy_optimizer = optim.Adam(policy_network.parameters(), lr=1e-4)

q_network = QNetwork(state_dim=2, hidden_dim=16, action_dim=4)
q_optimizer = optim.Adam(q_network.parameters(), lr=1e-4)


num_episodes = 100
episode_idx = 0
with tqdm(total=num_episodes, desc=f"Episode {episode_idx+1}/{num_episodes}") as pbar:
    for episode_idx in range(num_episodes):
        # reset episode
        env.reset()
        done = False

        # Generate an episode
        trajectory = []
        q_loss_list = []
        policy_loss_list = []
 
        i = 0
        while (done is not True):
            # select action from policy
            cur_state_tensor = torch.tensor(env.cur_state).type(torch.float32)
            with torch.no_grad():
                action = policy_network.explore_action(cur_state_tensor)
                # action = policy_network.explore_action(cur_state_tensor, possible_actions=env.get_possible_actions())
            
            from_state, next_state, reward, done = env.step(action)
            with torch.no_grad():
                next_state_tensor = torch.tensor(next_state).type(torch.float32)
                next_action = policy_network.explore_action(next_state_tensor)
                next_q_value = q_network(next_state_tensor)
                td_target = reward + gamma * (next_q_value[next_action] if not done else torch.tensor(0).type(torch.float32)) 

            # q_network
            q_optimizer.zero_grad()
            cur_q_value = q_network(cur_state_tensor)
            q_loss = nn.functional.mse_loss(cur_q_value[action], td_target)
            q_loss.backward()
            q_optimizer.step()
            
            # Update policy network
            policy_optimizer.zero_grad()
            action_outputs = policy_network(cur_state_tensor)
            action_dist = Categorical(logits = action_outputs)
            prob = action_dist.probs[action]
            policy_loss = -torch.log(prob + 1e-10) * cur_q_value[action].detach()
            policy_loss.backward()
            policy_optimizer.step()

            prob

            # save history
            trajectory.append([str(from_state), action, reward])
            q_loss_list.append(q_loss.item())
            policy_loss_list.append(policy_loss.item())

            i += 1
            if i >=50:
                break

        if episode_idx % 1 == 0:
            pbar.set_postfix(Q_loss=np.mean(q_loss_list), Policy_loss=np.mean(policy_loss_list), Num_steps=len(trajectory))
        pbar.update(1)



# Simulation Test ---------------------------------------------------------------------------------
env.reset()
# env.render()
i = 0
done = False
while (done is not True):
    
    with torch.no_grad():
        cur_state_tensor = torch.tensor(env.cur_state).type(torch.float32)
        action = policy_network.greedy_action(cur_state_tensor)
        # action_outputs = policy_network(cur_state_tensor)
        # action_prob = torch.softmax(action_outputs, dim=-1)
        # action_possible_prob = torch.full_like(action_prob, float('-inf'))
        # action_possible_prob[..., env.get_possible_actions()] = action_prob[..., env.get_possible_actions()]
        # action = torch.argmax(action_possible_prob, dim=-1).item()

        from_state, next_state, reward, done = env.step(action)
        env.render()
        time.sleep(0.2)
        clear_output(wait=True)
    i += 1
    if i >=30:
        break
################################################################################################################




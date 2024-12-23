
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

# policy-network 정의 (Actor Network)
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

# # Q-network 정의 (Critic Network)
# class QNetwork(nn.Module):
#     def __init__(self, state_dim, hidden_dim, action_dim):
#         super().__init__()
#         self.fc_block = nn.Sequential(
#             nn.Linear(state_dim, hidden_dim)
#             ,nn.ReLU()
#             ,nn.Linear(hidden_dim, hidden_dim)
#             ,nn.ReLU()
#             ,nn.Linear(hidden_dim, action_dim)
#         )

#     def forward(self, x):
#         return self.fc_block(x)

# (BaseLine) state를 input으로 입력받아 그 state의 value function(BaseLine)을 계산하는 함수 (Critic Network)
class ValueEstimator(nn.Module):
    def __init__(self, state_dim, hidden_dim, output_dim=1):
        super(ValueEstimator, self).__init__()
        self.linear1 = torch.nn.Linear(state_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim)
        self.act = torch.nn.ReLU()        
        
    def forward(self, x):
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        x = self.linear3(x)
        return x
        
    def estimate(self, state):
        state = torch.FloatTensor(state)
        return self.forward(state)



################################################################################################################

# (Step 단위 Update) --------------------------------------------------------------------------------------------

# env = CustomGridWorld()
# env = CustomGridWorld(grid_size=5)
# env = CustomGridWorld(grid_size=5, traps=[], obstacles=[], treasures=[])
# env = CustomGridWorld(grid_size=5, obstacles=obstacles)
# env = CustomGridWorld(grid_size=5, obstacles=obstacles, traps=traps)
# env = CustomGridWorld(grid_size=5, obstacles=obstacles, traps=traps, treasures=treasures)
env = CustomGridWorld(grid_size=4, reward_step=-1)

env.reset()
env.render()

gamma = 0.9

policy_network = PolicyNetwork(state_dim=2, hidden_dim=16, action_dim=4)
policy_optimizer = optim.Adam(policy_network.parameters(), lr=1e-4)

value_network = ValueEstimator(state_dim=2, hidden_dim=16, output_dim=1)
value_optimizer = optim.Adam(value_network.parameters(), lr=1e-4)


num_episodes = 300
episode_idx = 0
with tqdm(total=num_episodes, desc=f"Episode {episode_idx+1}/{num_episodes}") as pbar:
    for episode_idx in range(num_episodes):
        # reset episode
        env.reset()
        done = False

        # Generate an episode
        trajectory = []
        value_loss_list = []
        policy_loss_list = []
 
        i = 0
        while (done is not True):
            # select action from policy
            cur_state_tensor = torch.tensor(env.cur_state).type(torch.float32)
            
            action_probs = policy_network.predict_prob(cur_state_tensor)
            action = torch.multinomial(action_probs, 1).item()

            from_state, next_state, reward, done = env.step(action)

            # estimate value
            cur_value = value_network(cur_state_tensor)
            next_state_tensor = torch.tensor(next_state).type(torch.float32)
            next_value = value_network(next_state_tensor) if not done else torch.tensor(0).type(torch.float32)

            # Compute Advantage and Targets
            td_target = reward + gamma * next_value.detach()
            td_error = td_target - cur_value
   
            # policy loss
            log_prob = torch.log(action_probs[action] + 1e-10) 
            policy_loss = -log_prob * td_target

            # value update
            value_loss = nn.functional.mse_loss(cur_value, td_target)
            # value_loss = td_error.pow(2)
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

            # policy loss update
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()


            # save history
            trajectory.append([str(from_state), action, reward])
            value_loss_list.append(value_loss.item())
            policy_loss_list.append(policy_loss.item())

            i += 1
            if i >=50:
                break

        if episode_idx % 1 == 0:
            pbar.set_postfix(Value_loss=np.mean(value_loss_list), Policy_loss=np.mean(policy_loss_list), Num_steps=len(trajectory))
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

















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
# ( REINFORCE Monte Carlo ) ####################################################################################


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
        

################################################################################################################

# (Step 단위 Update) --------------------------------------------------------------------------------------------

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

policy_network = PolicyNetwork(state_dim=2, hidden_dim=32, action_dim=4)
# policy_network(torch.tensor(env.cur_state).type(torch.float32))
# policy_network.predict_prob(torch.tensor(env.cur_state).type(torch.float32))
# policy_network.predict_action(torch.tensor(env.cur_state).type(torch.float32))
# policy_network.predict_action(torch.tensor(env.cur_state).type(torch.float32), possible_actions=env.get_possible_actions())

optimizer = optim.Adam(policy_network.parameters(), lr=1e-3)


num_episodes = 100
episode_idx = 0
with tqdm(total=num_episodes, desc=f"Episode {episode_idx+1}/{num_episodes}") as pbar:
    for episode_idx in range(num_episodes):
        # reset episode
        env.reset()
        done = False

        # Generate an episode
        trajectory = []
        i = 0
        while (done is not True):
            # select action from policy
            cur_state_tensor = torch.tensor(env.cur_state).type(torch.float32)
            # action = policy_network.explore_action(cur_state_tensor, possible_actions=env.get_possible_actions())
            action = policy_network.explore_action(cur_state_tensor)
            # action_outputs = policy_network(cur_state_tensor)
            # action_prob = torch.softmax(action_outputs, dim=-1)
            # action_possible_prob = torch.full_like(action_prob, 0)
            # action_possible_prob[..., env.get_possible_actions()] = action_prob[..., env.get_possible_actions()]
            # action_dist = Categorical(logits=action_possible_prob)
            # action = action_dist.sample().item()
            
            from_state, next_state, reward, done = env.step(action)

            # display
            # env.render()
            # time.sleep(0.2)
            # clear_output(wait=True)

            # save trajectory
            trajectory.append([str(from_state), action, reward])

            i += 1
            if i >=50:
                break

        # Compute returns
        G = 0
        policy_loss_list = []
        for ei, (state_str, action ,reward) in enumerate(reversed(trajectory)):
            G = reward + gamma * G
            
            state_tensor = torch.tensor(eval(state_str)).type(torch.float32)
            action_outputs = policy_network(state_tensor)
            action_dist = Categorical(logits = action_outputs)
            prob = action_dist.probs[action]

            # Update policy network
            optimizer.zero_grad()
            policy_loss = -torch.log(prob + 1e-10) * G
            policy_loss.backward()
            optimizer.step()

            policy_loss_list.append(policy_loss.item())

        if episode_idx % 1 == 0:
            pbar.set_postfix(Loss=f"{np.mean(policy_loss_list):.3f}", Num_steps=len(trajectory))
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



































################################################################################################################
# (Episode 단위 Update) --------------------------------------------------------------------------------------------

# (BaseLine) state를 input으로 입력받아 그 state의 value function(BaseLine)을 계산하는 함수
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


policy_network = PolicyNetwork(state_dim=2, hidden_dim=32, action_dim=4)
# policy_network(torch.tensor(env.cur_state).type(torch.float32))
# policy_network.predict_prob(torch.tensor(env.cur_state).type(torch.float32))
# policy_network.predict_action(torch.tensor(env.cur_state).type(torch.float32))
# policy_network.predict_action(torch.tensor(env.cur_state).type(torch.float32), possible_actions=env.get_possible_actions())

policy_optimizer = optim.Adam(policy_network.parameters(), lr=1e-3)

value_network = ValueEstimator(state_dim=2, hidden_dim=32, output_dim=1)
value_optimizer = optim.Adam(value_network.parameters(), lr=1e-3)


num_episodes = 100
episode_idx = 0
with tqdm(total=num_episodes, desc=f"Episode {episode_idx+1}/{num_episodes}") as pbar:
    for episode_idx in range(num_episodes):
        # reset episode
        env.reset()
        done = False

        # Generate an episode
        trajectory = []
        i = 0
        while (done is not True):
            # select action from policy
            cur_state_tensor = torch.tensor(env.cur_state).type(torch.float32)
            # action = policy_network.explore_action(cur_state_tensor, possible_actions=env.get_possible_actions())
            action = policy_network.explore_action(cur_state_tensor)
            # action_outputs = policy_network(cur_state_tensor)
            # action_prob = torch.softmax(action_outputs, dim=-1)
            # action_possible_prob = torch.full_like(action_prob, 0)
            # action_possible_prob[..., env.get_possible_actions()] = action_prob[..., env.get_possible_actions()]
            # action_dist = Categorical(logits=action_possible_prob)
            # action = action_dist.sample().item()
            
            from_state, next_state, reward, done = env.step(action)

            # display
            # env.render()
            # time.sleep(0.2)
            # clear_output(wait=True)

            # save trajectory
            trajectory.append([str(from_state), action, reward])

            i += 1
            if i >=50:
                break

        # Compute returns
        trajectory_returns = []
        G = 0
        for ei, (state_str, action ,reward) in enumerate(reversed(trajectory)):
            G = reward + gamma * G
            trajectory_returns.insert(0, G)


        # # Normalize returns for stability
        # normalized_returns = torch.tensor( (np.array(trajectory_returns) - np.mean(trajectory_returns)) / (np.std(trajectory_returns) + 1e-6) ).type(torch.float32)

        returns = torch.tensor(trajectory_returns).type(torch.float32)
        # returns = normalized_returns
        
        # Compute policy loss
        policy_loss_list = []
        state_tensors = []
        baseline_values_list = []
        for (state_str, action, reward), G_t in zip(trajectory, returns):
            state_tensor = torch.tensor(eval(state_str)).type(torch.float32)
            action_outputs = policy_network(state_tensor)
            action_prob = torch.softmax(action_outputs, dim=-1)
            # action_prob = torch.clamp(action_prob, min=1e-10)
            log_action_prob = torch.log(action_prob[..., action] + 1e-6) 
            
            baseline_value = value_network(state_tensor)
            advantage = G_t - baseline_value.detach().item()   # baseline trick

            policy_loss_list.append( (-log_action_prob * advantage).unsqueeze(0) )
            state_tensors.append(state_tensor)
            baseline_values_list.append(baseline_value)
        policy_loss = torch.cat(policy_loss_list).sum()
        
        # Update policy network
        policy_optimizer.zero_grad()
        policy_loss.backward()
        # torch.nn.utils.clip_grad_norm_(policy_network.parameters(), max_norm=5)
        policy_optimizer.step()

        value_optimizer.zero_grad()
        baseline_value_loss = nn.functional.mse_loss(torch.cat(baseline_values_list), returns)
        baseline_value_loss.backward()
        value_optimizer.step()

        if episode_idx % 1 == 0:
            pbar.set_postfix(Policy_loss=f"{policy_loss.item():.3f}", 
                            BaseLine_loss=f"{baseline_value_loss.item():.3f}",
                            Num_steps=len(trajectory))
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
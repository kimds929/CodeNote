import os
import sys
sys.path.append(r'D:\DataScience\★Git_CodeNote\85_Reinforcement_Learning')

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
    from Environments.RL00_Env01_CustomGridWorld import CustomGridWorld
except:
    import httpimport
    remote_url = 'https://raw.githubusercontent.com/kimds929/'

    with httpimport.remote_repo(f"{remote_url}/CodeNote/main/85_Reinforcement_Learning/"):
        from Environments.RL00_Env01_CustomGridWorld import CustomGridWorld


try:
    from Environments.RL00_Env02_CartPole import CustomCartPole
except:
    import httpimport
    remote_url = 'https://raw.githubusercontent.com/kimds929/'

    with httpimport.remote_repo(f"{remote_url}/CodeNote/main/85_Reinforcement_Learning/"):
        from Environments.RL00_Env02_CartPole import CustomCartPole




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
policy_optimizer = optim.AdamW(policy_network.parameters(), lr=1e-4)

value_network = ValueEstimator(state_dim=2, hidden_dim=16, output_dim=1)
value_optimizer = optim.AdamW(value_network.parameters(), lr=5e-4)


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
            advantage = td_target - cur_value
   
            # policy loss
            log_prob = torch.log(action_probs[action] + 1e-10) 
            
            # policy_loss = -log_prob * td_target           # 기존 actor-critic
            policy_loss = -log_prob * advantage.detach()    # Advantage function을 이용한 actor-critic

            # value update
            value_loss = nn.functional.mse_loss(cur_value, td_target)
            # value_loss = advantage.pow(2)
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














































################################################################################################
################################################################################################
################################################################################################



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
import gymnasium as gym


try:
    from Environments.RL00_Env01_FrozenLake_v1 import generate_frozenlake_map
    from utils.DS_RL import ReplayMemory
    custom_map = generate_frozenlake_map(5,5, hole=0.15)
except:
    custom_map=None










################################################################################################################
################################################################################################################
env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode="rgb_array")  # deterministic

from stable_baselines3 import A2C
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000)


# model.predict(obs, deterministic=True)

# Simulation Test ---------------------------------------------------------------------------------
obs, info = env.reset()
# env.render()
i = 0
done = False
while (done is not True):
    action, _ = model.predict(obs, deterministic=True)
    action = action.item()  
    
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

























################################################################################################################
################################################################################################################
# [ Off-policy Actor Critic ]###################################################################################
class Actor(nn.Module):
    def __init__(self, action_dim, hidden_dim, max_states=100, embed_dim=1):
        super().__init__()
        self.state_embedding = nn.Embedding(num_embeddings=max_states, embedding_dim=embed_dim)
        self.policy_network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, action_dim)
        )
    
    def execute_model(self, obs, actions=None, temperature=None):
        embed_x = self.state_embedding(obs).squeeze(-2)
        logits = self.policy_network(embed_x)
        action_dist = Categorical(logits=logits)
        entropy = action_dist.entropy()
        
        if actions is None:
            if temperature is None:
                action = torch.argmax(logits, dim=-1)
            else:
                explore_dist = Categorical(logits=logits/temperature)
                action = explore_dist.sample()
            log_prob = action_dist.log_prob(action)
            return action, log_prob, entropy
        
        else:
            log_prob = action_dist.log_prob(actions)
            return log_prob, entropy
    
    def forward(self, obs, temperature=None):
        action, log_prob, entropy = self.execute_model(obs, temperature=temperature)
        return action, log_prob, entropy
    
    def evaluate_actions(self, obs, actions, temperature=None):
        log_prob, entropy = self.execute_model(obs, actions=actions, temperature=temperature)
        return log_prob, entropy
    
    def predict(self, obs, temperature=None):
        action, log_prob, entropy = self.execute_model(obs, temperature=temperature)
        return action
        
# Q-network 정의(StateValue)
class Critic(nn.Module):
    def __init__(self, hidden_dim, max_states=100, embed_dim=1):
        super().__init__()
        self.state_embedding = nn.Embedding(num_embeddings=max_states, embedding_dim=embed_dim)
        
        self.value_network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs):
        embed_obs = self.state_embedding(obs).squeeze(-2)
        value = self.value_network(embed_obs)
        return value
##############################################################################################################


env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode="rgb_array")  # deterministic

np.array(dir(env.unwrapped))

# env.unwrapped.desc.ravel()[19] == b'H'

obs, info = env.reset()
plt.imshow(env.render())
# env.observation_space.n

memory_size = 1024
batch_size = 64
sample_only_until = 500
n_epochs = 1
target_update_interval = 100

gamma = 0.9

# Actor Network
actor_network = Actor(action_dim=env.action_space.n, hidden_dim=32,
                       max_states=env.observation_space.n, embed_dim=2).to(device)
actor_optimizer = optim.AdamW(actor_network.parameters(), lr=1e-4, weight_decay=1e-2)

# main network
main_critic_network = Critic(hidden_dim=32,
                       max_states=env.observation_space.n, embed_dim=2).to(device)
critic_optimizer = optim.AdamW(main_critic_network.parameters(), lr=5e-4, weight_decay=1e-2)

# target network
target_critic_network = Critic(hidden_dim=32,
                       max_states=env.observation_space.n, embed_dim=2).to(device)
target_critic_network.load_state_dict(main_critic_network.state_dict())

# ★ (Prioritize Replay Buffer)
memory = ReplayMemory(max_size=memory_size, batch_size=batch_size, method='random')
# memory = ReplayMemory(max_size=memory_size, batch_size=batch_size, method='priority')

# memory.batch_size
# memory.sample_pointer
# memory.size
# memory.index
# len(memory.shuffled_indices)
# memory.shuffled_indices[memory.sample_pointer:memory.sample_pointer+memory.batch_size]

# loss_function = nn.MSELoss()
critic_loss_function = nn.SmoothL1Loss(reduction='none')

num_episodes = 300
total_step = 0

with tqdm(total=num_episodes) as pbar:
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        i = 0
        
        T = 1
        # T = np.logspace(2, 0, num=num_episodes)[episode]
        # T = max(0.5, 5 * (0.97 ** episode))
        cumulative_reward = 0
        while(not done):
            obs_tensor = torch.LongTensor([obs]).to(device)
            action, log_prob, entropy  = actor_network(obs_tensor, temperature=T)
            action = action.item()
            log_prob = log_prob.item()
            
            value = main_critic_network(obs_tensor)
            value = value.item()
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # if env.unwrapped.desc.ravel()[next_obs] == b'H':
            #     reward -= 1
            # elif env.unwrapped.desc.ravel()[next_obs] == b'G':
            #     reward += 100
            # elif next_obs == obs:
            #     reward -= 1
            # else:
            reward -= 0.01  # step-penalty
            done = terminated or truncated
            
            experience = (obs, action, log_prob, next_obs, reward, done, value)
            
            
            # buffer에 experience 저장
            memory.push(experience)

            obs = next_obs
            i += 1
            if i >=100:
                break
            cumulative_reward += reward

            avg_critic_loss = 0
            avg_actor_loss = 0
            if len(memory) >= sample_only_until:
                for epoch in range(n_epochs):
                    # for batch in memory:
                    sampled_exps, indices, weights = memory.sample()
                    # print(epoch, len(sampled_exps), memory.size, memory._iter_sample_pointer)

                    weights = torch.FloatTensor(weights).to(device)
                    batch_obs = torch.LongTensor(np.stack([sample[0] for sample in sampled_exps])).view(-1,1).to(device)
                    batch_actions = torch.LongTensor(np.stack([sample[1] for sample in sampled_exps])).view(-1,1).to(device)
                    batch_log_probs = torch.FloatTensor(np.stack([sample[2] for sample in sampled_exps])).view(-1,1).to(device)
                    batch_next_obs = torch.LongTensor(np.stack([sample[3] for sample in sampled_exps])).view(-1,1).to(device)
                    batch_rewards = torch.FloatTensor(np.stack([sample[4] for sample in sampled_exps])).view(-1,1).to(device)
                    batch_dones = torch.FloatTensor(np.stack([sample[5] for sample in sampled_exps])).view(-1,1).to(device)
                    batch_values = torch.FloatTensor(np.stack([sample[6] for sample in sampled_exps])).view(-1,1).to(device)
                    
                    # compute actor
                    log_prob, entropy = actor_network.evaluate_actions(batch_obs, batch_actions, temperature=T)
                    
                    # compute critic
                    value = main_critic_network(batch_obs)
                    next_value = target_critic_network(batch_next_obs)
                    
                    # target
                    td_target = batch_rewards + gamma * next_value.detach() * (1-batch_dones)
                    
                    # critic_loss
                    critic_loss_unreduced = critic_loss_function(value, td_target)
                    critic_loss = (critic_loss_unreduced * weights).mean()
                    
                    # actor_loss
                    advantage = (td_target - value).detach()    # advantage = td_error
                    if len(advantage) > 1:
                        advantage = (advantage - advantage.mean()) / (advantage.std()+1e-8)
                    # actor_loss = -(log_prob * advantage + 0.2 * entropy).mean()
                    actor_loss = -(log_prob * advantage).mean()
                    
                    # critic update
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_optimizer.step()
                    
                    # actor update
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # with torch.no_grad():
                    #     td_errors = critic_loss_unreduced.detach().cpu().numpy().reshape(-1)
                    #     memory.update_priorities(indices, td_errors)
                    
                    avg_critic_loss += critic_loss.to('cpu')
                    avg_actor_loss += actor_loss.to('cpu')
                    # break

            if total_step % target_update_interval == 0:
                target_critic_network.load_state_dict(main_critic_network.state_dict())
                # print('target_network update')

            total_step += 1
            
        if episode % 1 == 0:
            pbar.set_postfix(critic_loss=f"{avg_critic_loss/(n_epochs):.3f}", 
                            actor_loss=f"{avg_actor_loss/(n_epochs):.3f}",
                            Len_episodes=f"{i}",
                            total_reward = f"{cumulative_reward:.2f}"
                            )
        pbar.update(1)


# Simulation Test ---------------------------------------------------------------------------------
obs, info = env.reset()
# env.render()
i = 0
done = False
while (done is not True):
    
    with torch.no_grad():
        actor_network.eval()
        action, _, _ = actor_network(torch.LongTensor([obs]).to(device))
        action = action.item()  
        
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


















































################################################################################################################
################################################################################################################
# [ OnPolicy ActorCritic GAE ] #########################################################################################
import gymnasium as gym

try:
    from Environments.RL00_Env01_FrozenLake_v1 import generate_frozenlake_map
    # from utils.DS_RL import ReplayMemory
    custom_map = generate_frozenlake_map(5,5, hole=0.15)
except:
    custom_map=None

class Actor(nn.Module):
    def __init__(self, action_dim, hidden_dim, max_states=100, embed_dim=1):
        super().__init__()
        self.state_embedding = nn.Embedding(num_embeddings=max_states, embedding_dim=embed_dim)
        self.policy_network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, action_dim)
        )
    
    def execute_model(self, obs, actions=None, temperature=None):
        embed_x = self.state_embedding(obs).squeeze(-2)
        logits = self.policy_network(embed_x)
        action_dist = Categorical(logits=logits)
        entropy = action_dist.entropy()
        
        if actions is None:
            if temperature is None:
                action = torch.argmax(logits, dim=-1)
            else:
                explore_dist = Categorical(logits=logits/temperature)
                action = explore_dist.sample()
            log_prob = action_dist.log_prob(action)
            return action, log_prob, entropy
        
        else:
            log_prob = action_dist.log_prob(actions)
            return log_prob, entropy
    
    def forward(self, obs, temperature=None):
        action, log_prob, entropy = self.execute_model(obs, temperature=temperature)
        return action, log_prob, entropy
    
    def evaluate_actions(self, obs, actions, temperature=None):
        log_prob, entropy = self.execute_model(obs, actions=actions, temperature=temperature)
        return log_prob, entropy
    
    def predict(self, obs, temperature=None):
        action, log_prob, entropy = self.execute_model(obs, temperature=temperature)
        return action
        
# Q-network 정의(StateValue)
class Critic(nn.Module):
    def __init__(self, hidden_dim, max_states=100, embed_dim=1):
        super().__init__()
        self.state_embedding = nn.Embedding(num_embeddings=max_states, embedding_dim=embed_dim)
        
        self.value_network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs):
        embed_obs = self.state_embedding(obs).squeeze(-2)
        value = self.value_network(embed_obs)
        return value
##############################################################################################################


# compute_gae
def compute_gae(rewards, values, next_values, gamma=0.99, lmbda=0.95):
    """
    ReplayMemory에 저장된 sequential rollout에서 GAE 계산

    Args:
        rewards:      list or array of shape (N,)    ← [r0, r1, ..., r_{N-1}]
        values:       list or array of shape (N,)    ← [v0, v1, ..., v_{N-1}]
        next_value: : list or array of shape (N,)    ← [None, ...]  (terminated or truncated시 next_value )
        gamma:      할인률 (default=0.99)
        lmbda:      GAE λ 파라미터 (default=0.95)

    Returns:
        advantages: numpy.ndarray of shape (N,)   ← GAE(λ) 기반 advantage
        returns:    numpy.ndarray of shape (N,)   ← advantage + values
    """
    N = len(rewards)
    advantages = np.zeros(N, dtype=np.float32)
    returns = np.zeros(N, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(N)):
        if next_values[t] is None:
            next_value = values[t + 1]
        else:       # terminated or truncated 
            next_value = next_values[t]
            gae = 0.0

        # δ_t = r_t + γ * V(s_{t+1}) * mask - V(s_t)
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lmbda * gae
        advantages[t] = gae

    # return = advantage + V(s_t)
    for i in range(N):
        returns[i] = advantages[i] + values[i]

    return advantages, returns
################################################################################################

env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode="rgb_array")  # deterministic
obs, info = env.reset()
plt.imshow(env.render())
# env.observation_space.n

memory_size = 1024
batch_size = 64
sample_only_until = 500
n_epochs = 3
gamma = 0.9
gae_gamma = 0.99
gae_lambda = 0.95


# Actor Network
actor_network = Actor(action_dim=env.action_space.n, hidden_dim=32,
                       max_states=env.observation_space.n, embed_dim=2).to(device)
actor_optimizer = optim.AdamW(actor_network.parameters(), lr=1e-4, weight_decay=1e-2)


# main network
critic_network = Critic(hidden_dim=32,
                       max_states=env.observation_space.n, embed_dim=2).to(device)
critic_optimizer = optim.AdamW(critic_network.parameters(), lr=5e-4, weight_decay=1e-2)
critic_loss_function = nn.SmoothL1Loss(reduction='none')


# memory
memory = ReplayMemory(max_size=memory_size, method='sequential')


truncated_step = 50
N_ITER = 10
TOTAL_TIMESTEPS = 1000        # Rollout Timstep
N_REPLAY_STEPS = 250       # Rollout 횟수
N_EPOCHS = 3               # Model Learning 횟수


for iter in range(N_ITER):
    print(f"\r({iter+1}/{N_ITER} ITER) ", end='')
    replay_time_step = 0
    
    total_loop = (TOTAL_TIMESTEPS-1)//N_REPLAY_STEPS+1
    loop_count = 1
    
    while (replay_time_step < TOTAL_TIMESTEPS):
        # (Collect Rollout) ##################################################################################
        memory.reset()
        obs, info = env.reset()
        len_episodes = []
        episode_step = 0
        
        
        # (RollOut) ------------------------------------------------------------------------------------------------
        for _ in range(N_REPLAY_STEPS):
            obs_tensor = torch.LongTensor([obs]).to(device)
            # T = np.logspace(2, 0, num=num_episodes)[episode]
            T = 1
            action, log_prob, entropy  = actor_network(obs_tensor, temperature=T)
            log_prob = log_prob.detach().cpu().item()
            action = action.detach().cpu().item()
            
            value = critic_network(obs_tensor)
            value = value.detach().cpu().item()
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            if (episode_step >= truncated_step):
                truncated = True
            
            # if env.unwrapped.desc.ravel()[next_obs] == b'H':
            #     reward -= 1
            # elif env.unwrapped.desc.ravel()[next_obs] == b'G':
            #     reward += 100
            # elif next_obs == obs:
            #     reward -= 1
            # else:
            reward -= 0.01  # step-penalty
            done = terminated or truncated
            
            if terminated:
                next_value = 0.0
            elif truncated:
                next_value = critic_network(torch.LongTensor([next_obs]).to(device)).detach().cpu().item()
            else:
                next_value = None
            
            # buffer
            
            experience = (obs, action, log_prob, next_obs, reward, value, next_value)
            memory.push(experience)
            
            if done:
                obs, info = env.reset()
                len_episodes.append(episode_step)
                episode_step = 0
            else:
                obs = next_obs
                episode_step += 1
            replay_time_step += 1
        # -------------------------------------------------------------------------------------------------------
        
        
        batch, indices, weights = memory.sample()
        obss, actions, log_probs, next_obss, rewards, values, next_values = (np.array(batch, dtype='object').T).tolist()
        
        # 마지막 스텝에서의 next_value 확보 (부트스트랩용)
        with torch.no_grad():
            next_values[-1] = critic_network(torch.LongTensor([next_obs]).to(device)).item()
            
        # (compute GAE)
        advantages, returns = compute_gae(rewards, values, next_values, gamma=gae_gamma, lmbda=gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # dataset & dataloader
        obss_tensor = torch.LongTensor(obss).view(-1,1).to(device)
        actions_tensor = torch.LongTensor(actions).view(-1,1).to(device)
        log_probs_tensor = torch.FloatTensor(log_probs).view(-1,1).to(device)
        next_obss_tensor = torch.LongTensor(next_obss).view(-1,1).to(device)
        values_tensor = torch.FloatTensor(values).view(-1,1).to(device)
        advantages_tensor = torch.FloatTensor(advantages).view(-1,1).to(device)
        returns_tensor = torch.FloatTensor(returns).view(-1,1).to(device)
        
        dataset = TensorDataset(obss_tensor, actions_tensor, log_probs_tensor, values_tensor, advantages_tensor, returns_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        
        
        
        # ---------------------------------------------------------------------------------------------------------
        cumulative_critic_loss = 0
        cumulative_actor_loss = 0
        for epoch in range(N_EPOCHS):
            for batch_data in data_loader:
                obss, actions, old_log_probs, values, advantages, returns = batch_data
                
                # compute_actor
                # log_probs, entropy = actor_network.evaluate_actions(obss, actions, temperature=T)
                log_probs, entropy = actor_network.evaluate_actions(obss, actions.squeeze(-1), temperature=T)
                
                # compute_critic
                values_pred = critic_network(obss)
                # next_value = critic_network(next_obss)
                
                # critic_loss
                critic_loss = nn.functional.smooth_l1_loss(values_pred, returns)
                
                # actor_loss
                actor_loss = -(log_probs * advantages).mean()
                # actor_loss = -(log_probs * advantages + 0.01 * entropy).mean()
                
                # critic update
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                
                # actor update
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                
                # verbose
                cumulative_critic_loss += critic_loss.to('cpu') * len(obss)
                cumulative_actor_loss += actor_loss.to('cpu') * len(obss)

            print(f"\r[{iter+1}/{N_ITER}ITER](Loop {loop_count}/{total_loop}) Rollout: {replay_time_step+1}/{TOTAL_TIMESTEPS}, \Episode_len: {np.mean(len_episodes):.1f} Epoch: {epoch+1}/{N_EPOCHS} (CriticLoss: {cumulative_critic_loss/(N_EPOCHS * len(dataset)):.3f}, ActorLoss: {cumulative_actor_loss/(N_EPOCHS * len(dataset)):.3f})", end='')

        loop_count += 1



# Simulation Test ---------------------------------------------------------------------------------
obs, info = env.reset()
# env.render()
i = 0
done = False
while (done is not True):
    
    with torch.no_grad():
        actor_network.eval()
        action, _, _ = actor_network(torch.LongTensor([obs]).to(device))
        action = action.item()  
        
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
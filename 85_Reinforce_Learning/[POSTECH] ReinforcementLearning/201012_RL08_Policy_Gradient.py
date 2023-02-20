import gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, MultivariateNormal


def print_history(history):
    plt.figure(figsize=(12,4))
    plt.plot(history)
    plt.title('episode reward over iteration')
    plt.show()


env = gym.make('CartPole-v0')
env.reset()
# env.render()
env.observation_space
env.action_space

# ## 0. PyTorch 라이브러리를 이용하여 function approximator를 만들어 줍니다.
# ### Softmax policy with neural network approximator를 구현해봅시다!


# discrete action 환경에 사용할 수 있는 softmax policy입니다
# state를 input으로 받아서 각 action에 대한 확률을 output으로 계산합니다
class PolicyEstimator(nn.Module):
    def __init__(self, nS, nA, hidden_dim):
        super(PolicyEstimator, self).__init__()
        ## TODO
        self.layer1 = torch.nn.Linear(nS, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = torch.nn.Linear(hidden_dim, nA)
        
    def forward(self, x):
        # TODO
        # 현재 state에서 각 action을 선택할 확률을 return합니다.
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = torch.nn.functional.softmax(self.layer3(x), dim=-1)
        return x
        
    def select_action(self, state):
        # 현재 state에서 action 하나를 선택합니다.
        # TODO
        state = torch.FloatTensor(state)
        probs = self.forward(state)
        
        dist = torch.distributions.Categorical(probs)       # action에 대한 확률로 선택
        action = dist.sample()                              # 확률분포에 따라 sampling

        # 해당 action에 대한 확률의 log값도 함께 return해줍니다
        # Loss = -G_t · log π(A_t | S_t, θ)
        return action.item(), dist.log_prob(action)

env = gym.make('CartPole-v0')
nS = env.observation_space.shape[0]
nA = env.action_space.n

policy = PolicyEstimator(nS, nA, hidden_dim=16)
state = env.reset()

policy.select_action(state)
# prob = policy(torch.FloatTensor(state))
# dist = Categorical(prob)
# dist.sample()
# dist.log_prob(dist.sample())





# ## 1. REINFORCE Algorithm
def reinforce(env, 
              policy_estimator, 
              policy_lr=1e-3, 
              num_episodes=1000, 
              gamma=.99, 
              log_interval=50):
    
    policy_optimizer = optim.Adam(policy_estimator.parameters(), lr=policy_lr)
    
    history = []
    for i_episode in range(1, num_episodes+1):
        # episode 초기화
        state = env.reset()
        
        ep_reward = 0
        rewards = []
        log_probs = []

        # TODO
        # 현재 policy에 따라 episode를 생성합니다.
        # 학습에 사용하기 위해 reward와 각 action의 log probability를 저장합니다.
        while True: 
            action, log_prob = policy_estimator.select_action(state)
            next_state, reward, done, _ = env.step(action)

            if done:
                break
            state = next_state
            ep_reward += reward
            rewards.append(ep_reward)
            log_probs.append(log_prob)

        # TODO
        # episode의 각 step에 해당하는 cummulative return G_t를 계산합니다.
        returns = []
        G = 0        

        for r in rewards[::-1]:
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        
        # TODO
        # REINFORCE 알고리즘의 update 식을 참고하여 loss function을 정의해줍니다.
        log_probs = torch.stack(log_probs)      # 여러개의 tensor를 하나로 묶어줌
        policy_loss = (-log_probs * returns).sum()

        # parameter update를 합니다
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        # 학습 진행 상황을 history list에 저장하고 출력합니다
        # 마지막 50 episodes의 평균 reward를 저장하여 문제가 해결되었는지 판단합니다.
        history.append(ep_reward)
        avg_reward = sum(history[-50:])/len(history[-50:])
        
        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, avg_reward))
            
        if avg_reward > env.spec.reward_threshold:
            print("Solved! Average reward: {:.2f}".format(avg_reward))
            break
            
    return history



# ### CartPole-v0 환경에 적용해봅시다!
env = gym.make('CartPole-v0')
print('state space:', env.observation_space)
print('action space:', env.action_space)

nS = env.observation_space.shape[0]
nA = env.action_space.n

policy_estimator = PolicyEstimator(nS, nA, hidden_dim=16)

history = reinforce(env, 
                    policy_estimator, 
                    num_episodes=2000)

print_history(history)




# cart problem display
import cv2

def renderCart(state):
    cart_pos, _, pole_ang, _ = state
    pos_range = (-5, 5)
    ang_range = (-4.2, 4.2)
    
    size = 5
    screen = 255*np.ones([size*40, size*40], dtype=np.uint8)
    cv2.line(screen, (0, size*30), (size*40, size*30), 0, size//2)
    
    cart_pos = (int(np.interp(cart_pos, pos_range, (0, size*40))), size*30)
    cv2.rectangle(screen, (cart_pos[0]-size*3, cart_pos[1]-size*2), 
                  (cart_pos[0]+size*3, cart_pos[1]+size*2), 255, -1)
    cv2.rectangle(screen, (cart_pos[0]-size*3, cart_pos[1]-size*2), 
                  (cart_pos[0]+size*3, cart_pos[1]+size*2), 0, size//5, cv2.LINE_AA)
    
    pole_len = size*14
    pole_tip = (int(cart_pos[0]+pole_len*math.cos(pole_ang-math.pi/2)), 
                int(cart_pos[1]+pole_len*math.sin(pole_ang-math.pi/2)))
    
    cv2.line(screen, cart_pos, pole_tip, 0, size, cv2.LINE_AA)
    
    return screen


# cart-pole exec by learned Q_network_torch
from IPython import display
import math
state = env.reset()

img = plt.imshow(renderCart(state), cmap='gray')
while True:
    img.set_data(renderCart(state))
    display.display(plt.gcf())
    display.clear_output(wait=True)
    
    action, log_prob = policy_estimator.select_action(state)
    
    next_state, _, done, _ = env.step(action)
    state = next_state
    if done: 
        break
env.close()



























# ==================================================================================
import gym
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, MultivariateNormal


def print_history(history):
    plt.figure(figsize=(12,4))
    plt.plot(history)
    plt.title('episode reward over iteration')
    plt.show()


env = gym.make('CartPole-v0')
env.reset()
# env.render()
env.observation_space
env.action_space

# ## 0. PyTorch 라이브러리를 이용하여 function approximator를 만들어 줍니다.
# ### Softmax policy with neural network approximator를 구현해봅시다!


# discrete action 환경에 사용할 수 있는 softmax policy입니다
# state를 input으로 받아서 각 action에 대한 확률을 output으로 계산합니다
class PolicyEstimator(nn.Module):
    def __init__(self, nS, nA, hidden_dim):
        super(PolicyEstimator, self).__init__()
        ## TODO
        self.layer1 = torch.nn.Linear(nS, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = torch.nn.Linear(hidden_dim, nA)
        
    def forward(self, x):
        # TODO
        # 현재 state에서 각 action을 선택할 확률을 return합니다.
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = torch.nn.functional.softmax(self.layer3(x), dim=-1)
        return x
        
    def select_action(self, state):
        # 현재 state에서 action 하나를 선택합니다.
        # TODO
        state = torch.FloatTensor(state)
        probs = self.forward(state)
        
        dist = torch.distributions.Categorical(probs)       # action에 대한 확률로 선택
        action = dist.sample()                              # 확률분포에 따라 sampling

        # 해당 action에 대한 확률의 log값도 함께 return해줍니다
        # Loss = -G_t · log π(A_t | S_t, θ)
        return action.item(), dist.log_prob(action)







# ### CartPole-v0 환경에 적용해봅시다!
env = gym.make('CartPole-v0')
print('state space:', env.observation_space)
print('action space:', env.action_space)

nS = env.observation_space.shape[0]
nA = env.action_space.n

policy_estimator = PolicyEstimator(nS, nA, hidden_dim=16)



# ## 1. REINFORCE Algorithm
policy_lr=1e-3 
num_episodes=10
gamma=.99
log_interval=50
    
policy_optimizer = optim.Adam(policy_estimator.parameters(), lr=policy_lr)


history = []
ep_states = defaultdict(lambda: [])
for i_episode in range(1, num_episodes+1):

    # episode 초기화
    state = env.reset()

    ep_reward = 0
    rewards = []
    log_probs = []

    # TODO
    # 현재 policy에 따라 episode를 생성합니다.
    # 학습에 사용하기 위해 reward와 각 action의 log probability를 저장합니다.
    while True: 
        ep_states[i_episode].append(state)
        action, log_prob = policy_estimator.select_action(state)
        next_state, reward, done, _ = env.step(action)

        if done:
            break
        state = next_state
        ep_reward += reward
        rewards.append(ep_reward)
        log_probs.append(log_prob)

    # TODO
    # episode의 각 step에 해당하는 cummulative return G_t를 계산합니다.
    returns = []
    G = 0        

    for r in rewards[::-1]:
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.FloatTensor(returns)
    
    # TODO
    # REINFORCE 알고리즘의 update 식을 참고하여 loss function을 정의해줍니다.
    log_probs = torch.stack(log_probs)      # 여러개의 tensor를 하나로 묶어줌
    policy_loss = (-log_probs * returns).sum()

    # parameter update를 합니다
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    # 학습 진행 상황을 history list에 저장하고 출력합니다
    # 마지막 50 episodes의 평균 reward를 저장하여 문제가 해결되었는지 판단합니다.
    history.append(ep_reward)
    avg_reward = sum(history[-50:])/len(history[-50:])
    
    if i_episode % log_interval == 0:
        print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, avg_reward))
        
    if avg_reward > env.spec.reward_threshold:
        print("Solved! Average reward: {:.2f}".format(avg_reward))
        break



# ==================================================================================

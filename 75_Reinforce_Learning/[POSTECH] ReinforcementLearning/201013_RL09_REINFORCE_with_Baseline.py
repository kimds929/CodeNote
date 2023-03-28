import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
from tqdm import tqdm, tqdm_notebook
import cv2
from matplotlib import animation, rc
import atari_py
# pip install cmake
# pip install gym-super-mario-bros
# pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T



env = gym.make('CartPole-v0')
print('state space:', env.observation_space)
print('action space:', env.action_space)

nS = env.observation_space.shape[0]
nA = env.action_space.n



def print_history(history):
    plt.figure(figsize=(12,4))
    plt.plot(history)
    plt.title('episode reward over iteration')
    plt.show()



# ## 2. REINFORCE with baseline
# ### value function approximator with neural network도 만들어 봅시다!

# Policy_Estimator **** -----------------------------------------------------------
# discrete action 환경에 사용할 수 있는 softmax policy입니다
# state를 input으로 받아서 각 action에 대한 확률을 output으로 계산합니다
class PolicyEstimator(nn.Module):
    def __init__(self, nS, nA, hidden_dim):
        super(PolicyEstimator, self).__init__()
        self.layer1 = torch.nn.Linear(nS, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = torch.nn.Linear(hidden_dim, nA)
        
    def forward(self, x):
        # 현재 state에서 각 action을 선택할 확률을 return합니다.
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = torch.nn.functional.softmax(self.layer3(x), dim=-1)
        return x
        
    def select_action(self, state):
        # 현재 state에서 action 하나를 선택합니다.
        state = torch.FloatTensor(state)
        probs = self.forward(state)
        
        dist = torch.distributions.Categorical(probs)       # action에 대한 확률로 선택
        action = dist.sample()                              # 확률분포에 따라 sampling

        # 해당 action에 대한 확률의 log값도 함께 return해줍니다
        # Loss = -G_t · log π(A_t | S_t, θ)
        return action.item(), dist.log_prob(action)


# Value_Estimator **** -----------------------------------------------------------
# state를 input으로 입력받아 그 state의 value function을 계산하는 함수입니다.
class ValueEstimator(nn.Module):
    def __init__(self, nS, hidden_dim):
        super(ValueEstimator, self).__init__()
        self.linear1 = torch.nn.Linear(nS, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, 1)
        self.act = torch.nn.ReLU()        
        
    def forward(self, x):
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        x = self.linear3(x)
        return x
        
    def estimate(self, state):
        state = torch.FloatTensor(state)
        return self.forward(state)


value = ValueEstimator(3, 16)
value(torch.FloatTensor([1,2,3]))
value.estimate([1,2,3])




# BaseLine -----------------------------------------------------------------------
def reinforce_baseline(env, 
                       policy_estimator, 
                       value_estimator, 
                       policy_lr=1e-3, 
                       value_lr=1e-3, 
                       num_episodes=1000, 
                       gamma=.99, 
                       log_interval=50):
    
    policy_optimizer = optim.Adam(policy_estimator.parameters(), lr=policy_lr)
    value_optimizer = optim.Adam(value_estimator.parameters(), lr=value_lr)
        
    history = []
    for i_episode in range(1, num_episodes+1):
        # episode 초기화
        state = env.reset()

        ep_reward = 0
        states = []
        rewards = []
        log_probs = []

        # 현재 policy에 따라 episode를 생성합니다.
        # 학습에 사용하기 위해 reward와 각 action의 log probability를 저장합니다.
        # state value(baseline)를 계산하기 위해 state도 추가로 저장합니다.
        while True: 
            action, log_prob = policy_estimator.select_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward

            states.append(state)
            rewards.append(reward)
            log_probs.append(log_prob)

            if done:
                break
            state = next_state

        # episode의 각 step에 해당하는 cummulative return G_t를 계산합니다.
        returns = []
        G = 0
        for r in rewards[::-1]:
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        # baseline으로 사용할 state value를 계산합니다.
        values = value_estimator.estimate(states)
        values = values.reshape(-1)
        
        # REINFORCE with baseline 알고리즘의 update 식을 참고하여 loss function을 정의해줍니다.
        # policy_loss = - ∑ (G_t - V(S_t, w)) · log π(A_t | S_t, θ)
        # value_loss = mse( V(S_t, w), G_t)
        policy_loss = -((returns - values.detach()) * torch.stack(log_probs)).sum()
        value_loss = torch.nn.functional.mse_loss(values, returns)
        
        # parameter update를 합니다
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()  

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

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




# Model Learning -----------------------------------------------------------------------
policy_estimator = PolicyEstimator(nS, nA, hidden_dim=16)
value_estimator = ValueEstimator(nS, hidden_dim=32)

history = reinforce_baseline(env, 
                             policy_estimator, 
                             value_estimator, 
                             num_episodes=600)

print_history(history)


# torch.stack(history)
# [tensor(-0.7516, grad_fn=<SqueezeBackward1>),
#  tensor(-0.6328, grad_fn=<SqueezeBackward1>),
#  tensor(-0.6400, grad_fn=<SqueezeBackward1>),
#  tensor(-0.7416, grad_fn=<SqueezeBackward1>),
#  tensor(-0.6410, grad_fn=<SqueezeBackward1>),
#  tensor(-0.6471, grad_fn=<SqueezeBackward1>),
#  tensor(-0.7401, grad_fn=<SqueezeBackward1>),
#  tensor(-0.6464, grad_fn=<SqueezeBackward1>),
#  tensor(-0.7399, grad_fn=<SqueezeBackward1>),
#  tensor(-0.7427, grad_fn=<SqueezeBackward1>),
#  tensor(-0.6408, grad_fn=<SqueezeBackward1>),
#  tensor(-0.6462, grad_fn=<SqueezeBackward1>),
#  tensor(-0.6512, grad_fn=<SqueezeBackward1>),
#  tensor(-0.6513, grad_fn=<SqueezeBackward1>),
#  tensor(-0.7378, grad_fn=<SqueezeBackward1>),
#  tensor(-0.7370, grad_fn=<SqueezeBackward1>),
#  tensor(-0.6517, grad_fn=<SqueezeBackward1>)]

# tensor([-0.7516, -0.6328, -0.6400, -0.7416, -0.6410, -0.6471, -0.7401, -0.6464,
#         -0.7399, -0.7427, -0.6408, -0.6462, -0.6512, -0.6513, -0.7378, -0.7370,
#         -0.6517], grad_fn=<StackBackward>)










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
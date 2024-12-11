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






# ## 3. TD(0) Actor-Critic ----------------------------------------------------------------
def actor_critic(env, 
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

        # one step in the environment
        # 앞선 MC 방법들과 달리 episode가 끝난 후에 parameter update하는 것이 아니라, 매 step마다 학습합니다
        while True: 
            action, log_prob = policy_estimator.select_action(state)
            value = value_estimator.estimate(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            
            # Temporal difference(TD) target을 계산합니다.
            # δ ← R + γ·V(S', w) - V(S, w)
            # w ← w + aw·δ·▽

            # δ ← R + γ·V(S', w) - V(S, w)
            td_target = reward
            
            if not done:
                next_value = value_estimator.estimate(next_state)
                td_target += gamma * next_value.detach()
            
            td_error = td_target - value.detach()

            # TD(0) Actor-Critic의 update 식을 참고하여 loss function를 정의해줍니다
            # policy_loss = - td_error × log π(A_t | S_t, θ)
            # value_loss = loss (value, td_target)
            policy_loss = - td_error * log_prob
            value_loss = torch.nn.functional.mse_loss(value, torch.FloatTensor([td_target]) )

            # θ Update Rule
            # θ_t+1 = θ_t + α(δ_t) · ▽θπ(A_t | S_t, θ) / π(A_t | S_t, θ)            
            
            # parameter update를 합니다
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

            if done:
                break
            state = next_state

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




policy_estimator = PolicyEstimator(nS, nA, hidden_dim=16)
value_estimator = ValueEstimator(nS, hidden_dim=32)

history = actor_critic(env, 
                       policy_estimator, 
                       value_estimator,
                       policy_lr=1e-4,
                       num_episodes=2000)

print_history(history)













# ### 다른 environment에 대해서도 시도해봅시다! (e.g. LunarLander-v2)
# #### - approximator의 구조 혹은 hyperparameter를 바꿔보면서 최적의 모델을 찾아봅시다



# conda install swig # needed to build Box2D in the pip install
# pip install box2d-py # a repackaged version of pybox2d




# LunarLander Discrete ---------------------------------------------------------------------
env = gym.make('LunarLander-v2')
print('state space:', env.observation_space)
print('action space:', env.action_space)


nS = env.observation_space.shape[0]
nA = env.action_space.n


# 새로운 환경에 REINFORCE 알고리즘 적용해보기
policy_estimator = PolicyEstimator(nS, nA, hidden_dim=16)
value_estimator = ValueEstimator(nS, hidden_dim=32)

history = actor_critic(env, 
                       policy_estimator, 
                       value_estimator,
                       policy_lr=1e-4,
                       num_episodes=500)

print_history(history)





# LunarLander Continuous ---------------------------------------------------------------------
# ## (추가 실습) Continuous action space
# ### Gaussian policy with neural network approximator를 구현해봅시다!

env = gym.make('LunarLanderContinuous-v2')
print('state space:', env.observation_space)
print('action space:', env.action_space)


nS = env.observation_space.shape[0]
nA = env.action_space.shape[0]



# Gaussian policy를 사용하여 continuous policy approximator를 만듭니다.
# 편의상 isotropic covariance matrix를 사용하여 parameterize합니다.
class NewPolicyEstimator(nn.Module):
    def __init__(self, nS, nA, hidden_dim):
        super(NewPolicyEstimator, self).__init__()
        self.linear1 = torch.nn.Linear(nS, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.mu_head = torch.nn.Linear(hidden_dim, nA)
        self.sigma_head = torch.nn.Linear(hidden_dim, nA)

        self.act = torch.nn.ReLU()

    def forward(self, x):
        # 현재 state에서 연속 action에 대한 확률분포의 parameters(e.g. mean, covariance matrix)를 return합니다.
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))

        mu = self.mu_head(x)
        sigma = self.sigma_head(x)
        sigma = torch.nn.functional.softplus(sigma)

        return mu, sigma

    def select_action(self, state):
        # 현재 state에서 action 하나를 선택합니다.
        state = torch.FloatTensor(state)
        mu, sigma = self.forward(state)

        dist = torch.distributions.MultivariateNormal(mu, torch.diag(sigma))
        action = dist.sample()

        return np.array(action), dist.log_prob(action)

policy = NewPolicyEstimator(nS, nA, 16)
policy.select_action(env.reset())



# ### 위에서 작성한 알고리즘을 그대로 적용해 볼 수 있습니다.
# #### - approximator의 구조 혹은 hyperparameter를 바꿔보면서 최적의 모델을 찾아봅시다


policy_estimator = NewPolicyEstimator(nS, nA, hidden_dim=128)
value_estimator = ValueEstimator(nS, hidden_dim=256)

history = actor_critic(env, 
                    policy_estimator,
                    policy_lr=1e-3,
                    num_episodes=500, 
                    log_interval=10)

print_history(history)












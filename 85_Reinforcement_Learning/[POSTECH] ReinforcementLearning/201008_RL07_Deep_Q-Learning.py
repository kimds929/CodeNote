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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# import sys
# RLpath = r'D:\Python\★★Python_POSTECH_AI\Postech_AI 6) Reinforce_Learning\강의) 실습자료'
# sys.path.append(RLpath) # add project root to the python path
# from code_FA.lib import plotting
# # get_ipython().run_line_magic('matplotlib', 'inline')



env = gym.make('CartPole-v0')
env.reset()     
# array([-0.00146775, -0.03005972, -0.00392236, -0.04529694])
#    cart_position, cart_velocity, pole_angle, pole_angular_velocity

env.observation_space
env.action_space

# device를 설정합니다.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Current device : {}'.format(device))


# cartpole은 2가지 action으로 작동합니다.
# 0 (left)
# 1 (right)
VALID_ACTIONS = [0, 1]
LR = 1e-4

class DQN(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions):
        super(DQN, self).__init__()
        self.linear1 = torch.nn.Linear(state_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = torch.nn.functional.relu(self.linear3(x))
        return x


# class DQN(nn.Module):
#     def __init__(self, state_dim, hidden_dim, num_actions):
#         super(DQN, self).__init__()
#         self.linear1 = nn.Linear(state_dim, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, hidden_dim)
#         self.linear3 = nn.Linear(hidden_dim, num_actions)
        
#     def forward(self, x):
#         x = F.relu(self.linear1(x))
#         x = F.relu(self.linear2(x))
#         return F.relu(self.linear3(x))


def array2tensor(array):
    return torch.FloatTensor(array).to(device)


# epsilon greedy policy 입니다.
def select_action(state, q_network, eps):
    """
    state : 현재 state
    eps : epsilon-greedy를 위한 epsilon
    """
    if random.random() < eps:
        return random.choice(VALID_ACTIONS)
    else:       # 현state를 DQN forward하여 action_value → argmax → value
        q_network.eval()        # evaluation mode
        # q_network.train()     # train mode
        with torch.no_grad():
            action_values = q_network(array2tensor(state)).squeeze()
        return torch.argmax(action_values).item()

q_network = DQN(state_dim=4, hidden_dim=16, num_actions=2).to(device)
optimizer = optim.RMSprop(q_network.parameters(), lr=LR, momentum=0.95, alpha=0.95, eps=0.01)




eps = 0.1
discount_factor = 0.9
reward_list = []
num_episodes = 1000

for i_episode in tqdm_notebook(range(num_episodes)):
    # 환경을 initialize 합니다.
    state = env.reset()
    REWARD = 0
    # for문 안에서 episode의 한스텝을 진행합니다.
    # (state, action, next_state, reward) 를 사용해서 network를 업데이트 합니다.
    for t in count():
        # Select and perform an action        
        action = select_action(state, q_network, eps=eps)
        next_state, reward, done, _ = env.step(action)
        REWARD += reward
        
        # Optimize Model
        q_value_current = q_network(torch.FloatTensor(state))[action]
        q_value_next = q_network(torch.FloatTensor(next_state)).detach()

        # td_target = reward +  discount_factor * q_value_next.max()
        td_target = reward
        if not done:
            td_target = td_target +  discount_factor * q_value_next.max()
        td_error = (td_target - q_value_current) **2

        # td_target = reward
        # if not done:
        #     td_target = td_target + discount_factor * q_network(array2tensor(next_state)).max().detach()
        # td_error = (q_network(array2tensor(state))[action] - td_target)**2

        optimizer.zero_grad()
        td_error.backward()
        optimizer.step()

        if done:
            break
        # Move to the next state
        state = next_state
            
    reward_list.append(REWARD)
    print('{}th eposide reward : {}, Max Reward : {}'.format(i_episode+1, REWARD, max(reward_list)), end='\r')
    if max(reward_list) == REWARD:
        torch.save(q_network.state_dict(), 'best_model.bin')

print('Complete')
env.close()


plt.plot(reward_list)
plt.show()

# best모델 불러오기
q_network.load_state_dict(torch.load('best_model.bin'))





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
state = env.reset()

img = plt.imshow(renderCart(state), cmap='gray')
while True:
    img.set_data(renderCart(state))
    display.display(plt.gcf())
    display.clear_output(wait=True)
    
    action = q_network(torch.FloatTensor(state)).argmax().item()
    
    next_state, _, done, _ = env.step(action)
    state = next_state
    if done: 
        break
env.close()











# tensorflow --------------------------------------
import tensorflow as tf

class DQN_tf(tf.keras.Model):
    def __init__(self, state_dim, hidden_dim, num_actions):
        super(DQN_tf, self).__init__()
        self.linear1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.linear2 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.linear3 = tf.keras.layers.Dense(num_actions)
    
    def call(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

q_network_tf = DQN_tf(state_dim=4, hidden_dim=16, num_actions=2)
optimizer_tf = tf.keras.optimizers.RMSprop(learning_rate=LR, momentum=0.95, rho=0.95, epsilon=0.01)


eps = 0.1
discount_factor = 0.9
reward_list = []
num_episodes = 1000

for _ in tqdm_notebook(range(num_episodes)):
    state = env.reset()
    total_reward = 0
    for t in count():
        if np.random.rand() < eps:
            action = env.action_space.sample()
        else:
            q_value_current = q_network_tf(tf.constant(state, dtype=tf.float32)[tf.newaxis, ...])
            action = tf.argmax(q_value_current, axis=-1).numpy()[0]

        next_state, reward, done, _ = env.step(action)

        with tf.GradientTape() as Tape:
            q_value_current = q_network_tf(tf.constant(state, dtype=tf.float32)[tf.newaxis, ...])
            q_value_next = q_network_tf(tf.constant(next_state, dtype=tf.float32)[tf.newaxis, ...])

            td_target = reward
            if not done:
                td_target = td_target + discount_factor * tf.reduce_max(q_value_next)
            td_error = tf.math.square(td_target - q_value_current[0][action])

        gradients = Tape.gradient(td_error, q_network_tf.trainable_variables)
        optimizer_tf.apply_gradients(zip(gradients, q_network_tf.trainable_variables))

        if done:
            break
        state = next_state
        total_reward +=reward
    reward_list.append(total_reward)
print('Complete')
env.close()


plt.plot(reward_list)
plt.show()


# cart-pole exec by learned Q_network_tf
state = env.reset()

img = plt.imshow(renderCart(state), cmap='gray')
while True:
    img.set_data(renderCart(state))
    display.display(plt.gcf())
    display.clear_output(wait=True)
    
    q_value_current = q_network_tf(tf.constant(state, dtype=tf.float32)[tf.newaxis, ...])
    action = tf.argmax(q_value_current, axis=-1).numpy()[0]

    next_state, _, done, _ = env.step(action)      
    state = next_state
    if done: 
        break
env.close()
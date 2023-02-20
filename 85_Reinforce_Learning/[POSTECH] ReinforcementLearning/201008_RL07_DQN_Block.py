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


# env = gym.envs.make("Breakout-v0")
# env = gym.make('BreakoutDeterministic-v4')

# device를 설정합니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Current device : {}'.format(device))


#env = gym.envs.make("Breakout-v0")
env = gym.make('BreakoutDeterministic-v4')

# device를 설정합니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Current device : {}'.format(device))

# env.render()
# Atari breakout은 4가지 action으로 작도합니다.
# 0 (no action)
# 1 (fire)
# 2 (left)
# 3 (right)
# 필요한 parameter들을 설정해줍니다.
VALID_ACTIONS = [0, 1, 2, 3]
num_episodes = 20000
BATCH_SIZE=32

GAMMA=0.99
TARGET_UPDATE=10000

epsilon_start=1     # epsilon decay 1 → 0.1
epsilon_end=0.1
epsilon_decay_steps=1000000

LR = 2.5e-4
K=4     # 몇개의 frame을 붙여서 쓸 것 인지?


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# 학습을 위한 데이터를 저장할 Memory class를 생성합니다.
class ReplayMemory():
    def __init__(self, capacity):       # capacity: memory 용량
        self.capacity = capacity
        self.memory = []
        self.position = 0               # 현재 memory에서 어떤 위치에서 데이터가 쓰여지고 있는지

    def push(self, *args):
        """
        (state, action, nextstate, reward)를 받아서 메모리에 추가하는 함수입니다.
        memory가 가득차지 않았다면 tuple을 memory에 추가를하고 가득찼다면 
        가장 오래된 메모리를 지우고 그 자리에 새로운 tuple을 추가합니다.
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        memory에서 batch_size만큼 sampling을 합니다.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



# Shape:
#  $$\text{- Input : }(N, C_{in}, H_{in}, W_{in})$$
#  $$\text{- Output : }(N, C_{out}, H_{out}, W_{out})$$
# 
#   $$H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
#             \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor$$
#   $$W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
#             \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor$$
# 


class DQN(nn.Module):
    """
    Deep Q-Network
    State(image)를 처리하는 CNN network입니다.
    State는 이전 frame 4개로 이루어져있습니다. 즉 input_channel=4입니다.
    3개의 convolutional layer와 3개의 linear layer로 이루어진 Network입니다.
    """
    def __init__(self, h, w, outputs):
        """
        Network에 필요한 layer들을 선언합니다.
        필요한 변수들도 받습니다.
        h : state image의 높이
        w : state image의 넓이
        outputs : action의 개수
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(w,kernel_size=8, stride=4), kernel_size=4, stride=2)
        convw = conv2d_size_out(convw, kernel_size=3, stride=1)
        convh = conv2d_size_out(conv2d_size_out(h,kernel_size=8, stride=4), kernel_size=4, stride=2)
        convh = conv2d_size_out(convh, kernel_size=3, stride=1)
        linear_input_size = convw * convh * 64
        self.linear1 = nn.Linear(linear_input_size, 512)
        self.linear2 = nn.Linear(512, outputs)
        
    def forward(self, x):
        """
        x: (B, C, W, H)     # torch
        x: (B, W, H, C)     # Tensorflow
        self.__init__에서 생성한 layer들을 사용하여
        input x가 들어왔을때 실행되는 계산을
        실행합니다.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)


# state를 전처리하는 코드입니다.
def rgb2gray(state):        # rgb → gray
    return cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)

def resize(state):          # resize: 210,160 → 110, 84
    return cv2.resize(state, (84, 110))

def crop_box(state):
    return state[17:17+84,:]

def preprocess(state):
    out = crop_box(resize(rgb2gray(state)))/255
    return out


# epsilon greedy policy 입니다.
def select_action(state, eps):
    """
    state : torch tensor shape of (B, C, W, H)
    eps : epsilon for epsilon greedy policy
    """
    if random.random()<eps:
        return random.choice(VALID_ACTIONS)
    else:
        policy_net.eval()
        with torch.no_grad():
            action_prob = policy_net(state).squeeze()
        return torch.argmax(action_prob)


# 학습에 사용할 policy_net과 학습과정에서 target을 생성하기 위한 target_net을 만듭니다.
init_state = env.reset()
init_state = preprocess(init_state)
screen_height, screen_width = init_state.shape
n_actions = len(VALID_ACTIONS)

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)     # 학습에 사용하지 않음
# target_net의 weight들을 train_net의 weight와 똑같이 만들어줍니다.

target_net.load_state_dict(policy_net.state_dict())

# target_net은 학습이 되지 않기 때문에 evaluation mode로 전환합니다.
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters(), lr=LR, momentum=0.95, alpha=0.95, eps=0.01)


# memory를 만들고 capacity의 10%를 랜덤한 state들로 채워줍니다.
# 모델에서 사용할 state는 하나의 frame이 아니라 최근 4개의 프레임을 사용한다는 것에 주의합니다.
memory = ReplayMemory(120000)
state = env.reset()
state = preprocess(state)
state = np.expand_dims(state, 0)
# state.shape       # (1, 84, 84)


for i in range(K-1):        # k개의 frame을 쌓아줌
    action = random.choice(VALID_ACTIONS)
    next_state, _, _, _ = env.step(action)
    next_state = preprocess(next_state)
    next_state = np.expand_dims(next_state, 0)              # (1, 84, 84)
    state = np.concatenate([state, next_state], axis=0)     # (4, 84, 84)
state = np.expand_dims(state,0)
# state.shape       # (1, 4, 84, 84)        # (batch, image_set, w, h)


for i in tqdm_notebook(range(int(memory.capacity/10))):
    action = select_action(state, eps=1)
    next_state, reward, done, _  = env.step(action)
    next_state = preprocess(next_state)             # (84, 84)
    next_state = np.expand_dims(next_state, 0)      # (1, 84, 84)
    next_state = np.expand_dims(next_state, 0)      # (1, 1, 84, 84)
    next_state = np.concatenate([state[:,1:,:,:], next_state], axis=1)        # (1, 3, 84, 84) | (1, 1, 84, 84) = (1, 4, 84, 84)
    
    if done:
        next_state = None
    reward = np.sign(reward)            # Reward의 부호값만 사용하겠다.
    
    # 메모리에 state, action, next_state, reward를 저장
    memory.push(torch.tensor(state).float().to(device),
                torch.tensor([action]).to(device), 
                next_state, 
                torch.tensor([reward]).float().to(device))
    
    if done:
        state = env.reset()
        state = preprocess(state)
        state = np.expand_dims(state, 0)
        for i in range(K-1):
            action = random.choice(VALID_ACTIONS)
            next_state, _, _, _ = env.step(action)
            next_state = preprocess(next_state)
            next_state = np.expand_dims(next_state, 0)
            state = np.concatenate([state, next_state], axis=0)
        state = np.expand_dims(state,0)
    else:
        state = next_state


# 결과를 영상으로 보기위한 rendering 함수입니다.
def get_render(frames):
    fig, ax = plt.subplots()
    line, = ax.plot([],[])
    def init():
        line.set_data([],[])
        return (line, )
    
    def animate(i):
        ax.clear()
        ax.imshow(frames[i], 'gray', clim=(0,1))
        line, = ax.plot([], [])
        return (line,)
    anim = animation.FuncAnimation(fig, animate, frames = len(frames), interval=30, blit=True)
    rc('animation', html='html5')
    plt.close()
    return anim

transitions = memory.sample(BATCH_SIZE)
batch = Transition(*zip(*transitions)) # batch.state : 길이가 batch_size인 tuple
len(batch)
batch[0][0].shape
batch[0][1].shape
len(batch[0])

len(batch.next_state)


def optimize_model():
    """
    memory에서 batch 만큼의 sample을 뽑아와서 학습을 진행합니다.
    """
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions)) # batch.state : 길이가 batch_size인 tuple
    # batch.state = ((1, 4, 84, 84), (1, 4, 84, 84), ...)
    # batch.action = (1, 1, 1, ...)
    # batch.next_state = ((1, 4, 84, 84), (1, 4, 84, 84), ...)
    # batch.reward = (1, -1, 1, ...)

    # Sampling한 sample 각각이 Terminate state에 도달 하였는지 확인해야합니다.
    # Terminate state에서는 reward의 기댓값이 0이기 때문에 target policy의
    # 값 대신 0으로 업데이트를 진행하여야 하기 때문입니다.
    # Terminate state는 None으로 주어질 것입니다.
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,        # final: False / not final: True
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = np.stack([s for s in batch.next_state
                                                if s is not None])
    non_final_next_states = torch.tensor(non_final_next_states).squeeze().float().to(device)
    
    state_batch = torch.cat(batch.state).squeeze().float().to(device)       # torch.cat : 두 tensor를 이어붙인다.
    action_batch = torch.cat(batch.action).squeeze().to(device)
    reward_batch = torch.cat(batch.reward).squeeze().float().to(device)
    
    # policy_net을 사용하여 state_action_value를 target_net을 사용하여 next_state_value(target_value)를 계산합니다.
    policy_net.train()
    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))      # Q(s_t, a_t)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Error를 계산합니다.
    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # 학습을 진행합니다.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()




epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
update_count = 0
eps_count = 0
reward_list = [0]
best_episode = None
# num_episodes = 20000

for i_episode in tqdm_notebook(range(num_episodes)):
    # 환경을 initialize 합니다.
    state_episode = []
    state = env.reset()
    state_episode.append(state)
    state = preprocess(state)
    state = np.expand_dims(state, 0)
    # K개의 첫 state를 만들어서 합쳐줍니다.
    for i in range(K-1):
        action = random.choice(VALID_ACTIONS)
        next_state, _, _, _ = env.step(action)
        next_state = preprocess(next_state)
        next_state = np.expand_dims(next_state, 0)
        state = np.concatenate([state, next_state], axis=0)
    state = np.expand_dims(state,0)
    REWARD = 0

    # for문 안에서 episode의 한스텝을 진행합니다.
    # 이전 state S = (s1, s2, s3, s4)라고 한다면 다음 state S`= (s2, s3, s4, s5)가 됩니다.
    # 새로운 sample (S, action, S`, reward) 를 메모리에 추가시켜줍니다.
    # optimize_model 함수를 사용하여 한번의 update를 진행합니다.
    for t in count():
        # Select and perform an action
        if eps_count<len(epsilons):
            epsilon = epsilons[eps_count]
            eps_count+=1
        else:
            epsilon = epsilon_end
        
        action = select_action(state, eps=1)
        next_state, reward, done, _  = env.step(action)
        next_state = preprocess(next_state)
        next_state = np.expand_dims(next_state, 0)
        next_state = np.expand_dims(next_state, 0)
        next_state = np.concatenate([state[:,1:,:,:], next_state], axis=1)
        if done:
            next_state = None
        reward = np.sign(reward)
        memory.push(torch.tensor(state).float().to(device), 
                    torch.tensor([action]).to(device), 
                    next_state, 
                    torch.tensor([reward]).float().to(device))
        REWARD += reward

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        update_count+=1
        if done:
            break

    # Update the target network, copying all weights and biases in DQN
    if update_count % TARGET_UPDATE == 0:
        print('Update Target Network')
        target_net.load_state_dict(policy_net.state_dict())
    reward_list.append(REWARD)
    print('{}th eposide reward : {}, Max Reward : {}'.format(i_episode+1, REWARD, max(reward_list)), end='\r')
    if max(reward_list)==REWARD:
        best_episode=state_episode

print('Complete')
env.close()



state = env.reset()
state = preprocess(state)
state = np.stack([state] * 4, axis=0)
state = torch.from_numpy(state).unsqueeze(0).float().to(device)



policy_net.eval()
frame_list = []
for t in count():
    action = select_action(state, eps=0.05)
    next_state, reward, done, _ = env.step(action)
    # print(reward)
    frame_list.append(next_state)
    reward = torch.tensor([reward], device=device)
    next_state = preprocess(next_state)
    next_state = torch.from_numpy(next_state).unsqueeze(0).unsqueeze(0).float().to(device)
    next_state = torch.cat([state[:,1:,:,:], next_state], dim=1)
        
    if done:
        break



[frame for i, frame in enumerate(frame_list) if i%4==0]



get_render([frame for i, frame in enumerate(frame_list) if i%4==0])



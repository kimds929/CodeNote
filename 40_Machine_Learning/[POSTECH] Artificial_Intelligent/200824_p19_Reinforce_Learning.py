# 데이비드 실버 교수님의 강화학습 강의를 한국어로 풀어준 동영상입니다
# https://www.youtube.com/playlist?list=PLpRS2w0xWHTcTZyyX8LMmtbcMXpd3s4TU

# https://wordbe.tistory.com/entry/RL-%EA%B0%95%ED%99%94%ED%95%99%EC%8A%B5-part1-policy-value-function




import numpy as np
import gym
import matplotlib.pyplot as plt
# %matplotlib inline

import time
from IPython.display import clear_output


# 5.1. 1-D Observation Space
# 5.1.1. Taxi

env = gym.make('Taxi-v3',render_mode="human")

# env 설명 ****
# There are four designated locations in the grid world indicated by R(ed), G(reen), Y(ellow), and B(lue). 
# When the episode starts, the taxi starts off at a random square and the passenger is at a random location. 
# The taxi drives to the passenger's location, picks up the passenger, drives to the passenger's destination 
# (another one of the four specified locations),
#  and then drops off the passenger. Once the passenger is dropped off, the episode ends.

# Rendering:
# - blue: passenger
# - magenta: destination
# - yellow: empty taxi
# - green: full taxi
# - other letters (R, G, Y and B): locations for passengers and destinations

# Passenger locations:
# - 0: R(ed)
# - 1: G(reen)
# - 2: Y(ellow)
# - 3: B(lue)
# - 4: in taxi

# Destinations:
# - 0: R(ed)
# - 1: G(reen)
# - 2: Y(ellow)
# - 3: B(lue)


env.reset()
env.render()                # 내부 현재 state를 사용자가 보기 편하도록 해준 method
# ? gym.envs.toy_text.taxi.TaxiEnv
env.observation_space.n     # 구현된 state의 갯수
# There are 500 discrete states since there are 25 taxi positions, 
# 5 possible locations of the passenger (including the case 
# when the passenger is in the taxi), and 4 destination locations. 

env.action_space.n      # 구현된 action의 갯수
env.action_space.sample()   # 0~6까지 random하게 state가 나옴

env.reset()     # environment reset
r_action = env.action_space.sample()
print(r_action)

env.step(r_action)  
# (1, -1, False, {'prob': 1.0})
# (state, reward, done(환경이끝났는지?), info)




# env 실행 (action execution cycle)
print(env.reset())
env.render()
action = env.action_space.sample()
state, reward, done, truncated, info = env.step(action)
env.render()
print(f'action : {action} / state: {state}, reward: {reward}')




# RL Taxi-v3
n_states = env.observation_space.n
n_actions = env.action_space.n
print(n_actions, n_actions)




# Q_table update modeling ----------------------------
Q_table = np.random.uniform(0, 1, (n_states, n_actions))
# Q_table
print(Q_table.shape, np.min(Q_table), np.max(Q_table))

alpha = 0.3
gamma = 0.9
epsilone = 0.1  # 탐험 비율

n_episode = 1000

# 1000번 episode에 대하여 taxi 실행하면서 Q_table 학습
for episode in range(n_episode + 1):
    s, reset_info = env.reset() # state initialize
    # env.render()

    count = 0
    while True:
        # ε-greedy
        if np.random.uniform() < epsilone:      # exploration(탐험)
            a = env.action_space.sample()
        else:                                   # exploitation
            a = np.argmax(Q_table[s, :])
        next_state, reward, done, truncated, info = env.step(a)
        Q_table[s, a] = (1-alpha) * Q_table[s,a] + alpha * (reward + gamma + np.max(Q_table[next_state, :]))
        s = next_state  # state update

        # 결과 display
        # env.render()
        # clear_output(wait=True)
        # time.sleep(0.1)

        count += 1
        if done:
            break
    
    if episode % 100 == 0:
        print(f"Episode: {episode} steps: {count}")


# 학습된 모델을 활용하여 taxi 실행
s = env.reset()
while True:
    env.render()
    a = np.argmax(Q_table[s, :])
    s, _, done, _ = env.step(a)

    clear_output(wait=True)
    time.sleep(0.1)
    if done:
        break


env.close()












# 5.1.2. Frozen Lake ----------------------------------------------------------------------

import numpy as np
import gym
import matplotlib.pyplot as plt
# %matplotlib inline
from IPython.display import clear_output
from time import sleep

env = gym.make('FrozenLake-v1')

# ?gym.envs.toy_text.frozen_lake.FrozenLakeEnv
# Winter is here. You and your friends were tossing around a frisbee at the park
# when you made a wild throw that left the frisbee out in the middle of the lake.
# The water is mostly frozen, but there are a few holes where the ice has melted.
# If you step into one of those holes, you'll fall into the freezing water.
# At this time, there's an international frisbee shortage, so it's absolutely imperative that
# you navigate across the lake and retrieve the disc.
# However, the ice is slippery, so you won't always move in the direction you intend.
# The surface is described using a grid like the following

#     SFFF
#     FHFH
#     FFFH
#     HFFG

# S : starting point, safe
# F : frozen surface, safe
# H : hole, fall to your doom
# G : goal, where the frisbee is located

env.reset()
env.render()
env.step(env.action_space.sample())
env.render()




# Q_table update modeling ----------------------------
n_states = env.observation_space.n
n_actions = env.action_space.n
print(n_actions, n_actions)

Q_table = np.random.uniform(0, 1, (n_states, n_actions))
# Q_table
print(Q_table.shape, np.min(Q_table), np.max(Q_table))

alpha = 0.3
gamma = 0.9
epsilone = 0.1  # 탐험 비율

n_episode = 10000


hole_penalty = -15
counts = []

# 1000번 episode에 대하여 taxi 실행하면서 Q_table 학습
for episode in range(n_episode + 1):
    s = env.reset() # state initialize
    # env.render()

    count = 0
    while True:
        # ε-greedy
        if np.random.uniform() < epsilone:      # exploration(탐험)
            a = env.action_space.sample()
        else:                                   # exploitation
            a = np.argmax(Q_table[s, :])
        next_state, reward, done, info = env.step(a)
        Q_table[s, a] = (1-alpha) * Q_table[s,a] + alpha * (reward + gamma + np.max(Q_table[next_state, :]))
        s = next_state  # state update

        # 결과 display
        # env.render()
        # clear_output(wait=True)
        # time.sleep(0.1)

        count += 1
        if done:
            if next_state != 15:
                Q_table[s, a] = hole_penalty
            break

        counts.append(count)
        if len(counts) > 100:
            counts.pop(0)

    if episode % 1000 == 0:
        print(f"Episode: {episode} steps: {count}")


env.close()



# 학습된 모델을 활용하여 Frozon_Lake 실행
s = env.reset()
while True:
    a = np.argmax(Q_table[s, :])
    s, _, done, _ = env.step(a)

    env.render()

    clear_output(wait=True)
    time.sleep(0.1)
    if done:
        break



# Q Table Map show
actions = ['Left', 'Down', 'Right', 'Up']
print('Actions:', actions)

with np.printoptions(precision=2):
    print(Q_table)

print('\nMap:')
for i in range(4):
    for j in range(4):
        n = j+4*i
        if n in [5, 7, 11, 12, 15]:
            print('■', end=' ')
        else:
            print(actions[np.argmax(Q_table[n])][0], end=' ')

    print()






# Q_table update modeling2 ----------------------------
n_states = env.observation_space.n
n_actions = env.action_space.n
print(n_actions, n_actions)

Q_table = np.random.uniform(0, 1, (n_states, n_actions))
# Q_table
print(Q_table.shape, np.min(Q_table), np.max(Q_table))

alpha = 0.3
gamma = 0.9
epsilone = 0.1  # 탐험 비율

n_episode = 10000

hole_penalty = -15
counts = []

# ** --------------------
def adaptive_rate(t, reach_at, minimum):
    return max(minimum, min(1, 1.0 - np.log10((t+1)/(reach_at//10))))
# ** --------------------

for episode in range(n_episode + 1):
    s = env.reset() # state initialize
    # env.render()
    epsilone = adaptive_rate(episode, 10000, 0.2)   # epsilon 값을 adaptive_rate 적용

    count = 0
    while True:
        # ε-greedy
        if np.random.uniform() < epsilone:      # exploration(탐험)
            a = env.action_space.sample()
        else:                                   # exploitation
            a = np.argmax(Q_table[s, :])
        next_state, reward, done, info = env.step(a)
        Q_table[s, a] = (1-alpha) * Q_table[s,a] + alpha * (reward + gamma + np.max(Q_table[next_state, :]))
        s = next_state  # state update

        # 결과 display
        # env.render()
        # clear_output(wait=True)
        # time.sleep(0.1)

        count += 1
        if done:
            if next_state != 15:
                Q_table[s, a] = hole_penalty
            break

        counts.append(count)
        if len(counts) > 100:
            counts.pop(0)

    if episode % 1000 == 0:
        print(f"Episode: {episode} steps: {count}")

# epsilone 감수
x = range(1000)
plt.figure(figsize=(6, 4))
plt.plot(x, [adaptive_rate(t, 500, 0.01) for t in x], label='epsilon')
plt.plot(x, [adaptive_rate(t, 800, 0.1) for t in x], label='alpha')
plt.xlabel('episode'); plt.ylabel('value'); plt.legend()
plt.grid()
plt.show()





# 학습된 모델을 활용하여 Frozon_Lake 실행
s = env.reset()
while True:
    a = np.argmax(Q_table[s, :])
    s, _, done, _ = env.step(a)

    env.render()

    clear_output(wait=True)
    time.sleep(0.1)
    if done:
        break



# Q Table Map show
actions = ['Left', 'Down', 'Right', 'Up']
print('Actions:', actions)

with np.printoptions(precision=2):
    print(Q_table)

print('\nMap:')
for i in range(4):
    for j in range(4):
        n = j+4*i
        if n in [5, 7, 11, 12, 15]:
            print('■', end=' ')
        else:
            print(actions[np.argmax(Q_table[n])][0], end=' ')

    print()







# Q_table update modeling3 ----------------------------    
n_states = env.observation_space.n
n_actions = env.action_space.n
print(n_actions, n_actions)

Q_table = np.random.uniform(0, 1, (n_states, n_actions))
# Q_table
print(Q_table.shape, np.min(Q_table), np.max(Q_table))

alpha = 0.3
gamma = 0.9
epsilone = 0.1  # 탐험 비율

n_episode = 10000

hole_penalty = -15
counts = []


# ** --------------------
def getReward(next_state):
    rw_map = np.array([[0, 1, 2, 0],
                       [1, -10, 3, -10],
                       [2, 3, 4, -10],
                       [-10, 4, 5, 10000]]).reshape(-1)
    return rw_map[next_state]
# ** --------------------

for episode in range(n_episode + 1):
    s = env.reset() # state initialize
    # env.render()
    epsilone = adaptive_rate(episode, 10000, 0.2)   # epsilon 값을 adaptive_rate 적용

    count = 0
    while True:
        # ε-greedy
        if np.random.uniform() < epsilone:      # exploration(탐험)
            a = env.action_space.sample()
        else:                                   # exploitation
            a = np.argmax(Q_table[s, :])
        next_state, reward, done, info = env.step(a)

        reward = getReward(next_state)  # ****

        Q_table[s, a] = (1-alpha) * Q_table[s,a] + alpha * (reward + gamma + np.max(Q_table[next_state, :]))
        s = next_state  # state update

        # 결과 display
        # env.render()
        # clear_output(wait=True)
        # time.sleep(0.1)

        count += 1
        if done:
            if next_state != 15:
                Q_table[s, a] = hole_penalty
            break

        counts.append(count)
        if len(counts) > 100:
            counts.pop(0)

    if episode % 1000 == 0:
        print(f"Episode: {episode} steps: {count}")


# 학습된 모델을 활용하여 Frozon_Lake 실행
s = env.reset()
while True:
    a = np.argmax(Q_table[s, :])
    s, _, done, _ = env.step(a)

    env.render()

    clear_output(wait=True)
    time.sleep(0.1)
    if done:
        break



# Q Table Map show
actions = ['Left', 'Down', 'Right', 'Up']
print('Actions:', actions)

with np.printoptions(precision=2):
    print(Q_table)

print('\nMap:')
for i in range(4):
    for j in range(4):
        n = j+4*i
        if n in [5, 7, 11, 12, 15]:
            print('■', end=' ')
        else:
            print(actions[np.argmax(Q_table[n])][0], end=' ')

    print()









# --------------------------------------------------------------------------------------------------------------------
# 8 × 8 --------------------------------------------------------------------------------------------------------------
env = gym.make(id='FrozenLake8x8-v0', map_name=None, is_slippery=True)
env.render()
n_states = env.observation_space.n
n_actions = env.action_space.n


# Q_table update modeling4 ---------------------------

n_states = env.observation_space.n
n_actions = env.action_space.n
print(n_actions, n_actions)

Q_table = np.random.uniform(0, 1, (n_states, n_actions))
# Q_table
print(Q_table.shape, np.min(Q_table), np.max(Q_table))

alpha = 0.3
gamma = 0.9
epsilone = 0.1  # 탐험 비율

n_episode = 10000

hole_penalty = -1
counts = []


# ** --------------------
def getReward(next_state):   # goal 지점에 가까이갈수록 reward를 증가시켜 잘 학습되도록 조정
    x = []
    for i in range(64):
            x.append(next_state//8 + next_state%8)

    x[-1] = 1000
    return x[next_state]
# ** --------------------

for episode in range(n_episode + 1):
    s = env.reset() # state initialize
    # env.render()
    epsilone = adaptive_rate(episode, 10000, 0.2)   # epsilon 값을 adaptive_rate 적용

    count = 0
    while True:
        # ε-greedy
        if np.random.uniform() < epsilone:      # exploration(탐험)
            a = env.action_space.sample()
        else:                                   # exploitation
            a = np.argmax(Q_table[s, :])
        next_state, reward, done, info = env.step(a)

        reward = getReward(next_state)  # ****

        Q_table[s, a] = (1-alpha) * Q_table[s,a] + alpha * (reward + gamma + np.max(Q_table[next_state, :]))
        s = next_state  # state update

        # 결과 display
        # env.render()
        # clear_output(wait=True)
        # time.sleep(0.1)

        count += 1
        if done:
            if next_state != 15:
                Q_table[s, a] = hole_penalty
            break

        counts.append(count)
        if len(counts) > 100:
            counts.pop(0)

    if episode % 1000 == 0:
        print(f"Episode: {episode} steps: {count}")


# 학습된 모델을 활용하여 Frozon_Lake 실행
s = env.reset()
while True:
    a = np.argmax(Q_table[s, :])
    s, _, done, _ = env.step(a)

    env.render()

    clear_output(wait=True)
    time.sleep(0.1)
    if done:
        break



# Q Table Map show
actions = ['Left', 'Down', 'Right', 'Up']
print('Actions:', actions)

with np.printoptions(precision=2):
    print(Q_table)

print('\nMap:')
for i in range(8):
    for j in range(8):
        n = j+4*i
        if n in [5, 7, 11, 12, 15]:
            print('■', end=' ')
        else:
            print(actions[np.argmax(Q_table[n])][0], end=' ')

    print()













# 5.2. Multi-dimensional Observation Space ------------------------------------------------------
# 5.2.1. Cartpole
# 연속적인 문제의 경우 이산화를 통해 문제를 해결

import numpy as np
import gym
import matplotlib.pyplot as plt
# %matplotlib inline
import math

env = gym.make('CartPole-v0')
# ?gym.envs.classic_control.cartpole.CartPoleEnv
# A pole is attached by an un-actuated joint to a cart, which moves along
# a frictionless track. The pendulum starts upright, and the goal is to
# prevent it from falling over by increasing and reducing the cart's
# velocity.

n_actions = env.action_space.n

print('States:', env.observation_space)
print('Actions:', n_actions)

print('\nCurrent State:', observation)
print('Random Action:', env.action_space.sample())

print('States_shape:', env.observation_space.shape)
print('State_low:', env.observation_space.low)
print('State_high:', env.observation_space.high)







def discretize(states, minmax, n_bins):
    '''
    states = [Cart Position, Cart Velocity, Pole Angle, Pole Velocity At Tip]
    '''
    
    e = 1e-6
    idx_cart_pos = math.floor(np.interp(x=states[0], xp=minmax[0], fp=(0, n_bins[0]-e)))
    idx_cart_vel = math.floor(np.interp(x=states[1], xp=minmax[1], fp=(0, n_bins[1]-e)))
    idx_pole_ang = math.floor(np.interp(x=states[2], xp=minmax[2], fp=(0, n_bins[2]-e)))
    idx_pole_vel = math.floor(np.interp(x=states[3], xp=minmax[3], fp=(0, n_bins[3]-e)))
    
    indices = [idx_cart_pos, idx_cart_vel, idx_pole_ang, idx_pole_vel]
    index_flatten = np.ravel_multi_index(multi_index=indices, dims=n_bins)
    return index_flatten

def adaptive_rate(t, reach_at, minimum):
    return max(minimum, min(1, 1.0 - np.log10((t+1)/(reach_at//10))))

x = range(1000)
plt.figure(figsize=(6, 4))
plt.plot(x, [adaptive_rate(t, 500, 0.01) for t in x], label='epsilon')
plt.plot(x, [adaptive_rate(t, 800, 0.1) for t in x], label='alpha')
plt.xlabel('episode'); plt.ylabel('value'); plt.legend()
plt.grid()
plt.show()


# Modeling --------------------------
n_bins_pos = 10
n_bins_vel = 10
n_bins_ang = 10
n_bins_anv = 10
n_states = n_bins_pos*n_bins_vel*n_bins_ang*n_bins_anv

n_actions = 2

gamma = 0.99
s_minmax = [(-4, 4), (-2, 2), (-0.4, 0.4), (-2, 2)]
n_bins = [n_bins_pos, n_bins_vel, n_bins_ang, n_bins_anv]

env._max_episode_steps = 250

Q_table = np.random.uniform(0, 1, (n_states, n_actions))


# learning
for episode in range(1001):
    done = False
    state = env.reset()
    
    alpha = adaptive_rate(episode, 200, 0.1)
    epsilon = adaptive_rate(episode, 200, 0.01)      
    
    count = 0  
    while not done:
        count += 1
        
        s = discretize(state, s_minmax, n_bins)    
        
        if np.random.uniform() < epsilon:
            a = env.action_space.sample()
        else:
            a = np.argmax(Q_table[s, :])       

        # next state and reward
        next_state, reward, done, _ = env.step(a) 
                
        if done:
            pass
        else:                                               
            next_s = discretize(next_state, s_minmax, n_bins)
            Q_table[s, a] = (1 - alpha)*Q_table[s, a] + alpha*(reward + gamma*np.max(Q_table[next_s, :]))

        state = next_state
    
    if episode % 100 == 0:
        print("Episode: {} steps: {}".format(episode, count))

        
env.close()









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





from IPython import display

state = env.reset()

img = plt.imshow(renderCart(state), cmap='gray')
while True:
    img.set_data(renderCart(state))
    display.display(plt.gcf())
    display.clear_output(wait=True)
    
    s = discretize(state, s_minmax, n_bins)    
    a = np.argmax(Q_table[s,:])
    
    next_state, _, done, _ = env.step(a)      
    state = next_state
    if done: 
        break

env.close()







# 6. Other Tutorials
# https://www.youtube.com/embed/JgvyzIkgxF0?rel=0
# https://www.youtube.com/embed/lvoHnicueoE?rel=0
# https://www.youtube.com/embed/xWe58WGWmlk?rel=0
# https://www.youtube.com/embed/s5qqjyGiBdc?rel=0
# https://www.youtube.com/embed/2pWv7GOvuf0?rel=0
# https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js










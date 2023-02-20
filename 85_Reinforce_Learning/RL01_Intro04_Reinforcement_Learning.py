import numpy as np
import matplotlib.pyplot as plt

# Company Advertising Episode -----------------------------------------------------------
state1 = ['PU', 'PF', 'RU', 'RF']
state_no1 = [0, 1, 2, 3]

P1 = {
 0: {0: [(1, 0)], 
     1: [(0.5, 0), (0.5, 1)]},
 1: {0: [(0.5, 0), (0.5, 3)], 
     1: [(1, 1)]},
 2: {0: [(0.5, 0), (0.5, 2)], 
     1: [(0.5, 0), (0.5, 1)]},
 3: {0: [(0.5, 2), (0.5, 3)], 
     1: [(1, 1)]},
}
# {from_state : { action: [(probablity, to_state), (probablity, to_state) ]}}

R1 = [0, 0, 10, 10]
g1 = 0.9

action1 = [0, 1]     # 0 : save_money, 1 : advertising



# Bellman Optimal Equation =====================================================================
# value_iteration ------------------------------------------------------------------------
# (step1) : value_function initialize  ****
v1 = np.random.rand(4)
# v1 = np.zeros(4)

# (step2) : v(s) ← R(s) + γ·max{ ∑P(s'|s,a)v(x') }  ****
v_temp1 = v1.copy()
v1_history = [v1]
for _ in range(100):
    for s in state_no1:
        q_0 = sum([prob *v1[state] for prob, state in P1[s][0]])    # action : 0 (save_money)
        q_1 = sum([prob *v1[state] for prob, state in P1[s][1]])    # action : 1 (advertising)
    
        v_temp1[s] = R1[s] + g1 * max(q_0, q_1)
    v1 = v_temp1.copy()
    v1_history.append(v1)
print(v1)   # 각각의 state에 대한 value_function

# v1 history plot
for i in range(4):
    plt.plot(np.array(v1_history)[:,i], label=state1[i])
plt.xscale('symlog')
plt.yscale('log')
plt.legend()
plt.show()


# simplify
v1 = np.random.rand(4)
# v1 = np.zeros(4)

v1_history = [v1]
for _ in range(100):
    for s in state_no1:
        v1[s] = R1[s] + g1 * max([sum([prob *v1[state] for prob, state in P1[s][a]]) for a in action1])
    v1_history.append(v1)
print(v1)   # 각각의 state에 대한 value_function

# v1 history plot
for i in range(4):
    plt.plot(np.array(v1_history)[:,i], label=state1[i])
plt.xscale('symlog')
plt.yscale('log')
plt.legend()
plt.show()


# (step3) optimal policy ****
# v1 : optimal value_function
pi1 = {}

# π(s) = argmax(a) { ∑P(s'|s,a)v(x') }
for s in state_no1:
    q_0 = sum([prob * v1[state] for prob, state in P1[s][0]])    # action : 0 (save_money)
    q_1 = sum([prob * v1[state] for prob, state in P1[s][1]])    # action : 1 (advertising)
    
    pi1[s] = [q_0, q_1]

print(pi1)  # 각각의 state에서 해당 action을 했을때 얻을 수 있는 최대 예상 reward
print([np.argmax(pi1[k]) for k in pi1]) # optimal policy


# simplify
pi1 = {}
for s in state_no1:
    pi1[s] = [sum([prob * v1[state] for prob, state in P1[s][a]]) for a in action1]    # action : 0 (save_money)
print(pi1)
print([np.argmax(pi1[k]) for k in pi1]) # optimal policy





# Temporal Difference(TD) Methods ============================================================
v1      # bellman optimal equation: value_function
pi1     # bellman optimal equation: optimal policy

# TD Method Harsh --------------------------------------------
v2 = np.random.rand(4)
v2_history =[v2]
for _ in range(100):
    v2_next = R1 + g1*v2
    v2 = v2_next.copy()

    v2_history.append(v2_next)

print(v2)
# v2 history plot
for i in range(4):
    plt.plot(np.array(v2_history)[:,i], label=state1[i])
plt.xscale('symlog')
plt.yscale('log')
plt.legend()
plt.show()


pi2 = {}
for s in state_no1:
    pi2[s] = [sum([prob * v2[state] for prob, state in P1[s][a]]) for a in action1]    # action : 0 (save_money)
print(pi2)
print([np.argmax(pi2[k]) for k in pi2]) # optimal policy



# TD Method Smooth --------------------------------------------
alpha = 0.7

v3 = np.random.rand(4)
v3_history =[v3]

for _ in range(100):
    v3_next = R1 + g1*v3
    v3 = (1-alpha) * v3 + alpha * v3_next

    v3_history.append(v3_next)

print(v3)
# v3 history plot
for i in range(4):
    plt.plot(np.array(v3_history)[:,i], label=state1[i])
plt.xscale('symlog')
plt.yscale('log')
plt.legend()
plt.show()

pi3 = {}
for s in state_no1:
    pi3[s] = [sum([prob * v3[state] for prob, state in P1[s][a]]) for a in action1]    # action : 0 (save_money)
print(pi3)
print([np.argmax(pi3[k]) for k in pi3]) # optimal policy


n_states = 10
n_actions = 2

alpha = 0.3
gamma = 0.9

Q_table = np.random.uniform(0, 1, (n_states, n_actions))
Q_table







# ==========================================================================================================

import numpy as np
import gym
import matplotlib.pyplot as plt
# %matplotlib inline
from IPython.display import clear_output
import time
from time import sleep

# 데이비드 실버 교수님의 강화학습 강의를 한국어로 풀어준 동영상입니다
# https://www.youtube.com/playlist?list=PLpRS2w0xWHTcTZyyX8LMmtbcMXpd3s4TU

# https://wordbe.tistory.com/entry/RL-%EA%B0%95%ED%99%94%ED%95%99%EC%8A%B5-part1-policy-value-function

# [ Other Tutorials ]
# https://www.youtube.com/embed/JgvyzIkgxF0?rel=0
# https://www.youtube.com/embed/lvoHnicueoE?rel=0
# https://www.youtube.com/embed/xWe58WGWmlk?rel=0
# https://www.youtube.com/embed/s5qqjyGiBdc?rel=0
# https://www.youtube.com/embed/2pWv7GOvuf0?rel=0
# https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js




# Frozen Lake ------------------------------------------------------------------------------------------------------------------
env = gym.make('FrozenLake-v0')

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












# Taxi ------------------------------------------------------------------------------------------------------------------

env = gym.make('Taxi-v3')

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
state, reward, done, info = env.step(action)
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
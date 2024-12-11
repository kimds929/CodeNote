# Markov Reward Process(MRP)
import numpy as np

# MC Episode -----------------------------------------------------------
state1 = ['s1', 's2', 's3']
P1 = np.array([[0,   0,   1],
              [1/2, 1/2, 0],
              [1/3, 2/3, 0]])

R1 = [10, -1, 1]
g1 = 0.5


# directly
v_d1 = np.linalg.inv(np.eye(3) - g1 * P1) @ R1
print(v_d1)

print(state1[np.argmax(v_d1)])  # result


# fixed point iteration
v_i1 = np.random.rand(3)
v_i1 = np.zeros(3)
v_i1

for _ in range(100):
    v_i1 = R1 + g1 * P1 @ v_i1
print(v_i1)

print(state1[np.argmax(v_i1)]) # result




# Studnet Markov Chain Ep_isode -----------------------------------------------------------
# [C1 C2 C3 Pass Pub FB Sleep] = [0 1 2 3 4 5 6]
state2 = ['C1','C2','C3','Pass','Pub','FB','Sleep']
P2 = np.array([[0, 0.5, 0, 0, 0, 0.5, 0],
               [0, 0, 0.8, 0, 0, 0, 0.2],
               [0, 0, 0, 0.6, 0.4, 0, 0],
               [0, 0, 0, 0, 0, 0, 1],
               [0.2, 0.4, 0.4, 0, 0, 0, 0],
               [0.1, 0, 0, 0, 0, 0.9, 0],
               [0, 0, 0, 0, 0, 0, 1]])
R2 = [-2, -2, -2, 10, 1, -1, 0]
g2 = 0.9


# 특정 episode에 대한 예상 reward -----------------------
episodes = ['C1', 'C2', 'C3', 'Pub', 'C3', 'Pub']

gain = 0        # first_gain

for si, s in enumerate(episodes):
    state_index = state2.index(s)
    print(f'iter : {si} / state : {s} / plus_gain : {g2**si * R2[state_index]}')
    gain = gain + g2**si * R2[state_index]
print(gain)

def gain_total(reward, sinario, gamma):
    G = 0
    for i, state in enumerate(sinario):
        G = G + (gamma**i) * reward[state]
    return G

gain2 = gain_total(reward=R2, sinario=list(map(lambda x: state2.index(x), episodes)), gamma=g2)
print(gain2)



# 각 state별 Gain 기대치 (value_function) ---------------
# directly
v_d2 = np.linalg.inv(np.eye(P2.shape[0]) - g2 * P2) @ R2
print(v_d2)

print(state2[np.argmax(v_d2)])  # result


# fixed point iteration
v_i2 = np.random.rand(P2.shape[0])
# v_i2 = np.zeros(P2.shape[0])

for _ in range(100):
    v_i2 = R2 + g2 * P2 @ v_i2
print(v_i2)

print(state2[np.argmax(v_i2)])  # result

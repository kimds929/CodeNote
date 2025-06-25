import gym
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
# get_ipython().run_line_magic('matplotlib', 'inline')

from time import sleep
from IPython.display import clear_output
from tqdm import tqdm_notebook

import sys
RLpath = r'D:\DataScience\★Git_CodeNote\85_Reinforce_Learning\[POSTECH] ReinforcementLearning'
sys.path.append(RLpath) # add project root to the python path

from code_DP.lib.envs.gridworld import GridworldEnv
# from src.common.gridworld import GridworldEnv # Gridworld Environment

env = GridworldEnv(shape=[3,3])
env.reset()
env._render()



# --------------------------------------------------------------------------------------------------
# 1) Implement Policy Evaluation in python
# - 주어진 Policy에대해 Optimal한 Value Function을 Iterative한방법으로 Evaluation하기.
# π(a|s) → Vπ(s), Qπ(s,a)

result = []
# delta=0
# theta=0.00001
discount_factor = 0.9
policy = np.ones([env.nS, env.nA]) / env.nA
V = np.zeros(env.nS).reshape(-1,1)

# Vπ_new(s) = Σ(a|s) Σp(s',r|s,a)[r + γVπ(s')]
for _ in range(30):
    for s in range(env.nS):     # 모든 state에 대해
        v = 0
        for a, action_prob in enumerate(policy[s]):     # 해당 state의 각 action들에 대해
            for prob, next_state, reward, done in env.P[s][a]:  # 해당 state의 action에서 Probabilty에 따라
                # Vπ_new(s) = Σ(a|s) Σp(s',r|s,a)[r + γVπ(s')]
                v += action_prob * prob * (reward + discount_factor * V[next_state])

        V[s] = v
        # delta = max(delta, np.abs(v - V[s]))
        # if delta < theta:
        #     break
    result.append(V.copy())

V_np = np.squeeze(np.array(result))

plt.figure(figsize=(16,7))
for i in range(env.nS):
    plt.plot(V_np[:,i], label=str(i))
plt.legend()
plt.show()



# # State_value_function ------------------------------------------------------------------
# delta=0
# theta=0.00001
# discount_factor = 0.9
# policy = np.ones([env.nS, env.nA]) / env.nA
# V = np.zeros(env.nS).reshape(-1,1)

# # Vπ_new(s) = Σ(a|s) Σp(s',r|s,a)[r + γVπ(s')]
# while True:
#     for s in range(env.nS):
#         v = 0
#         for a, action_prob in enumerate(policy[s]):
#             for prob, next_state, reward, done in env.P[s][a]:
#                 v += action_prob * prob * (reward + discount_factor * V[next_state])

#         V[s] = v
#     delta = max(delta, np.abs(v - V[s]))
#     if delta < theta:
#         break
# # ----------------------------------------------------------------------------------------








# --------------------------------------------------------------------------------------------------
# 2) Implement policy iteration in python
# - 단계에서만든Policy Evaluation 알고리즘을 이용.
# - 다음두가지 단계를반복하여 Optimal한Policy와Value Function찾기.
#   Vπ_new(s) = Σ(a|s) Σp(s',r|s,a)[r + γVπ(s')]
#    π_new(s) = argmax Σp(s', r|s,a)[r + γVπ(s')]
#        * Qπ(s,a) = Σp(s', r|s,a)[r + γVπ(s')]

# π0 → V1 → Q1 → π1 → .... → π* → V* → Q*

delta=0
theta=0.00001
discount_factor = 0.9
policy = np.ones([env.nS, env.nA]) / env.nA
V = np.zeros(env.nS).reshape(-1,1)

n = 0
while True:
    # State_value_function: Vπ_new(s) = Σ(a|s) Σp(s',r|s,a)[r + γVπ(s')] -------------------
    # input: V, policy
    # output: V_new
    print(f'step: {n+1} | V_update', end='')
    while True:
        for s in range(env.nS):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + discount_factor * V[next_state])

            V[s] = v
        delta = max(delta, np.abs(v - V[s]))
        if delta < theta:
            break
    # ----------------------------------------------------------------------------------------

    policy_stable = True
    print('| Q_update | Pi_update', end='')
    for s in range(env.nS):
        chosen_a = np.argmax(policy[s])     # 해당 state에서 action확률이 가장 높은 정책 action

        # Action_value_function: Qπ_new(s) = R(s) + γ·∑P(s'|s,a)v(x') -----------------------
        # input: V
        # output: Q
        Q = np.zeros(env.nA)        # 각 action별로
        for a in range(env.nA):     # 해당 state의 action별로 q값을 계산
            for prob, next_state, reward, done in env.P[s][a]:
                Q[a] +=  prob * (reward + discount_factor * V[next_state])
        # -----------------------------------------------------------------------------------

        best_a = np.argmax(Q)
        if chosen_a != best_a:      # Q_function 계산 전 / 후 값이 같을때 종료
            policy_stable = False

        policy[s] = np.eye(env.nA)[best_a]      # policy update
    
    print(f' | stable: {policy_stable}')
    n += 1
    if policy_stable :
        break
    
print('[ policy ]\n', policy)
print('[ value_function ]\n', V)






# -----------------------------------------------------------------------------------------------------------
# 3) Implement Value iteration in python
# 전 단계Policy Iteration알고리즘에서 나누어진두가지 단계를조금더simple하게 변형하고 합친 간단한 Iterative 알고리즘.
# -----------------------------------------------------------------------------------------------------------

# V* = max Q*
# V*(from Q) = max R(s) + γ·∑P(s'|s,a)v(x')
# V* → Q* → π*



delta=0
theta=0.00001
discount_factor = 0.9
V = np.zeros(env.nS).reshape(-1,1)
n = 0
while True:
    for s in range(env.nS):
        # Action_value_function: Qπ_new(s) = R(s) + γ·∑P(s'|s,a)v(x') -----------------------
        # input: V
        # output: Q
        Q = np.zeros(env.nA)        # 각 action별로
        for a in range(env.nA):     # 해당 state의 action별로 q값을 계산
            for prob, next_state, reward, done in env.P[s][a]:
                Q[a] +=  prob * (reward + discount_factor * V[next_state])
        # -----------------------------------------------------------------------------------
        best_action_value = np.max(Q)

        delta = np.max([delta, np.abs(best_action_value - V[s])])
        V[s] = best_action_value
    if delta < theta:
        break
    n += 1
    if n > 100:
        break

for s in range(env.nS):
    # Action_value_function: Qπ_new(s) = R(s) + γ·∑P(s'|s,a)v(x') -----------------------
    # input: V
    # output: Q
    Q = np.zeros(env.nA)        # 각 action별로
    for a in range(env.nA):     # 해당 state의 action별로 q값을 계산
        for prob, next_state, reward, done in env.P[s][a]:
            Q[a] +=  prob * (reward + discount_factor * V[next_state])
    # -----------------------------------------------------------------------------------
    policy[s][np.argmax(Q)] = 1
print('[ policy ]\n', policy)
print('[ value_function ]\n', V)

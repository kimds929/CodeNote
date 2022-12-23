# Markov Reward Process(MRP)
import numpy as np

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


P1[0][0]
# [(1, 0)] : state 0(PU) 에서 0의 action(save_money) 했을경우, 1의 확률로 state 0으로 간다.

P1[0][1]    
# [(0.5, 0), (0.5, 1)] : state 0(PU) 에서 0의 action(advertising) 했을경우, 
#                        0.5의 확률로 state 0으로 가고, 0.5의 확률로 state 1로 간다.


# value_function이 주어졌을때 → MRP문제가 됨

# state =2, action = 0 일때, ----------------------------------------------
# q(s) = R(s) + γ·∑P(s'|s,a)v(x')
# ∑P(s'|s,a)v(x')

v1 = np.array([0, 0, 10, 10])      # value_function이 해당과 같이 주어졌다고 가정하면,
P1[2][0]

# next_expectation_values = ∑P(s'|s,a)v(x')
nev1 = 0
for prob, state in P1[2][0]:
    nev1 = nev1 + prob * v1[state]
print(nev1)

# simplify
sum([prob *v1[state] for prob, state in P1[2][0]])

# q(s) = R(s) + γ·∑P(s'|s,a)v(x')
q_s1 = R1[2] + g1 * sum([prob *v1[state] for prob, state in P1[2][0]])
print(q_s1)




# Bellman Expectaion Equation =====================================================================
# (step1) : value_function initialize  ****
v_exp1 = np.zeros(4)
v_exp_temp1 = v_exp1.copy()

# (step2) : 
# v(s) ← ∑π(a|s)·[R(s) + γ· ∑P(s'|s,a)v(x')]  ****
#       v(s) = ∑π(a|s)·q(s|a)
#       q(s,a) = R(s) + γ· ∑P(s'|s,a)v(x')

for _ in range(100):
    for s in state_no1:
        q_0 = sum([prob * v_exp1[state] for prob, state in P1[s][0]])    # action : 0 (save_money)
        q_1 = sum([prob * v_exp1[state] for prob, state in P1[s][1]])    # action : 1 (advertising)

        v_exp_temp1[s] = R1[s] + g1 * np.mean([q_0, q_1])
    v_exp1 = v_exp_temp1.copy()
print(v_exp1)


# (step3) expectaion of policy ****
pi_exp1 = {}
for s in state_no1:
    q_0 = sum([prob * v_exp1[state] for prob, state in P1[s][0]])    # action : 0 (save_money)
    q_1 = sum([prob * v_exp1[state] for prob, state in P1[s][1]])    # action : 1 (advertising)
    
    pi_exp1[s] = [q_0, q_1]

print(pi_exp1)  # 각각의 state에서 해당 action을 했을때 얻을 수 있는 reward 기대치 (항상 최선의 선택을 하지는 않음)
print([np.mean(pi1[k]) for k in pi1]) # optimal policy







# Bellman Optimal Equation =====================================================================
# value_iteration ------------------------------------------------------------------------
# (step1) : value_function initialize  ****
# v1 = np.random.rand(4)
v1 = np.zeros(4)

# (step2) : v(s) ← R(s) + γ·max{ ∑P(s'|s,a)v(x') }  ****
v_temp1 = v1.copy()
for _ in range(100):
    for s in state_no1:
        q_0 = sum([prob *v1[state] for prob, state in P1[s][0]])    # action : 0 (save_money)
        q_1 = sum([prob *v1[state] for prob, state in P1[s][1]])    # action : 1 (advertising)
    
        v_temp1[s] = R1[s] + g1 * max(q_0, q_1)
    v1 = v_temp1.copy()
print(v1)   # 각각의 state에 대한 value_function


# simplify
# v1 = np.random.rand(4)
v1 = np.zeros(4)
for _ in range(100):
    for s in state_no1:
        v1[s] = R1[s] + g1 * max([sum([prob *v1[state] for prob, state in P1[s][a]]) for a in action1])
print(v1)   # 각각의 state에 대한 value_function


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





# policy iteration --------------------------------------------------------------------------------
# (step1) initialize policy
pi2 = np.random.randint(0, 2, size=4)
pi2

# (step2) compute a value_function of policy(π) and update policy, v_π
def calc_value_function(policy):
    v2 = np.random.rand(4)
    v2_temp = v2.copy()
    for _v in range(100):
        for s in state_no1:
            # q(s) = R(s) + γ·∑P(s'|s,a)v(x')
            v2_temp[s] = R1[s] + g1 * np.mean([prob * v2[state] for prob, state in P1[s][policy[s]]])
        v2 = v2_temp.copy()
    return v2


v2 = calc_value_function(pi2)
v2

# (step3) update policy(π) to be greedy policy with respect to value_funcion
for _ in range(100):
    for s in state_no1:
        pi2[s] = np.argmax([sum([prob * v2[state] for prob, state in P1[s][a]]) for a in action1])
    v2 = calc_value_function(pi2)
print(pi2)








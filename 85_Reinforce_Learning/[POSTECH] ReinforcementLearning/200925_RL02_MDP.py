import gym
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
# get_ipython().run_line_magic('matplotlib', 'inline')

from time import sleep
from IPython.display import clear_output
from tqdm import tqdm_notebook

import sys
RLpath = r'C:\Users\Admin\Desktop\DataScience\Reference8) POSTECH_AI_Education\Postech_AI 6) Reinforce_Learning\강의) 실습자료'
sys.path.append(RLpath) # add project root to the python path

from code_DP.lib.envs.gridworld import GridworldEnv
# from src.common.gridworld import GridworldEnv # Gridworld Environment

# !unzip code_DP.zip -d code_DP      # 압축 해제


# [ Today Goals ] ----------------------------------------------------------------------------
# 1) Implement Policy Evaluation in python
# - 주어진 Policy에대해 Optimal한 Value Function을 Iterative한방법으로 Evaluation하기.
# π(a|s) → Vπ(s), Qπ(s,a)


# 2) Implement policy iteration in python
# - 단계에서만든Policy Evaluation 알고리즘을 이용.
# - 다음두가지 단계를반복하여 Optimal한Policy와Value Function찾기.
#   Vπ_new(s) = Σ(a|s) Σp(s',r|s,a)[r + γVπ(s')]
#    π_new(s) = argmax Σp(s', r|s,a)[r + γVπ(s')]
#        * Qπ(s,a) = Σp(s', r|s,a)[r + γVπ(s')]


# 3) Implement Value iteration in python
# 전 단계Policy Iteration알고리즘에서 나누어진두가지 단계를조금더simple하게 변형하고 합친 간단한 Iterative 알고리즘.

# ----------------------------------------------------------------------------------------------


env_grid = GridworldEnv()
env_grid.reset()
env_grid.render()

env_grid.observation_space
env_grid.action_space

env_grid.P
env_grid.P[6]
# env_grid.P[s][a] (prob, next_state, reward, is_terminal)

# (Actions) 0: up, 1: right, 2: left, 3: down
# (Reward) -1 at each step until reaching terminal step



# Environment Go
env_grid.reset()
obs_log_grid = []
cumulate_reward_grid = []
do_grid = []

for _ in range(20):
    # Render current state : Visualize
    print(f'step: {_+1}')
    env_grid.render()        # 현상태
    
    # (your agent here)
    # choose random action
    action = env_grid.action_space.sample() # action_sample
    
    ## output : (next_state, reward, is_terminal, debug_info)
    observation, reward, done, info = env_grid.step(action)

    # result_save
    obs_log_grid.append(observation)
    if cumulate_reward_grid:
        cumulate_reward_grid.append(cumulate_reward_grid[-1] + reward)
    else:
        cumulate_reward_grid.append(reward)
    do_grid.append(done)

    # if is_terminal == True, then break for loop
    if done:
        clear_output(wait=True)    
        print(f'-- end (steps: {_+1}) -- ')
        env_grid.render()
        break

    clear_output(wait=True)
    sleep(0.2)

env_grid.close()













# --------------------------------------------------------------------------------------------------------------------------------------
# 1) Implement Policy Evaluation in python
# - 주어진 Policy에대해 Optimal한 Value Function을 Iterative한방법으로 Evaluation하기.
# π(a|s) → Vπ(s), Qπ(s,a)


# env.P = {state: {action:[prob, next_state, reward, done],
#                 action:[prob, next_state, reward, done],
#                 ...                                     },
#         state: {action:[prob, next_state, reward, done],
#                 action:[prob, next_state, reward, done],
#                 ...                                     },
#         }

# policy eval 
def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    iter=0
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        iter+=1
        delta = 0
        print(iter," iteration!")
        #각 state마다 기존 값 V와 새로 계산되는 v값을 비교한다(수렴할때까지)
        for s in range(env.nS):
            v = 0
            # Vπ_new(s) = Σ(a|s) Σp(s',r|s,a)[r + γVπ(s')]
            #빈칸1_for문 작성_해당 state에서 취할 수 있는 각 액션마다(policy 기반)
            for a, action_prob in enumerate(policy[s]):
                # 빈칸2_for문 작성_해당 state의 action마다 P[s][a]를 가져와서
                for prob, next_state, reward, done in env.P[s][a]:
                    #빈칸3_실습pdf의 빨간 빈칸식을 v에 더해주기
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
                    
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
        clear_output(wait=True)
    return np.array(V)




#빈칸4_초기  random한 policy 만들어주기
random_policy = np.ones([env_grid.nS, env_grid.nA]) / env_grid.nA
random_policy

#빈칸5_policy_eval 함수 실행해서 v에 대입하기
v = policy_eval(policy=random_policy, env=env_grid)
print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env_grid.shape))
print("")


# Test: Make sure the evaluated policy is what we expected
expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
expected_v.reshape(env_grid.shape)
np.testing.assert_array_almost_equal(v, expected_v, decimal=3)


















# --------------------------------------------------------------------------------------------------------------------------------------
# 2) Implement policy iteration in python
# - 단계에서만든Policy Evaluation 알고리즘을 이용.
# - 다음두가지 단계를반복하여 Optimal한Policy와Value Function찾기.
#   Vπ_new(s) = Σ(a|s) Σp(s',r|s,a)[r + γVπ(s')]
#    π_new(s) = argmax Σp(s', r|s,a)[r + γVπ(s')]
#        * Qπ(s,a) = Σp(s', r|s,a)[r + γVπ(s')]

def policy_improvement(env, policy_eval=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """

    # q(s,a)를 구하는 부분: Σp(s', r|s,a)[r + γVπ(s')]
    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.
        
        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS
        
        Returns:
            A vector of length env.nA containing the expected value(Q) of each action.
        """
        #빈칸1_실습pdf의 첫번째 빨간 빈칸식
        #현재 state와 value function V가 주어졌을때,  각 액션의 기대값 구하기
        Q = np.zeros(env.nA)

        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                Q[a] +=  prob * (reward + discount_factor * V[next_state])
        return Q

    
    #policy 초기값을 설정
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:     #policy_stable이 True가 될 때까지 계속 반복
        # 위에서 만든  policy_eval함수로 V 계산하기
        V = policy_eval(policy, env, discount_factor)
        
        # Policy에 변화가 생기면 False로 바꾸어 while문 다시 반복 시행할 예정
        policy_stable = True
        
        # 각 state마다
        for s in range(env.nS):
            #빈칸2_해당 s state에서 policy상 가장 best action을 chosen_a에 저장
            chosen_a = np.argmax(policy[s])
            
            #빈칸3_위의 one_step_lookahead함수를 이용해서 가장 best action을 best_a에 저장
            best_a = np.argmax(one_step_lookahead(s, V))
            
            #chosen_a가 best_a가 아니면...
            if chosen_a != best_a:
                policy_stable = False
            #빈칸4_policy를 새로운 best_a에 맞게 update하기
            #(Hint : 우리가 원하는  best action이 3번째면, policy[a]에 [0,0,1,0]) 대입하기
            policy[s] = np.eye(env.nA)[best_a]
            
        
        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, V





policy, v = policy_improvement(env_grid)
print("Policy Probability Distribution:")
print(policy)
print("")

Grid_policy=np.reshape(np.argmax(policy, axis=1), env_grid.shape)
print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(Grid_policy)
print("")

# value function by ideal policy
print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env_grid.shape))
print("")









# Environment Go
env_grid.reset()
obs_log_grid = []
cumulate_reward_grid = []
do_grid = []

for _ in range(20):
    # Render current state : Visualize
    print(f'step: {_+1}')
    env_grid.render()        # 현상태
    
    # (your agent here)
    # choose random action
    # action = env_grid.action_space.sample() # action_sample
    action = Grid_policy[int(env_grid.s/4)][env_grid.s%4]
    
    ## output : (next_state, reward, is_terminal, debug_info)
    observation, reward, done, info = env_grid.step(action)

    # result_save
    obs_log_grid.append(observation)
    if cumulate_reward_grid:
        cumulate_reward_grid.append(cumulate_reward_grid[-1] + reward)
    else:
        cumulate_reward_grid.append(reward)
    do_grid.append(done)

    # if is_terminal == True, then break for loop
    if done:
        clear_output(wait=True)    
        print(f'-- end (steps: {_+1}) -- ')
        env_grid.render()
        break

    clear_output(wait=True)
    sleep(1)

env_grid.close()






# -----------------------------------------------------------------------------------------------------------
# 3) Implement Value iteration in python
# 전 단계Policy Iteration알고리즘에서 나누어진두가지 단계를조금더simple하게 변형하고 합친 간단한 Iterative 알고리즘.
# -----------------------------------------------------------------------------------------------------------
env_grid.P[3][2]



def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """
    

    #앞의 step2의 one_step_lookahead함수와 완전히 같은 함수!
    # q(s,a)를 구하는 부분: Σp(s', r|s,a)[r + γVπ(s')]
    def one_step_lookahead(state, V):
        # q(s) = R(s) + γ·∑P(s'|s,a)v(x')
        """
        Helper function to calculate the value for all action in a given state.
        
        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS
        
        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A
    
    #Value function 초기값 설정
    V = np.zeros(env.nS)
    
    while True:
        delta = 0
        # 각 state마다
        for s in range(env.nS):
            # 빈칸1_해당 state s 상태에서 one_step_lookahead를 이용하여 최대값을 best_action_value에 대입
            # (실습pdf의 빨간 첫번째 빈칸식)
            A = one_step_lookahead(state=s, V=V)    # A = [Q(s,0), Q(s,1), Q(s,2), Q(s,3)]
            
            # 여기서 이제 가장 큰 값을 골라 best_action_value에 대입
            best_action_value = np.max(A)
            
            # 빈칸2_기존 V와 새로운 V값의 차이로 delta값 구하기
            delta = np.max([delta, np.abs(best_action_value - V[s])])

            # Update the value function.
            V[s] = best_action_value
        # Check if we can stop 
        if delta < theta:
            break
    
    #policy 만들기
    policy = np.zeros([env.nS, env.nA])
    
    #빈칸3__실습pdf의 빨간 두번째 빈칸_위에서 찾은 optimal한 V를 이용하여 policy 찾기
    #(힌트 : one_step_lookahead이용)
    for s in range(env.nS):
        A = one_step_lookahead(state=s, V=V)    # A = [Q(s,0), Q(s,1), Q(s,2), Q(s,3)]
        policy[s][np.argmax(A)] = 1
        
    #이제 optimal한 policy와 value function return하기
    return policy, V


env_grid.reset()

policy, v = value_iteration(env_grid)
print("Policy Probability Distribution:")
print(policy)
print("")

Grid_policy=np.reshape(np.argmax(policy, axis=1), env_grid.shape)
print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(Grid_policy)
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env_grid.shape))
print("")


expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)





env_grid.reset()

obs_log = []
cumulative_reward = []
do = []

for t in range(200):
    env_grid.render()

    # your agent here
    action = Grid_policy[int(env_grid.s/4)][env_grid.s%4]

    ## output : (next_state, reward, is_terminal, debug_info)
    observation, reward, done, info = env_grid.step(action)
    
    obs_log.append(observation)
    if(t==0):
        cumulative_reward.append(reward)
    else:
        cumulative_reward.append(reward + cumulative_reward[-1])

    ## if is_terminal == True, then break for loop
    if(done):
        clear_output(wait = True)
        sleep(0.2)
        print("Episode finished after {} timesteps".format(t + 1))
        sleep(0.05)
        env_grid.render()
        break
        
    sleep(0.2)
    clear_output(wait = True)
        
env_grid.close()

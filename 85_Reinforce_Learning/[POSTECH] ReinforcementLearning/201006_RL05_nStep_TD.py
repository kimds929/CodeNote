import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import display
# get_ipython().run_line_magic('matplotlib', 'inline')

import time
from time import sleep
from IPython.display import clear_output
from tqdm import tqdm_notebook

# from collections import defaultdict
import sys
RLpath = r'D:\Python\★★Python_POSTECH_AI\Postech_AI 6) Reinforce_Learning\강의) 실습자료'
sys.path.append(RLpath) # add project root to the python path


from code_TD.lib.envs.cliff_walking import CliffWalkingEnv
from code_TD.lib.envs.windy_gridworld import WindyGridworldEnv
# from src.common.gridworld import GridworldEnv # Gridworld Environment

from collections import defaultdict, namedtuple
import itertools



# n-step TD Prediction ---------------------------------------------------------------------------------
env = CliffWalkingEnv()


def random_policy(state):
    action_probs = np.array([0.25,0.25,0.25,0.25])
    return np.random.choice(np.arange(len(action_probs)), p=action_probs)


def nstep_prediction(policy, env, num_episodes, n_step = 2, alpha = 0.5, discount_factor=0.5):    
    # The final value function
    V = defaultdict(float)

    for i_episode in tqdm_notebook(range(num_episodes)):

        state = env.reset() ### S_0

        # n번후에 업데이트 해야하므로 history를 저장해야함
        T = np.inf
        updating_time = 0
        state_history = [] ### S_0, S_1,...
        state_history.append(state)
        reward_history = [] ### R_1, R_2,...

        for t in itertools.count():
            if t< T:
                ###action_probs = policy(state)
                # action_probs = np.array([0.25,0.25,0.25,0.25])
                # action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                action = random_policy(state)

                # S_1
                next_state, reward, done, _ = env.step(action)
                reward_history.append(reward)       # R_1
                state_history.append(next_state)

                if done:
                    T=t+1

            updating_time = t - n_step + 1
            
            if updating_time >= 0:
                G = 0

                for i in range(updating_time + 1, int(np.min([updating_time + n_step, T])) + 1):
                    G+= (discount_factor** (i-updating_time - 1)) * reward_history[i-1]
                
                if updating_time +n_step < T:
                    G += ( discount_factor**n_step ) * V[state_history[updating_time + n_step]]

                V[state_history[updating_time]] += alpha * (G - V[state_history[updating_time]])
            if updating_time == T-1:
                break
            state = next_state
    return V


def ViewValue(V):
    value_table = np.zeros(48)
    for key in V.keys():
        value_table[key] = V[key]
    value_table = value_table.reshape(4,12)
    value_table = np.around(value_table,2)
    
    return pd.DataFrame(value_table)



V = nstep_prediction(random_policy, env, 5000, n_step = 4)
ViewValue(V).T


V = nstep_prediction(random_policy, env, 5000, n_step = 2)
ViewValue(V)















# n-step SARSA ---------------------------------------------------------------------------------
env = WindyGridworldEnv()


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action. Float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn



def nstep_sarsa(env, num_episodes, n_step = 2, alpha = 0.5, discount_factor=0.5, epsilon=0.1):    

    
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    EpisodeStats = namedtuple('Stats', ['episode_lengths', 'episode_rewards'])
                              
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    
    for i_episode in tqdm_notebook(range(num_episodes)):
        

        state = env.reset() ### S_0
        
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs) ### A_0
        
        T = np.inf
        updating_time = 0
        state_history = [] ### S_0, S_1,...
        state_history.append(state)
        reward_history = [] ### R_1, R_2,...
        action_history = [] ### A_0, A_1,...
        action_history.append(action)
        
        for t in itertools.count():
            if t< T:
                
                next_state, reward, done, _ = env.step(action)
                reward_history.append(reward)
                state_history.append(next_state)
                state = next_state

                if done: T=t+1
                else: 
                    action = np.random.choice(np.arange(len(policy(next_state))), p=policy(next_state))
                    action_history.append(action)
            updating_time = t-n_step + 1
            
            
            if updating_time >= 0:
                G = 0

                for i in range(updating_time + 1, int(np.min([updating_time + n_step, T])) + 1):
                    G+= (discount_factor** (i-updating_time - 1)) * reward_history[i-1]
                
                if updating_time +n_step < T:
                    G += ( discount_factor**n_step ) * Q[state_history[updating_time + n_step]][action_history[updating_time + n_step]]

                Q[state_history[updating_time]][action_history[updating_time]] += alpha * (G - Q[state_history[updating_time]][action_history[updating_time]])
                policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
            
            if updating_time == T-1:
                break

        stats.episode_rewards[i_episode] = np.sum(reward_history)
        stats.episode_lengths[i_episode] = t
                
    return Q, stats



Q, stats = nstep_sarsa(env, 1000, 3)


def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)

    return fig1, fig2, fig3


plot_episode_stats(stats)



def greedy_policy(observation):
    best_action = np.argmax(Q[observation])
    return best_action





env.reset()

obs_log =[]
cumulative_reward = []

for t in range(200):
    env.render()
    
    # your agent here
    # Q로부터 policy \pi_*를 얻어 action을 얻어봅시다.
    
    #action = np.random.choice(np.arange(len(optimal_policy(env.s))), p=optimal_policy(env.s))
    action = greedy_policy(env.s)
    ## output: (next_state, reward, is_terminal, debug_info)
    next_state, reward, done, _ = env.step(action)
    obs_log.append(next_state)
    cumulative_reward.append(reward)
    
    
    
    ## if is_terminal == True, then break for loop
    if done:
        
        env.render()
        print("Episode finished after {} timestpes".format(t+1))
        break
    
env.close

































# ============================================================================
# env = WindyGridworldEnv()
env = CliffWalkingEnv()
# Q_table = np.random.uniform(0, 1, (env.nS, env.nA))
# Q_table.shape     # (state, action)
Q_table = defaultdict(lambda: np.random.rand(env.nA))

n_step = 3

gamma = 1
alpha = 0.5
epsilon = 0.1
n_episode = 200
# method = 'SARSA'
# method = 'ExpSARSA'
method = 'Qlearning'

result = {}
result['n_episodes'] = []
result['total_reward'] = []


# ( greedy function ) ----------------------------------------------
# def epsilon_greedy(Q, state, epsilon):
#     if np.random.rand() < epsilon:
#         a = env.action_space.sample()
#     else:
#         a = np.argmax(Q[state, :])       # 각 state별 최상의 선택
#     return a

def epsilon_greedy_policy(Q, state, epsilon, nA):
    A = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[state])
    A[best_action] += (1.0 - epsilon)
    return A
# --------------------------------------------------------------




# s = env.s     # 현재위치
for episode in range(n_episode + 1):
    s = env.reset() # Environment 초기화, (return) 초기 위치

    # Epsilon Greedy ****
    # a = epsilon_greedy(Q=Q_table, state=s, epsilon=epsilon)
    a_probs = epsilon_greedy_policy(Q=Q_table, state=s, nA=env.nA, epsilon=epsilon)
    a = np.random.choice(np.arange(env.nA), p=a_probs)

    T = np.inf
    updating_time = 0

    state_history = [] ### S_0, S_1,...
    state_history.append(s)
    reward_history = [] ### R_1, R_2,...
    action_history = [] ### A_0, A_1,...
    action_history.append(a)

    count = 0
    total_reward = 0
    while True:
        if count < T:
            next_state, reward, done, info = env.step(a)   # step

            reward_history.append(reward)
            state_history.append(next_state)
            s = next_state

            if done:
                T = count + 1
            else: 
                next_a_probs = epsilon_greedy_policy(Q=Q_table, state=next_state, nA=env.nA, epsilon=epsilon)
                a = np.random.choice(np.arange(env.nA), p=next_a_probs)
                action_history.append(a)

        updating_time = count - n_step + 1
        if updating_time >= 0:      # Update Time이 도래하면,
            G = 0
            for i in range(updating_time + 1, int(np.min([updating_time + n_step, T])) + 1):
                G+= (gamma** (i-updating_time - 1)) * reward_history[i-1]
            
            if updating_time +n_step < T:
                G += ( gamma**n_step ) * Q_table[state_history[updating_time + n_step]][action_history[updating_time + n_step]]

            Q_table[state_history[updating_time]][action_history[updating_time]] += alpha * (G - Q_table[state_history[updating_time]][action_history[updating_time]])
        
        if updating_time == T-1:
            break
        count += 1
        total_reward += reward
    
    # result save in dict
    result['n_episodes'].append(count)
    result['total_reward'].append(total_reward)

Q_table
# result      # 

# # Learning ****
# if method.lower() == 'sarsa':
#     Q_table[s][a] = (1-alpha) * Q_table[s][a] + alpha * (reward + gamma * Q_table[next_state][next_a])    # SARSA
# elif method.lower() == 'expsarsa':
#     Q_table[s][a] = (1-alpha) * Q_table[s][a] + alpha * (reward + gamma * next_a_probs @ Q_table[next_state])    # Expectation_SARSA
# elif method.lower() == 'qlearning':
#     Q_table[s][a] = (1-alpha) * Q_table[s][a] + alpha * (reward + gamma * np.max(Q_table[next_state]))    # Q-Learning

# # s = next_state
# for i in range(updating_time + 1, int(np.min([updating_time + n_step, T])) + 1):

# updating_time + 1
# int(np.min([updating_time + n_step, T])) + 1


# -----------------------------------------------------------------------
# 결과 plotting
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.title(method)
plt.plot(result['n_episodes'])
plt.subplot(122)
plt.title(method)
plt.plot(result['total_reward'])




# -----------------------------------------------------------------------
# 학습된 모델을 활용하여 Environment 실행 ***
s = env.reset()
while True:
    env.render()
    a = np.argmax(Q_table[s])
    s, _, done, _ = env.step(a)

    clear_output(wait=True)
    time.sleep(0.2)
    if done:
        env.render()
        break
env.close()

# ============================================================================

Q_table


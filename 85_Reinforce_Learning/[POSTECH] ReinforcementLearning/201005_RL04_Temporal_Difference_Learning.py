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
# matplotlib.style.use('ggplot')

# Cliff_Walking ----------------------------------------------------------
env = CliffWalkingEnv()

print(env.reset())
env.render()

# action : 0 위쪽, 1 오른쪽, 2 아래, 3 왼쪽
print(env.step(0))      # (return) state, reward, is_terminal, information
env.render()

print(env.step(1))
env.render()

print(env.step(1))
env.render()

print(env.step(2))
env.render()


# Windy_Grid_World ----------------------------------------------------------
env = WindyGridworldEnv()

print(env.reset())
env.render()

print(env.step(1))
env.render()

print(env.step(1))
env.render()

print(env.step(1))
env.render()

print(env.step(2))
env.render()

print(env.step(1))
env.render()

print(env.step(1))
env.render()











# SARSA Windy_Grid_World =======================================================================
env_windy = WindyGridworldEnv()

print(env_windy.reset())
env_windy.render()

Q_windy = defaultdict(lambda: np.zeros(env_windy.nA))


# make_epsilon_greedy_policy
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


# ε-greedy
# π(a|s) = { ε/m + 1 - ε    if a* = argmax Q(s,a)
#          { ε/m              otherwise

# policy_func = make_epsilon_greedy_policy(Q=Q_windy, epsilon=0.1, nA=env_windy.nA)
# policy_func(1)        # array([0.925, 0.025, 0.025, 0.025]) 
# Q_windy[3]
# Q_windy




# sarsa
def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.
    
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    #The final action-value function
    ## 각자 사용할 Q table을 정의해 줍시다.
    # 예시 : A nested dictionary that maps state -> (action-> action-value)
    ## 혹은 simply 2d numpy array
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    ## stats 변수를 정의하고, 여기에 매 episode 길이와 episode reward를 기록합시다.
    EpisodeStats = namedtuple('Stats', ['episode_lengths', 'episode_rewards'])
                              
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    

    # 매 Episode 마다 ..
      ## initialize가 필요합니다.
      ## 1. environment를 reset해주고 initial state를 얻습니다.
        # The policy we're following
        ## epsilon-greedy policy from Q로부터 policy를 얻습니다.
      ## 2. 이를 앞서 정의한 policy에 넣어 probability vectgor를 얻고,
      ## 3. 위에서 얻은 probability vector에 기반하여 action을 sampling합니다. REFER: np.random.choice
    
    for i_episode in tqdm_notebook(range(num_episodes)):
        policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
        
        # Reset the environment and pick the first action
        state = env.reset()
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        
        # One step in the environment
        # 매 time step마다..
        # a. 주어진 action으로 step을 진행하고,
        # b. 위의 initialize한 과정과 비슷하게 앞서 정의한 policy에 넣어 probability vector를 얻은 뒤
        # c. next_action 및 next_state를 얻습니다.
        # d. stats 변수에 episode reward와 episode length에 관련된 사항들을 기록합시다.
        # e. TD error를 계산하고, SARSA update 식을 토대로 Q를 update해 줍시다.
        # f. 만약 현재 environment가 terminal이라면, for문을 break해 주고,
        # g. 아니라면 action과 state를 update해줍시다.
        for t in itertools.count():
            # Take a step
            next_state, reward, done, _ = env.step(action)
            
            # Pick the next action
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
            
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # TD Update
            td_target = reward + discount_factor * Q[next_state][next_action]
            ### EXPECTED Sarsa : td_target = reward + discount_factor * np.dot(policy(next_state), Q[next_state])
            
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
    
            if done:
                break
                
            action = next_action
            state = next_state        
    
    return Q, stats




Q_windy, stats_windy = sarsa(env_windy, 200)
stats_windy




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




plot_episode_stats(stats_windy)













# Q-Learning Cliff =======================================================================
env_cliff = CliffWalkingEnv()

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



def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.
    
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    EpisodeStats = namedtuple('Stats', ['episode_lengths', 'episode_rewards'])
                              
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))


    for i_episode in tqdm_notebook(range(num_episodes)):
        
        policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
        
        # Reset the environment and pick the first action
        state = env.reset()
        
        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():
            
            # Take a step
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # TD Update
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
                
            if done:
                break
                
            state = next_state
    
    return Q, stats




Q_cliff, stats_cliff = q_learning(env_cliff, 500)

plot_episode_stats(stats_cliff)








env.reset()

obs_log =[]
cumulative_reward = []

for t in range(200):
    env.render()
    
    # your agent here
    # Q로부터 policy \pi_*를 얻어 action을 얻어봅시다.
    
    action = np.argmax(Q_cliff[env.s])
    
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

    count = 0
    total_reward = 0
    while True:
        # Epsilon Greedy ****
        # a = epsilon_greedy(Q=Q_table, state=s, epsilon=epsilon)
        a_probs = epsilon_greedy_policy(Q=Q_table, state=s, nA=env.nA, epsilon=epsilon)
        a = np.random.choice(np.arange(env.nA), p=a_probs)

        next_state, reward, done, info = env.step(a)   # step
        # next_a = epsilon_greedy(Q=Q_table, state=next_state, epsilon=epsilon)
        next_a_probs = epsilon_greedy_policy(Q=Q_table, state=next_state, nA=env.nA, epsilon=epsilon)
        next_a = np.random.choice(np.arange(env.nA), p=next_a_probs)

        # Learning ****
        if method.lower() == 'sarsa':
            Q_table[s][a] = (1-alpha) * Q_table[s][a] + alpha * (reward + gamma * Q_table[next_state][next_a])    # SARSA
        elif method.lower() == 'expsarsa':
            Q_table[s][a] = (1-alpha) * Q_table[s][a] + alpha * (reward + gamma * next_a_probs @ Q_table[next_state])    # Expectation_SARSA
        elif method.lower() == 'qlearning':
            Q_table[s][a] = (1-alpha) * Q_table[s][a] + alpha * (reward + gamma * np.max(Q_table[next_state]))    # Q-Learning

        s = next_state
        count += 1
        total_reward += reward
        
        if done:
            break
    
    # result save in dict
    result['n_episodes'].append(count)
    result['total_reward'].append(total_reward)

Q_table
result      # 


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

# pd.DataFrame(Q_table).T.sort_index().to_numpy().argmax(axis=1)




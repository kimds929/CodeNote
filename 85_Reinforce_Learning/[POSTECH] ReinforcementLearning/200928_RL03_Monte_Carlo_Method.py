import gym
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
# get_ipython().run_line_magic('matplotlib', 'inline')

from time import sleep
from IPython.display import clear_output
from tqdm import tqdm_notebook


from collections import defaultdict
import sys
RLpath = r'D:\Python\★★Python_POSTECH_AI\Postech_AI 6) Reinforce_Learning\강의) 실습자료'
sys.path.append(RLpath) # add project root to the python path

from code_MC.lib.envs.blackjack import BlackjackEnv
from code_MC.lib import plotting
# from src.common.gridworld import GridworldEnv # Gridworld Environment

# !unzip code_DP.zip -d code_DP      # 압축 해제

# Black-Jack Simulation Game
# http://www.allingamer.net/kor/%EB%B8%94%EB%9E%99%EC%9E%AD/

env_black = BlackjackEnv()

observation = env_black.reset()
observation     # (score, dealer_score, usable_ace)

env_black.action_space          # 1: hit, 0: stick
env_black.observation_space     # (score(21+11), dealer_score(17+11), usable_ace)

# env_black._step(0)
# env_black._step(1)



# observation(state을 보기좋게 출력하기
def print_observation(observation):
    score, dealer_score, usable_ace = observation
    print("Player Score: {} (Usable Ace: {}), Dealer Score: {}".format(
          score, usable_ace, dealer_score))

print_observation(observation)


# 각 observation마다 어떤 액션을 할지 전략
# 아주 간단한 전략(strategy)
def strategy(observation):
    score, dealer_score, usable_ace = observation
    # Stick (action 0) if the score is >= 20, hit (action 1) otherwise
    return 0 if score >= 20 else 1

observation = env_black.reset()

print_observation(observation)
print(strategy(observation))
result = env_black._step(strategy(observation))
print(result)
observation = result[0]
print(strategy(observation))



win=0
#블랙잭 5판을 하자!
for i_episode in range(5):
    observation = env_black.reset()
    for t in range(100):
        print_observation(observation)
        #위에 전략을 이용하여 블랙잭을 한다!
        action = strategy(observation)
        print("Taking action: {}".format( ["Stick", "Hit"][action])) #If action=0, stick. Else action=1, Hit.
        observation, reward, done, _ = env_black.step(action)
        
        #블랙잭 게임이 끝이나면...
        if done:
            print_observation(observation)
            print("Game end. Reward: {}\n".format(float(reward)))
            if reward==1:
                win+=1
            break
print("we win total ",win,"times!")




# ## 해보기1) 블랙잭을 100판? 1000판? 해보자
def blackjack(n_play=5, strategy=strategy):
    win=0
    #블랙잭 5판을 하자!
    for i_episode in range(n_play):
        observation = env_black.reset()
        for t in range(100):
            #위에 전략을 이용하여 블랙잭을 한다!
            action = strategy(observation)
            observation, reward, done, _ = env_black.step(action)
            
            #블랙잭 게임이 끝이나면...
            if done:
                if reward==1:
                    win+=1
                break
    return win
blackjack(100)
blackjack(1000)

# ## 해보기2) strategy를 더 좋은 전략을 짜서 승률을 높여보자
class StrategyN():
    def __init__(self, criteria=20):
        self.criteria = criteria
    
    def __call__(self, observation):
        score, dealer_score, usable_ace = observation
        return 0 if score >= self.criteria else 1

s20 = StrategyN(20)
blackjack(10000, s20)

s17 = StrategyN(17)
blackjack(10000, s17)

s15 = StrategyN(15)
blackjack(10000, s15)

















# 1 Step_MC_Prediction ----------------------------------------------------------------------

def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    
    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """
    
    #우리는 MC방법을 이용해서 Value function V를 구할것이다.
    # How : 몇 번의 episode를 시행시켜보고, 그 episode들의 G(return)의 평균을 구해서
    # 그 평균값으로 V를 대체하자. 이를 위해 return G의 총 합과 횟수를 기록하자
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final value function
    V = defaultdict(float)
    
    #각각의 에피소드마다
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        #An episode will be an array of (state, action, reward) tuples        
        #episode를 빈 array로 만들자
        episode = []
        #현재 state를 리셋하자
        state = env.reset()
        
       #빈칸1__실습pdf의 빨간 첫번째 빈칸식_
        #policy를 이용하여 state를 계속 업데이트 하며, 이것들을 episode에 저장하자
        for t in range(100):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        
        #위에서 저장한 episode는 예를들면 다음과 같이 저장될 것이다
        # For example, epsiode is [((12, 10, False), 1, 0), ((15, 10, False), 1, 0), ((19, 10, False), 1, -1)]
        #states_in_episode에 위의 episode array에서 방문한 state들만 뽑아서 저장하자
        states_in_episode = [x[0] for x in episode]
        #각각의 방문했던 state마다
        for state in states_in_episode:
            #빈칸2_episode중에 처음으로 그 state에 방문한 idx를 first_occurence_idx에 저장하자.
            for i,x in enumerate(episode):
                if x[0]==state:
                    first_occurence_idx=i
                    break
            #빈칸3_처음 방문한 이후로 reward들을 G에 합하자
            G=0
            for i,x in enumerate(episode[first_occurence_idx:]):
                G += x[2]*(discount_factor**i)
            # 계산된 G를 해당 state의 returns_sum에 합하고, returns_count까지 이용하여 평균을 V에 대입하자
            returns_sum[state] += G
            returns_count[state] += 1.0
            #빈칸4_결국 우리가 구하고자 하는 state-value function V에는 어떤값을 대입??
            V[state] = returns_sum[state] / returns_count[state]

    return V    


def sample_policy(observation):
    """
    A policy that sticks if the player score is >= 20 and hits otherwise.
    """
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1



env_black = BlackjackEnv()


# mc_prediction(policy, env, num_episodes, discount_factor=1.0)
V_10k = mc_prediction(sample_policy, env_black, num_episodes=10000)
plotting.plot_value_function(V_10k, title="10,000 Steps")

V_500k = mc_prediction(sample_policy, env_black, num_episodes=500000)
plotting.plot_value_function(V_500k, title="500,000 Steps")
























def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    #빈칸1_Input Q를 maximize하는 action을 얻어내서 채워보자 (이해: 1-epsilon을 나중에 더해주기)
    #단, 실습pdf의 빨간 두번째 빈칸식에 나와있는 epsilon-greedy policy로 만들기
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A #A는 각 action별 확률(action개수만큼의 길이를 가짐)
    return policy_fn




def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities
    """
    
    #우리는 MC방법을 이용해서 action-value function Q를 구할것이다.
    # How : 몇 번의 episode를 시행시켜보고, 그 episode들의 G(return)의 평균을 구해서
    # 그 평균값으로 Q를 대체하자. 이를 위해 return G의 총 합과 횟수를 기록하자
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    #즉 쉽게 말하면, Q는 state x action 크기의 행렬이라고 생각하자
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    #우리는 위에서 만든 epsilon_greedy_policy를 이용할 것이다
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    #각각의 에피소드마다
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # 빈칸_2_Generate an episode
        #('1step_MC Prediction'의 빈칸_1 과 거의 비슷, 단 policy만 epsilon_greedy_policy확률에 맞춰서 action 선택)
        #주의: 현재 policy는 각 action별 확률을 나타냄
        episode = []
        state = env.reset()
        for t in range(100):
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        #sa_in_episode에 위의 episode array에서 방문한 state와 action들만 (state, action) pair를 저장하자
        sa_in_episode = [(x[0], x[1]) for x in episode]
        #각각의 방문했던 (state,action) 마다
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            #밑의빈칸3&4는 실습pdf의 첫번째 빨간 빈칸식 참고
            # 빈칸_3_episode중에 처음으로 그 state에 방문하고 action을 취한 idx를 first_occurence_idx에 저장하자.
            for i,x in enumerate(episode):
                if x[0]==state and x[1]==action:
                    first_occurence_idx=i
                    break
            
            # 빈칸_4_처음 방문한 이후로 reward들을 G에 합하자
            G=0
            for i,x in enumerate(episode[first_occurence_idx:]):
                G += x[2]*(discount_factor**i)
            
            # 계산된 G를 해당 state의 returns_sum에 합하고, returns_count까지 이용하여 평균을 Q에 대입하자
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            #빈칸5_결국 우리가 구하고자 하는 action-value function Q에는 어떤값을 대입??
            ###Q[state][action] = 으로 시작
            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
    return Q, policy



Q, policy = mc_control_epsilon_greedy(env, num_episodes=500000, epsilon=0.1)

V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")



def print_observation(observation):
    score, dealer_score, usable_ace = observation
    print("Player Score: {} (Usable Ace: {}), Dealer Score: {}".format(
          score, usable_ace, dealer_score))


win=0
for i_episode in range(1000):
    observation = env.reset()
    for t in range(100):
        print_observation(observation)
        action = np.argmax(policy(observation))
        print("Taking action: {}".format( ["Stick", "Hit"][action])) #If action=0, stick. Else action=1, Hit.
        observation, reward, done, _ = env.step(action)
        if done:
            print_observation(observation)
            print("Game end. Reward: {}\n".format(float(reward)))
            if reward==1:
                win+=1
            break
print("we win total ", win, "times!")









# -----------------------------------------------------------------------------------------

env = BlackjackEnv()
num_episodes = 10
discount_factor = 1.0 

returns_sum = defaultdict(float)
returns_count = defaultdict(float)
V = defaultdict(float)

def sample_policy(observation):
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1

policy = sample_policy

#각각의 에피소드마다
# for i_episode in range(1, num_episodes + 1):
episode = []
state = env.reset()

for t in range(10):
    action = policy(state)
    next_state, reward, done, _ = env.step(action)
    episode.append((state, action, reward))
    if done:
        break
    state = next_state

states_in_episode = [x[0] for x in episode]


for state in states_in_episode:
    for i, x in enumerate(episode):  # episode중에 처음으로 그 state에 방문한 idx를 first_occurence_idx에 저장
        if x[0]==state:
            first_occurence_idx = i
            break

    #처음 방문한 이후로 reward들을 G에 합하자
    G=0
    for i, x in enumerate(episode[first_occurence_idx:]):
        G += x[2]*(discount_factor**i)
    returns_sum[state] += G
    returns_count[state] += 1.0
    V[state] = returns_sum[state] / returns_count[state]


for state in states_in_episode:
    for i, x in enumerate(episode):  # episode중에 처음으로 그 state에 방문한 idx를 first_occurence_idx에 저장
        print(state, i, x)
        if x[0]==state:
            pass








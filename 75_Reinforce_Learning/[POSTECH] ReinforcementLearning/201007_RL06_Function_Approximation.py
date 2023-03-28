import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import matplotlib
matplotlib.use('Agg')
matplotlib.style.use('ggplot')
# conda install -c conda-forge ffmpeg
from IPython import display
from tqdm import tqdm, tqdm_notebook
from collections import namedtuple
import itertools

import sys
RLpath = r'D:\Python\★★Python_POSTECH_AI\Postech_AI 6) Reinforce_Learning\강의) 실습자료'
sys.path.append(RLpath) # add project root to the python path
from code_FA.lib import plotting
# get_ipython().run_line_magic('matplotlib', 'inline')






# Mountain-Car Playground ========================================================================

# kernel : 𝑥(𝑠)=(1, 𝑠_1, 𝑠_2, 𝑠_1 𝑠_2, 𝑠_1^2, 𝑠_2^2, ⋯)
# s_1 : position
# s_2 : velocity

env = gym.envs.make("MountainCar-v0")
env.reset()

# 0 : push left 
# 1 : no push
# 2 : push right
R = 0
init_position, init_velocity = env.reset()
position = init_position
velocity = init_velocity
position_list = [init_position]


for i in range(200):
    if position<=np.pi/6 and velocity<=0:
        action = 0
    elif position<=np.pi/6 and velocity>0:
        action = 2
    elif position>np.pi/6 and velocity<=0:
        action = 0
    elif position>np.pi/6 and velocity>0:
        action = 2
        
    state, reward, done, _ = env.step(2)
    position, velocity = state[0], state[1]
    position_list.append(position)
    if done:
        break
    R+=reward
env.close()
print('Reward : {}'.format(R))






def get_render(position_list):
    """
    Get result from list of position
    """
    def _height(xs):
        return np.sin(3 * xs)*.45+.55
    
    min_x, max_x, min_y, max_y = -1.2, 0.6, 0, 1.1
    hill_x = np.linspace(min_x, max_x, 100)
    hill_y = [_height(xs) for xs in hill_x]
    
    fig, ax = plt.subplots()
    ax.set_xlim((min_x, max_x))
    ax.set_ylim((min_y, max_y))
    line, = ax.plot([], [])
    
    def init():
        line.set_data(hill_x, hill_y)
        return (line,)

    def animate(i):
        ax.clear()
        ax.set_xlim((min_x, max_x))
        ax.set_ylim((min_y, max_y))
        ax.scatter(position_list[i], _height(position_list[i]), color='red', s=100)
        line, = ax.plot([], [])
        line.set_data(hill_x, hill_y)
        return (line,)
    anim = animation.FuncAnimation(fig, animate,# init_func=init,
                                   frames=len(position_list), interval=30, blit=True)
    rc('animation', html='html5')
    plt.close()
    return anim


get_render(position_list)


plt.plot(position_list)
plt.show()








# Mountain-Car Playground Function_Approximation ========================================================================

import sklearn.preprocessing
from collections import namedtuple

from sklearn import pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import PolynomialFeatures


env = gym.envs.make("MountainCar-v0")

# 0 : push left 
# 1 : no push
# 2 : push right


# Feature(state) 전처리, 평균을 0, 표준편차를 1로 만들어줍니다
# 환경에서 임의의 샘플 10000개를 뽑아와서 작업을 시작합니다.
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)



# Polynomial features
featurizer = PolynomialFeatures(degree=1)
featurizer.fit(scaler.transform(observation_examples))

#  Feature construction
# RBF kernel을 사용하여 새로운 Feature 들을 만들어냅니다.
# 다양한 gamma 값을 사용하여 다양한 Feature를 만들어냅니다.
# featurizer = sklearn.pipeline.FeatureUnion([
#         ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
#         ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
#         ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
#         ("rbf4", RBFSampler(gamma=0.5, n_components=100))
#         ])
# featurizer.fit(scaler.transform(observation_examples))



class Estimator():
    """
    Value Function approximator. 
    """
    
    def __init__(self):
        # action space가 discreate 하기 때문에  action 마다
        # 모델을 따로 만들어 줄것입니다.
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            model.partial_fit([self.featurize_state(env.reset())], [0])     # Weight Dimension Transform
            self.models.append(model)
    
    def featurize_state(self, state):
        """
        state(position, velocity)를 input으로 받고 Feature construction을 한
        결과를 ouput으로 내는 함수입니다.
        """
        scaled = scaler.transform([state])
        featurized = featurizer.transform(scaled)
        return featurized[0]
    
    def predict(self, s):
        """
        value에 대한 예측을 합니다.
        
        Args:
            s: value를 estimate하고 싶은 state
            
        Returns
            state s에서 각각의 action이 갖는 value를 numpy.array로 반환
            
        """
        features = self.featurize_state(s)
        return np.array([m.predict([features])[0] for m in self.models])
    
    def update(self, s, a, y):
        """
        주어진 state s, action a와 target y를 사용하여 estimator를 update 합니다.
        """
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])


def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    epsilon greedy policy 정의
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn





def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.sys.stdout.flush()
    """

    # 통계(episode별 길이와 reward)를 저장
    EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
    stats = EpisodeStats(episode_lengths=np.zeros(num_episodes),
                         episode_rewards=np.zeros(num_episodes))
    position_list = []
    
    for i_episode in range(num_episodes):
        
        # policy 정의
        policy = make_epsilon_greedy_policy(
            estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)

        last_reward = stats.episode_rewards[i_episode - 1]
        print("\rEpisode {}/{} ({})".format(i_episode + 1, num_episodes, last_reward), end="")
        sys.stdout.flush()
        
        # 환경은 reset합니다.
        state = env.reset()
        
        # One step in the environment
        for t in itertools.count():
            position_list.append(state[0])
                        
            # action을 선택합니다.
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            
            # action을 취합니다.
            next_state, reward, done, _ = env.step(action)

            # 통계를 기록합니다.
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # next_state에서의 q_value를 계산합니다.
            q_values_next = estimator.predict(next_state)

            # TD target을 계산합니다.
            td_target = reward + discount_factor * np.max(q_values_next)

            # estimator을 업데이트합니다.
            estimator.update(state, action, td_target)
                
            if done:
                break
                
            state = next_state
    
    return stats, position_list





estimator = Estimator()

stats, position_list = q_learning(env, estimator, 100, epsilon=0.0)

plotting.get_render(position_list[-1000:])


plotting.plot_cost_to_go_mountain_car(env, estimator)
plotting.plot_episode_stats(stats, smoothing_window=25)












# =======================================================================================
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Polynomial features
featurizer = PolynomialFeatures(degree=7)
featurizer.fit(scaler.transform(observation_examples))

#  Feature construction
# RBF kernel을 사용하여 새로운 Feature 들을 만들어냅니다.
# 다양한 gamma 값을 사용하여 다양한 Feature를 만들어냅니다.

# featurizer = sklearn.pipeline.FeatureUnion([
#         ("rbf1", RBFSampler(gamma=5.0, n_components=100))
#         ])
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))


# X = featurizer.transform(scaler.transform([env.reset()]))
# X

# model = SGDRegressor(learning_rate="constant")
# model.partial_fit(X, [0])
# model.predict(X)


# Q_Table Functional Approximation : Estimator Model Class --------------------------------------------------
class Estimator():
    def __init__(self):
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit([self.featurize_state(env.reset())], [0])     # Weight Dimension Transform
            self.models.append(model)

    def featurize_state(self, state):
        scaled = scaler.transform([state])          # Scaler
        featurized = featurizer.transform(scaled)   # Kernel Transform
        return featurized[0]

    def predict(self, s):
        features = self.featurize_state(s)      # Scaler + Kernel
        return np.array([m.predict([features])[0] for m in self.models])

    def update(self, s, a, y):
        # s: 현재 state 
        # a: 현재 state에서 선택한 action
        # y: target_td = r + Υ * max Q(s', a)
        features = self.featurize_state(s)      # Scaler + Kernel
        self.models[a].partial_fit([features], [y])     # L2 Loss로 Gradient Descent

# estimator = Estimator()
# estimator.predict()
# [m.coef_ for m in estimator.models]
# dir(model)
# model.coef_
# model.intercept_




# policy function -------------------------------
def make_epsilon_greedy_policy(estimator, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn



# Q-Learning: Functional Approximation --------------------------------------------------------------
def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):

    # 통계(episode별 길이와 reward)를 저장
    EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
    stats = EpisodeStats(episode_lengths=np.zeros(num_episodes),
                         episode_rewards=np.zeros(num_episodes))
    position_list = []
    
    for i_episode in range(num_episodes):
        
        # policy 정의
        policy = make_epsilon_greedy_policy(
            estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)

        last_reward = stats.episode_rewards[i_episode - 1]
        print("\rEpisode {}/{} ({})".format(i_episode + 1, num_episodes, last_reward), end="")
        sys.stdout.flush()
        
        
        state = env.reset()     # (Initialize) 환경은 reset
        
        # One step in the environment
        for t in itertools.count():
            position_list.append(state[0])
                        
            # Select Action by Policy
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            
            next_state, reward, done, _ = env.step(action)  # Action Step

            # Recoding history
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # Q_table[s][a] = (1-alpha) * Q_table[s][a] + alpha * (reward + gamma * np.max(Q_table[next_state]))    # Q-Learning
            q_values_next = estimator.predict(next_state)   # next_state에서의 q_value를 계산합니다.
            td_target = reward + discount_factor * np.max(q_values_next)    ## TD target을 계산합니다.
            estimator.update(state, action, td_target)      # estimator update

            if done:
                break
                
            state = next_state
    
    return stats, position_list



estimator = Estimator()
stats, position_list = q_learning(env, estimator, 100, epsilon=0.0)

# Result Plotting
plt.plot(position_list)
plt.show()


plotting.get_render(position_list[-1000:])

plotting.plot_cost_to_go_mountain_car(env, estimator)
plotting.plot_episode_stats(stats, smoothing_window=25)


# np.min(observation_examples[:,0])
# np.max(observation_examples[:,0])

# X = np.linspace(-1,2, 20)
# featurizer.transform(scaler.transform([env.reset()]))

# estimator.models[0].predict()
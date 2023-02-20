# https://www.gymlibrary.dev/environments/toy_text/frozen_lake/
#
# * terminated = True if environment terminates (eg. due to task completion, failure etc.)
# * truncated = True if episode truncates due to a time limit or a reason that is not defined as part of the task MDP.
#       This is done to remove the ambiguity in the done signal. 
#       done=True in the old API did not distinguish between the environment terminating & the episode truncating. 
#       This problem was avoided previously by setting info['TimeLimit.truncated'] in case of a timelimit through 
#       the TimeLimit wrapper. All that is not required now and the env.step() function returns us:
#         → next_state, reward, terminated, truncated , info = env.step(action)
#
#       How could this impact your code: If your game has some kind of max_steps or timeout, 
#       you should read the 'truncated' variable IN ADDITION to the 'terminated' variable to see if your game ended. 
#       Based on the kind of rewards that you have you may want to tweak things slightly.
#        A simplest option could just be to do a:
#         →  done = truncated OR terminated 
#
#       and then proceed to reuse your old code.


# LR format ================================================================================================
import gym

env = gym.make('FrozenLake-v1', map_name="4x4")   # 환경 불러오기

observation = env.reset()   # 환경 초기화
    # observation : 초기화된 첫 state로 세팅

for _ in range(100):
    env.render()                                        # 상태를 보는것 (화면 출력)
    
    # your agent here (this takes random actions)
    # 우리가 채워야할 전략
    action = env.action_space.sample()                  # 환경에서 취할수 있는 액션space에서 랜덤으로 선택하겠다.
    observation, reward, done, info = env.step(action)  # 액션의 step을 밟는 과정, step을 취하고 얻는 결과를 return
        # observation(object) : state (위치)  (Next_state)
        # reward(float) : 보상
        # done(bool: True / False) : Game의 종료 여부 (is_terminal)
        #   terminated, truncated
        # info(dict) : debug infomation
    if done:
        env.render()
        break
# ===========================================================================================================
# gym.Env.class : Create Environment




import gym
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
# get_ipython().run_line_magic('matplotlib', 'inline')

from time import sleep
from IPython.display import clear_output
from tqdm import tqdm_notebook



# ## 오늘의 목표 ----------------------------------------------------------------------
# 1. gym에서 pre-defined된 강화학습 환경 불러오고 문제 확인/이해하기(documentation을 적극 활용)
#  - reward
#  - observation_space
#  - action_space
# 
# 2. random한 action을 만들고 이를 반복적으로 진행(env.step)하여 다음 state와 reward, 그리고 현재 강화학습 환경이 종료되었는지 여부(is_terminal)을 확인합니다. 강화학습 환경이 종료되었다면, 반복문을 종료하고 결과를 얻습니다.
# 
# 3. timestep이 진행됨에 따라 얻는 reward의 합을 cumulative_reward에 기록해 보고 이를 plot해 봅시다.
# 
# 4. 이에 더하여, 주어진 강화학습 문제에 맞는 Custom Env를 직접 작성하고 위의 과정(1~3)을 반복해 봅시다.
# ### 1. Create a new pre-defined enviornment from the name with gym.make() function
# OpenAI의 gym package에서는 기존 강화학습 연구에서 사용되었던 toy example들이나 고전적인 문제들이 이미 구현되어 있으며, 이름을 넣어주는 것 만으로도 편리하게 pre-defined된 강화학습 환경을 만들어 줄 수 있습니다. 여기서는 Jupyter notebook에서 visualize 가능한 예시들을 살펴볼 예정입니다.
# - Taxi-v3
# - Copy-v0
# - FrozenLake8x8-v0
# - Breakout-v0
# 이외에도 다음 예시가 있으며,
# - CartPole-v0 : render 관련 문제로 jupyter notebook에서는 작동하지 않음. Linux terminal 상에서 확인 가능
# 더 자세한 예시들은 아래의 Documentation에서 확인 가능합니다.
# - http://gym.openai.com/envs

# ### Run random action until the enviorment arrives to the terminal state
# 아직 우리가 강화학습에서 어떻게 action을 학습하는지에 대해 배우지 않았기 때문에, 오늘은 random하게 action을 시행할 것입니다.
# - env.step(action)의 Output은 다음과 같이 주어집니다 : next_state, reward, is_terminal, debug_info
# - env의 "sample" method는 action space에서 uniform하게 random한 action을 sampling해 줍니다.
# ```
# env.step(action)
# env.action_space.sample() 
# env.render()
# ```




# ### 1-1. Taxi-v3 -------------------------------------------------------------------------------------------------------------------

# gym의 모든 강화학습 환경은 [EnvName]-[v#N] 과 같이 설정되어 있습니다.
# Taxi-v3 강화학습을 만들기 위해서, 다음 코드를 시키고,
# env = gym.make('Taxi-v3')
# 강화학습 환경을 초기화해 봅시다.
# env.reset()

env = gym.make('Taxi-v3')
env.reset()
env.render()


# env의 observation_space / action_space attribute / 현재의 state 들도 확인해 봅시다
env.observation_space       # discrete한 500개의 state를 가지고있다.
env.observation_space.n     # state 갯수
env.nS                      # state 갯수

env.action_space            # discrete한 6개의 action을 가지고 있다.
env.action_space.n          # action 갯수
env.action_space.sample()   # action_space에서 random한 action 1개를 추출
env.nA                      # action 갯수

env.s                       # 현재 state
env.render()                # 현재 상태를 display



# 이제, 주어진 강화학습 환경에서 random action을 넣어 state를 변화시켜 봅시다.
# - env.sample() method로 random action을 만들어 낸 뒤, 이를 env.step() method에 넣어 state를 변화시킵니다.
# - 매 순간의 state를 env.render() method를 이용하여 시각화합시다.
# - 강화학습 환경이 terminal 상태에 도달하였을 경우 반복문을 중단해야 합니다.
# - 처음 시점부터 현재까지 얻은 reward의 총량을 list에 기록해 봅시다.



# 현재 창에서 env.render()는 화면에 Taxi-v3 enviorment의 state를 text의 형태로 visualize하여 띄워줄 뿐, 이를 따로 output으로 내보내고 있지 않습니다. 
# 다음 Document를 참조하고 위에서 짠 코드를 그대로 응용하여, Taxi-v3 enviornment를 visualize해 봅시다.
    # 1. https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py 에서 render method 파트를 참조합시다.
    # 2. render method의 옵션을 수정하여, "Taxi-v3 enviorment의 state를 text의 형태로 visualize"한 결과물을 list에 하나씩 저장합시다. 총 200개가 저장되게 됩니다.
    # 3. 해당 결과물을 animation처럼 print해 봅시다. sleep() 함수로 시차를 두고, clear_output()으로 이전에 print한 결과물을 지워주면, Taxi_v2의 state 변화를 자연스럽게 애니메이션처럼 관찰할 수 있습니다.
# ++ https://ipython.org/ipython-doc/3/api/generated/IPython.display.html
# metadata = {'render.modes': ['human', 'ansi']}


## render의 mode를 ansi로 바꿔줍시다. 자세한 사항을 github를 참조합시다.
## 이렇게 되면, 매 step에서 print하는 state를 text 형태로 저장할 수 있습니다.
## sleep과 clear_output 함수를 이용하여, text 형태의 output을 애니메이션처럼 print해 봅시다.

env_taxi = gym.make('Taxi-v3')

env_taxi.reset()
obs_log_taxi = []
cumulate_reward_taxi = []
do_taxi = []
textoutput_taxi = []

for _ in range(50):
    # Render current state : Visualize
    txt = env_taxi.render(mode='ansi')        # 현상태
    textoutput_taxi.append(txt)
    
    # (your agent here)
    # choose random action
    action = env_taxi.action_space.sample() # action_sample
    
    ## output : (next_state, reward, is_terminal, debug_info)
    observation, reward, done, info = env_taxi.step(action)

    # result_save
    obs_log_taxi.append(observation)
    if cumulate_reward_taxi:
        cumulate_reward_taxi.append(cumulate_reward_taxi[-1] + reward)
    else:
        cumulate_reward_taxi.append(reward)
    do_taxi.append(done)

    # if is_terminal == True, then break for loop
    if done:
        txt = env_taxi.render(mode='ansi')
        textoutput_taxi.append(txt)
        break
env_taxi.close()





## Plot cumulative reward
plt.plot(cumulate_reward_taxi)
plt.show()


for i, rd_txt in enumerate(textoutput_taxi):
    print(f'TimeStep: {str(i+1).zfill(3)}')
    print(rd_txt)
    clear_output(wait=True)
    sleep(0.1)






# ### 1-2. Copy-v0 -------------------------------------------------------------------------------------------------------------------
# 위의 예시에서는 render한 결과물을 하나씩 저장하고 일괄적으로 print하였습니다. 하지만, 굳이
# env의 observation_space / action_space attribute / 현재의 state 들도 확인해 봅시다
# ```
# env.observation_space.n
# env.action_space.n
# env.s
# ```
env_copy = gym.make('Copy-v0')
env_copy.reset()

env_copy.observation_space
env_copy.action_space       
# Tuple(Discrete(2), Discrete(2), Discrete(5))
# Tuple(주어진 텍스트 위치: 왼쪽/오른쪽, whether write or not , A~E 중에 어떤걸 고르는지)



env_copy.reset()
obs_log_copy = []
cumulate_reward_copy = []
do_copy = []
textoutput_copy = []

for _ in range(100):
    # Render current state : Visualize
    txt = env_copy.render(mode='ansi')        # 현상태
    textoutput_copy.append(txt)
    
    # (your agent here)
    # choose random action
    action = env_copy.action_space.sample() # action_sample
    
    ## output : (next_state, reward, is_terminal, debug_info)
    observation, reward, done, info = env_copy.step(action)

    # result_save
    obs_log_copy.append(observation)
    if cumulate_reward_copy:
        cumulate_reward_copy.append(cumulate_reward_copy[-1] + reward)
    else:
        cumulate_reward_copy.append(reward)
    do_copy.append(done)

    # if is_terminal == True, then break for loop
    if done:
        txt = env_copy.render(mode='ansi')
        textoutput_copy.append(txt)
        break
env_copy.close()

## Plot cumulative reward
plt.plot(cumulate_reward_copy)
plt.show()


for i, rd_txt in enumerate(textoutput_copy):
    print(f'TimeStep: {str(i+1).zfill(3)}')
    print(rd_txt)
    clear_output(wait=True)
    sleep(0.1)



# ### 1-3. FrozenLake8x8-v0 -------------------------------------------------------------------------------------------------------------------
# env의 observation_space / action_space attribute / 현재의 state 들도 확인해 봅시다
# ```
# env.observation_space.n
# env.action_space.n
# env.s
# ```
env_lake = gym.make('FrozenLake8x8-v0')
env_lake.reset()

env_lake.observation_space  
env_lake.action_space          # (4) : 상하좌우



env_lake.reset()
obs_log_lake = []
cumulate_reward_lake = []
do_lake = []
textoutput_lake = []

for _ in range(50):
    # Render current state : Visualize
    txt = env_lake.render(mode='ansi')        # 현상태
    textoutput_lake.append(txt)
    
    # (your agent here)
    # choose random action
    action = env_lake.action_space.sample() # action_sample
    
    ## output : (next_state, reward, is_terminal, debug_info)
    observation, reward, done, info = env_lake.step(action)

    # result_save
    obs_log_lake.append(observation)
    if cumulate_reward_lake:
        cumulate_reward_lake.append(cumulate_reward_lake[-1] + reward)
    else:
        cumulate_reward_lake.append(reward)
    do_lake.append(done)

    # if is_terminal == True, then break for loop
    if done:
        txt = env_lake.render(mode='ansi')
        textoutput_lake.append(txt)
        break
env_lake.close()

## Plot cumulative reward
plt.plot(cumulate_reward_lake)
plt.show()


for i, rd_txt in enumerate(textoutput_lake):
    print(f'TimeStep: {str(i+1).zfill(3)}')
    print(rd_txt)
    clear_output(wait=True)
    sleep(0.1)






# ### 1-4. Render with matplotlib : Breakout-v0 -------------------------------------------------------------------------------------------------------------------

# 여기서는 atari의 Breakout(벽돌깨기) 예제를 보게 될 것입니다. 기존의 3개의 예시에서는 text로 output을 print해 주는 방식이었지만, 여기서는 조금 다른 방식으로(matplotlib의 pixel을 찍어주는 방식으로) 강화학습 환경의 state를 render해주게 됩니다.
# render method는 반드시 구현되어야 하는 step과는 다르게 구현되어 있지 않은 경우가 있습니다.

# env의 observation_space / action_space attribute / 현재의 state 들도 확인해 봅시다
# ```
# env.observation_space.n
# env.action_space.n
# env.s
# ```


# Text 형태가 아니기 때문에, render를 함에 있어서 조금 다른 방식을 취해야 합니다.
# 이는 각각의 Enviornment에 정의되어 있는 render 함수를 직접 읽으면서 파악하시는 것이 좋습니다.

env_BO = gym.make('Breakout-v0')
env_BO.reset()
env_BO.reset().shape
img = plt.imshow(env_BO.render(mode='rgb_array'))

env_BO.observation_space
env_BO.action_space

env_BO.reset()
obs_log_BO = []
cumulate_reward_BO = []
do_BO = []

for _ in range(50):
    img.set_data(env_BO.render(mode='rgb_array'))
    display.display(plt.gcf())
    display.clear_output(wait=True)
    
    # (your agent here)
    # choose random action
    action = env_BO.action_space.sample() # action_sample
    
    ## output : (next_state, reward, is_terminal, debug_info)
    observation, reward, done, info = env_BO.step(action)

    # result_save
    obs_log_BO.append(observation)
    if cumulate_reward_BO:
        cumulate_reward_BO.append(cumulate_reward_BO[-1] + reward)
    else:
        cumulate_reward_BO.append(reward)
    do_BO.append(done)

    # if is_terminal == True, then break for loop
    if done:
        break
env_BO.close()

## Plot cumulative reward
plt.plot(cumulate_reward_BO)
plt.show()









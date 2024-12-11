import gym
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
# get_ipython().run_line_magic('matplotlib', 'inline')

from time import sleep
from IPython.display import clear_output
from tqdm import tqdm_notebook





# ### [Advanced] Make your own gym enviorment by enherit gym.Env class ===================================================================================
# 
# 잘 알려져 있는 Benchmark를 돌려보는 데에는 위의 Enviornments들이 상당히 유용하게 쓰일 것입니다. 그러나, 각자의 문제에 맞게 강화학습 환경을 만들고 이를 training하기 위해서는, OpenAI의 gym package의 Env class를 이용하여 그에 맞는 강화학습 Enviorment를 직접 작성해 주셔야 합니다. 여기에는 State(env.observation_space)와 Action(env.action_space), 그에 해당하는 reward 및 필요에 따라서는 render와 같은 method까지 직접 작성해 주셔야 합니다.
# 여기에서는 코딩 연습을 겸하여, 다음과 같이 간단한 강화학습 환경을 만들어 보도록 하겠습니다.
# 
# - [0, 1, ..., 29] 까지 30개의 방이 일렬로 나열되어 있습니다.
# - 집에서 나가는 출구는 3번 방과 18번 방에 위치합니다.
# - 1, 11번 방에 들어가서 문을 열려고 시도하면, 설치되어 있던 폭탄이 작동합니다.
# - 29번 방에 들어가면 문이 잠기고 나올 수가 없습니다.
# - 취할 수 있는 Action은 총 3가지로, 좌/우로 한 칸씩 움직이거나 각 방에 있는 문을 열어보려 시도하는 것입니다.
# - Reward : 1개의 timestep이 지날 때마다 -0.3점, 문을 열었는데 반응이 없으면 -3점, 1, 11번 방에 들어가서 문을 열려고 하거나(폭탄작동) 29번 방에 들어가버리면(갇힘) -50점, 3, 18번 방에서 문을 확인하지 않으면 -20점, 3, 18번 방에서 탈출에 성공하면 50점

# 다음 BlackjackEnv(gym.Env)를 참조하여, 위의 예제를 완성하여 봅시다.
# 
# ++ 가능하신 분들께서는 render 함수도 새로 정의하여 작성해 봅시다


class HomeEscapeEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Discrete(30)
        self.s = np.random.randint(0,30)
        self.prev_action = None
        
        self.nS = 30
        self.nA = 3
        self.done = False
        
    def reset(self):
        return self._reset()

    def step(self, action):
        return self._step(action)
    
    def render(self):
        return self._render()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        reward = -0.3
        self.done = False
        
        if((action == 0) or (action == 2)):
            if((self.s == 18) or (self.s == 3)):
                reward = reward - 20
            self.s = min(max(0, self.s + action - 1), 29)
            
            if(self.s == 29):
                self.done = True
                reward = reward -50
        else:
            if((self.s == 18) or (self.s == 3)):
                reward = reward + 50
                self.done = True
            elif((self.s == 1) or (self.s == 11)):
                reward = reward - 50
                self.done = True
            else:
                reward = reward - 3
                
        self.prev_action = action
        return self.s, reward, self.done, None
            
    def _reset(self):
        self.s = np.random.randint(0,30)
        self.done = False
        return self.s
        
    def _render(self, mode = 'print'):
        if(self.prev_action == 0):
            prev = 'LEFT'
        elif(self.prev_action == 1):
            prev = 'OPEN'
        elif(self.prev_action == 2):
            prev = 'RIGHT'
        else:
            prev = 'NONE'
            
        txt_output = '|' + '__|' * (self.s) + 'P' + '|__' * (29 - self.s) + '| : Current Room # {}'.format(self.s) + ' / Prev Action : ' + prev
        
        if((self.s == 29) or
           ((self.s == 1) and (self.prev_action == 1)) or
           ((self.s == 11) and (self.prev_action == 1))):
            txt_output = txt_output[:self.s * 3 + 1] + 'X' + txt_output[self.s * 3 + 2:]
        elif(((self.s == 18) and (self.prev_action == 1)) or
            (self.s == 3) and (self.prev_action == 1)):
            
            txt_output = txt_output[:self.s * 3 + 1] + '★' + txt_output[self.s * 3 + 2:]
        
        if(mode == 'return'):
            return txt_output
        else:
            print(txt_output)




env = HomeEscapeEnv()
env.reset()

obs_log = []
cumulative_reward = []
do = []

for t in range(200):
    env.render()

    # your agent here
    action = env.action_space.sample()

    ## output : (next_state, reward, is_terminal, debug_info)
    observation, reward, done, info = env.step(action)
    
    obs_log.append(observation)
    if(t==0):
        cumulative_reward.append(reward)
    else:
        cumulative_reward.append(reward + cumulative_reward[-1])

    ## if is_terminal == True, then break for loop
    if(done):
        clear_output(wait = True)
        sleep(0.1)
        print("Episode finished after {} timesteps".format(t + 1))
        sleep(0.05)
        env.render()
        break
        
    sleep(0.15)
    clear_output(wait = True)
        
env.close()




plt.figure(figsize = (20,6))
plt.plot(cumulative_reward)
plt.show()




















# ### Blackjack Enviornment 예시




def cmp(a, b):
    return int((a > b)) - int((a < b))

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def draw_card(np_random):
    return np_random.choice(deck)


def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
            return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


class BlackjackEnv(gym.Env):
    """Simple blackjack environment
    Blackjack is a card game where the goal is to obtain cards that sum to as
    near as possible to 21 without going over.  They're playing against a fixed
    dealer.
    Face cards (Jack, Queen, King) have point value 10.
    Aces can either count as 11 or 1, and it's called 'usable' at 11.
    This game is placed with an infinite deck (or with replacement).
    The game starts with each (player and dealer) having one face up and one
    face down card.
    The player can request additional cards (hit=1) until they decide to stop
    (stick=0) or exceed 21 (bust).
    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust the player wins.
    If neither player nor dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.  The reward for winning is +1,
    drawing is 0, and losing is -1.
    The observation of a 3-tuple of: the players current sum,
    the dealer's one showing card (1-10 where 1 is ace),
    and whether or not the player holds a usable ace (0 or 1).
    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto (1998).
    https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html
    """
    def __init__(self, natural=False):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2)))
        self._seed()

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        # Start the first game
        self._reset()        # Number of 
        self.nA = 2

    def reset(self):
        return self._reset()

    def step(self, action):
        return self._step(action)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit: add a card to players hand and return
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1
            else:
                done = False
                reward = 0
        else:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1:
                reward = 1.5
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    def _reset(self):
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)

        # Auto-draw another card if the score is less than 12
        while sum_hand(self.player) < 12:
            self.player.append(draw_card(self.np_random))

        return self._get_obs()


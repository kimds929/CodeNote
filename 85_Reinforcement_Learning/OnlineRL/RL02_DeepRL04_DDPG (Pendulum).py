import sys
sys.path.append(r'D:\DataScience\★Git_CodeNote\85_Reinforcement_Learning')

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
import copy

import time
from IPython.display import clear_output
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader

import gymnasium as gym


# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

################################################################################################################

try:
    try:
        from Environments.RL00_Env01_FrozenLake_v1 import generate_frozenlake_map
        from DS_RL import pendulum_actor_evaluator, ReplayMemory
        custom_map = generate_frozenlake_map(5,5, hole=0.17)
    except:
        remote_url = 'https://raw.githubusercontent.com/kimds929/'

        with httpimport.remote_repo(f"{remote_url}/CodeNote/refs/heads/main/85_Reinforcement_Learning/Environments"):
            from RL00_Env01_FrozenLake_v1 import generate_frozenlake_map
        custom_map = generate_frozenlake_map(5,5, hole=0.17)
        
        with httpimport.remote_repo(f"{remote_url}/CodeNote/refs/heads/main/85_Reinforcement_Learning/utils"):
            from DS_RL import pendulum_actor_evaluator, ReplayMemory
except:
    custom_map=None


################################################################################################################

def pendulum_actor_evaluator(actor, device='cpu'):
    # θ : -np.pi ~ -np.pi/2 : 4사분면 / -np.pi/2 ~ 0 : 1사분면 / 0 ~ np.pi/2 : 2사분면 / np.pi/2 ~ np.pi : 3사분면
    # ang_velocity : -8(시계방향) ~ 8 (반시계방향)
    # action : -2 (시계방향 힘) ~ 2 (반시계방향 힘)

    radians = np.linspace(np.pi, -np.pi, num=9)
    # degrees = np.degrees(radians)
    ang_velocity = np.linspace(-8, 8, num=9)
    grid = np.stack(np.meshgrid(radians, ang_velocity)).reshape(2,-1).T
    grid_obs = np.concatenate([np.sin(grid[:, [0]]), np.cos(grid[:, [0]]),grid[:, [1]]], axis=1)

    grid_obs_tensor = torch.FloatTensor(grid_obs).to(device)
    pred_actions, _ = actor.predict(grid_obs_tensor)
    if 'torch' in str(type(pred_actions)):
        pred_actions = pred_actions.detach().to('cpu').numpy()
    grid_obs_pred = np.concatenate([np.degrees(grid[:,[0]]), grid[:,[0]], grid[:, [1]], pred_actions], axis=1)

    df_grid_obs_pred = pd.DataFrame(grid_obs_pred, columns=['degrees', 'radians','ang_velocity','pred_action'])

    df_predict_tb = df_grid_obs_pred.groupby(['radians', 'ang_velocity'])['pred_action'].mean().unstack('radians')
    df_predict_tb = df_grid_obs_pred.groupby(['degrees', 'ang_velocity'])['pred_action'].mean().unstack('degrees')
    df_predict_tb = np.round(df_predict_tb,2)
    df_predict_tb = df_predict_tb.sort_values('degrees', axis=1, ascending=False)
    return df_predict_tb


##############################################################################################################



# Replay Buffer
class ReplayMemory:
    def __init__(self, max_size, batch_size=64):
        # deque object that we've used for 'episodic_memory' is not suitable for random sampling
        # here, we instead use a fix-size array to implement 'buffer'
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0  # 어디까지 채워져있는지 check
        self.size = 0
        self.batch_size = batch_size

    def push(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size

        if self.size >= batch_size:
            # indices = sample(range(self.size), batch_size)
            indices = np.random.choice(range(self.size), size=batch_size, replace=False)
            return [self.buffer[index] for index in indices]

    def reset(self):
        self.__init__(max_size=self.max_size, batch_size=self.batch_size)
    
    def __len__(self):
        return self.size

    def __iter__(self):
        self._iter_indices = np.random.permutation(self.size)  # 셔플된 인덱스
        self._iter_pos = 0
        return self

    def __next__(self):
        if self._iter_pos + self.batch_size > self.size:
            raise StopIteration
        batch_indices = self._iter_indices[self._iter_pos:self._iter_pos + self.batch_size]
        self._iter_pos += self.batch_size
        return [self.buffer[idx] for idx in batch_indices]
###########################################################################################################




# pip install gymnasium[classic-control]
# pip install pygame

import gymnasium as gym

# 환경 생성
env = gym.make("Pendulum-v1", render_mode="rgb_array")  # render_mode='human'은 화면에 직접 그리기용
obs, info = env.reset()
plt.imshow(env.render())
env.step([0])



print(env.observation_space)
# Box([-1. -1. -8.], [1. 1. 8.], (3,), float32)
# θ : -np.pi ~ -np.pi/2 : 4사분면 / -np.pi/2 ~ 0 : 1사분면 / 0 ~ np.pi/2 : 2사분면 / np.pi/2 ~ np.pi : 3사분면
#   (0-dim) cos(θ): 진자의 각도에 대한 코사인
#   (1-dim) sin(θ): 진자의 각도에 대한 사인
#   (2-dim) θ_dot: 진자의 각속도        # -8(시계방향) ~ 8 (반시계방향)
# θ를 직접 주지 않고 cos(θ), sin(θ)로 표현 → 주기성을 반영하기 위함


print(env.action_space)
# Box(low=-2.0, high=2.0, shape=(1,), dtype=float32)
# 범위: -2.0 ~ +2.0 (좌/우로 회전시키는 힘)

# 보상구조
# reward = -(θ^2 + 0.1 * θ̇^2 + 0.001 * torque^2)
# 보상은 항상 음수이며, 0에 가까울수록 좋은 것
# 보상을 줄이기 위해:
#     진자가 위로 가까이 가야 하고
#     각속도가 작아야 하며
#     힘도 적게 써야 함

obs, info = env.reset()
done=False
i=0
for _ in range(500):
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    done = terminated or truncated
    
    print(reward)
    
    plt.figure()
    plt.imshow(env.render())
    plt.show()
    time.sleep(0.1)
    clear_output(wait=True)
    
    obs = next_obs
    i += 1
    if done or i >=50:
        break





##############################################################################################################
# pip install sb3-contrib
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.monitor import Monitor

env = gym.make("Pendulum-v1", render_mode="rgb_array")  # render_mode='human'은 화면에 직접 그리기용
obs, info = env.reset()
plt.imshow(env.render())

# env = make_vec_env("Pendulum-v1", n_envs=1, monitor_dir="./logs/")

# 2. 모니터링(로그 수집)용 래핑
# env = Monitor(env)      # 에피소드별 보상, 길이, 종료 여부 등의 기본 통계를 기록합니다.

# 3. 벡터 환경으로 래핑 (DDPG는 단일 env라도 VecEnv 형태여야 함)
# env = DummyVecEnv([lambda: env])



# 액션 공간 정보
n_actions = env.action_space.shape[-1]

# 노이즈 설정 (Gaussian noise)
gaussian_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
ou_noise = OrnsteinUhlenbeckActionNoise(
    mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
)

# DDPG 모델 생성
model = DDPG(
    "MlpPolicy",                # MLP 정책 사용
    env=env,
    action_noise=ou_noise,
    verbose=1,
    buffer_size=100000,         # 리플레이 버퍼 크기
    learning_rate=1e-3,         # 학습률
    batch_size=64,              # 미니배치 크기
    tau=0.005,                  # soft target update # 타깃 네트워크 업데이트 계수
    gamma=0.99,
    train_freq=(1, "step"),  # 얼마나 자주 학습할지
    gradient_steps=1,          # 한 번 학습 시 얼마나 반복할지 (-1이면 episode 길이만큼)
)

# 총 10만 스텝 동안 학습
model.learn(total_timesteps=10000)

# 학습된 모델 저장
# model.save("ddpg_pendulum")

# 저장된 모델 불러오기
# model = DDPG.load("ddpg_pendulum", env=env)



# Simulation Test ---------------------------------------------------------------------------------
obs, info = env.reset()
done=False
i=0
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    done = terminated or truncated
    
    print(reward)
    plt.figure()
    plt.imshow(env.render())
    plt.show()
    time.sleep(0.1)
    clear_output(wait=True)
    
    obs = next_obs
    i += 1
    if done or i >=50:
        break

























##############################################################################################################



# Replay Buffer
class ReplayMemory:
    def __init__(self, max_size, batch_size=64):
        # deque object that we've used for 'episodic_memory' is not suitable for random sampling
        # here, we instead use a fix-size array to implement 'buffer'
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0  # 어디까지 채워져있는지 check
        self.size = 0
        self.batch_size = batch_size

    def push(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size

        if self.size >= batch_size:
            # indices = sample(range(self.size), batch_size)
            indices = np.random.choice(range(self.size), size=batch_size, replace=False)
            return [self.buffer[index] for index in indices]

    def reset(self):
        self.__init__(max_size=self.max_size, batch_size=self.batch_size)
    
    def __len__(self):
        return self.size

    def __iter__(self):
        self._iter_indices = np.random.permutation(self.size)  # 셔플된 인덱스
        self._iter_pos = 0
        return self

    def __next__(self):
        if self._iter_pos + self.batch_size > self.size:
            raise StopIteration
        batch_indices = self._iter_indices[self._iter_pos:self._iter_pos + self.batch_size]
        self._iter_pos += self.batch_size
        return [self.buffer[idx] for idx in batch_indices]
###########################################################################################################



##############################################################################################################

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action=1):
        super().__init__()
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, hidden_dim)
            ,nn.Tanh()
            ,nn.Linear(hidden_dim, action_dim)
            ,nn.Tanh()
        )
        self.max_action = max_action    # 출력 볌위 설정
    
    def forward(self, obs):
        mu = self.max_action*self.policy_network(obs)
        return mu
    
    def predict(self, obs):
        mu = self.forward(obs)
        return mu, None

# Q-network 정의(StateValue)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.value_network = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, hidden_dim)
            ,nn.ReLU()
            ,nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, actions):
        critic_input = torch.cat([obs, actions], dim=-1)
        value = self.value_network(critic_input)
        return value

##############################################################################################################

# 3) Noise Processes
class OUNoise:
    def __init__(self, dim, mu=0.0, theta=0.15, sigma=0.2):
        self.mu, self.theta, self.sigma = mu, theta, sigma
        self.dim = dim
        self.reset()

    def reset(self):
        self.state = np.ones(self.dim) * self.mu

    def noise(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.dim)
        self.state += dx
        return self.state

class GaussianNoise:
    def __init__(self, dim, mu=0.0, sigma=0.1):
        self.mu, self.sigma = mu, sigma
        self.dim = dim

    def noise(self):
        return np.random.normal(self.mu, self.sigma, size=self.dim)
##############################################################################################################

env = gym.make("Pendulum-v1", render_mode="rgb_array")  # render_mode='human'은 화면에 직접 그리기용
obs, info = env.reset()
plt.imshow(env.render())

memory_size = 1024
batch_size = 64

sample_only_until = 500
gamma = 0.9
tau = 0.01


obs_dim = env.observation_space.shape[-1]
action_dim = env.action_space.shape[-1]
min_action = env.action_space.low.item()
max_action = env.action_space.high.item()
# Actor Network
actor_main_network = Actor(state_dim=obs_dim, hidden_dim=64, action_dim=action_dim,
                           max_action=max_action
                           ).to(device)
optimizer_main_actor = optim.AdamW(actor_main_network.parameters(), lr=3e-4)

actor_target_network = copy.deepcopy(actor_main_network)
actor_target_network.load_state_dict(actor_main_network.state_dict())


# Critic Network
critic_main_network = Critic(state_dim=obs_dim, action_dim=action_dim, hidden_dim=64).to(device)
optimizer_main_critic = optim.AdamW(critic_main_network.parameters(), lr=1e-3, weight_decay=0)

critic_target_network = copy.deepcopy(critic_main_network)
critic_target_network.load_state_dict(critic_main_network.state_dict())



# --------------------------------------------------------
N_ITER = 10
TOTAL_TIMESTEPS = 1200        # Rollout Timstep
N_REPLAY_STEPS = 600         # Rollout 횟수
N_EPOCHS = 3               # Model Learning 횟수
update_interval = 10

for iter in range(N_ITER):
    print(f"\r({iter+1}/{N_ITER} ITER) ", end='')
    # memory
    memory = ReplayMemory(max_size=memory_size, batch_size=batch_size)

    total_loop = (TOTAL_TIMESTEPS-1)//N_REPLAY_STEPS+1
    loop_count = 1
    replay_time_step = 0
    
    while (replay_time_step < TOTAL_TIMESTEPS):
        episode_step = 0
        obs, info = env.reset()
        total_rewards = []
        
        total_reward = 0
        # (Rollout) ---------------------------------------------------------------------
        for _ in range(N_REPLAY_STEPS):
            obs_tensor = torch.FloatTensor(obs).to(device)
            
            # (Exploration 1: Constant Exploration) ------------------------------------------------------------
            # exploration_noise = 0.1
            
            # (Exploration 2: Linear Decay) ------------------------------------------------------------
            max_total_steps = N_ITER * TOTAL_TIMESTEPS
            global_step = iter * TOTAL_TIMESTEPS + replay_time_step
            init_noise =0.3
            final_noise = 0.05
            exploration_noise = final_noise + (init_noise - final_noise) * (1 - global_step / max_total_steps)
            
            # (Exploration 3: Exponential decay) ------------------------------------------------------------
            # global_step = iter * TOTAL_TIMESTEPS + replay_time_step
            # init_noise =0.3
            # final_noise = 0.05
            # decay_rate = 1e-4
            # exploration_noise = final_noise + (init_noise - final_noise) * np.exp(-decay_rate * global_step)
            # ------------------------------------------------------------------------------------------
            
            action  = actor_main_network(obs_tensor)
            action = action.detach().to('cpu').numpy() + np.random.normal(0, exploration_noise, size=action_dim)
            action = action.clip(min_action, max_action)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            if reward >= 0.01:
                print('goal')
            done = terminated or truncated
            experience = (obs, action, next_obs, reward, done)
            # buffer에 experience 저장
            memory.push(experience)
            done
            # counting
            if done:
                obs, info = env.reset()
                total_rewards.append(total_reward)
                total_reward = 0
                episode_step += 1
            else:
                obs = next_obs
                total_reward += np.array(reward).item()
            replay_time_step += 1
            # ---------------------------------------------------------------------------------

            # (Training) ----------------------------------------------------------------------
            if len(memory) > sample_only_until and replay_time_step % update_interval == 0:
                avg_actor_loss = 0
                avg_critic_loss = 0
                for epoch in range(N_EPOCHS):
                    batch = memory.sample()
                    obss, actions, next_obss, rewards, dones = (np.array(batch, dtype='object').T).tolist()
                    
                    # dataset & dataloader
                    obss_tensor = torch.FloatTensor(obss).to(device)
                    actions_tensor = torch.FloatTensor(actions).view(-1, action_dim).to(device)
                    next_obss_tensor = torch.FloatTensor(next_obss).to(device)
                    rewards_tensor = torch.FloatTensor(rewards).view(-1,1).to(device)
                    dones_tensor = torch.FloatTensor(dones).view(-1,1).to(device)
                    
                    # comput value 
                    values = critic_main_network(obss_tensor, actions_tensor)
                    
                    with torch.no_grad():
                        next_actions = actor_target_network(next_obss_tensor).detach()
                        next_values = critic_target_network(next_obss_tensor, next_actions).detach()
                        td_target = (rewards_tensor + gamma * next_values * (1-dones_tensor))
                    
                    # critic_loss
                    critic_loss = nn.functional.mse_loss(values, td_target)
                    # critic_loss = nn.functional.smooth_l1_loss(value, td_target)
                    
                    # critic_learn
                    optimizer_main_critic.zero_grad()
                    critic_loss.backward()
                    optimizer_main_critic.step()
                    
                    # actor loss
                    actor_loss = -critic_main_network(obss_tensor, actor_main_network(obss_tensor)).mean()
                    
                    # actor_learn
                    optimizer_main_actor.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor_main_network.parameters(), max_norm=1.0)
                    optimizer_main_actor.step()
                    
                    # Soft update
                    for param, target_param in zip(critic_main_network.parameters(), critic_target_network.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    for param, target_param in zip(actor_main_network.parameters(), actor_target_network.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    
                    avg_critic_loss += critic_loss.item()
                    avg_actor_loss += actor_loss.item()
                # ---------------------------------------------------------------------------------
                print(f"\r[{iter+1}/{N_ITER}IT](L: {loop_count}/{total_loop}) Rep: {replay_time_step}/{TOTAL_TIMESTEPS}, Rewards: {np.mean(total_rewards).item():.2f} ([Epochs: {epoch+1}/{N_EPOCHS}] Critic: {avg_critic_loss/N_EPOCHS:.3f}, Actor: {avg_actor_loss/N_EPOCHS:.3f})", end='')
        
            else:
                print(f"\r[{iter+1}/{N_ITER}IT](L: {loop_count}/{total_loop}) Rep: {replay_time_step}/{TOTAL_TIMESTEPS}, Rewards: {np.mean(total_rewards).item():.2f}", end='')
        
        loop_count += 1    



# Simulation Test ---------------------------------------------------------------------------------
obs, info = env.reset()
done=False
i=0
for _ in range(500):
    obs_tensor = torch.FloatTensor(obs).to(device)
    action, _ = actor_main_network.predict(obs_tensor)
    action = action.detach().to('cpu').numpy()
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    done = terminated or truncated
    
    print(reward)
    plt.figure()
    plt.imshow(env.render())
    plt.show()
    time.sleep(0.1)
    clear_output(wait=True)
    
    obs = next_obs
    i += 1
    if done or i >=50:
        break



################################################################################################
eval_tb = pendulum_actor_evaluator(actor_main_network, device=device)
eval_tb
plt.imshow(eval_tb, cmap='coolwarm', vmin=-2, vmax=2)

################################################################################################
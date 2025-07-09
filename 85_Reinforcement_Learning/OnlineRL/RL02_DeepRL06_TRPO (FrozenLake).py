import sys
sys.path.append(r'D:\DataScience\★Git_CodeNote\85_Reinforcement_Learning')

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict

import time
from IPython.display import clear_output
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader

# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# import httpimport
# remote_url = 'https://raw.githubusercontent.com/kimds929/'

# with httpimport.remote_repo(f"{remote_url}/CodeNote/main/85_Reinforce_Learning/"):
#     from RL00_Env01_CustomGridWorld import CustomGridWorld



###########################################################################################################

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
    
    def sample_all(self):  # Sequential Sample all
        return self.buffer[:self.size]
    
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



################################################################################################################
################################################################################################################
# [ FrozenLake ] ###############################################################################################

import gymnasium as gym
import httpimport


try:
    try:
        from Environments.RL00_Env01_FrozenLake_v1 import generate_frozenlake_map
        from DS_RL import frozenlake_visualize_grid_probs, ReplayMemory
        custom_map = generate_frozenlake_map(5,5, hole=0.17)
    except:
        remote_url = 'https://raw.githubusercontent.com/kimds929/'

        with httpimport.remote_repo(f"{remote_url}/CodeNote/refs/heads/main/85_Reinforcement_Learning/Environments"):
            from RL00_Env01_FrozenLake_v1 import generate_frozenlake_map
        custom_map = generate_frozenlake_map(5,5, hole=0.17)
        
        with httpimport.remote_repo(f"{remote_url}/CodeNote/refs/heads/main/85_Reinforcement_Learning/utils"):
            from DS_RL import frozenlake_visualize_grid_probs, ReplayMemory
except:
    custom_map=None



##############################################################################################################
# policy-network 정의 (Actor Network)
class Actor(nn.Module):
    def __init__(self, action_dim, hidden_dim, max_states=100, embed_dim=1):
        super().__init__()
        self.state_embedding = nn.Embedding(num_embeddings=max_states, embedding_dim=embed_dim)
        self.policy_network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim)
            ,nn.Tanh()
            ,nn.Linear(hidden_dim, hidden_dim)
            ,nn.Tanh()
            ,nn.Linear(hidden_dim, action_dim)
        )
    
    def forward_logits(self, obs):
        embed_x = self.state_embedding(obs).squeeze(-2)
        logits = self.policy_network(embed_x)
        return logits
    
    def execute_model(self, obs, actions=None, temperature=1, deterministic=False):
        logits = self.forward_logits(obs)
        action_dist = Categorical(logits=logits)
        entropy = action_dist.entropy()
        
        if actions is None:
            if deterministic is True:
                action = torch.argmax(logits, dim=-1)
            elif temperature is None:
                action = torch.argmax(logits, dim=-1)
            else:
                explore_dist = Categorical(logits=logits/temperature)
                action = explore_dist.sample()
            log_prob = action_dist.log_prob(action)
            return action, log_prob, entropy
        
        else:
            log_prob = action_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
            return log_prob, entropy
    
    def forward(self, obs, temperature=1, deterministic=False):
        action, log_prob, entropy = self.execute_model(obs, temperature=temperature, deterministic=deterministic)
        return action, log_prob, entropy
    
    def evaluate_actions(self, obs, actions, temperature=1, deterministic=False):
        log_prob, entropy = self.execute_model(obs, actions=actions, temperature=temperature, deterministic=deterministic)
        return log_prob, entropy
    
    def predict(self, obs, temperature=1, deterministic=False):
        action, log_prob, entropy = self.execute_model(obs, temperature=temperature, deterministic=deterministic)
        return action
    

# Q-network 정의
class Critic(nn.Module):
    def __init__(self, hidden_dim, max_states=100, embed_dim=1):
        super().__init__()
        self.state_embedding = nn.Embedding(num_embeddings=max_states, embedding_dim=embed_dim)
        self.fc_block = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim)
            ,nn.SELU()
            ,nn.Linear(hidden_dim, hidden_dim)
            ,nn.SELU()
            ,nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.state_embedding(x).squeeze(-2)
        return self.fc_block(x)
##############################################################################################################

# compute_gae
def compute_gae(rewards, values, next_values, gamma=0.99, lmbda=0.95):
    """
    ReplayMemory에 저장된 sequential rollout에서 GAE 계산

    Args:
        rewards:      list or array of shape (N,)    ← [r0, r1, ..., r_{N-1}]
        values:       list or array of shape (N,)    ← [v0, v1, ..., v_{N-1}]
        next_value: : list or array of shape (N,)    ← [None, ...]  (terminated or truncated시 next_value )
        gamma:      할인률 (default=0.99)
        lmbda:      GAE λ 파라미터 (default=0.95)

    Returns:
        advantages: numpy.ndarray of shape (N,)   ← GAE(λ) 기반 advantage
        returns:    numpy.ndarray of shape (N,)   ← advantage + values
    """
    N = len(rewards)
    advantages = np.zeros(N, dtype=np.float32)
    returns = np.zeros(N, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(N)):
        if next_values[t] is None:
            next_value = values[t + 1]
        else:       # terminated or truncated 
            next_value = next_values[t]
            gae = 0.0

        # δ_t = r_t + γ * V(s_{t+1}) * mask - V(s_t)
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lmbda * gae
        advantages[t] = gae

    # return = advantage + V(s_t)
    for i in range(N):
        returns[i] = advantages[i] + values[i]

    return advantages, returns
##############################################################################################################

gamma = 0.99
lmbda = 0.95
batch_size = 64

env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode="rgb_array")  # deterministic
obs, info = env.reset()
plt.imshow(env.render())
# env.observation_space.n

state_dim = env.observation_space.n
action_dim = env.action_space.n


# policy_network
actor_network = Actor(hidden_dim=64, action_dim=env.action_space.n, embed_dim=4).to(device)

# critic_network
critic_network = Critic(hidden_dim=16, max_states=env.observation_space.n, embed_dim=2).to(device)
optimizer_critic = optim.AdamW(critic_network.parameters(), lr=1e-3, weight_decay=1e-2)

# Replay Buffer
memory = ReplayMemory(max_size=10000)

N_ITER = 10
TOTAL_TIMESTEPS = 1000        # Rollout Timstep
N_REPLAY_STEPS = 200         # Rollout 횟수
N_EPOCHS = 3               # Model Learning 횟수




for it in range(N_ITER):
    print(f"\r({it+1}/{N_ITER} ITER) ", end='')
    replay_time_step = 0
        
    total_loop = (TOTAL_TIMESTEPS-1)//N_REPLAY_STEPS+1
    loop_count = 1
    it=0
    while (replay_time_step < TOTAL_TIMESTEPS):
        # (Collect Rollout) ##################################################################################
        memory.reset()
        state, info = env.reset()
        
        for _ in range(N_REPLAY_STEPS):
            
            print(f"\r[{it+1}/{N_ITER}ITER](Loop {loop_count}/{total_loop}) Rollout: {replay_time_step+1}/{TOTAL_TIMESTEPS}, ", end='')
            # estimate policy, value
            obs_tensor = torch.LongTensor([obs]).to(device)
            
            logits = actor_network.forward_logits(obs_tensor)
            dist = Categorical(logits=logits)
            logits = logits.detach().to('cpu')
            action = dist.sample().item()
            
            value = critic_network(obs_tensor)
            value = value.item()
            
            # env_step
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            next_value = None
            if terminated:
                next_value = 0.0
            elif truncated:
                next_obs_tensor = torch.LongTensor([next_obs]).to(device)
                next_value = critic_network(next_obs_tensor).item()
            
            # buffer push
            experience = (obs, action, logits, reward, value, next_value)
            memory.push(experience)
            
            # counting
            if terminated or truncated:
                state, info = env.reset()
            else:
                state = next_obs
            replay_time_step += 1
        ######################################################################################################
        
        # RolloutData
        # batch, indices, weights = memory.sample()
        batch = memory.sample_all()
        obss, actions, logits, rewards, values, next_values  = (np.array(batch, dtype='object').T).tolist()
        obss_tensor = torch.LongTensor(obss).view(-1,1)
        
        
        
        # 마지막 스텝에서의 next_value 확보 (부트스트랩용)
        with torch.no_grad():
            next_obs_tensor = torch.LongTensor([next_obs]).to(device)
            next_values[-1] = critic_network(next_obs_tensor).item()

        # ComputeGAE
        advantages, returns = compute_gae(rewards=rewards, values=values, next_values=next_values, gamma=gamma, lmbda=lmbda)
        # advantage 정규화 (선택적)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # dataset & dataloader
        obss_tensor = torch.LongTensor(obss).view(-1,1).to(device)
        actions_tensor = torch.LongTensor(actions).view(-1,1).to(device)
        logits_tensor = torch.stack(logits).to(device)
        values_tensor = torch.FloatTensor(values).view(-1,1).to(device)
        advantages_tensor = torch.FloatTensor(advantages).view(-1,1).to(device)
        returns_tensor = torch.FloatTensor(returns).view(-1,1).to(device)
        
        dataset = TensorDataset(obss_tensor, actions_tensor, logits_tensor, values_tensor, advantages_tensor, returns_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        
        # (TRPO Learning) ---------------------------------------------------------------------------------------
        total_critic_loss = 0.0
        total_policy_loss = 0.0
        num_batches = 0
        
        for epoch in range(N_EPOCHS):
            for batch_data in data_loader:
                # break
                # 현재 policy, value 평가: evaluate value,log_prob, entropy
                obss, actions, old_logits, values, advantages, returns = batch_data
                
                # old policy
                old_dist = Categorical(logits=old_logits.detach())
                old_log_probs = old_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
                
                # new policy
                new_logits = actor_network.forward_logits(obss)
                new_dist = Categorical(logits=new_logits)
                new_log_probs = new_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
                
                
                # 【 Actor Loss & Backward 】##########################################################################################
                # (log_prob ratio)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr_loss = - (ratio * advantages).mean()
                
                damping = 1e-2
                kl_target = 1e-2
                n_optim_steps = 10
                
                # [Conjugate Gradient] ==================================================================
                grads = torch.autograd.grad(surr_loss, actor_network.parameters(), create_graph=True)
                #   torch.autograd.grad : 스칼라 출력 y에 대해 model의 파라미터에 대한 gradient들을 계산해서 리스트로 반환하는 PyTorch의 자동 미분 기능
                #       ㄴ retain_graph=True : 한 번 그래디언트를 구한 후에도 연산 그래프를 메모리에 남겨둠 (이후 또 backward나 grad를 호출하려면 필요)
                #       ㄴ create_graph=True : gradient 연산조차도 추적하게 만들어서 2차 미분이 가능하게 함 (2차 미분을 위한 그래프 생성)
                grad_flat = torch.cat([g.view(-1) for g in grads])
                negative_grad_flat = -grad_flat
                
                x = torch.zeros_like(negative_grad_flat)
                r = negative_grad_flat.clone()
                p = negative_grad_flat.clone()
                
                rs_old = r @ r
                for _ in range(n_optim_steps):
                    # (Fisher-Vector Product) ----------------------------------------------------
                    new_logits = actor_network.forward_logits(obss)
                    new_dist = Categorical(logits=new_logits)
                    
                    kl = torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()
                    grads_kl = torch.autograd.grad(kl, actor_network.parameters(), create_graph=True)
                    grad_flat_kl = torch.cat([g.contiguous().view(-1) for g in grads_kl])
                    kl_v = (grad_flat_kl * p).sum()     # p
                    grads_kl_v = torch.autograd.grad(kl_v, actor_network.parameters(), retain_graph=True)
                    Ap = torch.cat([g.contiguous().view(-1) for g in grads_kl_v]).detach() + damping * p         # p
                    # --------------------------------------------------------------------------
                    alpha = (r @ r) / (p @ Ap)
                    x += alpha * p
                    r_next = r - alpha * Ap
                    if r_next.norm() < 1e-10:
                        break
                    beta = (r_next @ r_next) / (r @ r)
                    p = r_next + beta * p
                    r = r_next
                step_dir = x
                # ====================================================================================
                
                # [Step-size Scaling] ==================================================================
                max_kl = 1e-2
                
                # (Fisher-Vector Product) ----------------------------------------------------
                new_logits = actor_network.forward_logits(obss)
                new_dist = Categorical(logits=new_logits)
                
                kl = torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()
                grads_kl = torch.autograd.grad(kl, actor_network.parameters(), create_graph=True)
                grad_flat_kl = torch.cat([g.contiguous().view(-1) for g in grads_kl])
                kl_v = (grad_flat_kl * step_dir).sum()  # step_dir
                grads_kl_v = torch.autograd.grad(kl_v, actor_network.parameters(), retain_graph=True)
                A_step_dir = torch.cat([g.contiguous().view(-1) for g in grads_kl_v]).detach() + damping * step_dir  # step_dir
                # --------------------------------------------------------------------------
                shs =  step_dir @ A_step_dir
                if shs <= 0:
                    print(f"[Warning] shs (step_dir^T * FVP(step_dir)) is non-positive: {shs.item():.6f}")
                    continue  # 또는 break
                step_scale = torch.sqrt((2 * kl_target) / (shs + 1e-8))
                full_step = step_scale * step_dir
                # ====================================================================================
                
                # [Apply update with line search] ======================================================
                def set_params(model, flat):
                    i = 0
                    for p in model.parameters():
                        n = p.numel()
                        p.data.copy_(flat[i:i+n].view_as(p)); i += n
            
                old_params = torch.cat([p.data.view(-1) for p in actor_network.parameters()])
                
                for frac in 0.5 ** np.arange(n_optim_steps):
                    set_params(actor_network, old_params + frac*full_step)
                    ls_logits = actor_network.forward_logits(obss)
                    ls_dist = Categorical(logits=ls_logits)
                    ls_log_probs = ls_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
                
                    loss_new = -(torch.exp(ls_log_probs - old_log_probs) * advantages).mean()
                    kl_new = torch.distributions.kl.kl_divergence(old_dist, ls_dist).mean()
                    if loss_new < surr_loss and kl_new <= kl_target:
                        break
                else:      # 실패 시 원래 파라미터 복구
                    set_params(actor_network, old_params)
                ###################################################################################################
                
                # 【 Critic Loss & Backward 】 #######################################################################
                critic_loss = nn.functional.mse_loss(critic_network(obss), returns)
                
                optimizer_critic.zero_grad()
                critic_loss.backward()
                optimizer_critic.step()
                ###################################################################################################
                # loss
                total_critic_loss += critic_loss.item()
                total_policy_loss += surr_loss.item()
                num_batches += 1
        # === 학습 epoch 전체 후 평균 loss 출력 ===
        avg_critic_loss = total_critic_loss / num_batches
        avg_policy_loss = total_policy_loss / num_batches
                
        loop_count += 1            
        # === 학습 현황 출력 ===
        print(f"\r[{it+1}/{N_ITER}ITER](Loop {loop_count}/{total_loop}) Rollout: {replay_time_step}/{TOTAL_TIMESTEPS}, "
            f"learning_epoch: {epoch+1}/{N_EPOCHS} (CriticLoss: {avg_critic_loss:.3f}, "
            f"ActorLoss: {avg_policy_loss:.3f})", end='')

   
   
# Simulation Test ---------------------------------------------------------------------------------
obs, info = env.reset()
# env.render()
i = 0
done = False
while (done is not True):
    
    with torch.no_grad():
        actor_network.eval()
        action = actor_network.predict(torch.LongTensor([obs]).to(device), deterministic=True)
        action = action.item()  
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        plt.imshow(env.render())
        plt.show()
        time.sleep(0.2)
        clear_output(wait=True)
        obs = next_obs
    i += 1
    if i >=30:
        break

################################################################################################################
################################################################################################################

# ← ↓ → ↑
actor_network.forward_logits(torch.LongTensor([0]).to(device))
# env.step(3)

allgrid_logits = actor_network.forward_logits(torch.arange(env.observation_space.n).to(device))
allgrid_logits
allgrid_probs = (torch.exp(allgrid_logits) / torch.exp(allgrid_logits).sum(dim=-1, keepdim=True)).to('cpu').detach().numpy()


actor_probs = allgrid_probs.reshape(env.unwrapped.nrow,env.unwrapped.ncol,env.action_space.n).copy()
actor_probs.shape
frozenlake_visualize_grid_probs(actor_probs, env)



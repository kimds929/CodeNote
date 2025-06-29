


    



# ################################################################################################################
# ################################################################################################################
# # [ On-policy Actor Critic ]###################################################################################
# class Actor(nn.Module):
#     def __init__(self, action_dim, hidden_dim, max_states=100, embed_dim=1):
#         super().__init__()
#         self.state_embedding = nn.Embedding(num_embeddings=max_states, embedding_dim=embed_dim)
#         self.policy_network = nn.Sequential(
#             nn.Linear(embed_dim, hidden_dim)
#             ,nn.ReLU()
#             ,nn.Linear(hidden_dim, hidden_dim)
#             ,nn.ReLU()
#             ,nn.Linear(hidden_dim, action_dim)
#         )
    
#     def execute_model(self, obs, actions=None, temperature=None):
#         embed_x = self.state_embedding(obs).squeeze(-2)
#         logits = self.policy_network(embed_x)
#         action_dist = Categorical(logits=logits)
#         entropy = action_dist.entropy()
        
#         if actions is None:
#             if temperature is None:
#                 action = torch.argmax(logits, dim=-1)
#             else:
#                 explore_dist = Categorical(logits=logits/temperature)
#                 action = explore_dist.sample()
#             log_prob = action_dist.log_prob(action)
#             return action, log_prob, entropy
        
#         else:
#             log_prob = action_dist.log_prob(actions)
#             return log_prob, entropy
    
#     def forward(self, obs, temperature=None):
#         action, log_prob, entropy = self.execute_model(obs, temperature=temperature)
#         return action, log_prob, entropy
    
#     def evaluate_actions(self, obs, actions, temperature=None):
#         log_prob, entropy = self.execute_model(obs, actions=actions, temperature=temperature)
#         return log_prob, entropy
    
#     def predict(self, obs, temperature=None):
#         action, log_prob, entropy = self.execute_model(obs, temperature=temperature)
#         return action
        
# # Q-network 정의(StateValue)
# class Critic(nn.Module):
#     def __init__(self, hidden_dim, max_states=100, embed_dim=1):
#         super().__init__()
#         self.state_embedding = nn.Embedding(num_embeddings=max_states, embedding_dim=embed_dim)
        
#         self.value_network = nn.Sequential(
#             nn.Linear(embed_dim, hidden_dim)
#             ,nn.ReLU()
#             ,nn.Linear(hidden_dim, hidden_dim)
#             ,nn.ReLU()
#             ,nn.Linear(hidden_dim, 1)
#         )

#     def forward(self, obs):
#         embed_obs = self.state_embedding(obs).squeeze(-2)
#         value = self.value_network(embed_obs)
#         return value
# ##############################################################################################################

# def check_gradients(model):
#     for name, param in model.named_parameters():
#         if param.grad is not None:
#             grad_norm = param.grad.norm().item()
#             print(f"{name}: grad norm = {grad_norm:.6f}")
            


# env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode="rgb_array")  # deterministic

# np.array(dir(env.unwrapped))

# env.unwrapped.desc.ravel()[19] == b'H'

# obs, info = env.reset()
# plt.imshow(env.render())
# # env.observation_space.n

# memory_size = 1024
# batch_size = 64
# sample_only_until = 500
# n_epochs = 1
# target_update_interval = 100

# gamma = 0.9

# # Actor Network
# actor_network = Actor(action_dim=env.action_space.n, hidden_dim=32,
#                        max_states=env.observation_space.n, embed_dim=2).to(device)
# actor_optimizer = optim.AdamW(actor_network.parameters(), lr=5e-5, weight_decay=1e-2)

# # main network
# main_critic_network = Critic(hidden_dim=32,
#                        max_states=env.observation_space.n, embed_dim=2).to(device)
# critic_optimizer = optim.AdamW(main_critic_network.parameters(), lr=1e-3, weight_decay=1e-2)

# # target network
# target_critic_network = Critic(hidden_dim=32,
#                        max_states=env.observation_space.n, embed_dim=2).to(device)
# target_critic_network.load_state_dict(main_critic_network.state_dict())

# # ★ (Prioritize Replay Buffer)
# memory = ReplayMemory(max_size=memory_size, batch_size=batch_size, method='sequential')




# # loss_function = nn.MSELoss()
# critic_loss_function = nn.SmoothL1Loss(reduction='none')

# num_episodes = 300
# total_step = 0

# with tqdm(total=num_episodes) as pbar:
#     for episode in range(num_episodes):
#         obs, info = env.reset()
#         done = False
#         i = 0
        
#         T = np.logspace(2, 0, num=num_episodes)[episode]
#         # T = max(0.5, 5 * (0.97 ** episode))
#         cumulative_reward = 0
#         while(not done):
#             obs_tensor = torch.LongTensor([obs]).to(device)
#             action, log_prob, entropy  = actor_network(obs_tensor, temperature=T)
#             action = action.item()
#             log_prob = log_prob.item()
            
#             value = main_critic_network(obs_tensor)
#             value = value.item()
            
#             next_obs, reward, terminated, truncated, info = env.step(action)
            
#             # if env.unwrapped.desc.ravel()[next_obs] == b'H':
#             #     reward -= 1
#             # elif env.unwrapped.desc.ravel()[next_obs] == b'G':
#             #     reward += 100
#             # elif next_obs == obs:
#             #     reward -= 1
#             # else:
#             reward -= 0.01  # step-penalty
#             done = terminated or truncated
            
#             experience = (obs, action, log_prob, next_obs, reward, done, value)
            
            
#             # buffer에 experience 저장
#             memory.push(experience)

#             obs = next_obs
#             i += 1
#             if i >=100:
#                 break
#             cumulative_reward += reward

#             avg_critic_loss = 0
#             avg_actor_loss = 0
#             if len(memory) >= sample_only_until:
#                 for epoch in range(n_epochs):
#                     # for batch in memory:
#                     sampled_exps, indices, weights = memory.sample()
#                     # print(epoch, len(sampled_exps), memory.size, memory._iter_sample_pointer)

#                     weights = torch.FloatTensor(weights).to(device)
#                     batch_obs = torch.LongTensor(np.stack([sample[0] for sample in sampled_exps])).view(-1,1).to(device)
#                     batch_actions = torch.LongTensor(np.stack([sample[1] for sample in sampled_exps])).view(-1,1).to(device)
#                     batch_log_probs = torch.FloatTensor(np.stack([sample[2] for sample in sampled_exps])).view(-1,1).to(device)
#                     batch_next_obs = torch.LongTensor(np.stack([sample[3] for sample in sampled_exps])).view(-1,1).to(device)
#                     batch_rewards = torch.FloatTensor(np.stack([sample[4] for sample in sampled_exps])).view(-1,1).to(device)
#                     batch_dones = torch.FloatTensor(np.stack([sample[5] for sample in sampled_exps])).view(-1,1).to(device)
#                     batch_values = torch.FloatTensor(np.stack([sample[6] for sample in sampled_exps])).view(-1,1).to(device)
                    
#                     # compute actor
#                     log_prob, entropy = actor_network.evaluate_actions(batch_obs, batch_actions, temperature=T)
                    
#                     # compute critic
#                     value = main_critic_network(batch_obs)
#                     next_value = target_critic_network(batch_next_obs)
                    
#                     # target
#                     td_target = batch_rewards + gamma * next_value.detach() * (1-batch_dones)
                    
#                     # critic_loss
#                     critic_loss_unreduced = critic_loss_function(value, td_target)
#                     critic_loss = (critic_loss_unreduced * weights).mean()
                    
#                     # actor_loss
#                     advantage = (td_target - value).detach()    # advantage = td_error
#                     if len(advantage) > 1:
#                         advantage = (advantage - advantage.mean()) / (advantage.std()+1e-8)
#                     # actor_loss = -(log_prob * advantage + 0.2 * entropy).mean()
#                     actor_loss = -(log_prob * advantage).mean()
                    
#                     # critic update
#                     critic_optimizer.zero_grad()
#                     critic_loss.backward()
#                     critic_optimizer.step()
                    
#                     # actor update
#                     actor_optimizer.zero_grad()
#                     actor_loss.backward()
#                     actor_optimizer.step()

#                     # with torch.no_grad():
#                     #     td_errors = critic_loss_unreduced.detach().cpu().numpy().reshape(-1)
#                     #     memory.update_priorities(indices, td_errors)
                    
#                     avg_critic_loss += critic_loss.to('cpu')
#                     avg_actor_loss += actor_loss.to('cpu')
#                     # break

#             if total_step % target_update_interval == 0:
#                 target_critic_network.load_state_dict(main_critic_network.state_dict())
#                 # print('target_network update')

#             total_step += 1
            
#         if episode % 1 == 0:
#             pbar.set_postfix(critic_loss=f"{avg_critic_loss/(n_epochs):.3f}", 
#                             actor_loss=f"{avg_actor_loss/(n_epochs):.3f}",
#                             Len_episodes=f"{i}",
#                             total_reward = f"{cumulative_reward:.2f}"
#                             )
#         pbar.update(1)


# # Simulation Test ---------------------------------------------------------------------------------
# obs, info = env.reset()
# # env.render()
# i = 0
# done = False
# while (done is not True):
    
#     with torch.no_grad():
#         actor_network.eval()
#         action, _, _ = actor_network(torch.LongTensor([obs]).to(device))
#         action = action.item()  
        
#         next_obs, reward, terminated, truncated, info = env.step(action)
#         done = terminated or truncated
        
#         plt.imshow(env.render())
#         plt.show()
#         time.sleep(0.1)
#         clear_output(wait=True)
#         obs = next_obs
#     i += 1
#     if i >=30:
#         break


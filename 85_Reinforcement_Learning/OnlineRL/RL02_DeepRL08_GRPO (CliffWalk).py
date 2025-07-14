
## 0. Define MDP environment: Cliff Walk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

""" Tile layout (36=start, 47=goal, 37-46=cliff)
0	1	2	3	4	5	6	7	8	9	10	11
12	13	14	15	16	17	18	19	20	21	22	23
24	25	26	27	28	29	30	31	32	33	34	35
36	37	38	39	40	41	42	43	44	45	46	47
"""

cliff_states = np.arange(37, 47)  # States for cliff tiles
goal_state = 47 # Goal state


# def get_reward(state: int, cliff_pos: np.array, goal_pos: int) -> int:
def get_reward(state): #when arriving at state
    """
    Compute reward for given state
    """
    if state == goal_state:       # Reward of +100 for reaching goal
        return 100
    elif state in cliff_states:    # Reward of -100 for falling down cliff
        return -100
    else:                       # Otherwise, reward of -1 for each move
        return -1



def get_state(agent_pos):
    """
    (x,y)-position -> state integer in [0,47]
    """
    return 12 * agent_pos[0] + agent_pos[1]


def get_position(state):
    """
    state integer in [0,47] -> (x,y)-position
    """
    return (int(np.floor(state / 12)), state % 12)



def move_agent(agent_pos, action):
    """
    Move agent to new position based on current position and action
    """
    # Retrieve agent position
    (pos_x, pos_y) = agent_pos

    if action == 0:  # Up
        pos_x = pos_x - 1 if pos_x > 0 else pos_x
    elif action == 1:  # Down
        pos_x = pos_x + 1 if pos_x < 3 else pos_x
    elif action == 2:  # Left
        pos_y = pos_y - 1 if pos_y > 0 else pos_y
    elif action == 3:  # Right
        pos_y = pos_y + 1 if pos_y < 11 else pos_y
    else:  # Infeasible move
        raise Exception("Infeasible move")

    agent_pos = (pos_x, pos_y)

    return agent_pos


def get_Q_from_V(V_table, gamma):
    """
    Given V_table, output Q_table of size (4,48)

    """
    Q_table = np.zeros((4, 48))

    for state in range(37): 
        for action in range(4):
            pos = get_position(state)
            pos_next = move_agent(pos, action)
            state_next = get_state(pos_next)
            r = get_reward(state_next)
            Q_table[action, state] = r + gamma * V_table[state_next] # Q (a, s) = r + gamma * V(s')
    
    return Q_table


def visualize_value_function(V, title="Value Function"):
    """
    Visualizes the value function as a heatmap with *every* value annotated in black.
    V must be a flat vector of length 48.
    """
    V_grid = V.reshape((4, 12))
    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    im = ax.imshow(V_grid, cmap="viridis", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Value", rotation=270, labelpad=15)
    for (i, j), val in np.ndenumerate(V_grid):
        ax.text(
            j, i,                # column (x), row (y)
            f"{val:.1f}",        # formatted label
            ha="center", va="center",
            color="black",       # guaranteed to contrast
            fontsize=8
        )
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def visualize_policy(policy, title="Policy (Greedy)"):
    """
    Visualizes the policy as arrows on the grid.
    The policy is assumed to be of shape (4,37) for nonterminal states.
    Terminal states (37-47) are left blank.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(np.zeros((4,12)), cmap="Greys", vmin=0, vmax=1)
    action_arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    for state in range(37):
        pos = get_position(state)
        best_action = np.argmax(policy[:,state])
        arrow = action_arrows[best_action]
        ax.text(pos[1], pos[0], arrow, ha="center", va="center", color="red", fontsize=16)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()




#################################################################################################################
# K-step TD #####################################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import trange

print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

class ValueNetwork(nn.Module):
    def __init__(self, input_dim=2, fourier_dim=32, hidden_dim=128):
        super().__init__()
        B = torch.randn(fourier_dim, input_dim) * 10  
        self.register_buffer('B', B)  

        self.net = nn.Sequential(
            nn.Linear(2 * fourier_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def fourier_features(self, x): # x: (batch_size, 2)
        proj = 2 * np.pi * x @ self.B.T
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, x):
        phi = self.fourier_features(x)
        return self.net(phi).squeeze(-1)


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=2, fourier_dim=32, hidden_dim=128, n_actions=4):
        super().__init__()
        B = torch.randn(fourier_dim, input_dim) * 10
        self.register_buffer('B', B)

        self.net = nn.Sequential(
            nn.Linear(2 * fourier_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def fourier_features(self, x):
        proj = 2 * np.pi * x @ self.B.T
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, x):
        phi = self.fourier_features(x)
        logits = self.net(phi)
        return torch.softmax(logits, dim=-1)

""" Tile layout (36=start, 47=goal, 37-46=cliff)
0	1	2	3	4	5	6	7	8	9	10	11
12	13	14	15	16	17	18	19	20	21	22	23
24	25	26	27	28	29	30	31	32	33	34	35
36	37	38	39	40	41	42	43	44	45	46	47
"""

Vnet = ValueNetwork().to(device)
Pnet = PolicyNetwork().to(device)

value_optimizer = optim.SGD(Vnet.parameters(), lr=1e-4)
policy_optimizer = optim.SGD(Pnet.parameters(), lr=1e-4)

gamma = 0.99     
num_episodes = 10000


for episode in trange(num_episodes):
    s = np.random.randint(37)
    agent_pos = get_position(s)  
    trajectory = []  #  tuples (state_tensor, action, reward)

    done = False
    run = 0
    while not done:
        if run > 101:
            break
        x0, y0 = agent_pos
        state_tensor = torch.tensor([(x0 / 3.0, y0 / 11.0)], dtype=torch.float32)       # normalize (0~1) state coordinates

        probs = Pnet(state_tensor.to(device)).squeeze(0)  
        action = torch.multinomial(probs.to('cpu').detach(), 1).item()

        agent_pos = move_agent(agent_pos, action)
        s_next = get_state(agent_pos)
        reward = get_reward(s_next)

        trajectory.append((state_tensor, action, reward))

        if (s_next == goal_state) or (s_next in cliff_states):
            done = True
        run +=1 

    returns = []
    G = 0.0
    for (_, _, r) in reversed(trajectory):
        G = r + gamma * G  
        returns.append(G)
    returns.reverse()

    loss_P = torch.tensor(0.0)
    loss_V = torch.tensor(0.0)
    for idx, (state_tensor, action, _) in enumerate(trajectory):
        G_t = returns[idx]                      
        V_t = Vnet(state_tensor.to(device)).squeeze(0)     
        # Policy gradient loss: log π(a|s) * (G_t − V(s))
        logp = torch.log(Pnet(state_tensor.to(device))[0, action])
        advantage = (G_t - V_t.detach())
        loss_P = loss_P - logp * advantage
        # Value function loss: ½ (G_t − V(s))^2
        loss_V = loss_V + 0.5 * (G_t - V_t) ** 2

    policy_optimizer.zero_grad()
    loss_P.backward()
    policy_optimizer.step()

    value_optimizer.zero_grad()
    loss_V.backward()
    value_optimizer.step()




grid_input = torch.tensor([
    (x / 3.0, y / 11.0) for x in range(4) for y in range(12)
], dtype=torch.float32)

with torch.no_grad():
    predicted_V = Vnet(grid_input.to(device)).to('cpu').detach().numpy()
predicted_V[-11:] = 0

visualize_value_function(predicted_V, title="Optimal Value Function (MC‐A2C)")

with torch.no_grad():
    pi = torch.zeros((4, 48))
    for state_idx in range(48):
        x, y = get_position(state_idx)
        inp = torch.tensor([(x / 3.0, y / 11.0)], dtype=torch.float32)
        pi[:, state_idx] = Pnet(inp.to(device)).squeeze(0).to('cpu').detach()
visualize_policy(pi.numpy(), title="Optimal Policy (MC‐A2C)")









############################################################################################################
# GRPO #####################################################################################################
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import trange


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=2, fourier_dim=32, hidden_dim=128, n_actions=4):
        super().__init__()
        B = torch.randn(fourier_dim, input_dim) * 10
        self.register_buffer('B', B)

        self.net = nn.Sequential(
            nn.Linear(2 * fourier_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def fourier_features(self, x):
        proj = 2 * np.pi * x @ self.B.T
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, x):
        phi = self.fourier_features(x)
        logits = self.net(phi)
        return torch.softmax(logits, dim=-1)


Pnet = PolicyNetwork().to(device)
policy_optimizer = optim.Adam(Pnet.parameters(), lr=1e-3)

gamma = 1.0                # undiscounted (only terminal reward is used)
N = 1280                    # # of outer “batches”
clip_epsilon = 0.2         # PPO‐style clipping parameter
ppo_epochs = 4             # # of mini‐epochs per batch
trajectory_batch_size = 8  # how many “groups” of episodes per batch; effective batch size = traj batch size * group size 
PPO_batch_size = 32        # minibatch size for gradient steps
group_size = 4             # G = # of episodes to sample per initial state



for _ in trange(N):
    trajectory_batch = []  # will store tuples (state, action, old_prob, advantage)

    # (C.1) For each “group” (i.e. each initial state), collect G full episodes
    for _ in range(trajectory_batch_size):
        # Pick a random starting state in [0..36]
        s0 = np.random.randint(37)
        agent_pos = get_position(s0)

        # We will store all G “full” trajectories (each a list of (state,action,old_prob,_))
        group_trajectories = []
        group_rewards = []  # terminal reward of each trajectory

        for g in range(group_size):
            agent_pos = get_position(s0)
            single_traj = []  

            step = 0
            while True:
                if step > 101:
                    break
                x0, y0 = agent_pos
                state_tensor = torch.tensor([(x0 / 3.0, y0 / 11.0)], dtype=torch.float32)

                with torch.no_grad():
                    probs = Pnet(state_tensor.to(device)).squeeze(0) 

                action = torch.multinomial(probs.to('cpu').detach(), 1).item()

                agent_pos = move_agent(agent_pos, action)
                s = get_state(agent_pos)
                reward = get_reward(s)

                prob_old = probs[action].detach()
                single_traj.append((state_tensor, action, prob_old, reward))

                if (s == goal_state) or (s in cliff_states):
                    break
                step +=1 
            terminal_reward = single_traj[-1][3] 
            group_rewards.append(terminal_reward)
            group_trajectories.append(single_traj)

        r_array = np.array(group_rewards, dtype=np.float32)
        mu = float(r_array.mean())
        sigma = float(r_array.std(ddof=0))
        if sigma == 0.0:
            sigma = 1.0  # A=0 if all rewards are identical throughout group
        for g in range(group_size):
            A_g = (group_rewards[g] - mu) / sigma

            # For every step in the g-th trajectory, use advantage A_g
            for (st, act, old_p, _) in group_trajectories[g]:
                trajectory_batch.append((st, act, old_p, A_g))

    states = torch.cat([item[0] for item in trajectory_batch], dim=0)          
    actions = torch.tensor([item[1] for item in trajectory_batch], dtype=torch.long) 
    old_probs = torch.stack([item[2] for item in trajectory_batch])             
    advantages = torch.tensor([item[3] for item in trajectory_batch], dtype=torch.float32)  


    for _ in range(ppo_epochs):
        idx = torch.randperm(len(states))
        for i in range(0, len(states), PPO_batch_size):
            b_idx = idx[i:i+PPO_batch_size]
            b_states = states[idx].to(device)
            b_actions = actions[idx].to(device)
            b_old_probs = old_probs[idx].to(device)
            b_adv = advantages[idx].to(device)

            # Ratio
            ratio = Pnet(b_states.to(device))[range(len(b_states)), b_actions] / b_old_probs    

            # Clipped surrogate objective
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * b_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Optimize policy
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

with torch.no_grad():
    pi_vis = torch.zeros((4, 48))
    for state_idx in range(48):
        x, y = get_position(state_idx)
        inp = torch.tensor([(x / 3.0, y / 11.0)], dtype=torch.float32)
        pi_vis[:, state_idx] = Pnet(inp.to(device)).squeeze(0).to('cpu').detach()

visualize_policy(pi_vis.numpy(), "Optimal policy (GRPO)")

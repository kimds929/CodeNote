###########################################################################################################
# print("gym version", gym.__version__)

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym

import time
from IPython.display import clear_output

###########################################################################################################

exec_example = False
###########################################################################################################
from collections import deque
def is_path_available(grid):
    """S → G 경로가 존재하는지 BFS로 확인"""
    height, width = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    q = deque()

    start = np.argwhere(grid == 'S')[0]
    goal = tuple(np.argwhere(grid == 'G')[0])
    q.append(tuple(start))
    visited[start[0], start[1]] = True

    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                if not visited[nr, nc] and grid[nr, nc] != 'H':
                    visited[nr, nc] = True
                    q.append((nr, nc))
    return False


def generate_frozenlake_map(width, height, hole=None, max_retries=100):
    """S→G 경로가 존재하는 안전한 FrozenLake 맵을 생성 (문제 없는 맵이 나올 때까지 반복)"""
    size = width * height
    if hole is None:
        hole = int(np.sqrt(size))
    elif type(hole) == float:
        hole = int(size * hole)

    for attempt in range(max_retries):
        # 1. 맵 기본 설정
        grid = np.full((height, width), 'F', dtype='<U1')
        grid[0, 0] = 'S'
        grid[height - 1, width - 1] = 'G'

        # 2. 구멍 위치 무작위 설정
        all_positions = np.array([
            (r, c) for r in range(height) for c in range(width)
            if (r, c) not in [(0, 0), (height - 1, width - 1)]
        ])
        indices = np.random.choice(len(all_positions), size=min(hole, len(all_positions)), replace=False)
        for r, c in all_positions[indices]:
            grid[r, c] = 'H'

        # 3. 경로 확인
        if is_path_available(grid):
            return [''.join(row) for row in grid]
    raise ValueError(f"경로를 확보할 수 있는 맵 생성 실패 (시도 횟수: {max_retries})")


# class SightMask():
#     def __init__(self, env):
#         self.n_row = env.unwrapped.nrow
#         self.n_col = env.unwrapped.ncol
#         self.pathes = []
#         self.render = env.render()
#         self.reset()
    
#     def append(self, state, terminated=False, truncated=False, done=False):
#         self.pathes[-1].append(state)
        
#         if terminated or truncated or done:
#             self.pathes.append([])
    
#     def reset(self):
#         self.pathes = [[0],[]]
    
#     def mask(self, pathes=None):
#         pathes = self.pathes if pathes is None else pathes
            
#         # 방문 마스크 생성 (방문하면 True)
#         visited = np.zeros((self.n_row, self.n_col), dtype=bool)
#         for path in pathes:
#             for s in path:
#                 r, c = divmod(s, self.n_col)
#                 visited[r, c] = True

#         # 배경 크기에 맞춘 RGBA 오버레이 생성
#         H, W, _ = self.render.shape
#         cell_h = H // self.n_row
#         cell_w = W // self.n_col

#         overlay = np.zeros((H, W, 4), dtype=np.float32)

#         # 기본: 검정 + 불투명(α=1)로 가려버림
#         overlay[..., :3] = 0.0
#         overlay[..., 3]  = 1.0

#         # 방문한 칸은 투명(α=0)으로 뚫기
#         for r in range(self.n_row):
#             for c in range(self.n_col):
#                 if visited[r, c]:
#                     overlay[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w, 3] = 0.0
#         return overlay

class SightMask:
    def __init__(self, env):
        self.n_row = env.unwrapped.nrow
        self.n_col = env.unwrapped.ncol
        self.render = env.render()
        self.visited = np.zeros((self.n_row, self.n_col), dtype=bool)
        self.reset()

    def append(self, state, terminated=False, truncated=False, done=False):
        r, c = divmod(state, self.n_col)
        self.visited[r, c] = True   # 경로 대신 Boolean mask만 갱신

    def reset(self):
        # 방문한 셀을 저장하는 Boolean mask
        self.visited = np.zeros((self.n_row, self.n_col), dtype=bool)
        # 시작점(0번 state)은 자동 방문 처리
        self.visited[0, 0] = True

    def mask(self):
        # 배경 크기에 맞춘 RGBA 오버레이 생성
        H, W, _ = self.render.shape
        cell_h = H // self.n_row
        cell_w = W // self.n_col

        overlay = np.zeros((H, W, 4), dtype=np.float32)
        # 기본: 검정색, 알파=1로 가림
        overlay[..., :3] = 0.0
        overlay[..., 3] = 1.0

        # 방문한 셀은 투명(α=0)
        for r in range(self.n_row):
            for c in range(self.n_col):
                if self.visited[r, c]:
                    overlay[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w, 3] = 0.0

        return overlay


def frozenlake_visualize_grid_probs(prob_map, env, ax=None, z_base=5, return_plot=False):
    n_rows = env.unwrapped.nrow
    n_cols = env.unwrapped.ncol
    grid = env.unwrapped.desc.astype(str)
    
    
    import matplotlib.cm as cm
    # 방향: ← ↓ → ↑
    dirs = [(-0.3, 0), (0, -0.3), (0.3, 0), (0, 0.3)]
    color_dict = {'S': 'mediumseagreen', 'H':'blue', 'G':'red'}

    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        created_ax = True
    
    norm = plt.Normalize(vmin=0.2, vmax=0.3)
    cmap = cm.get_cmap('jet')

    for i in range(n_rows):
        for j in range(n_cols):
            cx, cy = j, (n_rows-1) -i  # 좌측 상단이 index 0
            cell_probs = prob_map[i, j]
            max_dir = np.argmax(cell_probs)
            
            label = grid[i, j]
            if label != 'F':
                ax.text(cx, cy, label, fontsize=12, color=color_dict[label], 
                        ha='center', va='center', weight='bold', zorder=z_base+3)
            for d, (dx, dy) in enumerate(dirs):
                prob = cell_probs[d]
                color = cmap(norm(prob))
                ax.arrow(cx, cy, dx * prob * 2, dy * prob * 2,
                        head_width=0.05, head_length=0.05,
                        fc=color, ec=color, alpha=0.5, zorder=z_base+2)

                # 색상 조건: max 확률이면 빨간색, 아니면 검정
                text_color = 'red' if d == max_dir else 'black'
                alpha = 1 if d == max_dir else 0.5
                # 확률 수치 annotation
                offset_x, offset_y = dx * 0.9, dy * 0.9
                ax.text(cx + offset_x, cy + offset_y, f"{prob:.3f}",
                        fontsize=9, ha='center', va='center', color=text_color, alpha=alpha, zorder=z_base+3)
    # 컬러바 추가
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Action Probability')

    # 축 및 스타일
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(np.arange((n_rows-1),-1,-1))
    ax.xaxis.set_ticks_position('top')     # x축 눈금을 위쪽으로
    ax.xaxis.set_label_position('top')     # x축 라벨도 위쪽으로
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=z_base+1)
    ax.set_xlim(-0.5, (n_cols-1) + 0.5)
    ax.set_ylim(-0.5, (n_rows-1) + 0.5)
    ax.set_aspect('equal')
    ax.set_title("Policy Action Probabilities (← ↓ → ↑) with Values")
    plt.tight_layout()

    if created_ax:
        return ax
    elif return_plot:
        plt.close()
        return fig
    else:
        plt.show()

###########################################################################################################



# example 1
if exec_example:

    custom_map = [
        "SFFFF",
        "FHFHF",
        "FFFFF",
        "HFFHF",
        "FFFHG"
    ]
    # S	시작 지점
    # F	평지 (Free)
    # H	구멍 (Hole)
    # G	목적지 (Goal)

    custom_map = generate_frozenlake_map(4,4, hole=0.2)

    # env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="ansi")  # deterministic
    env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode="ansi")  # deterministic
    observation, info = env.reset()
    print(env.render())

    env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode="rgb_array") 
    observation, info = env.reset()
    plt.imshow(env.render())

    ###########################################################################################################
    custom_map = generate_frozenlake_map(4,4, hole=0.15)
    env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode="rgb_array")
    obs, info = env.reset()
    plt.imshow(env.render())
    plt.show()


    obs, info = env.reset()
    for i in range(50):
        # Animation
        if i % 1 == 0:
            plt.imshow(env.render())
            plt.show()
            time.sleep(0.05)
            clear_output(wait=True)
            
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
            
        if terminated:
            if obs == env.observation_space.n:
                print("Goal reached!", "Reward:", reward)
            else:
                print("Fail.", "Reward:", reward)
        elif truncated:
            print("Fail to find goal", "Reward:", reward)
            break

    plt.imshow(env.render())
    plt.show()




# example 2
if exec_example:
    custom_map = generate_frozenlake_map(5,5, hole=0.15)
    env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode="rgb_array")
    obs, info = env.reset()


    sight_mask = SightMask(env)
    plt.imshow(env.render())
    plt.imshow(sight_mask.mask())
    plt.show()


    # Episode
    sight_mask.reset()


    obs, info = env.reset()

    i=0
    while True:
        # Animation
        if i % 1 == 0:
            plt.imshow(env.render())
            plt.imshow(sight_mask.mask())
            plt.show()
            time.sleep(0.05)
            clear_output(wait=True)
            
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        sight_mask.append(obs, terminated=terminated, truncated=truncated)
        
        if terminated:
            if obs == env.observation_space.n:
                print("Goal reached!", "Reward:", reward)
            else:
                print("Fail.", "Reward:", reward)
            break
        elif truncated or i > 100:
            print("Fail to find goal", "Reward:", reward)
            break
        i+=1
    
    plt.imshow(env.render())
    plt.imshow(sight_mask.mask())
    plt.show()

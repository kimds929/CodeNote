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
###########################################################################################################

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
            print("Goal reached!", "Reward:", reward)
            break
        elif truncated:
            print("Fail to find goal", "Reward:", reward)
            break

    plt.imshow(env.render())
    plt.show()


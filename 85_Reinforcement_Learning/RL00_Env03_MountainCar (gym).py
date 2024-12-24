
### Mountaincar : RGB Control #####################################################################
import matplotlib.pyplot as plt
from IPython.display import clear_output
import gym

# env = gym.make("MountainCar-v0", render_mode="human")
# env = gym.make("MountainCar-v0", render_mode="rgb_array")
env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
# action space : -1 : 1
state = env.reset()

for _ in range(180):
    if _ < 40:
        next_state, reward, done, truncated, info = env.step([-1.0])
    elif _ < 80:
        next_state, reward, done, truncated, info = env.step([1.0])
    elif _ < 120:
        next_state, reward, done, truncated, info = env.step([-1.0])
    else:
        next_state, reward, done, truncated, info = env.step([1.0])
    
    plt.title(f"{reward} / {done}")
    plt.imshow(env.render())
    plt.show()
    clear_output(wait=True)
    if done:
        break

env.state
import numpy as np
np.array(dir(env))
env.spec

env.action_space.sample()

# random action
env.reset()
plt.imshow(env.render())
plt.show()
i = 0
done = False
while not done:
    env.render()
    action = env.action_space.sample()  # Random action for testing
    next_state, reward, done, truncated, info = env.step(action)

    if i >= 100:
        break
    i +=1
env.close()

plt.imshow(env.render())
plt.show()


# state
#   position : 위치 (-1.2~0.6)
#   velocity : 현재 속도(-0.07~0.07)

# action
#   0 : 왼쪽으로 가속
#   1 : 아무것도 하지 않음
#   2 : 오른쪽으로 가속

# reward
#   목표위치 도달 : +1
#   목표위치 미도달 : -1








### Mountaincar : Human Control #####################################################################
import gym
import pygame

# 키보드 입력을 처리하기 위한 초기화
pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("MountainCar Manual Control")
pygame.time.delay(30)       # rendering 속도조절

# 환경 초기화
env = gym.make("MountainCarContinuous-v0", render_mode="human")
state = env.reset()
done = False

history = []
# 실행 루프
while not done:
    # 이벤트 처리
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # 키보드 입력 처리
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        action = [-1]  # 왼쪽 가속
    elif keys[pygame.K_RIGHT]:
        action = [1]  # 오른쪽 가속
    else:
        action = [0]  # 중립 (가속하지 않음)

    # 환경 한 단계 진행
    info = env.step(action)
    state, reward, done, truncated, _ = info
    history.append(info)

env.close()
pygame.quit()

len(history)





















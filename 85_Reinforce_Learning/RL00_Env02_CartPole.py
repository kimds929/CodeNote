# import gym
# env = gym.make('CartPole-v1')

# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()








import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math

class CartPoleEnv:
    def __init__(self):
        # 물리 상수들
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = self.mass_cart + self.mass_pole
        self.length = 0.5  # pole length
        self.polemass_length = self.mass_pole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # time step for simulation

        # 상태 변수들 (카트 위치, 속도, 막대 각도, 각속도)
        self.state = None
        self.reset()

    def reset(self):
        # 상태 초기화 (작은 무작위 초기값 부여)
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return self.state

    def step(self, action):
        # action = 0 (왼쪽으로 힘을 가함), action = 1 (오른쪽으로 힘을 가함)
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # 물리 공식 계산
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        theta_acc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.mass_pole * costheta ** 2 / self.total_mass))
        x_acc = temp - self.polemass_length * theta_acc * costheta / self.total_mass

        # 새로운 상태 갱신
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc
        self.state = (x, x_dot, theta, theta_dot)

        # 완료 여부 (극이 일정 각도를 넘으면 실패)
        done = bool(
            x < -5
            or x > 5
            or theta < -20 * math.pi / 180
            or theta > 20 * math.pi / 180
        )

        return np.array(self.state), done

    def render(self):
        # 상태 시각화
        x, _, theta, _ = self.state

        plt.figure(figsize=(5, 5))
        ax = plt.gca()
        ax.clear()

        # 바닥 선
        ax.add_patch(Rectangle((-2.4, -0.05), 4.8, 0.1, color='gray'))

        # 카트
        cart_width = 0.4
        cart_height = 0.2
        ax.add_patch(Rectangle((x - cart_width / 2, 0), cart_width, cart_height, color='blue'))

        # 막대 (극)
        pole_x = x + self.length * math.sin(theta)
        pole_y = cart_height + self.length * math.cos(theta)
        ax.plot([x, pole_x], [cart_height, pole_y], color='red', linewidth=5)

        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-0.5, 1.5)
        plt.pause(0.001)

# 간단한 시뮬레이션
env = CartPoleEnv()
state = env.reset()
env.render()

from IPython.display import clear_output
for _ in range(200):
    env.render()
    action = np.random.choice([0, 1])  # 임의의 행동 선택
    state, done = env.step(action)
    if done:
        break
    clear_output(wait=True)
plt.show()


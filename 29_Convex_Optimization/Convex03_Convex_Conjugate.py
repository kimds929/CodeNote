import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sympy as sp
from scipy.optimize import fsolve
from scipy.misc import derivative

# 접선의 기울기와 y절편을 구하는 함수
def tangent_line_point(func, x_value):
    # 접점에서의 함수 값 f(x_value)
    f_at_x = func(x_value)
    
    # 접점에서의 기울기 f'(x_value) (수치적 미분 사용)
    slope = derivative(func, x_value, dx=1e-6)
    
    # 접선의 y절편 계산: y = slope * x + intercept 이므로, intercept = f(x_value) - slope * x_value
    intercept = f_at_x - slope * x_value
    
    return slope, intercept

# 주어진 기울기에 해당하는 x 값을 찾고 접선의 y절편을 구하는 함수
def tangent_line_slope(func, target_slope):
    # x 값에서의 미분값과 target_slope의 차이를 계산하는 함수
    def slope_diff(x):
        return derivative(func, x, dx=1e-6) - target_slope
    
    # fsolve를 사용해 slope_diff(x) = 0을 만족하는 x 값을 찾음 (즉, 기울기가 target_slope인 x 값을 찾음)
    x0 = fsolve(slope_diff, 0)[0]  # 초기 추정치를 0으로 설정
    
    # 해당 x0에서 함수 값 계산
    f_at_x0 = func(x0)
    
    # 접선의 y절편 계산: intercept = f(x0) - slope * x0
    intercept = f_at_x0 - target_slope * x0
    
    return x0, target_slope, intercept


def square(x):
    return 0.5* (x-1)**2+3

def exponential(x):
    return np.exp(x)

# Conjugate function f*(y) 정의
def convex_conjugate(y, f):
    # 목적 함수 정의: - (x * y - f(x)) -> maximize (이것을 minimize로 변환)
    def objective(x):
        return -(x * y - f(x))
    
    # 초기 추정치 (0부터 시작하는 예시)
    x0 = 0.0
    
    # minimize를 사용하여 maximize 문제를 푼다
    result = minimize(objective, x0, method='BFGS')
    
    # 최적의 conjugate 값을 반환
    return -result.fun  # minimize의 음수 값을 취한다 (maximize)


func = square
# func = exponential
x_p = np.linspace(-2,5, 100)

colors = ['coral', 'goldenrod', 'mediumseagreen', 'steelblue', 'navy', 'purple']

plt.figure()
plt.title('Original function of f(x) = 0.5*(x-1)^2 + 3')
plt.plot(x_p, func(x_p), c="blue", label='f')
for ci, i in enumerate(range(-1,3)):
    point, slope, b = tangent_line_slope(func, i)
    plt.plot(x_p, x_p*slope + b, color=colors[ci], alpha=0.5, label=f"slope {i}")
    plt.scatter(0, b, color=colors[ci], alpha=0.5)
    # plt.text(0, b, f"{b:.1f}")
    plt.text(0, b, f"{b:.1f}", color=colors[ci])
plt.legend(loc='upper right')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.axvline(0, color='black')
plt.axhline(0, color='black')
plt.grid(True, alpha=0.2)
plt.show()





# y_p = np.linspace(-2, 5, 100)


y_p = np.linspace(0, 3.3, 100)
conjugate_vals = [convex_conjugate(y, func) for y in y_p]


# 결과를 출력
plt.figure()
plt.title('Convex Conjugate of f(x) = exp(x)')
plt.plot(y_p, conjugate_vals, c="green", label='f*')
for ci, i in enumerate(range(0,4)):
    conj = convex_conjugate(i, func)
    plt.scatter(i, conj, color=colors[ci], alpha=0.5, label=f"slope {i}")
    plt.text(i, conj, f"{conj:.1f}", color=colors[ci])

plt.legend(loc='upper left')
plt.xlabel('y')
plt.ylabel('f*(y)')
plt.axvline(0, color='black')
plt.axhline(0, color='black')
plt.grid(True, alpha=0.2)
plt.ylim(-2.1, 2)
plt.show()
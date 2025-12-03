################################################################################
import pyswarms as ps
import numpy as np
import pandas as pd

# 목적 함수 (PySwarms는 여러 입자를 동시에 평가하므로 최소 2D 배열 입력)
def simple_curve(x):
    # x.shape = (n_particles, dimensions)
    return 2*(x[:, 0]-0.5)**2 + 2       # output.shape #(n_particles,)

xp = np.linspace(-3,3, 100).reshape(-1,1)
plt.plot(xp, simple_curve(xp))
plt.show()

# 옵션 설정
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k':3, 'p':2}
# c1: 개인 학습 계수 (자신의 최적해를 향한 가중치)
# c2: 사회 학습 계수 (이웃/전역 최적해를 향한 가중치)
# w: 관성 계수 (이전 속도를 유지하는 정도)
# 'k': 3,      # (Local) 각 입자가 참고하는 이웃 수
# 'p': 2       # (Local) 이웃 중 몇 명을 선택할 확률

# 경계 설정 (예: -3 ~ 3)
bounds = (np.array([-3]), np.array([3]))


# 최적화기 생성
# Objective
#   ps.single :  하나의 목적 함수
#   ps.multi : 여러개의 목적함수
# GlobalPSO vs. LocalPSO
#   GlobalBestPSO: 모든 입자가 전역 최적해를 공유
#   LocalBestPSO: 일부 이웃 입자끼리만 정보 공유 → 지역 탐색 강화

optimizer = ps.single.GlobalBestPSO(
# optimizer = ps.single.LocalBestPSO(
    n_particles=10,
    dimensions=1,
    options=options,
    # bounds=bounds
    bounds=None
)


# 최적화 실행
cost, pos = optimizer.optimize(simple_curve, iters=50)

print("최적 비용:", cost)
print("최적 위치:", pos)

################################################################################
def bowl(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    return (x1-2)**2 + (x2+3)**2

x1 = np.linspace(-10, 10, 10)
x2 = np.linspace(-10, 10, 10)
X_mesh = np.array(np.array(np.meshgrid(x1,x2)).reshape(2,-1).T)
pred_banana = bowl(X_mesh).reshape(-1,1)

contour_df = pd.DataFrame(np.concatenate([X_mesh, pred_banana], axis=1), columns=['x1','x2','pred'])
contour = contour_df.set_index(['x1','x2']).unstack('x1')

plt.contourf([b for a, b in contour.columns], contour.index, contour, cmap='jet')
plt.colorbar()


# 옵션 설정
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# 경계 설정 (예: -3 ~ 3)
bounds = search_bound = np.array([[-10, 10], [-10,10]])

# 최적화기 생성
optimizer = ps.single.GlobalBestPSO(
    n_particles=10,
    dimensions=2,
    options=options,
    bounds=search_bound.T.tolist()
)

# 최적화 실행
cost, pos = optimizer.optimize(bowl, iters=50)

print("최적 비용:", cost)
print("최적 위치:", pos)


# 시각화 
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt

cost_history = optimizer.cost_history
plot_cost_history(cost_history)
plt.show()


################################################################################

def complex1(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    return 20 + (x1**2 - 10*np.cos(2*np.pi*x1)) + (x2**2 - 10*np.cos(2*np.pi*x2))

x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
X_mesh = np.array(np.array(np.meshgrid(x1,x2)).reshape(2,-1).T)
pred_banana = complex1(X_mesh).reshape(-1,1)

contour_df = pd.DataFrame(np.concatenate([X_mesh, pred_banana], axis=1), columns=['x1','x2','pred'])
contour = contour_df.set_index(['x1','x2']).unstack('x1')

plt.contourf([b for a, b in contour.columns], contour.index, contour, cmap='jet')
plt.colorbar()



# 옵션 설정
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# 경계 설정 (예: -3 ~ 3)
bounds = search_bound = np.array([[-10, 10], [-10,10]])

# 최적화기 생성
optimizer = ps.single.GlobalBestPSO(
    n_particles=10,
    dimensions=2,
    options=options,
    bounds=search_bound.T.tolist()
)

# 최적화 실행
cost, pos = optimizer.optimize(complex1, iters=50)

print("최적 비용:", cost)
print("최적 위치:", pos)



################################################################################

def complex2(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    return (x1**2 + x2 - 11)**2 + (x1+x2**2-7)**2
# 전역 최소점: 4개 존재 : (3, 2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)

x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
X_mesh = np.array(np.array(np.meshgrid(x1,x2)).reshape(2,-1).T)
pred_banana = complex2(X_mesh).reshape(-1,1)

contour_df = pd.DataFrame(np.concatenate([X_mesh, pred_banana], axis=1), columns=['x1','x2','pred'])
contour = contour_df.set_index(['x1','x2']).unstack('x1')

plt.contourf([b for a, b in contour.columns], contour.index, contour, cmap='jet')
plt.colorbar()



# 옵션 설정
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# 경계 설정 (예: -3 ~ 3)
bounds = search_bound = np.array([[-10, 10], [-10,10]])

# 최적화기 생성
optimizer = ps.single.GlobalBestPSO(
    n_particles=10,
    dimensions=2,
    options=options,
    bounds=search_bound.T.tolist()
)

# 최적화 실행
cost, pos = optimizer.optimize(complex2, iters=50)

print("최적 비용:", cost)
print("최적 위치:", pos)
################################################################################







# 목적 함수 정의
def objective_function(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    return (x1**2 + x2 - 11)**2 + (x1+x2**2-7)**2

# def objective_function(position):
#     x, y = position
#     return (x - 2)**2 + (y + 3)**2

# PSO 파라미터
num_particles = 5
num_iterations = 10
w = 0.5   # 관성 계수
c1 = 1.5  # 개인 최적 계수
c2 = 1.5  # 전역 최적 계수

# 초기 위치 지정 (일부는 직접 지정, 일부는 무작위)
positions = np.array([
    [0.0, 0.0],   # 직접 지정
    [2.0, -3.0],  # 직접 지정
    [4.0, 4.0],   # 직접 지정
    np.random.uniform(-5, 5, 2),  # 무작위
    np.random.uniform(-5, 5, 2)   # 무작위
])

# 초기 속도 (0으로 시작)
velocities = np.zeros_like(positions)

# 개인 최적값 초기화
personal_best_positions = positions.copy()
personal_best_scores = objective_function(positions)  # 벡터 반환

# 전역 최적값 초기화
global_best_index = np.argmin(personal_best_scores)
global_best_position = personal_best_positions[global_best_index]

# PSO 메인 루프
for iteration in range(num_iterations):
    # 속도 업데이트
    r1 = np.random.rand(num_particles, 2)
    r2 = np.random.rand(num_particles, 2)
    velocities = (w * velocities +
                  c1 * r1 * (personal_best_positions - positions) +
                  c2 * r2 * (global_best_position - positions))
    
    # 위치 업데이트
    positions += velocities
    
    # 새로운 점수 계산 (전체 입자 평가)
    scores = objective_function(positions)
    
    # 개인 최적 업데이트
    improved = scores < personal_best_scores
    personal_best_scores[improved] = scores[improved]
    personal_best_positions[improved] = positions[improved]
    
    # 전역 최적 업데이트
    global_best_index = np.argmin(personal_best_scores)
    global_best_position = personal_best_positions[global_best_index]
    
    print(f"Iteration {iteration+1}, Global Best Position: {global_best_position}, Score: {personal_best_scores[global_best_index]}")
################################################################################










################################################################################
# Gaussian Process + 최적화


import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# 1. 목적 함수 정의 (Himmelblau's function)
def objective_function(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

# 2. 초기 데이터 (임의의 3개 점)
X_train = np.array([[0.0, 0.0],
                    [2.0, 2.0],
                    [3.0, 1.0]])
y_train = objective_function(X_train)

# 3. GP 모델 설정
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

# 4. UCB 파라미터
beta = 2.0  # 탐색-활용 균형

# 5. 반복 탐색
for iteration in range(15):
    # GP 학습
    gp.fit(X_train, y_train)

    # 후보 점 생성 (2D grid)
    grid_size = 50
    x1_range = np.linspace(-5, 5, grid_size)
    x2_range = np.linspace(-5, 5, grid_size)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    X_candidates = np.column_stack([X1.ravel(), X2.ravel()])

    # GP 예측
    mu, sigma = gp.predict(X_candidates, return_std=True)

    # UCB 계산 (여기서는 최소값을 찾기 위해 -mu 사용)
    ucb_values = -mu + beta * sigma
    
    # UCB가 최대인 점 선택
    next_x = X_candidates[np.argmax(ucb_values)].reshape(1, -1)

    # 실제 함수 값 평가
    next_y = objective_function(next_x)

    # 데이터 업데이트
    X_train = np.vstack([X_train, next_x])
    y_train = np.append(y_train, next_y)

    print(f"Iteration {iteration+1}: next_x={next_x[0]}, next_y={next_y[0]:.4f}")

# 최종 결과
best_index = np.argmin(y_train)
print(f"Best found: x={X_train[best_index]}, y={y_train[best_index]:.4f}")











################################################################################
# Thompson Sampling + 최적화


import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize

# 1. 목적 함수 정의
def objective_function(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

# 2. MLP 모델 정의 (Dropout 포함)
class DropoutMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, dropout_p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

# 3. 초기 데이터
X_train = np.random.uniform(-5, 5, (5, 2))
y_train = objective_function(X_train)

# 4. 학습 함수
def train_model(model, X, y, epochs=200, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = loss_fn(output, y_tensor)
        loss.backward()
        optimizer.step()

# 5. Thompson Sampling 반복
model = DropoutMLP()
beta = 0.0  # TS에서는 beta 대신 샘플링된 함수 사용
n_iter = 15

for iteration in range(n_iter):
    # 모델 학습
    train_model(model, X_train, y_train)

    # 함수 샘플링 (MC Dropout)
    model.train()  # Dropout 활성화
    def sampled_function(x):
        x_t = torch.tensor(x.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            return model(x_t).item()

    # 샘플 함수 최적화 (minimize 사용)
    res = minimize(sampled_function, x0=np.random.uniform(-5, 5, 2),
                   bounds=[(-5, 5), (-5, 5)], method='L-BFGS-B')
    next_x = res.x.reshape(1, -1)
    next_y = objective_function(next_x)

    # 데이터 업데이트
    X_train = np.vstack([X_train, next_x])
    y_train = np.append(y_train, next_y)

    print(f"Iter {iteration+1}: next_x={next_x[0]}, next_y={next_y[0]:.4f}")

# 최종 결과
best_idx = np.argmin(y_train)
print(f"Best found: x={X_train[best_idx]}, y={y_train[best_idx]:.4f}")
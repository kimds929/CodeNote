import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
import torch

import time
from IPython.display import clear_output
# clear_output(wait=True)



# --------------------------------------------------------------------------------------------------------
# Tensorflow Regressor Problem ---------------------------------------------------------------------


# Dataset Define ----
test_dict = {'y': [10, 8, 20, 17, 15],
        'x1': [3, 2, 5, 4, 6],
        'x2': [10, 8, 5, 12, 7]}

df = pd.DataFrame(test_dict)

# Dataset 정의 ----
y_cols = ['y']
# X_cols = df.columns.drop(y_cols)
X_cols = ['x1']

y = df[y_cols]
X = df[X_cols]
y
X


# numpy data ----
X_np0 = X.copy()
X_np0['const'] = 1
X_np0

X_np1 = X_np0[['const'] + list(X_cols)]

X_np = X_np1.to_numpy()
y_np = y.to_numpy()
X_np
y_np


# w_init
w_init = np.array([15, -2]).reshape(-1,1)  # w initialize (Random)
print(f'X.shape: {X_np.shape} / w_init.shape : {w_init.shape}  /  y.shape : {y_np.shape}')


# Linear_Regression (sklearn)
LR = LinearRegression(fit_intercept=False)
LR.fit(X_np, y_np)
LR.coef_


# plotting
def plotting(w=False):
    if type(w) != bool:
        line_x = np.linspace(-1, 10, 50).reshape(-1,1)
        line_x = np.hstack([line_x ** 0, line_x])
        line_y = line_x @ w
        # print(w)

    plt.figure()
    plt.hlines(y=0, xmin=-5, xmax=10, alpha=0.5)
    plt.vlines(x=0, ymin=-5, ymax=30, alpha=0.5)
    plt.scatter(data=df, x='x1', y='y', s=50)

    if type(w) != bool:
        plt.plot(line_x[:,1], line_y, 'r--', alpha=0.3)
    plt.plot(line_x[:,1], LR.predict(line_x), 'orange', alpha=0.7) 
    plt.xlim([-1, 10])
    plt.ylim([-1, 30])
    
    plt.grid(alpha=0.5)
    plt.show()

# result summary plotting
def result_plot(w_history, mse_history, grad_history):
    w_history_reshape = np.array(w_history).reshape(-1, 2)
    grad_history_reshape = np.array(grad_history).reshape(-1,2)

    plt.figure(figsize=(15, 4.5))

    # mse plot
    plt.subplot(1, 3, 1)
    plt.title('loss_function')
    plt.plot(mse_history, label='mse')
    plt.xscale('log')
    plt.legend()

    # weight hitory plot
    plt.subplot(1, 3, 2)
    plt.title('weight\n' + str(w_history_reshape[-1]))
    plt.plot(w_history_reshape[:,0], label='const')
    plt.plot(w_history_reshape[:,1], label='x1')
    plt.xscale('log')
    plt.legend()

    # gradient history plot
    plt.subplot(1, 3, 3)
    plt.title('weight_gradient\n' + str(grad_history_reshape[-1]))
    plt.plot(grad_history_reshape[:,0], label='const')
    plt.plot(grad_history_reshape[:,1], label='x1')
    plt.xscale('log')
    plt.legend()
    plt.show()



# plotting
# w_init = np.array([15, -2]).reshape(-1,1)  # w initialize (Random)
plotting(w=w_init)






# Gradient Descent ----------------------------------
learning_rate = 0.003
epoch = 10000



# numpy ---------------------------------
w_init = np.array([15, -2]).reshape(-1,1)  # w initialize
w_np = w_init.copy()

w_np_history = []
mse_np_history = []
grad_np_history = []

for _ in range(epoch):
    # y_pred = X_np @ w_np

    # # gradient 구하기
    #     # small delta 를 활용한 미분  ★★★
    # mse_np = np.mean((y_np - X_np @ w_np) **2) # mse
    # mse_grad_list = []
    # delta = 1e-6
    # for i in range(len(w_np)):
    #     delta_list = [0, 0]
    #     delta_list[i] = delta
    #     delta_np = np.array(delta_list).reshape(-1,1)
    #     mse_grad = (np.mean((y_np - X_np @ w_np) **2) - np.mean((y_np - X_np @ (w_np - delta_np)) **2) ) / delta
    #     mse_grad_list.append(mse_grad)
    # mse_grad_w_np = np.array(mse_grad_list).reshape(-1,1)

        # 미분 공식을 활용한 미분  ★★★
    mse_np = np.mean((y_np - X_np @ w_np) **2) # mse
    mse_grad_w_np = - 2 * X_np.T @ (y_np - X_np @ w_np)

    # history 저장
    w_np_history.append(w_np)
    mse_np_history.append(mse_np)
    grad_np_history.append(mse_grad_w_np)

    # w값 업데이트
    w_np = w_np - learning_rate * mse_grad_w_np

    if _ <10 or (_ <100 and (_+1) % 20 == 0) or (_ < 1000 and (_+1) % 100 == 0) or ( (_+1) % 1000 == 0):
        print(f'{_+1}번째 epoch')
        print(f'w : {w_np.ravel()}')
        plotting(w_np)
        clear_output(wait=True)
        time.sleep(0.2)


result_plot(w_np_history, mse_np_history, grad_np_history)




################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# numpy -> torch
X_t = torch.tensor(X_np, dtype=torch.float32, device=device)  # (N, 2)
y_t = torch.tensor(y_np, dtype=torch.float32, device=device)  # (N, 1)



# Pytorch Autograd---------------------------------------------------------------
# https://tutorials.pytorch.kr/beginner/pytorch_with_examples.html
y_torch = torch.as_tensor(y_np, dtype=torch.float)
X_torch = torch.as_tensor(X_np, dtype=torch.float)

learning_rate = 0.003
epoch = 10000

w_init = np.array([15, -2]).reshape(-1,1)  # w initialize
# w_torch = torch.as_tensor(w_init, dtype=torch.float)
# w_torch.requires_grad_(True)
w_torch = torch.tensor(w_init, requires_grad=True, dtype=torch.float)

w_torch_history = []
mse_torch_history = []
grad_torch_history = []
for _ in range(epoch):
    y_pred = X_torch @ w_torch              # forward
    loss = ((y_torch - y_pred)**2).mean()   # loss
    loss.backward()                         # backward
    
    with torch.no_grad():
        # history 저장
        w_torch_history.append(w_torch.numpy())
        mse_torch_history.append(loss.numpy())
        grad_torch_history.append(w_torch.grad.numpy())
    
        w_torch -= learning_rate * w_torch.grad # weight update
        w_torch.grad.zero_()                    # weight initialize

    if _ <10 or (_ <100 and (_+1) % 20 == 0) or (_ < 1000 and (_+1) % 100 == 0) or ( (_+1) % 1000 == 0):
        print(f'{_+1}번째 epoch')
        print(f'w : {w_torch.detach().numpy().ravel()}')
        plotting(w_torch.detach().numpy())
        clear_output(wait=True)
        time.sleep(0.2)







# pytorch optimizer 활용------------------------------------------------
import torch.optim as optim
learning_rate = 0.003
epoch = 10000

w_init = np.array([15, -2]).reshape(-1,1)  # w initialize
w_opt = torch.tensor(w_init, dtype=torch.float32, device=device, requires_grad=True)
optimizer = optim.SGD([w_opt], lr=learning_rate)
def torch_mse(y_pred, y_true):
    return torch.mean((y_true - y_pred) ** 2)

w_history_opt = []
mse_history_opt = []
grad_history_opt = []

for ep in range(epoch):
    # forward
    y_pred = X_t @ w_opt            # forward
    loss = torch_mse(y_pred, y_t)   # loss

    # backward
    optimizer.zero_grad()   # zero_grad()
    loss.backward()         # backward

    # history
    w_history_opt.append(w_opt.detach().cpu().numpy().copy())
    mse_history_opt.append(loss.detach().cpu().item())
    grad_history_opt.append(w_opt.grad.detach().cpu().numpy().copy())

    # update
    optimizer.step()

    if ep < 10 or (ep < 100 and (ep+1) % 20 == 0) or (ep < 1000 and (ep+1) % 100 == 0) or ((ep+1) % 1000 == 0):
        print(f'{ep+1}번째 epoch')
        print(f'w : {w_opt.detach().cpu().numpy().ravel()}')
        plotting(w_opt.detach().cpu().numpy())
        clear_output(wait=True)
        time.sleep(0.2)

result_plot(w_history_opt, mse_history_opt, grad_history_opt)






# Pytorch Model ---------------------------------------------------------------
y_torch = torch.as_tensor(y_np, dtype=torch.float)
X_torch = torch.as_tensor(X_np, dtype=torch.float)

class Torch_LR(torch.nn.Module):
    def __init__(self):
        super(Torch_LR, self).__init__()
        self.linear = torch.nn.Linear(2,1, bias=False)
    
    def forward(self, x):
        x = self.linear(x)
        return x

torch_model = Torch_LR()


dir(torch_model)
list(torch_model.parameters())    # weight

epoch = 10000
learning_rate = 0.003
optimizer = torch.optim.SGD(torch_model.parameters(), lr=learning_rate)

w_torch_mdl_history = []
mse_torch_mdl_history = []
grad_torch_mdl_history = []
for _ in range(epoch):
    optimizer.zero_grad()                   # gradients를 clear해서 새로운 최적화 값을 찾기 위해 준비

    pred = torch_model(X_torch)                         # forward
    loss = torch.nn.functional.mse_loss(pred, y_torch)  # loss
    loss.backward()                                     # backward

    optimizer.step()                                    # weight update

    with torch.no_grad():
        # history 저장
        w_torch_mdl_history.append(torch_model.linear.weight.numpy())
        mse_torch_mdl_history.append(loss.numpy())
        grad_torch_mdl_history.append(torch_model.linear.weight.grad.numpy())

    if _ <10 or (_ <100 and (_+1) % 20 == 0) or (_ < 1000 and (_+1) % 100 == 0) or ( (_+1) % 1000 == 0):
        print(f'{_+1}번째 epoch')
        print(f'w : {torch_model.linear.weight.grad.numpy().ravel()}')
        plotting(torch_model.linear.weight.detach().numpy().reshape(-1,1))
        clear_output(wait=True)
        time.sleep(0.2)
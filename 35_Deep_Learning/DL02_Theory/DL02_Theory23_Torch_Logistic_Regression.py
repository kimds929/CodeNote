import random
import numpy as np
import matplotlib.pylab as plt
import torch

from IPython.display import clear_output
import time

# seed
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Plot 함수 (numpy / torch 모두 대응)
def plot_graph(X, y, X_hat=None, y_hat=None, str_title=None):
    # torch -> numpy 변환
    if torch.is_tensor(X): X = X.detach().cpu().numpy()
    if torch.is_tensor(y): y = y.detach().cpu().numpy()
    if X_hat is not None and torch.is_tensor(X_hat): X_hat = X_hat.detach().cpu().numpy()
    if y_hat is not None and torch.is_tensor(y_hat): y_hat = y_hat.detach().cpu().numpy()

    fig = plt.figure(figsize=(8, 6))
    if str_title is not None:
        plt.title(str_title, fontsize=20, pad=20)

    plt.plot(X, y, ls='none', marker='o', markeredgecolor='white')

    if X_hat is not None and y_hat is not None:
        plt.plot(X_hat, y_hat)

    plt.tick_params(axis='both', labelsize=14)
    plt.show()


# ----- dataset -----
x_0 = 2 + np.random.randn(5)
y_0 = np.zeros(5)

x_1 = 6 + np.random.randn(5)
y_1 = np.ones(5)

x = np.concatenate((x_0, x_1))
y = np.concatenate((y_0, y_1))

print(x)
print(y)

plot_graph(x, y, str_title='dataset')

# numpy -> torch
x_t = torch.tensor(x, dtype=torch.float32, device=device)  # (N,)
y_t = torch.tensor(y, dtype=torch.float32, device=device)  # (N,)


# ----- Weight, bias Initialize -----
W = torch.nn.Parameter(torch.tensor(np.random.randn(), dtype=torch.float32, device=device))
b = torch.nn.Parameter(torch.tensor(np.random.randn(), dtype=torch.float32, device=device))


def cross_entropy(y_pred, y_true):
    """
    [Cross Entropy]
    -sum( y*log(p) + (1-y)*log(1-p) )
    """
    y_pred = torch.clamp(y_pred, 1e-9, 1.0 - 1e-9)
    return -torch.sum(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))


lr = 0.003
optimizer = torch.optim.SGD([W, b], lr=lr)


# 초기 곡선
x_fl = np.linspace(np.min(x), np.max(x), num=100)
x_fl_t = torch.tensor(x_fl, dtype=torch.float32, device=device)

with torch.no_grad():
    y_fl = torch.sigmoid(W * x_fl_t + b)

plot_graph(x, y, X_hat=x_fl, y_hat=y_fl, str_title='Logistic Regression')


# ----- training loop -----
steps = 0
display_step = 1000
training_interval = 10000

for step in range(steps, steps + training_interval + 1):
    # y_pred = sigmoid(W*x + b)
    logits = W * x_t + b                # forward 1
    y_pred = torch.sigmoid(logits)      # forward 2

    loss = cross_entropy(y_pred=y_pred, y_true=y_t) # loss
    
    optimizer.zero_grad()       # zero_grad
    loss.backward()             # backward
    optimizer.step()            # step

    if step % display_step == 0:
        with torch.no_grad():
            y_fl = torch.sigmoid(W * x_fl_t + b)

        plot_graph(x, y, X_hat=x_fl, y_hat=y_fl,
                   str_title=f'Logistic Regression\nstep: {step} / loss: {loss.item():.6f}')
        clear_output(wait=True)

steps = step

with torch.no_grad():
    y_fl = torch.sigmoid(W * x_fl_t + b)

plot_graph(x, y, X_hat=x_fl, y_hat=y_fl,
           str_title=f'Logistic Regression\nstep: {step} / loss: {loss.item():.6f}')


# ----- Mesh plot -----
def sigmoid(w, x, b):
    # 정상 로지스틱 회귀식: w*x + b
    return torch.sigmoid(w * x + b)

x_mesh = np.linspace(0, 7, 100)
y_mesh = np.linspace(0, 1, 100)

xs, ys = np.meshgrid(x_mesh, y_mesh)
k = np.hstack([xs.reshape(-1, 1), ys.reshape(-1, 1)])  # (10000, 2)

k_t = torch.tensor(k, dtype=torch.float32, device=device)
xk = k_t[:, 0]
yk = k_t[:, 1]

with torch.no_grad():
    cond = (sigmoid(w=W, x=xk, b=b) < yk).detach().cpu().numpy()

fig = plt.figure(figsize=(8, 6))
plt.title('Logistic_Regression', fontsize=20, pad=20)
plt.plot(x, y, ls='none', marker='o', markeredgecolor='white')
plt.plot(x_fl, y_fl.detach().cpu().numpy())
plt.plot(k[cond][:, 0], k[cond][:, 1], marker='s', color='orange', alpha=0.1, markeredgecolor='white')
plt.show()

print("Final W, b:", W.detach().cpu().item(), b.detach().cpu().item())

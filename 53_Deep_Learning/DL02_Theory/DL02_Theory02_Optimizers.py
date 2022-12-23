import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Perceptron

import time
from IPython.display import clear_output
clear_output(wait=True)

# https://bit.ly/3bJ7a4G                # SGD, Momentum video
# https://gomguard.tistory.com/187      # optimizers1
# https://ucsd.tistory.com/50           # optimizers2
# https://distill.pub/2017/momentum/    # momentum visualization


def loss_function(x):
    # return (x - 0.1) **2 + 0.3
    return (x+0.6)*(x+0.23)*(x+0.2)*(x-0.75)

def gradient(x, function, eps=1e-6):
    return (function(x) - function(x-eps) )/ eps

x_min = -1
x_max = 1

xp = np.linspace(x_min, x_max, 100)
yp = loss_function(x=xp)
y_min, y_max = yp.min(), yp.max()

# loss_function plotting
plt.plot(xp, yp)



# gradient plotting
def gradient_graph(x_history, y_history, grad_history, title=False):
    plt.figure(figsize=(10,3.5))
    plt.subplot(1,2,1)
    if title:
        plt.title(title + '\nOptimizing')
    plt.plot(xp, yp)
    plt.plot(x_history, y_history, 'o-', color='red', alpha=0.2)
    plt.scatter(x_history[-1], y_history[-1], color='red', edgecolors='white', s=100)
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])

    plt.subplot(1,2,2)
    if title:
        plt.title(title + '\nGradient Change')
    plt.plot(grad_history, 'o-')
    plt.xlim([0, epoch])
    plt.show()








# Gradient Descent ---------------------------------------------------------
def update(x, grad, lr):
    return x - lr * grad

x_init = np.random.rand(1) * (x_max - x_min) + x_min      # random starting point
# x_init = -0.95
# x_init = 0
# x_init = 0.95

# learning_rate = 0.01     # X**2  너무작은 lr
# learning_rate = 0.1     # X**2  수렴
# learning_rate = 1.1     # X**2  발산

# learning_rate = 0.01      # x**4 너무작은 lr
learning_rate = 0.05     # X**4 수렴
# learning_rate = 1.5     # X**4 발산

epoch = 15

x_move = x_init

# history list
x_history = [x_move]
y_history = [loss_function(x_move)]
grad_history = []


for _ in range(epoch):
    # Gradient_Descent
    grad_x = gradient(x=x_move, function=loss_function, eps=1e-6)
    x_move = update(x=x_move, grad=grad_x, lr=learning_rate)
    y_move = loss_function(x_move)

    x_history.append(x_move)
    y_history.append(y_move)
    grad_history.append(grad_x)

    if (_+1) % 1 ==0:
        gradient_graph(x_history, y_history, grad_history, title='Gradient_Descent')
        clear_output(wait=True)







# Momentum  ---------------------------------------------------------
def update_momentum(x, v, gamma, grad, lr):
    v_new = gamma * v + lr * grad
    x_new = x - v_new
    return x_new, v_new

x_init = np.random.rand(1) * (x_max - x_min) + x_min      # random starting point
x_init = -0.95
# x_init = 0
# x_init = 0.95

# learning_rate = 0.01     # X**2  너무작은 lr
# learning_rate = 0.1     # X**2  수렴
# learning_rate = 1.1     # X**2  발산

# learning_rate = 0.01      # x**4 너무작은 lr
learning_rate = 0.05     # X**4 수렴
# learning_rate = 1.5     # X**4 발산

epoch = 15


# initial define
x_grad = x_init
x_momentum = x_init

    # Momentum
v = 0    # v_init
gamma = 0.8


# history list
x_grad_history = [x_grad]
y_grad_history = [loss_function(x_grad)]
grad_gd_history = []

x_momentum_history = [x_momentum]
y_momentum_history = [loss_function(x_momentum)]
momentum_history = []


for _ in range(epoch):
    # Gradient_Descent
    grad_gd_x = gradient(x=x_grad, function=loss_function, eps=1e-6)
    x_grad = update(x=x_grad, grad=grad_gd_x, lr=learning_rate)
    y_grad = loss_function(x_grad)

    x_grad_history.append(x_grad)
    y_grad_history.append(y_grad)
    grad_gd_history.append(grad_gd_x)

    # Momentum
    grad_momentum_x = gradient(x=x_momentum, function=loss_function, eps=1e-6)
    x_momentum, v = update_momentum(x=x_momentum, v=v, gamma=gamma, grad=grad_momentum_x, lr=learning_rate)
    y_momentum = loss_function(x_momentum)
    
    x_momentum_history.append(x_momentum)
    y_momentum_history.append(y_momentum)
    momentum_history.append(grad_momentum_x)
    

    if (_+1) % 1 ==0:
        gradient_graph(x_grad_history, y_grad_history, grad_gd_history, title='Gradient_Descent')
        gradient_graph(x_momentum_history, y_momentum_history, momentum_history, title='Momentum')
        clear_output(wait=True)











# RMSProp  ---------------------------------------------------------
def update_rmsprop(x, g, gamma, grad, lr, eps=1e-6):
    g_new = gamma * g + (1 - gamma) *(grad)**2
    x_new = x - lr/(np.sqrt(g_new)+ eps) * grad
    return x_new, g_new

x_init = np.random.rand(1) * (x_max - x_min) + x_min      # random starting point
# x_init = -0.95
# x_init = 0
# x_init = 0.95

# learning_rate = 0.01     # X**2  너무작은 lr
# learning_rate = 0.1     # X**2  수렴
# learning_rate = 1.1     # X**2  발산

# learning_rate = 0.01      # x**4 너무작은 lr
learning_rate = 0.05     # X**4 수렴
# learning_rate = 1.5     # X**4 발산

epoch = 15


# initial define
x_grad = x_init
x_momentum = x_init
x_rmsprop = x_init

    # Momentum
v = 0    # v_init
gamma = 0.8

    # RMSProp
g = 0     # g_init
eps = 1e-6


# history list
x_grad_history = [x_grad]
y_grad_history = [loss_function(x_grad)]
grad_gd_history = []

x_momentum_history = [x_momentum]
y_momentum_history = [loss_function(x_momentum)]
momentum_history = []

x_momentum_history = [x_rmsprop]
y_momentum_history = [loss_function(x_rmsprop)]
momentum_history = []

x_rmsprop_history = [x_rmsprop]
y_rmsprop_history = [loss_function(x_rmsprop)]
rmsprop_history = []


for _ in range(epoch):
    # Gradient_Descent
    grad_gd_x = gradient(x=x_grad, function=loss_function, eps=1e-6)
    x_grad = update(x=x_grad, grad=grad_gd_x, lr=learning_rate)
    y_grad = loss_function(x_grad)

    x_grad_history.append(x_grad)
    y_grad_history.append(y_grad)
    grad_gd_history.append(grad_gd_x)

    # Momentum
    grad_momentum_x = gradient(x=x_momentum, function=loss_function, eps=1e-6)
    x_momentum, v = update_momentum(x=x_momentum, v=v, gamma=gamma, grad=grad_momentum_x, lr=learning_rate)
    y_momentum = loss_function(x_momentum)
    
    x_momentum_history.append(x_momentum)
    y_momentum_history.append(y_momentum)
    momentum_history.append(grad_momentum_x)

    # RMSProp
    grad_rmsprop_x = gradient(x=x_rmsprop, function=loss_function, eps=1e-6)
    x_rmsprop, g = update_rmsprop(x=x_rmsprop, g=g, gamma=gamma, eps=eps, grad=grad_rmsprop_x, lr=learning_rate)
    y_rmsprop = loss_function(x_rmsprop)
    
    x_rmsprop_history.append(x_rmsprop)
    y_rmsprop_history.append(y_rmsprop)
    rmsprop_history.append(grad_rmsprop_x)
    
    if (_+1) % 1 ==0:
        gradient_graph(x_grad_history, y_grad_history, grad_gd_history, title='Gradient_Descent')
        gradient_graph(x_momentum_history, y_momentum_history, momentum_history, title='Momentum')
        gradient_graph(x_rmsprop_history, y_rmsprop_history, rmsprop_history, title='RMSProp')
        clear_output(wait=True)













# Adam  ---------------------------------------------------------
def update_adam(x, m, g, grad, lr, n_iter, b1=0.9, b2=0.999, eps=1e-8):
    m_new = b1 * m + (1 - b1) * grad
    g_new = b2 * g + (1 - b2) * (grad)**2

    m_hat = m_new / (1 - b1 ** n_iter)
    g_hat = g_new / (1 - b2 ** n_iter)

    x_new = x - lr/(np.sqrt(g_hat) + eps) * m_hat
    return x_new, m_new, g_new

x_init = np.random.rand(1) * (x_max - x_min) + x_min      # random starting point
# x_init = -0.95
x_init = 0
# x_init = 0.95

# learning_rate = 0.01     # X**2  너무작은 lr
# learning_rate = 0.1     # X**2  수렴
# learning_rate = 1.1     # X**2  발산

# learning_rate = 0.01      # x**4 너무작은 lr
learning_rate = 0.05     # X**4 수렴
# learning_rate = 1.5     # X**4 발산

epoch = 15


# initial define
x_grad = x_init
x_momentum = x_init
x_rmsprop = x_init
x_adam = x_init

    # Momentum
v = 0    # v_init
gamma = 0.8

    # RMSProp, Adam
g_rms = 0     # g_init
g_adam = 0     # g_init
eps = 1e-8

m_adam = 0
b1 = 0.9
b2 = 0.999


# history list
x_grad_history = [x_grad]
y_grad_history = [loss_function(x_grad)]
grad_gd_history = []

x_momentum_history = [x_momentum]
y_momentum_history = [loss_function(x_momentum)]
momentum_history = []

x_momentum_history = [x_rmsprop]
y_momentum_history = [loss_function(x_rmsprop)]
momentum_history = []

x_rmsprop_history = [x_rmsprop]
y_rmsprop_history = [loss_function(x_rmsprop)]
rmsprop_history = []

x_adam_history = [x_adam]
y_adam_history = [loss_function(x_adam)]
adam_history = []


for _ in range(epoch):
    # Gradient_Descent
    grad_gd_x = gradient(x=x_grad, function=loss_function, eps=1e-6)
    x_grad = update(x=x_grad, grad=grad_gd_x, lr=learning_rate)
    y_grad = loss_function(x_grad)

    x_grad_history.append(x_grad)
    y_grad_history.append(y_grad)
    grad_gd_history.append(grad_gd_x)

    # Momentum
    grad_momentum_x = gradient(x=x_momentum, function=loss_function, eps=1e-6)
    x_momentum, v = update_momentum(x=x_momentum, v=v, gamma=gamma, grad=grad_momentum_x, lr=learning_rate)
    y_momentum = loss_function(x_momentum)
    
    x_momentum_history.append(x_momentum)
    y_momentum_history.append(y_momentum)
    momentum_history.append(grad_momentum_x)

    # RMSProp
    grad_rmsprop_x = gradient(x=x_rmsprop, function=loss_function, eps=1e-6)
    x_rmsprop, g_rms = update_rmsprop(x=x_rmsprop, g=g_rms, gamma=gamma, eps=eps, grad=grad_rmsprop_x, lr=learning_rate)
    y_rmsprop = loss_function(x_rmsprop)
    
    x_rmsprop_history.append(x_rmsprop)
    y_rmsprop_history.append(y_rmsprop)
    rmsprop_history.append(grad_rmsprop_x)

    # Adam
    grad_adam_x = gradient(x=x_adam, function=loss_function, eps=1e-6)
    x_adam, m_adam, g_adam = update_adam(x=x_adam, m=m_adam, g=g_adam, n_iter=(_+1),
                                grad=grad_adam_x, lr=learning_rate,
                                b1=b1, b2=b2, eps=eps)
    y_adam = loss_function(x_adam)
    
    x_adam_history.append(x_adam)
    y_adam_history.append(y_adam)
    adam_history.append(grad_adam_x)
    
    if (_+1) % 1 ==0:
        gradient_graph(x_grad_history, y_grad_history, grad_gd_history, title='Gradient_Descent')
        gradient_graph(x_momentum_history, y_momentum_history, momentum_history, title='Momentum')
        gradient_graph(x_rmsprop_history, y_rmsprop_history, rmsprop_history, title='RMSProp')
        gradient_graph(x_adam_history, y_adam_history, adam_history, title='Adam')
        clear_output(wait=True)

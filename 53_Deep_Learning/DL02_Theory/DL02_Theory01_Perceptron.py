import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Perceptron

import time
from IPython.display import clear_output
clear_output(wait=True)

# Dataset ----------------------------------------------------------------------
# Dataset Define ----
x_data = {'x1':[0,0,1,1], 'x2':[0,1,0,1]}
# y_data = [0,0,0,1]
y_data = [1,0,0,0]

x_df = pd.DataFrame(x_data)
y_df = pd.Series(y_data, name='y').to_frame()

df = pd.concat([x_df, y_df], axis=1)
df


# Dataset 정의 ----
y_cols = ['y']
X_cols = df.columns.drop(y_cols)

y = df[y_cols]
X = df[X_cols]


# numpy Data ----
X_np = X.to_numpy()
y_np = y.to_numpy()

# w_init
w_init = np.array([6, -1, -2]).reshape(-1,1)  # w initialize (Random)
print(f'X.shape: {X_np.shape} / w_init.shape : {w_init.shape}  /  y.shape : {y_np.shape}')



# plotting
def plotting(w=False, mesh=False):
    if type(w) != bool:
        line_x = np.linspace(-1, 3, 30).reshape(-1,1)
        line_x = np.hstack([line_x ** 0, line_x])
        coef_w = [-w[0]/w[2], -w[1]/w[2]]
        line_y = line_x @ coef_w
        # print(coef_w)

    plt.figure()
    plt.hlines(y=0, xmin=-5, xmax=5, alpha=0.5)
    plt.vlines(x=0, ymin=-5, ymax=5, alpha=0.5)
    for i, v in df.groupby('y'):
        plt.scatter(data=v, x='x1', y='x2', label=i, s=100)

    if type(w) != bool:
        plt.plot(line_x[:,1], line_y, 'r--', alpha=0.3)    
    plt.legend()
    plt.xlim([-1, 3])
    plt.ylim([-1, 3])

    if mesh:
        xp1 = np.linspace(-1,3, 100)
        xp2 = np.linspace(-1,3, 100)
        x1_mesh, x2_mesh = np.meshgrid(xp1, xp2)
        mesh_np = np.hstack([x1_mesh.reshape(-1,1) ** 0, x1_mesh.reshape(-1,1), x2_mesh.reshape(-1,1)])
        cond = np.sign((mesh_np @ w).ravel()) == 1
        plt.plot(mesh_np[cond][:,1], mesh_np[cond][:,2], color='orange', marker='o', alpha=0.05)

    plt.grid(alpha=0.5)
    plt.show()

# w_init = np.array([6, 1, 2]).reshape(-1,1)  # w initialize (Random)
plotting(w_init, mesh=True)



# Perceptron ----------------------------------------------------------------------
learning_rate = 0.001
epoch = 1000

# numpy ---------------------------------------------------------
w_init = np.array([6, 1, 2]).reshape(-1,1)  # w initialize (Random)

# function 정의
def activation_function(x):
    if x > 0:
        return 1
    else:
        return 0

def perceptron_np(X, w):
    return activation_function(X @ w[1:] + w[0])
    
def update_w(X, w, error, learning_rate=0.001):
    if error == 0:
        return w
    else:
        bias = (w[0].ravel() + learning_rate * error)
        weight = (w[1:].ravel() + learning_rate * error * X)
        return np.array([*bias, *weight]).reshape(-1,1)

learning_rate = 0.01
epoch = 300

# Perceptron
w_np = w_init.copy()
for _ in range(epoch):
    errors = []
    for r in range(len(X_np)):
        y_pred = perceptron_np(X_np[r], w_np)
        error = y_np[r] - y_pred
        w_np = update_w(X=X_np[r], w=w_np, error=error, learning_rate=learning_rate)
        # print(y_pred, error, w_np.ravel())   
        errors.append(error[0])

    if (_+1) % 20 == 0:
        print(f'{_+1}번째 epoch')
        print(f'w : {w_np.ravel()}')
        # print(errors)
        plotting(w_np, mesh=False)
        time.sleep(0.2)
        clear_output(wait=True)


w_np[0] + X_np @ w_np[1:]
print(w_np)
plotting(w_np, mesh=True)




# sklearn -----------------------------------------
sklearn_perceotron = Perceptron(fit_intercept=True, max_iter=epoch, random_state=1)
sklearn_perceotron.fit(X, y)


sklean_w = np.hstack([sklearn_perceotron.intercept_, *sklearn_perceotron.coef_]).reshape(-1,1)
sklean_w

X_np1 @ sklean_w

print(sklean_w)
plotting(w=sklean_w, mesh=True)









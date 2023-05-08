# (Python) Interpreter Optimizing 230419
# Gaussian-Distribution : f(x) = 1/(σ√(2π)) ·exp(-1/2·((x-μ)/σ)^2)


import numpy as np
import torch
import matplotlib.pyplot as plt

## 【 1D Optimization (1 variable) 】 #######################################################
arr = np.random.permutation(np.arange(10)).reshape(1,1,10)
arr_t1 = torch.Tensor(arr)
# arr_t1 = -torch.Tensor(arr)       # minimize problem

# max_layer = torch.nn.MaxPool1d(3, padding=1, stride=1)    # max 
avg_layer = torch.nn.AvgPool1d(3, padding=1, stride=1)      # average

# arr_pool = max_layer(arr_t1)
arr_pool = avg_layer(arr_t1).squeeze()

torch.max(arr_pool)     # maxinum value
arr_argmax = torch.argmax(arr_pool)     # maxinum position

# maximum coordinate
print(arr_argmax)

arr_pool[arr_argmax]

# maximum value with neighbors
arr.reshape(-1)[arr_argmax-1:arr_argmax+2]    # maximum position with neighbors



## 【 2D Optimization (2 variable) 】 #######################################################
mat = np.random.permutation(np.arange(25)).reshape(1,1,5,5)
mat_t1 = torch.Tensor(mat)
# mat_t2 = -torch.Tensor(mat)   # minimize problem

# max_layer = torch.nn.MaxPool2d(3, padding=1, stride=1)  # max
avg_layer = torch.nn.AvgPool2d(3, padding=1, stride=1)  # average

# mat_pool = max_layer(mat_t1).squeeze()
mat_pool = avg_layer(mat_t1).squeeze()

torch.max(mat_pool)     # maxinum value
mat_argmax = torch.argmax(mat_pool)     # maxinum position

# maximum coordinate
mat_row = mat_argmax // 5
mat_column = mat_argmax % 5
print(mat_row, mat_column)

mat_pool[mat_row, mat_column]

# maximum value with neighbors
mat.reshape(5,5)[mat_row-1:mat_row+2, mat_column-1:mat_column+2]    # maximum position with neighbors



## 【 3D Optimization (3 variable) 】 #######################################################
ten = np.random.permutation(np.arange(125)).reshape(1,1,5,5,5)
ten_t1 = torch.Tensor(ten)

# max3d_layer = torch.nn.MaxPool3d(3, padding=1, stride=1)  # max
avg3d_layer = torch.nn.AvgPool3d(3, padding=1, stride=1)  # average

# ten_pool = max3d_layer(ten_t1).squeeze()
ten_pool = avg3d_layer(ten_t1).squeeze()

torch.max(ten_pool)     # maxinum value
ten_argmax = torch.argmax(ten_pool)     # maxinum position

# maximum coordinate
ten_layer = ten_argmax // 25
ten_node = ten_argmax % 25
ten_row = ten_node // 5
ten_column = ten_node % 5
print(ten_layer, ten_row, ten_column)

ten_pool[ten_layer, ten_row, ten_column]

# maximum value with neighbors
ten.reshape(5,5,5)[ten_layer-1:ten_layer+2, ten_row-1:ten_row+2, ten_column-1:ten_column+2]    # maximum position with neighbors
##########################################################################








##########################################################################
# 【 example 】###########################################################
##########################################################################
import scipy.stats as stats
norm = sp.stats.norm(loc=0, scale=1)
1-norm.cdf(1)   # 불량률

def failure_prob(x, std, lsl, usl):
    # prob_arr = np.vstack([usl-func1d(x), func1d(x)-lsl]).min(0) / rmse
    prob_arr = np.vstack([usl-x, x-lsl]).min(0) / std
    return 1-norm.cdf(prob_arr)
          
# np.set_printoptions(suppress=True)



# ground truth function --------------------------
def func1d(x):
    return (x-2)*(x-17)*(x+5)*(x+15)+3

# condition
x = np.arange(-15, 15, 1)
rmse = 5000
usl = 10000
lsl = -30000

# graph ----------------------------------------------------
plt.figure()
plt.plot(x, func1d(x))
plt.fill_between(x, func1d(x)-2.6*rmse, func1d(x)+2.6*rmse, alpha=0.1)
for l in [lsl, usl]:
    plt.axhline(l, color='red', ls='--', alpha=0.5)
plt.show()

failure_prob_1d = failure_prob(func1d(x), rmse, lsl, usl)
failure_prob_1d
# pd.Series(failure_prob_1d, index=x).to_clipboard()



# apply with 1d optimization *** ###########################
arr = failure_prob_1d.reshape(1,1,-1)
# arr_t1 = torch.Tensor(arr)
arr_t1 = -torch.Tensor(arr)       # minimize problem

# max_layer = torch.nn.MaxPool1d(3, padding=1, stride=1)    # max 
avg_layer = torch.nn.AvgPool1d(3, padding=1, stride=1)      # average

# arr_pool = max_layer(arr_t1)
arr_pool = avg_layer(arr_t1).squeeze()

torch.max(arr_pool)     # maxinum value
arr_argmax = torch.argmax(arr_pool)     # maxinum position

# maximum coordinate
print(arr_argmax)

arr_pool[arr_argmax]

# maximum value with neighbors
arr.reshape(-1)[arr_argmax-1:arr_argmax+2]    # maximum position with neighbors
x[arr_argmax]       # maximum_point


# result plot -----------------------------------------------------------------
plt.figure()
fig, ax1 = plt.subplots()
ax1.plot(x, func1d(x), label='predict_plot')
ax1.fill_between(x, func1d(x)-2.6*rmse, func1d(x)+2.6*rmse, alpha=0.1)
for l in [lsl, usl]:
    ax1.axhline(l, color='red', ls='--', alpha=0.5)
plt.scatter(x[arr_argmax],  func1d(x)[arr_argmax], color='blue', label='optimal_points_y')
plt.legend(bbox_to_anchor=(1.1,1))

ax2 = ax1.twinx()
# ax2.plot(x, failure_prob_1d, color='orange', alpha=0.5, label='failure_prob')
ax2.plot(x, -arr_pool, color='coral', alpha=0.5, label='failure_prob_with_neighbor')
plt.scatter(x[arr_argmax],  -arr_pool[arr_argmax], color='red', label='optimal_points_prob')

plt.legend(bbox_to_anchor=(1.6,0.8))
plt.show()
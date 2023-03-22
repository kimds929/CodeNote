import pandas as pd
import numpy as np
import cvxpy 
import scipy
import cvxopt 
import matplotlib.pyplot as plt

df_pcm = pd.read_clipboard(sep='\t')
df_pcm.head()

df_pcm.groupby('길이방향위치').size().sort_values()
# y = pd.read_csv('coe_premium.csv')
# y = y['premium'].to_numpy()

plt.figure(figsize=(10,3))
plt.plot(df_pcm['길이방향위치'], df_pcm['출측_장력편차'])
plt.show()

y = df_pcm['출측_장력편차'].to_numpy()
n = y.size

ones_row = np.ones((1, n))
D = sp.sparse.spdiags(np.vstack((ones_row, -2*ones_row, ones_row)), range(3), n-2, n)
# D.toarray()

# [sp.sparse.spdiags] : Return a sparse matrix from diagonals.
# spd_data = np.array([[1, 2, 3, 4], 
#                     [1, 2, 3, 4], 
#                     [1, 2, 3, 4]])
# diags = np.array([0, -1, 2])
# spdiags_D = sp.sparse.spdiags(spd_data, diags, 4, 4).toarray()
# 
# # (result)
# # array([[1, 0, 3, 0],
# #        [1, 2, 0, 4],
# #        [0, 2, 3, 0],
# #        [0, 0, 3, 4]])



################################################################################################################################
# (L1 filtering Objective Function) min 1/2 ||y-x||₂² + λ||Dx|||₁       ########################################################
# . y is the actual time series
# . x is the estimated filtered time series
# . The first part of the loss function represents the objective of minimising the sum of squared residuals between the actual and fitted series.
# . The second part of the loss function represents the desire for smoothness. 
# . Dx captures the smoothness between every set of three points.
# . Finally 𝜆 is the regularisation parameter.
# . Think of it as capturing the trade off between our two objectives of minimising residuals and maximising smoothness. 
#   We will see examples later on of how 𝜆 affects the filtered trend.

lambda_list = [0, 0.1, 0.5, 1, 2, 5, 10, 50, 200, 500, 1000, 2000, 5000, 10000, 100000]


# solver = cvxpy.CVXOPT   # L2
solver = cvxpy.ECOS     # L1
reg_norm = 1

lambda_value = 1000

x = cvxpy.Variable(shape=n) 
objective = cvxpy.Minimize(0.5 * cvxpy.sum_squares(y-x) 
                  + lambda_value * cvxpy.norm(D@x, reg_norm))
problem = cvxpy.Problem(objective)
problem.solve(solver=solver, verbose=False)

plt.figure(figsize=(10,3))
plt.plot(np.arange(1, n+1), y/100)
plt.plot(np.arange(1, n+1), np.array(x.value)/100)
plt.ylim(-3,3)
plt.show()

pd.Series(np.array(x.value)).to_clipboard()


# fig, ax = plt.subplots(len(lambda_list)//3, 3, figsize=(20,20))
# ax = ax.ravel()

# for lambda_value in lambda_list:
#     x = cvxpy.Variable(shape=n) 
#     # x is the filtered trend that we initialize
#     objective = cvxpy.Minimize(0.5 * cvxpy.sum_squares(y-x) 
#                   + lambda_value * cvxpy.norm(D@x, reg_norm))
#     # Note: D@x is syntax for matrix multiplication
#     problem = cvxpy.Problem(objective)
#     problem.solve(solver=solver, verbose=False)
#     ax[ii].plot(np.arange(1, n+1), y, , linewidth=1.0, c='b')
#     ax[ii].plot(np.arange(1, n+1), np.array(x.value), 'b-', linewidth=1.0, c='r')
#     ax[ii].set_xlabel('Time')
#     ax[ii].set_ylabel('Log Premium')
#     ax[ii].set_title('Lambda: {}\nSolver: {}\nObjective Value: {}'.format(lambda_value, prob.status, round(obj.value, 3)))
#     ii+=1
    
# plt.tight_layout()
# plt.savefig('results/trend_filtering_L{}.png'.format(reg_norm))
# plt.show()
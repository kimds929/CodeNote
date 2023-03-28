# https://www.cvxpy.org/        # 최적화 solver

import numpy as np

# 1. Linear Equations
    # 1.1. Solving Linear Equations

    # Matrix Define
A = np.array([[4,-5], [-2,3]])
b = np.array([-13,9]).reshape(-1,1)

    # A-1 · b
# np.linalg.inv(A).dot(b)
np.linalg.inv(A) @ b

    # matrix 형태
# np.asmatrix(A).I.dot(b)
np.asmatrix(A).I @ b
# np.asmatrix(A).I * b



    # 1.4. Vector-Vector Products
x = np.array([1,1]).reshape(-1,1)       # column vector
y = np.array([2,3]).reshape(-1,1)       # column vector

# 내적 (inner product)
x.T.dot(y)
x.T @ y

# np.asmatrix(x).T * np.asmatrix(y)     # matrix
# np.asmatrix(x).T @ np.asmatrix(y)     # matrix




# 2. Norms (strenth or distance in linear space)

x = np.array([4,3]).reshape(-1,1)

np.linalg.norm(x,1)     # 1 norm
np.linalg.norm(x,2)     # 2 norm


x = np.matrix([[1],[2]])
y = np.matrix([[2],[-1]])

x.T * y     # 0 : orthogonal


# Linear Equation Approximate Solution ***
# np.linalg.inv(A.T @ A) @ A.T @ B

# 4.2. Geometic Point of View: Projection
# 4.2.1. Orthogonal Projection onto a Subspace
A = np.matrix([[1,0],
               [0,1],
               [0,0]])
B = np.matrix([[1],
               [1],
               [1]])


X = (A.T @ A).I @ A.T @ B   # Approximate Solution
B_star = A * X
B_star      # 수선의 발 위치







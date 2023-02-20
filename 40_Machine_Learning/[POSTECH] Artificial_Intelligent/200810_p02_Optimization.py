import numpy as np
import cvxpy as cvx
# conda install -c conda-forge cvxpy

# Derivatives_Matrix
    # A @ x         →   A.T
    # x.T @ A       →   A
    # x.T @ x       →   2 * x
    # x.T @ A @ x   →   A @ x + A.T @ x 


# Obejctive_function : 
# min (x1 - 3)**2 + (x2 - 3)**2
#  → x1**2 + x2**2 - 6*x1 - 6 * x2 + 18
    #   [x1**2   x1* X2]
    #   [x1* X2   x2**2]
# → 1/2 [x1 x2] @ [[2,0], @ [[x1], - [6, 6] @ [[x1], + 18
#                  [0,2]]    [x2]]             [x2]]
#   f = 1/2 X.T @ H @ X + g.T @ X
# ▽f = H @ X + g            # H : Identity Matrix

H = np.matrix( [[2, 0], [0, 2]] )
g = np.matrix( [[-6], [-6]] )

x = np.zeros( shape=(2,1) )   # (0,0) 으로 Initialize
alpha = 0.1                   # step_size


for i in range(100):
    dev = H @ x + g
    x = x - alpha * dev
print(x)



# ------------------------------------------------------ 실습문제
    # 실습 1 : Linear Programming  ****
# objective_function : (max) 3 * x1 + 3/2 * x2 → (min) -( 3 * x1 + 3/2 * x2 )
# subjective to : -1 ≤ x1 ≤ 2
#                  0 ≤ x2 ≤ 3

f1 = np.array([3, 3/2]).reshape(-1,1)
lb1 = np.array([-1, 0]).reshape(-1,1)
ub1 = np.array([2, 3]).reshape(-1,1)

x1 = cvx.Variable(shape=(2, 1))
obj1 = cvx.Minimize(-f1.T @ x1)
constrain1 = [lb1 <= x1, x1 <= ub1]
prob1 = cvx.Problem(obj1, constrain1)
result1 = prob1.solve()
print(f'optimized_x1: {x1.value}')
print(f'optimized_result1: {result1}')




    # 실습 2 : Quadratic Programming (convex) ****
# objective_function : (min) 1/2 * x1**2 + 3 * x1 + 4 * x2
# subjective to : x1 + 3 * x2 ≥ 15
#                 2 * x1 + 5 * x2  ≤ 100
#                 3 * x1 + 4 * x2  ≤ 80
#                 x1, x2  ≥ 0
h2 = np.array([[1/2, 0],
               [0, 0]])
f2 = np.array([3,4]).reshape(-1,1)
A2 = np.array([[-1, -3], [2, 5], [3, 4]] )
b2 = np.array([[-15], [100], [80]] )
lb2 = np.array([[0, 0]]).reshape(-1,1)

x2 = cvx.Variable(shape=(2,1))
obj2 = cvx.Minimize( cvx.quad_form(x2, h2) + f2.T @ x2) 
# obj2 = cvx.Minimize( x2.T @ h2 @ x2 + f2.T @ x2)    # DCP error
constrain2 = [ A2 @ x2 <= b2, lb2 <= x2 ]
prob2 = cvx.Problem(obj2, constrain2)
result2 = prob2.solve()
print(f'optimized_x2: {x2.value}')
print(f'optimized_result2: {result2}')






# ---- [ 과  제 ] -------------------------------------------------------------------------------
    # 실습 3 : Shortest Distance ****
# objective_function : (min) sqrt((x1 - 3)**2 + (x2 - 3)**2)
#   → objective_function : (min) (x1 - 3)**2 + (x2 - 3)**2
#          subjective to : x1 + x2 ≤ 3
#                          x1, x2 ≥ 0

h3 = np.array([[1, 0], [0, 1]])
f3 = np.array([-6, -6]).reshape(-1,1)
# c3 = np.array([18, 18]).reshape(-1,1)
A3 = np.array([1, 1]).reshape(-1,1)
b3 = np.array([[3]])
lb3 = np.array([0, 0]).reshape(-1,1)

x3 = cvx.Variable(shape=(2,1))
obj3 = cvx.Minimize(cvx.quad_form(x3, h3) + f3.T @ x3 + 18)

constrain3 = [A3.T @ x3 <= b3, x3 >= lb3]
prob3 = cvx.Problem(obj3, constrain3)
result3 = prob3.solve()
print(f'optimized_x3: {x3.value}')
print(f'optimized_result3: {result3}')




    # 실습 4 : Empty Bucket ****
result = {}
result['mu'] = []
result['x1'] = []
result['distance'] = []

for mu4 in np.linspace(1,5,20):
    # mu4 = 1
    a4 = np.array([0, 1]).reshape(-1,1)
    b4 = np.array([4, 2]).reshape(-1,1)

    A4 = np.array([0, 1]).reshape(-1,1)
    B4 = np.array([0, 0]).reshape(-1,1)

    x4 = cvx.Variable(shape=(2,1))
    obj4 = cvx.Minimize(cvx.norm(a4-x4, 2) + mu4 * cvx.norm(b4-x4, 2))
    constrain4 = [A4.T @ x4 == B4]
    prob4 = cvx.Problem(obj4, constrain4)
    result4 = prob4.solve()

    result['mu'].append(mu4)
    result['x1'].append(x4.value[0,0])
    result['distance'].append(result4)
# print(f'optimized_x4: {x4.value}')
# print(f'optimized_result4: {result4}')
pd.DataFrame(result)



    # 실습 5 : Supply Chain ****
a5 = np.array([np.sqrt(3), 0]).reshape(-1,1)
b5 = np.array([-np.sqrt(3), 0]).reshape(-1,1)
c5 = np.array([0, 3]).reshape(-1,1)

x5_L1 = cvx.Variable(shape=(2,1))
obj5_L1 = cvx.Minimize(cvx.norm(a5 - x5_L1, 1) + cvx.norm(b5 - x5_L1, 1) + cvx.norm(c5 - x5_L1,1))
prob5_L1 = cvx.Problem(obj5_L1)
result5_L1 = prob5_L1.solve()

x5_L2 = cvx.Variable(shape=(2,1))
obj5_L2 = cvx.Minimize(cvx.norm(a5 - x5_L2, 2) + cvx.norm(b5 - x5_L2, 2) + cvx.norm(c5 - x5_L2,2))
prob5_L2 = cvx.Problem(obj5_L2)
result5_L2 = prob5_L2.solve()

print(f'optimized_x5_L1: {x5_L1.value.ravel()}  /  optimized_x5_L2: {x5_L2.value.ravel()}')
print(f'optimized_result5_L1: {result5_L2}  /  optimized_result5_L2: {result5_L2}')





import sys
sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import DS_DF_Summary, DS_OneHotEncoder, DS_LabelEncoder
from DS_OLS import *

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Postech_AI 4) Aritificial_Intelligent/교재_실습_자료/'
# absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'


import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline



# 1. Fixed-Point Iteration
# 1.1. Numerical approach ----------------------------------------------------------

x = np.linspace(-2*np.pi, 2*np.pi, 100)
y = np.cos(x)

plt.plot(x, y, linewidth = 2)
plt.plot(x, x, linewidth = 2)
plt.xlim(-2*np.pi, 2*np.pi)
plt.axvline(x=0, color = 'k', linestyle = '--')
plt.axhline(y=0, color = 'k', linestyle = '--')
plt.legend(['cos(x)','x'])
plt.show()


# naive approach
x = 0.3
print (np.cos(x))
print (np.cos(np.cos(x)))
print (np.cos(np.cos(np.cos(x))))
print (np.cos(np.cos(np.cos(np.cos(x)))))
print (np.cos(np.cos(np.cos(np.cos(np.cos(x))))))
print (np.cos(np.cos(np.cos(np.cos(np.cos(np.cos(x)))))))
print (np.cos(np.cos(np.cos(np.cos(np.cos(np.cos(np.cos(x))))))))


# better way
x = 0.3
for i in range(24):
    tmp = np.cos(x)
    x = np.cos(tmp)
print (x)


# better way    # 해를 찾는과정을 기록
x = np.zeros((24, 1))
x[0] = 0.3
for i in range(23):
    x[i+1] = np.cos(x[i])
print (x)


# better way
x = 10
for i in range(24):
    x = np.cos(x)
print (x)



# Use an idea of a fixed point
x = 2
for i in range(10):
    x = 2/x
    
print (x)


# How to overcome 
# Use an idea of a fixed point +   kind of *|damping|*

x = 3
for i in range(10):
    x = (x + 2/x)/2
    
print(x)




# 1.2. System of Linear Equations ----------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

#  4x1 − x2 + x3 = 7
#  4x1 − 8x2 + x3 = −21
# −2x1 + x2 + 5x3 = 15


# matrix inverse
A = np.array([[4, -1, 1], [4, -8, 1], [-2, 1, 5]])
b = np.array([[7, -21, 15]]).T

x = np.linalg.inv(A).dot(b)
print(x)


# Iterative way
A = np.array(([[0, 1/4, -1/4], [4/8, 0, 1/8], [2/5, -1/5, 0]]))
b = np.array([[7/4, 21/8, 15/5]]).reshape(-1,1)

# initial point
x = np.array([[1, 1, 2]]).reshape(-1,1)

for i in range(10):
    x = A @ x + b
    
print (x)

np.linalg.eig(A)[0]
np.linalg.inv(A)


# think about why this one does not work    # 수렴하지 않을때
A = np.array(([[3, 1, -1 ], [4, 7, 1], [2, -1, -4]]))
b = np.array([[7, 21, 15]]).T

np.linalg.inv(A).dot(b)

# initial point
x = np.array([[1, 2, 2]]).T

for i in range(10):
    x = A.dot(x) + b
    
print(x)


import time


inv_start = time.time()
size = 5000
M = np.random.rand(size, size)
M_I = np.linalg.inv(M)
inv_end = time.time()
print(inv_end - inv_start)




epoch = 100

iter_start = time.time()
M = np.random.rand(size, size)
b = np.random.rand(size,1)
x = np.random.rand(size,1)

for _ in range(epoch):
    x = M.dot(x) + b

iter_end = time.time()
print(iter_end - iter_start)




# 2. Recursive Algorithm 
# https://www.youtube.com/embed/t4MSwiqfLaY

# 2.1. foo example    -------------------------------------------------------
print ('{c}\n'.format(c="Hello class"))

def foo(str):
    print(str)  

foo('Hello class')


def foo_recursion(str):
    print('{s}\n'.format(s=str))
    foo_recursion(str)

foo_recursion('hello')



# Do not run. It falls into infinite loop.
#foo_recursion('Hello Class')
# base line
def foo_recursion_correct(text, n):
    if n <= 0:
        return
    else:
        print(text)
        foo_recursion_correct(text, n-1)

foo_recursion_correct('hello', 4)




# 2.2. Factorial example -------------------------------------------------------
n = 5
m = 1
for i in range(n):
    m = m*(i+1)

print (m)


import math
math.factorial(5)

def fac(n):
    if n == 1:
        return 1
    else:
        return n*fac(n-1)

# recursive
fac(5)












# 3. Dynamic Programming
# 3.1. Naive Recursive algorithm -------------------------------------------------------


# 3.2. Memorized DP algorithm -------------------------------------------------------
# https://www.youtube.com/embed/OQ5jsbhAv_M


# 3.4. Examples -------------------------------------------------------
# 3.4.1. Fibonacci -------------------------------------------------------

# naive Fibonacci
def fib(n):
    if n <= 2:
        return 1
    else:
        return fib(n-1) + fib(n-2)

fib(10)

# Memorized DP Fibonacci
def mfib(n):
    global memo
    
    if memo[n-1] != 0:
        return memo[n-1]
        
    elif n <= 2:
        return 1
    else:
        memo[n-1] = mfib(n-1) + mfib(n-2)
        return memo[n-1]



n = 10
memo = np.zeros(n)
mfib(n)



n = 30
%timeit fib(30)

memo = np.zeros(n)
%timeit mfib(30)


np.log(np.exp(1))




# 3.4.2. Climbing a stair -------------------------------------------------------
# You are climbing a stair case. Each time you can either make 1 step, 2 steps, or 3 steps. How many distinct ways can you climb if the stairs has  n=30  steps?

import numpy as np

def stair(n):
    global memo
    
    if memo[n] != 0:
        m = memo[n]
    elif n == 1:
        m = 1
    elif n == 2:
        m = 2
    elif n == 3:
        m = 4
    else:
        m = stair(n-1) + stair(n-2) + stair(n-3)
        
    memo[n] = m
    return m


n = 5
global memo
memo = np.zeros(n+1)

stair(n)
print(memo)






# 3.4.3. Knapsack problem using Dynamic Programming (DP)
from collections import namedtuple
myTuple = namedtuple('item', 'name weight value')

names = ['Bag', 'Coat', 'Shoes', 'Ring', 'Bill', 'Coins']
weights = [10, 9, 4, 2, 1, 20]
values = [175, 90,20,50,10,200]

items = [myTuple(name=n, weight=w, value=v) for n, w, v in zip(names, weights, values)]
MAX_WEIGHT = 20

items




def chooseBest(items, w_capacity):
    if len(items) == 0 or w_capacity <= 0:
        value, taken = 0, []
        return value, taken
    else:
        first_item, rest_items = items[0], items[1:]
        
        v_leave, t_leave = chooseBest(rest_items, w_capacity) # do not take the first item
        
        v_taken, t_taken = chooseBest(rest_items, w_capacity - first_item.weight) # do take the first item
        v_taken += first_item.value
        t_taken += [first_item]
        
        if w_capacity - first_item.weight >= 0 and v_taken >= v_leave:
            return v_taken, t_taken
        else:
            return v_leave, t_leave
        
chooseBest(items, MAX_WEIGHT)














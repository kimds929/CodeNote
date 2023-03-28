import os

import numpy as np
import matplotlib.pyplot as plt

import cv2
import torch


# Numpy -----------------------------------------------
np.random.seed(0)

N, D = 3, 4

x = np.random.randn(N, D)
y = np.random.randn(N, D)
z = np.random.randn(N, D)

a = x * y
b = a + z
c = np.sum(b)

grad_c = 1.0
## TO DO : Calculate manually gradients
grad_b = grad_c * np.ones((N, D))
grad_a = grad_b.copy()
grad_z = grad_b.copy()
grad_x = grad_a * y
grad_y = grad_a * x
grad_y 




# Pytorch -----------------------------------------------

N, D = 3, 4

x = torch.randn(N, D, requires_grad=True)
y = torch.randn(N, D, requires_grad=True)
z = torch.randn(N, D, requires_grad=True)

a = x * y
b = a + z
c = torch.sum(b)

c.backward()

print(x.grad)   # d
print(y.grad)




# array indexing ------------------------------------------

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])  # a.shape (3, 4)
print(a.shape)
print(a, '\n')


# [[2 3]
#  [6 7]]
# call-by-reference가 된 경우
b = a[:2, 1:3]      # slicing
print(b)

print(a[0, 1])
b[0, 0] = 77    # b[0, 0] is the same piece of data as a[0, 1]
print(a[0, 1])



# Indexing -------------------------------------------------------------
# ####  Slicing을 할 때는 dimension이 낮아질 수 있다.
# - Slicing을 하는 방법에는 여러가지가 있는데, integer를 활용해 indexing을 할 때는 dimension이 낮아지고, 
#   slicing을 이용해 indexing 할 때는 dimension이 유지된다.


# Create the following rank 2 array with shape (3, 4)
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a, a.shape)

row_r1 = a[1, :]    # Rank 1 view of the second row of a  
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
row_r3 = a[[1], :]  # Rank 2 view of the second row of a
# row_r1 = np.expand_dims(row_r1, axis=0)
print("Slicing Row")
print(row_r1, row_r1.shape)
print(row_r2, row_r2.shape)
print(row_r3, row_r3.shape)


col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print("Slicing Column")
print(col_r1, col_r1.shape, '\n')
print(col_r2, col_r2.shape)




# #### Integer array를 이용해 indexing을 할 수 있다. 
# - Slicing을 할 때는 네모난 subarray만 추출할 수 있지만, 
#   integer array를 이용할 경우 임의의 수치들을 꺼내올 수 있다.


a = np.array([[1,2], [3, 4], [5, 6]])

print(np.array([a[0, 0], a[1, 1], a[2, 0]]))
print(a[[0, 1, 2], [0, 1, 0]])



# Create a new array from which we will select elements
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
print(a)

## TO DO 
# Select one element from each row of a using the indices
b = np.array([0, 2, 0, 1])
print(a[np.arange(4), b])  # Prints "[ 1  6  7 11]"

a[np.arange(4), b] += 10
print(a)





# Boolean array로도 indexing을 할 수 있다. 
print(a > 2)
print(a[a > 2])     # 순서대로 Return, dimension이 유지되지 않음






# ## Broadcasting
# - Broadcasting is strong!

x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)   

for i in range(4):
    y[i, :] = x[i, :] + v
print(y)

vv = np.tile(v, (4, 1))  # Stack 4 copies of v on top of each other
y = x + vv  
print(y)

y = x + v  # Add v to each row of x using broadcasting
print(y)



##Quiz
def checkbroadcasting(x, y):
    try:
        x+y
        print("correct")
    except:
        print("wrong")

x=np.empty((0))
y=np.empty((2,2))
checkbroadcasting(x,y)
        
x=np.empty((5,3,4,1))
y=np.empty((3,4,1))
checkbroadcasting(x,y)

x=np.empty((5,3,4,1))
y=np.empty((3,1,1))
checkbroadcasting(x,y)

x=np.empty((5,2,4,1))
y=np.empty((3,1,1))
checkbroadcasting(x,y)


# ## 지금까지 배운 indexing 과 Broadcasting 방법이 ***모두*** Pytorch에도 적용 된다.

# # Pytorch Tutorial

# ## Tensors
# * Tensorflow의 Tensor와 다르지 않다.
#   * Numpy의 ndarrays를 기본적으로 활용하고 있다.
#   * Numpy의 ndarrays의 대부분의 operation을 사용할 수 있도록 구성되어 있다.
# * Numpy의 operation은 CPU만을 이용해 느리지만 Tensor는 CUDA를 활용해 GPU를 이용하기 때문에 빠르게 연산을 진행할 수 있다.
# 











# Pytorch =========================================================
# Construct a 5 x 3 matrix, uninitialized
x = torch.Tensor(5, 3)        # 단순 Random
print(x, '\n')

# Construct a randomly initialized matrix 
x = torch.rand(5, 3)        # 분포에 맞게 initialize 됨
print(x, '\n')

# Construct a matrix with the list
x = torch.tensor([[3, 4, 5], [1, 2, 3]])
print(x, '\n')

# Get its size
print(x.size())
print(x.shape) #???? is it Numpy? kkkkk




# ### dtype and device 
#  * dtype - Tensor의 데이터 타입
#  * device - Tensor의 작업 위치 (cpu or cuda)

x = torch.tensor([[3, 4, 5], [1, 2, 3]], dtype=torch.float64)
print(x, '\n')

y = torch.tensor([[3, 4, 5], [1, 2, 3]])
print(y, '\n')

#error
print(x + y)

y = y.double() 
print(y, '\n')

print(x + y)





device = torch.device('cuda')
device = torch.device('cuda:1')
x = x.to(device)

print(x, '\n')
print(x.device, '\n')




device_0 = torch.device('cuda:0')
device_1 = torch.device('cuda:1')



x = torch.randn(4, 3, dtype=torch.float64)
y = torch.randn(4, 3, dtype=torch.float32)
z = torch.randint(0, 10, (4, 3), dtype=torch.int32)

z = z.to(device_1)

print('Before "to" method')

print(x.dtype, x.device)
print(y.dtype, y.device)
print(z.dtype, z.device, '\n')

print('After "to" method')
# to method with specific dtype and device 
x = x.to(dtype=torch.int32, device=device_0)

# to method with some tensor 
y = y.to(z)
z = z.to(device="cpu")

print(x.dtype, x.device)
print(y.dtype, y.device)
print(z.dtype, z.device, '\n')




x = torch.empty(3, 5)
print(x, '\n')

x = torch.zeros(3, 5)
print(x, '\n')

x = torch.ones(3, 5)
print(x, '\n')

x = torch.full((3, 5), 3.1415)
print(x, '\n')

x = torch.arange(0, 5, 2)
print(x, '\n')

y = torch.linspace(0, 5, 9)
print(y, '\n')

z = torch.logspace(-10, 10, 5)
print(z, '\n')

z = torch.eye(5)
print(z, '\n')

# Construct a 3 x 5 matrix with random value from uniform distribution, i.e. Uniform[0, 1)
x = torch.rand(3, 5)

# Construct a 3 x 5 matrix with random value from normal distribution, i.e. Normal(0, 1)
x = torch.randn(3, 5)

x = torch.randint(3, 10, (3, 5))
print(x, '\n')











# ===================================================================================================
# ## Autograd: automatic differentiation
# * Autograd package는 Tensors가 사용할 수 있는 모든 Operation의 Gradient를 자동으로 계산해준다.
# * Tensor의 required_grad attribute를 이용해 gradient의 계산여부를 결정할 수 있다.
#   * 계산이 완료된 이후에 .backward()를 호출하면 자동으로 gradient를 계산한다.
#   * .grad attribute를 통해 마찬가지로 gradient에 접근할 수 있다. 
#   * .grad_fn attribute를 통해 해당 Variable이 어떻게 생성되었는지 확인할 수 있다.
#   



# Create a variable
x = torch.ones(2, 2, requires_grad=True)

print(x)
print(x.grad)
print(x.requires_grad)

y = x + 2
print(y)

# y는 operation으로 생성된 결과이기 때문에 grad_fn이 있지만 , x는 없다.
print(x.grad_fn)
print(y.grad_fn)

# Do more operations on y 
z = y * y * 3
out = z.mean()

print(z, z.grad_fn, '\n')
print(out)


# ### Gradients 
# * out.backward()을 하면 out의 gradient를 1로 시작해 Back-propagation을 시작한다.
# * .backward()를 호출한 이후부터는 .grad를 통해 각 변수의 gradient를 구할 수 있다.


# out.backward() == out.backward(tr.Tensor([1.0]))
out.backward()      # Gradient 계산
print(x.grad)




# We can do many crazy thing with autograd
x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float32)
y.backward(gradients)

print(x.grad)











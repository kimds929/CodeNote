
# TensorFlow vs Pytorch
#  - TensorFlow 1.0 은 정적 그래프 , Pytorch 는 동적 그래프 가장 주된 차이였으나
#     TensorFlow 2.0 에서 동적 그래프 방식 채택하면서 비슷해짐
# - Pytorch 는 직관적이고 디버깅이 쉬운 코드를 제공해 TensorFlow 에 비해 
#     구현까지걸리는 시간이 훨씬 짧음
# - TensorFlow 가 Pytorch 에 비해서 좀 더 빠르고 차이가 크진 않음 ),
#     커뮤니티가 커서 예시 코드가 많음

# https://pytorch.org/tutorials/        # pytorch tutorial


# Pytorch ---------------------------------------------------------------
import numpy as np
import tensorflow as tf

import torch
print(torch.__version__)

# Tensor 다루기 ---------------------------------------------------------------
t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# tensor = tf.constant(t)
tensor = torch.tensor(t)    # tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])

print(tensor)                   # tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])
print(type(tensor))             # <class 'torch.Tensor'>
print()
print('shape: ', tensor.shape)  # shape:  torch.Size([12])
print('size: ', tensor.size())  # size:  torch.Size([12])  (== shape)

def print_tensor(tensor):
    print(tensor)
    print(type(tensor))
    print()
    print('shape: ', tensor.shape)
    print('size: ', tensor.size())



    # reshape ***
tensor2d = tensor.reshape(3,4)
print_tensor(tensor2d)
# tensor([[ 1,  2,  3,  4],
#         [ 5,  6,  7,  8],
#         [ 9, 10, 11, 12]])
# <class 'torch.Tensor'>

# shape:  torch.Size([3, 4])
# size:  torch.Size([3, 4])


tensor2d2 = tensor.view(3,4)        # (== reshape)
print_tensor(tensor2d2)
# tensor([[ 1,  2,  3,  4],
#         [ 5,  6,  7,  8],
#         [ 9, 10, 11, 12]])
# <class 'torch.Tensor'>

# shape:  torch.Size([3, 4])
# size:  torch.Size([3, 4])


tensor3d = tensor.view(3,2,-1)
# tensor3d = tensor.view(7,-1)  # error: 약수만 가능, 원데이터 크기보다 크게 불가
print_tensor(tensor3d)
# tensor([[[ 1,  2],
#          [ 3,  4]],

#         [[ 5,  6],
#          [ 7,  8]],

#         [[ 9, 10],
#          [11, 12]]])
# <class 'torch.Tensor'>

# shape:  torch.Size([3, 2, 2])
# size:  torch.Size([3, 2, 2])



# indexing & 데이터 접근 ***
print(tensor3d[0])
print(tensor3d[0,1])
print(tensor3d[0][1])

# tensor([[1, 2],
#         [3, 4]])
# tensor([3, 4])
# tensor([3, 4])



    # 데이터 추출 (numpy data) ***
print(tensor3d.numpy())
# [[[ 1  2]
#   [ 3  4]]
#  [[ 5  6]
#   [ 7  8]]
#  [[ 9 10]
#   [11 12]]]

print(tensor3d[0][1].numpy())
# [3 4]


    # tensor: tensorflow → torch
tf_tensor1 = tf.constant(np.random.rand(5,3))
torch_tensor1 = torch.tensor(tf_tensor1.numpy())
tf_tensor2 = tf.constant(torch_tensor1.numpy())
print(tf_tensor1)
print(torch_tensor1)
print(torch_tensor2)



# 원소 추출 -----------------------------------------------------------
print(tensor3d[0,0,1])          # tensor(2)
print(tensor3d[0,0,1].numpy())  # 2
print(tensor3d[0,0,1].item())   # 2



# tensor 생성 ----------------------------------------------------------
tensor2 = torch.empty(5,3)    # random숫자
print(tensor2)
# tensor([[0.7652, 0.7568, 0.4527],
#         [0.3647, 0.6968, 0.5788],
#         [0.5197, 0.2378, 0.6809],
#         [0.0209, 0.9468, 0.0388],
#         [0.8880, 0.9743, 0.0997]])
tensor3 = torch.rand(5,3)
print(tensor3)
# tensor([[0.8752, 0.6211, 0.7555],
#         [0.6243, 0.7179, 0.8598],
#         [0.1280, 0.6457, 0.6671],
#         [0.9453, 0.3984, 0.5642],
#         [0.8322, 0.4425, 0.9158]])

# np.empty((5,3))
# tf.empty((5,3)) # error



tensor4 = torch.zeros(5,3, dtype=torch.float)
print(tensor4)
# tensor([[0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.]])


tensor5 = torch.zeros(5,3, dtype=torch.int)
print(tensor5)
# tensor([[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]], dtype=torch.int32)


tensor6 = torch.ones(5,3)
print(tensor6)
# tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]])


tensor7 = torch.eye(5)
print(tensor7)
# tensor([[1., 0., 0., 0., 0.],
#         [0., 1., 0., 0., 0.],
#         [0., 0., 1., 0., 0.],
#         [0., 0., 0., 1., 0.],
#         [0., 0., 0., 0., 1.]])



# Torch Tensor Operation -----------------------------------------------------------
print(tensor2)
print(tensor3)

add1 = tensor2 + tensor3
print(add1)
# tensor([[1.6404, 1.3780, 1.2082],
#         [0.9891, 1.4147, 1.4386],
#         [0.6476, 0.8835, 1.3480],
#         [0.9662, 1.3452, 0.6029],
#         [1.7202, 1.4168, 1.0155]])

add2 = torch.add(tensor2, tensor3)
print(add2)
# tensor([[1.6404, 1.3780, 1.2082],
#         [0.9891, 1.4147, 1.4386],
#         [0.6476, 0.8835, 1.3480],
#         [0.9662, 1.3452, 0.6029],
#         [1.7202, 1.4168, 1.0155]])



























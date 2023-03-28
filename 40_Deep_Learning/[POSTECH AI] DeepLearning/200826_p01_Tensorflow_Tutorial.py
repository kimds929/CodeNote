import numpy as np
import tensorflow as tf
print(tf.__version__)

# Tensorflow Start -----------------------------------------------------------------
hello = tf.constant('hello world')

hello.ndim
hello.shape
hello.dtype
hello.numpy()           # byte형태로 저장
hello.numpy().decode()  # byte형태 → decoding


# Tensor 생성 --------------------------------------------------------------------------
tensor_zeros1 = tf.zeros(3)
print(tensor_zeros1)
tensor_zeros2 = tf.zeros((3,4))
print(tensor_zeros2)

tensor_ones1 = tf.ones((2,2))
print(tensor_ones1)

tensor_eye1 = tf.eye(5)
print(tensor_eye1)



# Tensor --------------------------------------------------------------------------
def print_tensor(tensor):
    print(tensor)
    print()
    print(f'type: {type(tensor)}')
    print(f'dtype: {tensor.dtype}')    # tensor 자체의 type
    print()
    print(f'tf.shape: {tf.shape(tensor)}  /  shape : {tensor.shape}')    # shape
    print(f'tf.size: {tf.size(tensor)}')     # size(원소갯수)
    print(f'tf.rank: {tf.rank(tensor)}  /  ndim: {tensor.ndim}\n')     # rank(차원)

    # 1차원 tensor *
t = list(np.arange(1,13))
tensor = tf.constant(t)
print_tensor(tensor)

    # 2차원 tensor *
tensor2d = tf.reshape(tensor, (3,-1))
print_tensor(tensor2d)

# ※ tensor 접근 (numpy와 동일)
tensor2d[0]
tensor2d[0,1]
tensor2d[0][1]

tensor2d[:,0]
tensor2d[:,::2]

    # 3차원 tensor *
tensor3d = tf.reshape(tensor, (3, 2, -1))
print_tensor(tensor3d)
tensor3d.numpy()
tensor3d[0].numpy()
tensor3d[0][0].numpy()
tensor3d[0][0][1].numpy()

tensor3d_2 = tf.reshape(tensor2d, (3,2,-1))
print_tensor(tensor3d_2)


    # 4차원 tensor *
tensor4d = tf.reshape(tensor, (3,2,2,-1))
print_tensor(tensor4d)



    # tensor_squeeze (1단위 차원은 제거)
tensor_sq = tf.squeeze(tensor4d)
print_tensor(tensor_sq)


    # tensor_expand (차원늘리기)
tensor_ex0 = tf.expand_dims(tensor2d, 0)    # 0: 0(맨 앞)의 위치에 차원추가(앞에)
tensor_ex1 = tf.expand_dims(tensor2d, 1)    # 1: 1의 위치에 차원추가(0~1사이)
tensor_ex2 = tf.expand_dims(tensor2d, 2)    # 2: 2의 위치에 차원추가(1~2사이)
tensor_ex_1 = tf.expand_dims(tensor2d, -1)    # 0 : 맨뒤 위치에 차원추가

print_tensor(tensor2d)
print_tensor(tensor_ex0)
print_tensor(tensor_ex1)
print_tensor(tensor_ex2)
print_tensor(tensor_ex_1)







# Basic Operation ----------------------------------------------------------------------
a = tf.constant(3)
b = tf.constant(2)
c = tf.constant(5)


    # add
add1 = tf.add(a, b)
add2 = a + b
print('-- add --')
print(add1)
print(add2)

    # subtract
sub1 = tf.subtract(a, b)
sub2 = a - b
print('-- subtract --')
print(sub1)
print(sub2)

    # multiply
print('-- multiply --')
mul1 = tf.multiply(a, b)
mul2 = a * b
print(mul1)
print(mul2)

    # divide
print('-- divide --')
div1 = tf.divide(a, b)
div2 = a / b
print(div1)
print(div2)

    # 평균(mean)
mean1 = tf.reduce_mean([a,b,c])
mean2 = tf.reduce_mean(
    [tf.cast(a, tf.float16), tf.cast(b, tf.float16), tf.cast(c, tf.float16)])
print(mean1)
print(mean2)


# Matrix Operation ----------------------------------------------------------------------
mat1 = tf.constant([[1,2], [3,4]])
mat2 = tf.constant([[5,6], [7,8]])
print(mat1)
print(mat2)

    # matrix add
add_mat1 = tf.add(mat1, mat2)
add_mat2 = mat1 + mat2
print('-- Matrix add --')
print(add_mat1)
print(add_mat2)

    # matrix multiply
mul_mat1 = tf.multiply(mat1, mat2)
mul_mat2 = mat1 * mat2
print('-- Matrix multiply --')
print(mul_mat1)
print(mul_mat2)

    # matrix matrix-multiply (dot-product)
print('-- Matrix matrix_multiply --')
dot_mat1 = tf.matmul(mat1, mat2)
dot_mat2 = mat1 @ mat2
print(dot_mat1)
print(dot_mat2)


    # reduce_mean ***
mat_mean1 = tf.reduce_mean(mat1)
mat_mean2 = tf.reduce_mean(tf.cast(mat1, tf.float16))
print(mat_mean1)
print(mat_mean2)


    # reduce_sum ***
tf.reduce_sum(mat1)
tf.reduce_sum(mat2)
tf.reduce_sum([mat1, mat2])

mat0 = tf.constant(np.arange(1,5).reshape(2,2))
r_sum_mat1 = tf.reduce_sum(mat0)
r_sum_mat2 = tf.reduce_sum(mat0, axis=0)
r_sum_mat3 = tf.reduce_sum(mat0, axis=1)

print(mat0)
print(r_sum_mat1)
print(r_sum_mat2)
print(r_sum_mat3)






# Broadcasting -------------------------------------------------------------------

# vector + constant
vec = tf.constant([1, 2])
print(vec)

vec1 = vec + 1
print(vec1)


# matrix + constant
mat = tf.constant(np.arange(1,7).reshape(3,2))
print(mat)

mat1 = mat + 1
print(mat1)


# matrix + vector
mat2 = mat + vec
print(mat2)

# matrix
vec2 = tf.constant([1,2,3])
print(vec2)
# mat3 = mat + vec2      # error
# print(mat3)

vec3 = tf.reshape(vec2, (-1,1))
print(vec3)
mat4 = mat + vec3
print(mat4)



# matrix broadcasting possible? 
#  * 뒤를 기준으로 붙여서 규칙이 맞으면 가능
#     # 하나가 1이거나, 숫자가 같으면 가능

# ex_1
mat5 = tf.constant(np.arange(3))
mat6 = tf.constant(np.arange(3).reshape(-1,1))
mat7 = mat5 + mat6
print(mat7)

# ex_2
mat8 = tf.ones([8,1,6,1])
mat9 = tf.ones([7,1,5])

mat10 = mat8 + mat9
print(mat10.shape)

# ex_3
mat11 = tf.ones([2,1])
mat12 = tf.ones([8,4,3])
# mat13 = mat11 + mat12     # error
# print(mat13.shape)

# ex_4
mat13 = tf.ones([15,3,5])
mat14 = tf.ones([15,1,5])
mat15 = mat13 + mat14
print(mat15.shape)


# ex_5
mat16 = tf.ones([8,3])
mat17 = tf.ones([8,2,3])

# mat18 = mat16 + mat17       # error
mat18 = tf.expand_dims(mat16, 1) + mat17    # possible
print(mat18.shape)
print(tf.expand_dims(mat16, 1).shape)






# tf.Variable : variable 정의 및 연산 -------------------------------------
c = tf.constant(tf.ones([3,3]))
v = tf.Variable(tf.ones([3,3]))

print(c)
print(v)
# print_tensor(c)
# print_tensor(v)
c[0]
v[0]

c[0].numpy()
v[0].numpy()


a = tf.Variable(np.random.randn(), dtype=float)
print(a)















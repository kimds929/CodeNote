import tensorflow as tf
# pip install --upgrade tensorflow      # tensorflow upgrade




print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))   # 비어있을경우 CPU버전



# tf.constant : constant 정의 및 연산 -------------------------------------
    # tf.constant creates a constant tensor specified by value, dtype, shape and so on.
a = tf.constant([1,2,3])            # constant정의
b = tf.constant(4, shape=[1,3])     # constant정의, shape 지정
print(a, '\n', b)


# tensorflow 연산
A = a + b
B = a*b
print(A, '\n', B)

a = tf.constant([1,2,3])
b = tf.constant([4,5,6])


output = tf.multiply(a, b)      # 곱셈  
# output = a * b
print(output)

a = tf.constant([[1,2,3],[4,5,6]])
a.shape     # shpae 확인

b = tf.constant([[1,2], [3,4], [5,6]])

# a + b   # shape error
# a * b   # shape error

# tf.matmul(a, b)   # dot-product not error 
a @ b               # dot-product not error


    # reshape
# a.reshape([3,2])    # error
a_T = tf.reshape(a, [3,2])        # reshape

# b.T       # error
b_T = tf.transpose(b)     # transpose


c = a_T @ b_T   # reshaped matrixes dot-product

c
c + 1
c + tf.constant([1,2,3])        # column마다 값을 더함
c + tf.constant([[1],[2],[3]])  # row마다 값을 더함



    # numpy 형태로 변환
c.numpy()
type(c.numpy())


# tf.Variable : variable 정의 및 연산 -------------------------------------
    # tf.Variable is regarded as the decision variable in optimization. We should initialize variables to use tf.Variable.
    # 학습할 수 있는 속성이 켜져 있음
# ?tf.Variable
x1 = tf.Variable([1, 1], dtype=tf.float32)      # [1,1] : 초기값,  일반적인 데이터 크기 : tf.float32
x2 = tf.Variable([2, 2], dtype=tf.float32)
y = x1 + x2

print(y)


x1 = tf.Variable([1, 1], dtype=tf.float32, name='x1')   # name지정
x2 = tf.Variable([2, 2], dtype=tf.float32, name='x2')   # name지정


tf.reshape(x1, [2,1])
tf.reshape(x2, shape=[2,1])


    # numpy 형태로 변환
x1.numpy()






# tf.placeholder : placeholder 정의 및 연산 -------------------------------------
    # The value of tf.placeholder must be fed using the feed_dict optional argument to Session.run().







# 2.3. TensorFlow as Optimization Solver ---------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

import tensorflow as tf


# object_function = min (ω-4)**2
w = tf.Variable(0, dtype=tf.float32, name='omega')
learning_rate = 0.05

w_record = [w.numpy()]
grad_record = []
cost_record = []
for _ in range(50):
    # gradient 계산 ----------------------------
    with tf.GradientTape() as tape:
        loss = (w - 4)**2                   # object function
        grad = tape.gradient(loss, w)       # loss를 w에 대해 미분해서 기울기를 계산해라
    # -----------------------------------------

    w.assign(w - learning_rate*grad)    # w값 업데이트 - assign : 할당
    # w = w - learning_rate*grad        # error w값 업데이트 - assign : 할당

    w_record.append(w.numpy())          # w값 저장
    grad_record.append(grad.numpy())    # grad값 저장
    cost_record.append(loss.numpy())    # loss 값 저장

print("\n optimal w =", w.numpy())



plt.figure(figsize=(15, 5))

# w plot
plt.subplot(1,3,1)
plt.plot(w_record)
plt.xlabel('iteration', fontsize = 15)
plt.ylabel('w', fontsize = 15)

# gradient plot
plt.subplot(1,3,2)
plt.plot(grad_record)
plt.xlabel('iteration', fontsize = 15)
plt.ylabel('gradient', fontsize = 15)

# loss function plot
plt.subplot(1,3,3)
plt.plot(cost_record)
plt.xlabel('iteration', fontsize = 15)
plt.ylabel('cost', fontsize = 15)
plt.show()

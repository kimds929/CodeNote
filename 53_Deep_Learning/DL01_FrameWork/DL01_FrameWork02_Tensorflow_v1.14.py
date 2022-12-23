import tensorflow as tf
print(tf.__version__)   # 1.14.0

# Python API 사용
# 데이터 플로우 그래프 (정적)
# Define & Run방식
# High Level API가 많이 존재

    # ○ Define and Run 방식 (v1)
    # - 데이터 플로우 그래프에 대한 정의와 연산이 구분되어 있음(computational graph 생성 tf.Graph )
    #  computational graph run ( tf.Session))
    # - TensorFlow 에서 텐서를 다루거나 생성하는 연산 tf.operation 은 그래프의 node 가 되고 ,
    #   그래프 안에서 흐를 값 tf.tensor 들은 그래프의 edge 가 나타낸다.
    # (edge가 값을 가지고 있는 것은 아님)

# tf 1.x -----------------------------------------------------------------
# 1.x 버전에서는 아래 코드가 그래프를 만들라는 명령어
tf.reset_default_graph()    # 그래프 초기화

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0)
print(a)    # Tensor("Const:0", shape=(), dtype=float32)
print(b)    # Tensor("Const_1:0", shape=(), dtype=float32)

total = a + b       # total이 a와 b를 더하는 역할을 하는 노드
print(total)    # Tensor("add:0", shape=(), dtype=float32)

# session을 통해 데이터를 흘려줘야 실제 연산이 일어남
sess = tf.Session()
print(sess.run([a,b]))  # [3.0, 4.0]
print(sess.run(total))  # 7.0
sess.close()


# Define by Run방식 (Eager execution)?
print(tf.executing_eagerly())   # False

tensor = tf.constant(0)
print(type(tensor))     # <class 'tensorflow.python.framework.ops.Tensor'>




# Constant ----------------------------
t1 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
print(t1)
print(type(t1))
    # Tensor("Const_3:0", shape=(2, 2), dtype=float32)
    # <class 'tensorflow.python.framework.ops.Tensor'>

sess = tf.Session()
t1_out = sess.run(t1)
print(t1_out)
print(type(t1_out))
    # [[1. 2.]
    #  [3. 4.]]
    # <class 'numpy.ndarray'>


def print_tf(x):
    print('Value : ', x)
    print('Type : ', type(x))





# Variable ------------------------------------------------------
w = tf.Variable(tf.random_normal([5,2], stddev=0.1))
print(w)    # <tf.Variable 'Variable:0' shape=(5, 2) dtype=float32_ref>

# w_out = sess.run(w) # error
# print_tf(w_out)    # error
    # Attempting to use uninitialized value Variable 
    # [[{{node _retval_Variable_0_0}}]]

sess.run(w.initializer)
w_out = sess.run(w)
print_tf(w_out)
# Value :  [[ 0.09468853  0.00060364]
#  [ 0.02099405  0.00924892]
#  [-0.12052071 -0.04318758]
#  [-0.05975599  0.00846222]
#  [-0.15336089  0.09051846]]
# Type :  <class 'numpy.ndarray'>










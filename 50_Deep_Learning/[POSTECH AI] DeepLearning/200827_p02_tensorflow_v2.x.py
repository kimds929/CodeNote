import tensorflow as tf
print(tf.__version__)       # 2.1.0

# Python API 사용
# 데이터 플로우 그래프 (동적)
# Define by Run방식 (Eager execution)
# API Cleanup / Keras Python API

    #  ○ AutoGraph (tf.function )(v2)
    # - TensorFlow 2.x 에서는 그래프 모듈을 버리고 Define by Run 방식으로 편리함 추구
    # - 하지만 이로 인해 속도 저하
    # - 속도 보완을 위해 도입된 것이 AutoGraph
    # - function 에 tf.function 이라는 데코레이터를 붙여 그래프 모듈로 복구 속도 향상


# tf 2.x -----------------------------------------------------------------
# 2.x 버전에서는 아래 코드가 해당 데이터를 만드는 의미
a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0)
print(a)        # tf.Tensor(3.0, shape=(), dtype=float32)
print(b)        # tf.Tensor(4.0, shape=(), dtype=float32)

total = a + b
print(total)    # tf.Tensor(7.0, shape=(), dtype=float32)


# Define by Run방식 (Eager execution)?
print(tf.executing_eagerly())   # True

tensor = tf.constant(0)
print(type(tensor))     # <class 'tensorflow.python.framework.ops.EagerTensor'>




# Constant ----------------------------
t1 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
print(t1)
print(type(t1))
    # [[1. 2.]
    #  [3. 4.]], shape=(2, 2), dtype=float32)
    # <class 'tensorflow.python.framework.ops.EagerTensor'>


t1_np = t1.numpy()
print(t1_np)
    # [[1. 2.]
    #  [3. 4.]]


def print_tf(x):
    print('Value : ', x)
    print('Type : ', type(x))




# Variable ------------------------------------------------------
w = tf.Variable(tf.random_normal_initializer(stddev=0.1)([5,2]))
print(w)
# <tf.Variable 'Variable:0' shape=(5, 2) dtype=float32, numpy=
# array([[ 0.20188141,  0.07414984],
#        [ 0.027543  , -0.11962394],
#        [ 0.14758836, -0.00174593],
#        [ 0.12245792, -0.14183752],
#        [-0.23274188,  0.07787116]], dtype=float32)>


print_tf(w)
# Value :  <tf.Variable 'Variable:0' shape=(5, 2) dtype=float32, numpy=
# array([[ 0.20188141,  0.07414984],
#        [ 0.027543  , -0.11962394],
#        [ 0.14758836, -0.00174593],
#        [ 0.12245792, -0.14183752],
#        [-0.23274188,  0.07787116]], dtype=float32)>
# Type :  <class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'>



# Auto Graph ------------------------------------------
import timeit

# Create an oveerride model to classify pictures
class SequentialModel(tf.keras.Model):
  def __init__(self, **kwargs):
    super(SequentialModel, self).__init__(**kwargs)
    self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
    self.dense_1 = tf.keras.layers.Dense(128, activation="relu")
    self.dropout = tf.keras.layers.Dropout(0.2)
    self.dense_2 = tf.keras.layers.Dense(10)

  def call(self, x):
    x = self.flatten(x)
    x = self.dense_1(x)
    x = self.dropout(x)
    x = self.dense_2(x)
    return x

# 실행
input_data = tf.random.uniform([60, 28, 28])

eager_model = SequentialModel()         # eager model
graph_model = tf.function(eager_model)  # graph model (function으로 감싸줘서 그래프로 계산)

iteration = 50000
print("Eager time:", timeit.timeit(lambda: eager_model(input_data), number=iteration))
print("Graph time:", timeit.timeit(lambda: graph_model(input_data), number=iteration))

    # iteration 10,000
    # Eager time: 13.620038199999954
    # Graph time: 8.949597100000119

    # iteration 50,000
    # Eager time: 68.07357860000002
    # Graph time: 22.563082499999837




# 간단한 계산의 경우 graph를 통해 계산하면 시간이 더 오래 걸림 
def sum1to100(x):
    for i in range(100):
        x +=(i+1)
    return x

@tf.function
def graph_sum(x):
    return sum1to100(x)

iteration = 100
print("Eager time:", timeit.timeit(lambda: sum1to100(0), number=iteration))
print("Graph time:", timeit.timeit(lambda: graph_sum(0), number=iteration))

# iteration 100
# Eager time: 0.0008634000000711239
# Graph time: 0.13925620000009076




# convert 1.0 code to 2.0 --------------------------------------------------------
import tensorflow.compat.v1 as tf       # tensorflow 1.0버전 불러오기
tf.disable_v2_behavior()        # 버전2처럼 행동하는것은 못하게 하겠음
                                # 이 방법 사용시 TensorFlow 2.0 의 장점을 활용하기 어려움

print(tf.__version__)
tf.reset_default_graph()    # 그래프 초기화

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0)
print(a)    # Tensor("Const:0", shape=(), dtype=float32)
print(b)    # Tensor("Const_1:0", shape=(), dtype=float32)

total = a + b       # total이 a와 b를 더하는 역할을 하는 노드
print(total)    # Tensor("add:0", shape=(), dtype=float32)

# session
sess = tf.Session()
print(sess.run([a,b]))  # [3.0, 4.0]
print(sess.run(total))  # 7.0
sess.close()

# ---------------------------------------------------------------------------------





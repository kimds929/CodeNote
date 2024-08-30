import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

import tensorflow as tf

from IPython.display import clear_output
# clear_output(wait=True)




# 2.3. Logistic Regression with Tensorflow
m = 1000
true_w = np.array([[-6], [2], [1]])
train_X = np.hstack([np.ones([m,1]), 5*np.random.rand(m,1), 4*np.random.rand(m,1)])

true_w = np.asmatrix(true_w)
train_X = np.asmatrix(train_X)

train_y = 1/(1 + np.exp(-train_X*true_w)) > 0.5 

C1 = np.where(train_y == True)[0]
C0 = np.where(train_y == False)[0]

n_mixed = 10
idx1 = np.random.randint(0, len(C1), size=n_mixed)
idx0 = np.random.randint(0, len(C0), size=n_mixed)
C1[idx1], C0[idx0] = C0[idx0], C1[idx1]

train_y = np.empty([m,1])
train_y[C1] = 1
train_y[C0] = 0

plt.figure(figsize = (10,8))
plt.plot(train_X[C1,1], train_X[C1,2], 'ro', alpha = 0.3, label='C1')
plt.plot(train_X[C0,1], train_X[C0,2], 'bo', alpha = 0.3, label='C0')
plt.xlabel(r'$x_1$', fontsize = 15)
plt.ylabel(r'$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 12)
plt.axis('equal')
plt.ylim([0,4])
plt.show()


# train_test_split
train_X = train_X.astype(np.float32)
train_y = train_y.astype(np.float32)
print(train_X.shape, train_y.shape)





# using tensorflow ---------------------------------------------------------
import tensorflow as tf

W = tf.Variable(tf.zeros([3, 1]), tf.float32, name='weights')

def logistic_fn(x):
    return tf.sigmoid(tf.matmul(x, W))

def cross_entropy(y_pred, y_true):
    return tf.reduce_mean( - y_true * tf.math.log(y_pred) - (1-y_true) * tf.math.log(1-y_pred) )

def accuracy_fn(pred, y):
    predicted = tf.cast(pred > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))
    return accuracy*100

# tf.constant([1,2,3,4]) > 2      # boolean 형태 
# tf.cast(tf.constant([1,2,3,4]) > 2, dtype=tf.float32)   # boolean → 0, 1

def grad(x, y):
    with tf.GradientTape() as tape:
        pred = logistic_fn(x)
        loss_value = cross_entropy(y_pred = pred, y_true = y)
    return tape.gradient(loss_value, W)


# tf.data.Dataset : Dataset만들기
# .batch : 배치의 크기
# .shuffle : 데이터를 섞어주기

dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y)).batch(len(train_X)).shuffle(len(train_X))#.repeat()
optimizer = tf.optimizers.SGD(learning_rate=0.05)

EPOCHS = 5001
for step in range(EPOCHS):
    for x, y in iter(dataset):
        W_grad = grad(x, y)         # gradient 함수 구하기
        optimizer.apply_gradients(grads_and_vars=[(W_grad, W)]) # W_gradient를 W에 대해 구해라
        
        if step % 250 == 0:     # 250번마다 print를 해라
            print("Iter: {:4d}, Loss: {:.4f}, Acc: {:.1f}%".format(step, cross_entropy(logistic_fn(x), y), accuracy_fn(logistic_fn(train_X), train_y)))



# 결과보기
w_hat = W.numpy()
x1p = np.arange(0, 8, 0.01).reshape(-1, 1)
x2p = - w_hat[1,0]/w_hat[2,0]*x1p - w_hat[0,0]/w_hat[2,0]

plt.figure(figsize = (10,8))
plt.plot(train_X[C1,1], train_X[C1,2], 'ro', alpha = 0.3, label='C1')
plt.plot(train_X[C0,1], train_X[C0,2], 'bo', alpha = 0.3, label='C0')
plt.plot(x1p, x2p, 'g', linewidth = 3, label = '')
plt.xlabel(r'$x_1$', fontsize = 15)
plt.ylabel(r'$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 12)
plt.axis('equal')
plt.xlim([0,5])
plt.ylim([0,4])
plt.show()


# tf.dataset ------------------------------------------------------------
train_X = np.random.randint(low=0, high=10, size=[4,2])
train_y = np.random.randint(low=0, high=2, size=[4,1])

print(train_X)
print(train_y)


# dataset
dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y)) 
for i, (x, y) in enumerate(dataset):
    print(i, '-' * 50)
    print(x,y)

# batch
dataset2 = tf.data.Dataset.from_tensor_slices((train_X, train_y)).batch(2) 
for i, (x, y) in enumerate(dataset2):
    print(i, '-' * 50)
    print(x,y)


# shuffle
dataset3 = tf.data.Dataset.from_tensor_slices((train_X, train_y)).batch(2).shuffle(len(train_X))
for i, (x, y) in enumerate(dataset3):
    print(i, '-' * 50)
    print(x,y)    

# repeat
dataset4 = tf.data.Dataset.from_tensor_slices((train_X, train_y)).batch(2).repeat(2)
for i, (x, y) in enumerate(dataset4):
    print(i, '-' * 50)
    print(x,y)






# ---------------------------------------------------------------------------------
# 3. Neural Network with a Single Neuron  

# 3.1. Logistic Regression in a Form of Neural Network ----------------------------

import tensorflow as tf

m = 1000
true_w = np.array([[-6], [2], [1]])
train_X = np.hstack([np.ones([m,1]), 5*np.random.rand(m,1), 4*np.random.rand(m,1)])

true_w = np.asmatrix(true_w)
train_X = np.asmatrix(train_X)

train_y = 1/(1 + np.exp(-train_X*true_w)) > 0.5 

# train_test_split
train_X = train_X.astype(np.float32)
train_y = train_y.astype(np.float32)

print(train_X.shape, train_y.shape)


train_X = train_X[:,1:]     # bias 항 제거
train_X

print(train_X.shape, train_y.shape)

W = tf.Variable(tf.zeros([2, 1]), tf.float32, name='weights')
b = tf.Variable(tf.zeros([1]), tf.float32, name='bias')


def logistic_fn(x):
    return tf.sigmoid(tf.add(tf.matmul(x, W), b))

def cross_entropy(y_pred, y):
    return tf.reduce_mean(- y*tf.math.log(y_pred) - (1-y)*tf.math.log(1-y_pred))

def accuracy_fn(pred, y):
    predicted = tf.cast(pred > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))
    return accuracy*100

dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y)).batch(len(train_X)).shuffle(len(train_X))#.repeat()
optimizer = tf.optimizers.SGD(learning_rate=0.05)



EPOCHS = 5001
for step in range(EPOCHS):
    for x, y  in iter(dataset):
        loss = lambda: cross_entropy(logistic_fn(x), y)
        optimizer.minimize(loss, [W, b])
        
        if step % 250 == 0:
            print("Iter: {:4d}, Loss: {:.4f}, Acc: {:.1f}%".format(step, cross_entropy(logistic_fn(x), y), accuracy_fn(logistic_fn(train_X), train_y)))



# result
w_hat = W.numpy()
b_hat = b.numpy()
w_hat, b_hat

# plotting
x1p = np.arange(0, 8, 0.01).reshape(-1, 1)
x2p = - w_hat[0,0]/w_hat[1,0]*x1p - b_hat[0]/w_hat[1,0]

plt.figure(figsize = (10,8))
plt.plot(train_X[C1,0], train_X[C1,1], 'ro', alpha = 0.3, label='C1')
plt.plot(train_X[C0,0], train_X[C0,1], 'bo', alpha = 0.3, label='C0')
plt.plot(x1p, x2p, 'g', linewidth = 3, label = '')
plt.xlabel(r'$x_1$', fontsize = 15)
plt.ylabel(r'$x_2$', fontsize = 15)
plt.legend(loc = 1, fontsize = 12)
plt.axis('equal')
plt.xlim([0,5])
plt.ylim([0,4])
plt.show()






# 3.2 XOR Problem with Tensorflow --------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import tensorflow as tf


X = np.array([[0, 0],
              [1, 1],
              [1, 0],
              [0, 1]], dtype=np.float32)
Y = np.array([0, 0, 1, 1], dtype=np.float32).reshape(-1, 1)

plt.scatter(X[:2,0], X[:2,1], label='class A')
plt.scatter(X[2:,0], X[2:,1], label='class B')
plt.xlim([-1, 2]); plt.ylim([-1, 2])
plt.legend()
plt.show()




# ---------------------------------------------
n_features = 2
n_hidden = 3
n_output = 1

W1 = tf.Variable(tf.random.normal(shape=[n_features, n_hidden], dtype=tf.float32), name='W1')
b1 = tf.Variable(tf.random.normal(shape=[n_hidden], dtype=tf.float32), name='b1')
W2 = tf.Variable(tf.random.normal(shape=[n_hidden, n_output], dtype=tf.float32), name='W2')
b2 = tf.Variable(tf.random.normal(shape=[n_output], dtype=tf.float32), name='b2')

def model(x):
    hidden = x @ W1 + b1
    hidden = tf.sigmoid(hidden)
    logits = hidden @ W2 + b2
    pred = tf.sigmoid(logits)
    return pred

def accuracy(X, Y):
    predicted = tf.cast(model(X) > 0.5, dtype=tf.float32)
#     print(predicted)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
    return accuracy*100



np.meshgrid(np.arange(0, 5, 1))     # 0 ~ 5 까지 1간격으로

xx, yy = np.meshgrid(np.arange(-1, 2, 0.01), np.arange(-1, 2, 0.01))
grid_X = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)]).astype(np.float32)
Z = (model(grid_X).numpy() > 0.5).astype(np.int32).reshape(xx.shape)

plt.title('Accracy: {:.1f}%'.format(accuracy(X, Y)))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.2)
plt.scatter(X[:2,0], X[:2,1], label='class A'); plt.scatter(X[2:,0], X[2:,1], label='class B')
plt.xlim([-1, 2]); plt.ylim([-1, 2]); plt.legend(); plt.show()

    

def plotModel():
    xp = np.arange(-1, 2, 0.01).reshape(-1, 1)
    yp1 = -W1[0,0]/W1[1,0]*xp - b1[0]/W1[1,0]
    yp2 = -W2[0,0]/W2[1,0]*xp - b2[0]/W2[1,0]
    
    xx, yy = np.meshgrid(np.arange(-1, 2, 0.01), np.arange(-1, 2, 0.01))
    grid_X = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)]).astype(np.float32)
    Z = (model(grid_X).numpy() > 0.5).astype(np.int32).reshape(xx.shape)
    plt.title('Accracy: {:.1f}%'.format(accuracy(X, Y)))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.2)
    plt.scatter(X[:2,0], X[:2,1], label='class A'); plt.scatter(X[2:,0], X[2:,1], label='class B')
    plt.xlim([-1, 2]); plt.ylim([-1, 2]); plt.legend(); plt.show()


loss = tf.losses.MeanSquaredError()
optm = tf.optimizers.Adam(learning_rate=0.01)

for epoch in range(2000):
    loss_val = lambda: loss(model(X), Y)
    optm.minimize(loss_val, var_list=[W1, b1, W2, b2])
    if epoch % 200 == 0:
        plotModel()
        clear_output(wait=True)









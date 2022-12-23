import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression

import tensorflow as tf
# import torch

from IPython.display import clear_output
# clear_output(wait=True)


# Dataset Load ============================================================================
# test_df = pd.read_clipboard()
test_dict = {'y1': [2, 12,  6, 19, 5,  5, 14, 8, 12, 20,  1, 10],
            'y2': [0, 0,  1, 1, 0,  0, 1, 0, 1, 1,  0, 1],
            'x1': [ 5,  5, 35, 38,  9, 19, 30,  2, 49,  30,  0, 14],
            'x2': ['a', 'c', 'a', 'b', 'b', 'b', 'a', 'c', 'c', 'a', 'b', 'c'],
            'x3': [46, 23, 23,  3, 36, 10, 14, 28,  5, 19, 42, 32],
            'x4': ['g1', 'g2', 'g1', 'g2', 'g1', 'g2', 'g1', 'g2', 'g1', 'g2', 'g2', 'g2']
            }

test_df = pd.DataFrame(test_dict)

y1_col = ['y1']     # Regressor
y2_col = ['y2']     # Classifier
x_col = ['x1']

y1 = test_df[y1_col]    # Regressor
y2 = test_df[y2_col]    # Classifier
X = test_df[x_col]

y1_np = y1.to_numpy()
y2_np = y2.to_numpy()    # Classifier
X_np = X.to_numpy()

plt.figure(figsize=(10,3))
plt.subplot(121)
plt.title('Regressor')
plt.plot(X, y1, 'o')

plt.subplot(122)
plt.title('Classifier')
plt.plot(X, y2, 'o')
plt.show()


# xp, yp
Xp = np.linspace(np.min(X_np), np.max(X_np), 100).reshape(-1,1)





# [ Sklearn ] ============================================================================
LR = LinearRegression()
LR.fit(X,y1)
print(LR.coef_[0,0], LR.intercept_[0])

plt.plot(X, y1, 'o')
plt.plot(X, X*LR.coef_[0,0] + LR.intercept_[0], linestyle='-', color='orange', alpha=0.5)
plt.show()



# [ Tensorflow ] ==========================================================================

# (Tensorflow Basic) ---------------------------------------------------------------
# model_01 (Sequential) ****
n_epoch = 20000
n_batch = 3

model_01 = tf.keras.Sequential([ tf.keras.layers.Dense(1) ])
model_01.compile(optimizer='adam', loss='mse', metrics=['mse'])
result_01 = model_01.fit(X_np, y1_np, epochs=n_epoch, shuffle=True, batch_size=n_batch, verbose=0)

len(model_01.weights)
print(model_01.weights[0].numpy()[0,0], model_01.weights[1].numpy()[0])


# model_02 (Class) ****
n_epoch = 20000
n_batch = 3

class Model02(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # self.input = 
        self.dense1 = tf.keras.layers.Dense(1, activation=None)
    
    def call(self, X, training=None, mask=None):
        self.d1 = self.dense1(X)
        return self.d1

model_02 = Model02()
model_02.compile(optimizer='adam', loss='mse', metrics=['mse'])
# model_01.summary()
result_02 = model_02.fit(X_np, y1_np, epochs=n_epoch, shuffle=True, batch_size=n_batch, verbose=0)


len(model_02.weights)
print(model_02.weights[0].numpy()[0,0], model_01.weights[1].numpy()[0])



# (Tensorflow Expert) ----------------------------------------------------------------
# Data Train_valid_split ****
df = pd.DataFrame(np.hstack([y1_np, X_np]), columns=['y1', 'X'])
indice = np.random.permutation(12)
train_indice = indice[:8]
test_indice = indice[8:]
print(f'train_indice: {train_indice}')
print(f'test_indice: {test_indice}')

train_x = X_np[train_indice]
train_y = y1_np[train_indice]
valid_x = X_np[test_indice]
valid_y = y1_np[test_indice]
print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)

# dataset ****
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset
n_batch = 3
n_shuffle = 3       # X_np.shape[0]
train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(n_batch).shuffle(train_x.shape[0])
valid_ds = tf.data.Dataset.from_tensor_slices((valid_x, valid_y)).batch(n_batch).shuffle(train_x.shape[0])

df.iloc[train_indice,:]
for i, (tx, ty) in enumerate(train_ds, 1):
    print(i, tx.numpy().ravel(), ty.numpy().ravel())


# Model_03 ****
start01 = time.time()   # time_start
class Model02(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # self.input = 
        self.dense1 = tf.keras.layers.Dense(1, activation=None)
    
    def call(self, X, training=None, mask=None):
        self.d1 = self.dense1(X)
        return self.d1

n_epoch = 20000
display_epoch = 1000

model_03 = Model02()        # ClassModel
loss_obj01 = tf.keras.losses.MeanSquaredError()
optimizer01 = tf.keras.optimizers.Adam()

@tf.function
def training_step(model, X, y1, loss_obj, optimizer_obj):
    with tf.GradientTape() as Tape:
        y_pred = model(X)
        loss = loss_obj(y_pred, y1)
    gradients = Tape.gradient(loss, model.trainable_variables)
    optimizer = optimizer_obj.apply_gradients(zip(gradients, model.trainable_variables))


for epoch in range(1, n_epoch+1):
    # Model_compile_fit
    for batch_x, batch_y in train_ds:
        with tf.GradientTape() as Tape:
            y_pred = model_03(batch_x)
            loss = loss_obj01(y_pred, tf.cast(batch_y, tf.float32))
        gradients = Tape.gradient(loss, model_03.trainable_variables)
        optimizer = optimizer01.apply_gradients(zip(gradients, model_03.trainable_variables))
        
        # tf.function
        # training_step(model_03, batch_x, tf.cast(batch_y, tf.float32),
        #     loss_obj01, optimizer01)

    # Display History
    if epoch % display_epoch == 0:
        y_pred = model_03(batch_x)
        loss = loss_obj01(y_pred, tf.cast(batch_y, tf.float32))

        coef = model_03.trainable_variables[0].numpy()[0,0]
        intercept = model_03.trainable_variables[1].numpy()[0]

        print(f'epoch: {epoch}, loss:{format(loss.numpy(), ".2f")}  /  coef: {format(coef,".5f")}, {format(intercept,".5f")}')

end01 = time.time()   # time_start
print(f'time_elapse: {format(end01 - start01, ".2f")}')
print()





# model_01.summary()
model_02.summary()

# plt.plot(X, y1, 'o')
# plt.plot(X, X*LR.coef_[0,0] + LR.intercept_[0], linestyle='-', color='orange',alpha=0.5)
# plt.plot(X, X*model_01.weights[0].numpy()[0,0] + model_01.weights[1].numpy()[0], linestyle='--', color='blue', alpha=0.5)
# plt.show()


# train_ds = tf.data.Dataset.from_tensor_slices((X_np, y1_np))
# train_ds = tf.data.Dataset.from_tensor_slices((X_np,y1_np)).shuffle(3, reshuffle_each_iteration=False)
# train_ds = tf.data.Dataset.from_tensor_slices((X_np,y1_np)).batch(2, drop_remainder=False)   # drop_remainder: 남는 원소 drop?


# import tensorflow as tf

# class MyModel(tf.keras.Model):

#   def __init__(self):
#     super(MyModel, self).__init__()
#     self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
#     self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
#     self.dropout = tf.keras.layers.Dropout(0.5)

#   def call(self, inputs, training=False):
#     x = self.dense1(inputs)
#     if training:
#       x = self.dropout(x, training=training)
#     return self.dense2(x)

# model = MyModel()
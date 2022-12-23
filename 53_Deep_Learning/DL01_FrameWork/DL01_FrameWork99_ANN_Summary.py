import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import torch
# import time
# time.time()


# test_df = pd.read_clipboard()
test_dict = {'y1': [2, 12,  6, 19, 5,  5, 14, 8, 12, 20,  1, 10],
            'y2': [0, 0,  1, 1, 0,  0, 1, 0, 1, 1,  0, 1],
            'x1': [ 5,  5, 35, 38,  9, 19, 30,  2, 49,  30,  0, 14],
            'x2': ['a', 'c', 'a', 'b', 'b', 'b', 'a', 'c', 'c', 'a', 'b', 'c'],
            'x3': [46, 23, 23,  3, 36, 10, 14, 28,  5, 19, 42, 32],
            'x4': ['g1', 'g2', 'g1', 'g2', 'g1', 'g2', 'g1', 'g2', 'g1', 'g2', 'g2', 'g2']
            }

test_df = pd.DataFrame(test_dict)
test_df


X_cols = ['x1']
y_col = 'y1'



# Tensorflow Keras -----------------------------------------------------------------------------
class TF_Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(1)
        # self.dense1 = tf.keras.layers.Dense(1, activation=tf.nn.relu)
        # self.act1 = tf.keras.layers.Activation('relu')
        self.act1 = lambda x: x 
    
    def call(self, X, training=None):
        self.h1 = self.dense1(X)
        self.a1 = self.act1(self.h1)
        return self.a1


# ----------------------------------------------------------------
# DataSet ***
# train_X = tf.constant(test_df[X_cols])
train_X = tf.constant(test_df[X_cols], dtype=tf.float32)
train_y = tf.constant(test_df[y_col], dtype=tf.float32)

# train_ready ***
n_epochs = 10
model = TF_Model()
loss_function = tf.keras.losses.MeanSquaredError()
optimizer_tf = tf.keras.optimizers.Adam()

# training ***
for e in range(n_epochs):
    with tf.GradientTape() as tape:
        pred = model(train_X)               # predict
        loss = loss_function(pred, train_y)       # loss
    gradients = tape.gradient(loss, model.trainable_variables)      # gradient
    optimize = optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # back-propagation
    print(f"({e+1} epoch) loss: {loss}", end='\r')
    time.sleep(0.3)


# ----------------------------------------------------------------
# DataSet *** (DataSet)
train_ds = tf.data.Dataset.from_tensor_slices((train_X, train_y)).batch(2)
# train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(n_batch).shuffle(train_x.shape[0])
# valid_ds = tf.data.Dataset.from_tensor_slices((valid_x, valid_y)).batch(n_batch).shuffle(train_x.shape[0])

# train_ready ***
n_epochs = 10
model = TF_Model()
loss_function = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

# training ***
for e in range(n_epochs):
    losses = []
    for batch_X, batch_y in train_ds:
        with tf.GradientTape() as tape:
            pred = model(batch_X)               # predict
            loss = loss_function(pred, batch_y)       # loss
        gradients = tape.gradient(loss, model.trainable_variables)      # gradient
        optimize = optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # back-propagation
        losses.append(loss.numpy())
    print(f"({e+1} epoch) loss: {np.mean(losses)}", end='\r')
    time.sleep(0.3)





# Torch -----------------------------------------------------------------------------
class Torch_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(in_features=1, out_features=1)
        self.act1 = lambda x: x
        # self.act1 = torch.nn.ReLU()

    def forward(self, X):
        self.h1 = self.dense1(X)
        self.a1 = self.act1(self.h1)
        return self.a1

model = Torch_Model()

# ----------------------------------------------------------------
# DataSet ***
# train_X = torch.tensor(np.array(test_df[X_cols]))
train_X = torch.tensor(np.array(test_df[X_cols]), dtype=torch.float32)
train_y = torch.tensor(np.array(test_df[y_col]), dtype=torch.float32)

# train_ready ***
n_epochs = 10
model = Torch_Model()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# training ***
for e in range(n_epochs):
    optimizer.zero_grad()               # weight_initialize
    pred = model(train_X)               # predict
    loss = loss_function(pred, train_y) # loss
    loss.backward()                     # backward
    optimizer.step()                    # update_weight
    
    print(f"({e+1} epoch) loss: {loss.item()}", end='\r')
    time.sleep(0.3)

# ----------------------------------------------------------------
# DataSet ***  (DataSet - DataLoader)
train_ds = torch.utils.data.TensorDataset(train_X, train_y)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=2)
# train_dl = torch.utils.data.DataLoader(train_ds, batch_size=len(train_X), shuffle=True)

# train_ready ***
n_epochs = 10
model = Torch_Model()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# training ***
for e in range(n_epochs):
    losses = []
    for batch_X, batch_y in train_dl:
        optimizer.zero_grad()               # weight_initialize
        pred = model(train_X)               # predict
        loss = loss_function(pred, train_y) # loss
        loss.backward()                     # backward
        optimizer.step()                    # update_weight
        
        losses.append(loss.item())
    print(f"({e+1} epoch) loss: {np.mean(losses)}", end='\r')
    time.sleep(0.3)

# ----------------------------------------------------------------
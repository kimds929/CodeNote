import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
import tensorflow as tf

import time
from IPython.display import clear_output
# clear_output(wait=True)

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
x_col = ['x1']

y1 = test_df[y1_col]    # Regressor
X = test_df[x_col]

y1_np = y1.to_numpy()
X_np = X.to_numpy()

plt.figure(figsize=(8,5))
plt.title('Regressor')
plt.plot(X, y1, 'o')



# xp, yp
Xp = np.linspace(np.min(X_np), np.max(X_np), 100).reshape(-1,1)

# Data Train_valid_split
df = pd.DataFrame(np.hstack([X_np, y1_np]), columns=['X', 'y1'])
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



# Dataset
n_batch = 3

train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(train_x.shape[0]).batch(3)

# valid_ds = tf.data.Dataset.from_tensor_slices((valid_x, valid_y)).batch(n_batch).shuffle(train_x.shape[0])


df.iloc[train_indice,:]
for i, (tx, ty) in enumerate(train_ds, 1):
    print(i, tx.numpy().ravel(), ty.numpy().ravel())




# ANN Modeling
# Class
class ModelClass(tf.keras.Model):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units=2, activation='relu')
        self.dense2 = tf.keras.layers.Dense(4, activation=None)
        self.dense3 = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, X):
        d1 = self.dense1(X)
        d2 = self.dense2(d1)
        d3 = self.dense3(d2)
        return d3

# Keras Basic =============================================================================
# compile
model = ModelClass()
model.compile(optimizer='adam', loss=tf.losses.binary_crossentropy)
# model.compile(optimizer='adam', loss=tf.losses.binary_crossentropy, metrics=['accuracy'])

# Normal_fit
# result = model.fit(train_ds, epochs=10)
result = model.fit(train_ds, validation_data=valid_ds, epochs=100, verbose=0)
result.epoch[-1]

# history
# result.history.keys()
# result.history['loss']
# result.history['accuracy']


# ## Early Stopping Callback -----------------------------------------------------------
model_es = ModelClass()
model_es.compile(optimizer='adam', loss=tf.losses.binary_crossentropy)

earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=1)
# ?tf.keras.callbacks.EarlyStopping
# tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     min_delta=0,
#     patience=0,
#     verbose=0,
#     mode='auto',
#     baseline=None,
#     restore_best_weights=False,
# )
result_es = model_es.fit(train_ds, validation_data=valid_ds, epochs=100, callbacks=[earlystopper], verbose=0)
result_es.epoch[-1]


# ## 모델 학습 -----------------------------------------------------------
# history = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=[earlystopper])










# Keras Expert =============================================================================
# Conv2D(kernel_initializer=tf.keras.initializers.he_normal())


# Keras Basic & Expert =============================================================================
# Kernel_initializer -----------------------------------------------------------



# Model Bulilding ---------------------------------------------------------------

# Functional Layer
def layer_function(node):
    funtional_layer = tf.keras.Sequential([tf.keras.layers.Dense(node)])
    return funtional_layer

# Class Layer
class ClassLayer(tf.keras.Model):
    def __init__(self, node):
        super(ClassLayer, self).__init__()
        self.dense = tf.keras.layers.Dense(node)
    
    def call(self, x, print_shape=False):
        dense_output = self.dense(x)
        if print_shape:
            print(f'Class_Layer Call: {x.shape}  → {dense_output.shape}')
        if training:
            print('Class_Layer_training')
        return dense_output

# Functional Class Layer
def functional_class_layer(node):
    funtional_layer = tf.keras.Sequential()
    funtional_layer.add(ClassLayer(node=node))
    return funtional_layer

# Main_model
class Model_Building(tf.keras.Model):
    def __init__(self):
        super(Model_Building, self).__init__()

        self.dense1 = tf.keras.layers.Dense(2)
        self.sequential1 = tf.keras.Sequential([
            tf.keras.layers.Dense(3),
            tf.keras.layers.Dense(5),
        ])
        self.sequential1.add(tf.keras.layers.Dense(7))

        self.funtaional1 = layer_function(node=4)

        self.ClassLayer1 = ClassLayer(node=1)

        self.functional_class_layer1 = functional_class_layer(node=5)

    def call(self, x, print_shape=False):
        input_layer = x
        dense1 = self.dense1(input_layer)
        sequential1 = self.sequential1(dense1)
        funtional1 = self.funtaional1(sequential1)
        ClassLayer1 = self.ClassLayer1(funtional1, print_shape=print_shape)
        functional_class_layer1 = self.functional_class_layer1(ClassLayer1)
        output = functional_class_layer1

        if print_shape:
            print(f'input_layer: {input_layer.shape}')
            print(f'dens1: {dense1.shape}')
            print(f'sequential1: {sequential1.shape}')
            print(f'funtional1: {funtional1.shape}')
            print(f'ClassLayer1: {ClassLayer1.shape}')
            print(f'functional_class_layer1: {functional_class_layer1.shape}')
            print(f'output: {output.shape}')
        return output

x = np.random.rand(4,1).astype('float32')
model = Model_Building()

model(x, print_shape=True)
model(x, training=True)
model(x, training=True, print_shape=True)



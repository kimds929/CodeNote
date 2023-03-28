import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression

import tensorflow as tf
import torch



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
# Regressor
LR_y1 = LinearRegression()
LR_y1.fit(X, y1)
LR_y1_weight = np.round((LR_y1.coef_[0,0], LR_y1.intercept_[0]), 5)      # weight

# Classifier
LR_y2 = LogisticRegression(penalty='none', tol=0.001, max_iter=10000, random_state=1)
LR_y2.fit(X, y2)
LR_y2_weight = np.round((LR_y2.coef_[0,0], LR_y2.intercept_[0]), 5)      # weight

# Plotting
plt.figure(figsize=(10,3))
plt.subplot(121)
plt.title(f'Regressor: {LR_y1_weight}')
plt.plot(X, y1, 'o')
plt.plot(Xp, LR_y1.predict(Xp), linestyle='-', color='orange', alpha=0.5)

plt.subplot(122)
plt.title(f'Classifier: {LR_y2_weight}')
plt.plot(X, y2, 'o')
plt.plot(Xp, LR_y2.predict_proba(Xp)[:,1], linestyle='-', color='orange', alpha=0.5)
plt.show()

















# Loss Function ContourMap -----------------------------------------------------------------
wa = np.linspace(-1000, 1000, 300)
wb = np.linspace(-1000, 1000, 300)
wa_2d, wb_2d = np.meshgrid(wa, wb)

# y1_np
# X_np @ Xc.reshape(1,-1) 
predict = X_np @ wa_2d.reshape(1,-1) + np.ones_like(X_np) @ wb_2d.reshape(1,-1)
contour = ((predict - y1_np)**2).sum(0).reshape(len(wb), len(wa)) / (len(X_np))
contour_df = pd.DataFrame(contour, index=wb, columns=wa)

contour_df.to_clipboard()
# plt.contourf(contour_df.columns, contour_df.index, contour_df, cmap='rainbow')
# plt.colorbar()

# pip install plotly
# pip install nbformat
import plotly.graph_objects as go

fig = go.Figure(data=[go.Surface(x=contour_df.columns, y=contour_df.index, z=contour_df )])
# fig = go.Figure(data=[go.Surface(x=contour_df.columns, y=contour_df.index, z=np.log2(contour_df) )])
fig.show()



2**20
2**14

# plt.scatter(X_np, y1_np)
# plt.plot(X_np, X_np*0.226 +5.02 , color='red')
# plt.plot(Xp, LR_y1.predict(Xp), linestyle='-', color='orange', alpha=0.5)



# [ Tensorflow ] ==========================================================================

# (Tensorflow Basic) ---------------------------------------------------------------
    # model = (Sequential or Class)
    # model.compile(...) → 어떻게 학습할지, option 설정(optimizer, loss, metric)
    # model.fit(x, y)
# ----------------------------------------------------------------------------------

# model_01 (Sequential) ****
n_epoch = 20000

# Regressor
model_01_1 = tf.keras.Sequential([ tf.keras.layers.Dense(1) ])
model_01_1.compile(optimizer='adam', loss='mse', metrics=['mse'])    
result_01_1 = model_01_1.fit(X_np, y1_np, epochs=n_epoch, verbose=0)
weight_01_1 = np.round((model_01_1.weights[0].numpy()[0,0], model_01_1.weights[1].numpy()[0]),5)  # weight

# Classifier
model_01_2 = tf.keras.Sequential([ tf.keras.layers.Dense(1, activation='sigmoid') ])
model_01_2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])    
result_01_2 = model_01_2.fit(X_np, y2_np, epochs=n_epoch, verbose=0)
weight_01_2 = np.round((model_01_2.weights[0].numpy()[0,0], model_01_2.weights[1].numpy()[0]),5)  # weight

# plt.title(f'Classifier: {LR_y2_weight}')
# plt.plot(X, y2, 'o')
# plt.plot(Xp, LR_y2.predict_proba(Xp)[:,1], linestyle='-', color='orange', alpha=0.5, label='sklearn_logistic')
# plt.plot(Xp, model_01_2.predict(Xp), linestyle='-', color='red', alpha=0.5, label='tf_keras_basic')
# plt.show()



# model_02 (Class) ****
n_epoch = 20000

class Model02(tf.keras.Model):
    def __init__(self, kind='regressor'):
        super(Model02, self).__init__()
        if kind == 'regressor':
            self.dense1 = tf.keras.layers.Dense(1, activation=None)
        elif kind == 'classifier':
            self.dense1 = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, X, training=None, mask=None):
        self.d1 = self.dense1(X)
        return self.d1

# Regressor
model_02_1 = Model02(kind='regressor')
model_02_1.compile(optimizer='adam', loss='mse', metrics=['mse'])
result_02_1 = model_02_1.fit(X_np, y1_np, epochs=n_epoch, verbose=0)
weight_02_1 = np.round((model_02_1.weights[0].numpy()[0,0], model_02_1.weights[1].numpy()[0]),5)  # weight

# Classifier
model_02_2 = Model02(kind='classifier')
model_02_2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
result_02_2 = model_02_2.fit(X_np, y2_np, epochs=n_epoch, verbose=0)
weight_02_2 = np.round((model_02_2.weights[0].numpy()[0,0], model_02_2.weights[1].numpy()[0]),5)  # weight







# (Tensorflow Expert) ----------------------------------------------------------------
    # Train_Dataset   : tf.data.Dataset.from_tensor_slices(...)

    # class Model_Class()                                                       # (Neural_Networks Design)
    #   ....

    # model = Model_Class()
    # loss_function = (define)
    # optimizer = (define)

    # @tf.funtion
    # def Training(...)
    #   with GradientTape() as Tape
    #       y_pred = model(X)                                                   # (forward)
    #       loss = loss_function(y_pred, y_true)                                # (calculation_loss)
    #   gradients = Tape.gradient(loss, model.trainable_variables)              # (backward) derivatative loss by weight
    #   optimizer.apply_gradients(zip(gradients, model.trainable_variables))    # (weight_update)
    #   gradients = Tape.gradient(loss, model.trainable_variables)

    # for epoch in range(1, n_epochs+1):
    #   for batch_x, batch_y in enumerate(Train_Dataset):
    #       Training(...)                                                       # (funtion)
    #
    #   if epoch % display_term == 0:
    #       ....                                                                # (displaying_code)
# ------------------------------------------------------------------------------


# dataset ****
train_ds1 = tf.data.Dataset.from_tensor_slices((X_np, y1_np)).batch(X_np.shape[0])
train_ds2 = tf.data.Dataset.from_tensor_slices((X_np, y2_np)).batch(X_np.shape[0])

# print(np.concatenate([X_np, y1_np], axis=1))
# for i, (tx, ty) in enumerate(train_ds1, 1):
#     print(i, tx.numpy(), ty.numpy())

# print(np.concatenate([X_np, y2_np], axis=1))
# for i, (tx, ty) in enumerate(train_ds2, 1):
#     print(i, tx.numpy(), ty.numpy())


# train_ds = tf.data.Dataset.from_tensor_slices((X_np, y1_np))
# for i, (tx, ty) in enumerate(train_ds, 1):
#     print(i, tx.numpy(), ty.numpy())




# Model_03 ****
class Model03(tf.keras.Model):
    def __init__(self, kind='regressor'):
        super(Model03, self).__init__()
        if kind == 'regressor':
            self.dense1 = tf.keras.layers.Dense(1, activation=None)
        elif kind == 'classifier':
            self.dense1 = tf.keras.layers.Dense(1, activation='sigmoid')
            
    def call(self, X, training=True, mask=None):
        self.d1 = self.dense1(X)
        return self.d1

n_epoch = 20000
display_epoch = 1000



# Regressor
model_03_1 = Model03(kind='regressor')
loss_obj_03_1 = tf.keras.losses.MeanSquaredError()
optimizer_03_1 = tf.keras.optimizers.Adam()

start_03_1 = time.time()   # time_start

@tf.function
def training_step(model, X, y, loss_obj, optimizer_obj):
    with tf.GradientTape() as Tape:
        y_pred = model(X)
        loss = loss_obj(y_pred=y_pred, y_true=y)
    gradients = Tape.gradient(loss, model.trainable_variables)
    optimizer = optimizer_obj.apply_gradients(zip(gradients, model.trainable_variables))

for epoch in range(1, n_epoch+1):
    # Model_compile_fit
    for batch_x, batch_y in train_ds1:
        # with tf.GradientTape() as Tape:
        #     y_pred = model_03_1(batch_x)
        #     loss = loss_obj_03_1(y_pred, tf.cast(batch_y, tf.float32))
        # gradients = Tape.gradient(loss, model_03_1.trainable_variables)
        # optimizer = optimizer_03_1.apply_gradients(zip(gradients, model_03_1.trainable_variables))
        
        # tf.function
        training_step(model_03_1, batch_x, tf.cast(batch_y, tf.float32),
            loss_obj_03_1, optimizer_03_1)

    # Display History
    if epoch % display_epoch == 0:
        y_pred = model_03_1(batch_x)
        loss = loss_obj_03_1(y_pred, tf.cast(batch_y, tf.float32))

        coef = model_03_1.trainable_variables[0].numpy()[0,0]
        intercept = model_03_1.trainable_variables[1].numpy()[0]

        print(f'epoch: {epoch}, loss:{format(loss.numpy(), ".2f")}  /  coef: {format(coef,".5f")}, {format(intercept,".5f")}')

weight_03_1 = np.round((model_03_1.weights[0].numpy()[0,0], model_03_1.weights[1].numpy()[0]),5)  # weight
end_03_1 = time.time()   # time_end
print(f'time_elapse: {format(end_03_1 - start_03_1, ".2f")}')
print()



# Classifier
model_03_2 = Model03(kind='classifier')
loss_obj_03_2 = tf.keras.losses.BinaryCrossentropy()
optimizer_03_2 = tf.keras.optimizers.Adam()
start_03_2 = time.time()   # time_start

@tf.function
def training_step2(model, X, y, loss_obj, optimizer_obj):
    with tf.GradientTape() as Tape:
        y_pred = model(X)
        loss = loss_obj(y_pred=y_pred, y_true=y)
    gradients = Tape.gradient(loss, model.trainable_variables)
    optimizer_obj.apply_gradients(zip(gradients, model.trainable_variables))

for epoch in range(1, n_epoch+1):
    # Model_compile_fit
    for batch_x, batch_y in train_ds2:
        # with tf.GradientTape() as Tape:
        #     y_pred = model_03_2(batch_x)
        #     loss = loss_obj_03_2(y_pred, tf.cast(batch_y, tf.float32))
        # gradients = Tape.gradient(loss, model_03_2.trainable_variables)
        # optimizer = optimizer_03_2.apply_gradients(zip(gradients, model_03_2.trainable_variables))
        
        # tf.function
        training_step2(model_03_2, batch_x, tf.cast(batch_y, tf.float32),
            loss_obj_03_2, optimizer_03_2)

    # Display History
    if epoch % display_epoch == 0:
        y_pred = model_03_2(batch_x)
        loss = loss_obj_03_2(y_pred, tf.cast(batch_y, tf.float32))

        coef = model_03_2.trainable_variables[0].numpy()[0,0]
        intercept = model_03_2.trainable_variables[1].numpy()[0]

        print(f'epoch: {epoch}, loss:{format(loss.numpy(), ".2f")}  /  coef: {format(coef,".5f")}, {format(intercept,".5f")}')

weight_03_2 = np.round((model_03_2.weights[0].numpy()[0,0], model_03_2.weights[1].numpy()[0]),5)  # weight
end_03_2 = time.time()   # time_end
print(f'time_elapse: {format(end_03_2 - start_03_2, ".2f")}')
print()

# loss_obj_03_2(model_02_2(batch_x), tf.cast(batch_y,tf.float32))
# loss_obj_03_2(model_03_2(batch_x), tf.cast(batch_y,tf.float32))

plt.title(f'Classifier: {LR_y2_weight}')
plt.plot(X, y2, 'o')
plt.plot(Xp, LR_y2.predict_proba(Xp)[:,1], linestyle='-', color='orange', alpha=0.5, label='sklearn_logistic')
plt.plot(Xp, model_02_2.predict(Xp), linestyle='-', color='red', alpha=0.5, label='tf_keras_basic')
plt.plot(Xp, model_03_2.predict(Xp), linestyle='-', color='blue', alpha=0.5, label='tf_keras_expert')
plt.legend()
plt.show()






























################################################################################################################################

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.python.client import device_lib
# device_lib.list_local_devices()

import time
from IPython.display import clear_output


tf.__version__
# tf.config.experimental.list_physical_devices('GPU')
# tf.debugging.set_log_device_placement(True)     # 해당 연산이 어떤 장치에 할당 되었는지 알려줌
# tf.debugging.set_log_device_placement(False)     # 해당 연산이 어떤 장치에 할당 되었는지 알려줌


wine_dict = {'Mo': [0.044, 0.16 , 0.146, 0.191, 0.363, 0.106, 0.479, 0.234, 0.058,
        0.074, 0.071, 0.147, 0.116, 0.166, 0.261, 0.191, 0.009, 0.027,
        0.05 , 0.268, 0.245, 0.161, 0.146, 0.155, 0.126],
 'Ba': [0.387, 0.312, 0.308, 0.165, 0.38 , 0.275, 0.164, 0.271, 0.225,
        0.329, 0.105, 0.301, 0.166, 0.132, 0.078, 0.085, 0.072, 0.094,
        0.294, 0.099, 0.071, 0.181, 0.328, 0.081, 0.299],
 'Cr': [0.029, 0.038, 0.035, 0.036, 0.059, 0.019, 0.062, 0.044, 0.022,
        0.03 , 0.028, 0.087, 0.041, 0.026, 0.063, 0.063, 0.021, 0.021,
        0.006, 0.045, 0.053, 0.06 , 0.1  , 0.037, 0.054],
 'Sr': [1.23 , 0.975, 1.14 , 0.927, 1.13 , 1.05 , 0.823, 0.963, 1.13 ,
        1.07 , 0.491, 2.14 , 0.578, 0.229, 0.156, 0.192, 0.172, 0.358,
        1.12 , 0.36 , 0.186, 0.898, 1.32 , 0.164, 0.995],
 'Pb': [0.561, 0.697, 0.73 , 0.796, 1.73 , 0.491, 2.06 , 1.09 , 0.048,
        0.552, 0.31 , 0.546, 0.518, 0.699, 1.02 , 0.777, 0.232, 0.025,
        0.206, 1.28 , 1.19 , 0.747, 0.604, 0.767, 0.686],
 'B': [2.63, 6.21, 3.05, 2.57, 3.07, 6.56, 4.57, 3.18, 6.13, 3.3 , 6.56,
        3.5 , 6.43, 7.27, 5.04, 5.56, 3.79, 4.24, 2.71, 5.68, 4.42, 8.11,
        6.42, 4.91, 6.94],
 'Mg': [128. , 193. , 127. , 112. , 138. , 172. , 179. , 145. , 113. ,
        140. , 103. , 199. , 111. , 107. ,  94.6, 110. ,  75.9,  80.9,
        120. ,  98.4,  87.6, 160. , 134. ,  86.5, 129. ],
 'Ca': [ 80.5,  75. ,  91. ,  93.6,  84.6, 112. , 122. ,  91.9,  70.2,
         74.7,  67.9,  66.3,  83.8,  44.9,  54.9,  64.1,  48.1,  57.6,
         64.8,  64.3,  70.6,  82.1,  83.2,  53.9,  85.9],
 'K': [1130, 1010, 1160,  924, 1090, 1290, 1170, 1020, 1240, 1100, 1090,
        1470, 1120,  854,  899,  976,  995,  876, 1050,  945,  820, 1220,
        1810, 1020, 1330],
 'Aroma': [3.3, 4.4, 3.9, 3.9, 5.6, 4.6, 4.8, 5.3, 4.3, 4.3, 5.1, 3.3, 5.9,
        7.7, 7.1, 5.5, 6.3, 5. , 4.6, 6.4, 5.5, 4.7, 4.1, 6. , 4.3]}

data = pd.DataFrame(wine_dict)


x_cols = ['Mo', 'Ba', 'Cr','B','K']
Y_col = ['Aroma']




from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(data, test_size=0.3)

df_train_X = tf.constant(df_train[x_cols], dtype=tf.float32)
df_train_Y = tf.constant(df_train[Y_col], dtype=tf.float32)

df_test_X = tf.constant(df_test[x_cols], dtype=tf.float32)
df_test_Y = tf.constant(df_test[Y_col], dtype=tf.float32)
# np.random.permutation(len(data)) # shuffle index


n_batch=8
train_set = tf.data.Dataset.from_tensor_slices((df_train_X, df_train_Y)).batch(n_batch).shuffle(len(df_train))
test_set = tf.data.Dataset.from_tensor_slices((df_test_X, df_test_Y)).batch(n_batch).shuffle(len(df_test))


# Tensorflow Model Class
class TF_Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8, name='layer1')
        self.dense2 = tf.keras.layers.Dense(64, name='layer2')
        self.dense3 = tf.keras.layers.Dense(1, name='layer3')

    def call(self, X, training=False):
        self.i = X
        self.h1 = tf.keras.activations.relu(self.dense1(self.i))
        self.h2 = tf.keras.activations.relu(self.dense2(self.h1))
        self.o = self.dense3(self.h2)
        
        # self.h1 = tf.keras.activations.relu(self.dense1(self.i))
        # self.o = self.dense3(self.h1)
        return self.o



# Tensorflow Step Function
@tf.function
def tf_step(model, X, Y, loss_function, optimizer, training=False):
    result = {'pred':None, 'loss':None}
    
    if training:
        with tf.GradientTape() as Tape:
            y_pred = model(X, training=training)  # forward
            loss = loss_function(Y, y_pred)   # loss
        gradients = Tape.gradient(loss, model.trainable_variables)  # backward
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))    # update weight
        
        result['gradients'] = [t.numpy() for t in gradients]
        result['weights'] = [t.numpy() for t in model.trainable_variables]
    else:
        y_pred = model(X, training=training)  # forward
        loss = loss_function(Y, y_pred)   # loss

    result['pred'] = y_pred.numpy()
    result['loss'] = loss.numpy()
    return result


# Model Learning
model = TF_Model()
model.summary()

# 5*8+8
# 8*64+64
# 64*1+1
# pd.DataFrame( model.i.numpy() @ model.dense1.get_weights()[0]  + model.dense1.get_weights()[1].reshape(1,-1) )
# pd.DataFrame(model.dense1(model.i).numpy())
# pd.DataFrame(model.h1.numpy())
# pd.DataFrame(model.h2.numpy())
# pd.DataFrame(model.o.numpy())





loss_function = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
# rmse(test_X_tf, test_result['pred'])

loss_train = []
loss_test = []
EPOCHS = 10
for epoch in range(EPOCHS):
    # train
    loss_train_batch = []
    for train_batch_X, train_batch_Y in train_set: 
        train_result = tf_step(model, train_batch_X, train_batch_Y, loss_function, optimizer, training=True)
        loss_train_batch.append(train_result['loss'])
    loss_train.append(np.mean(loss_train_batch))
    
    # valid
    loss_test_batch = []
    for test_batch_X, test_batch_Y in test_set:
        test_result = tf_step(model, test_batch_X, test_batch_Y, loss_function, optimizer)
        loss_test_batch.append(train_result['loss'])
    loss_test.append(np.mean(loss_test_batch))
        
    clear_output(wait=True)
    plt.figure()
    plt.plot(loss_train, 'o-', label='train')
    plt.plot(loss_test, 'o-', label='test')
    plt.legend(loc='upper right')
    plt.show()
    time.sleep(0.1)

# 【 parameter 저장 】
# np.savez_compressed('abcd.npz', input=np.array([1,2,3,4]), aaa=np.array(['a','b']))
# a = np.load('abcd.npz')
# list(a.keys())


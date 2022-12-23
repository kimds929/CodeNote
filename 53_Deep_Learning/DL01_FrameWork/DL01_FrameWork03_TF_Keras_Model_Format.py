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








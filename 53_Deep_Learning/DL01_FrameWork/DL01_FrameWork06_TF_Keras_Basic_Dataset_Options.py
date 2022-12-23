import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
# import torch

from sklearn.linear_model import LinearRegression, LogisticRegression
import time
from IPython.display import clear_output
# clear_output(wait=True)

import os
from datetime import datetime
import math


# --------------------------------------------------------------------------------------------------------
# Tensorflow Regressor Problem ---------------------------------------------------------------------



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

y1_np = y1.to_numpy()    # Regressor
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
# Classifier
LR_y2 = LogisticRegression(penalty='none', tol=0.001, max_iter=10000, random_state=1)
LR_y2.fit(X, y2)
LR_y2_weight = np.round((LR_y2.coef_[0,0], LR_y2.intercept_[0]), 5)      # weight

# Plotting
plt.figure(figsize=(5,3))
plt.title(f'Classifier: {LR_y2_weight}')
plt.plot(X, y2, 'o')
plt.plot(Xp, LR_y2.predict_proba(Xp)[:,1], linestyle='-', color='orange', alpha=0.5)
plt.show()



# Dataset =================================================================================================================================
# Data Train_valid_split ****
df = pd.DataFrame(np.hstack([y2_np, X_np]), columns=['y2', 'X'])
indice = np.random.permutation(12)
train_indice = indice[:8]
test_indice = indice[8:]
print(f'train_indice: {train_indice}')
print(f'test_indice: {test_indice}')

train_x = X_np[train_indice]
train_y = y2_np[train_indice]
valid_x = X_np[test_indice]
valid_y = y2_np[test_indice]
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



# Keras Basic Model =================================================================================================================================
# 
n_epoch = 20

# Classifier
model_01 = tf.keras.Sequential([ tf.keras.layers.Dense(1, activation='sigmoid') ])
model_01.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

result_01 = model_01.fit(X_np, y2_np, batch_size=3, epochs=n_epoch, verbose=2)              # batch_size
# result_01 = model_01.fit(train_ds, validation_data=valid_ds, epochs=n_epoch, verbose=2)   # validation_data
# result_01 = model_01.fit(train_ds, validation_split=0.1, epochs=n_epoch, verbose=2)       # validation_split

weight_01_1 = np.round((model_01.weights[0].numpy()[0,0], model_01.weights[1].numpy()[0]),5)  # weight




# [ Predict ] =============================================================================================================================



# [ History ] =================================================================================================================================
result_01.history.keys()
result_01.params
result_01.history['loss']


plt.figure(figsize=(10,3))
plt.subplot(121)
plt.title('Accuracy')
plt.plot(result_01.history['accuracy'], 'o-', label='train')
plt.plot(result_01.history['val_accuracy'], 'o-', label='valiation')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()

plt.subplot(122)
plt.title('Loss')
plt.plot(result_01.history['loss'], 'o-', label='train')
plt.plot(result_01.history['val_loss'], 'o-', label='valiation')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()







# Callbacks =================================================================================================================================

# [ Tensorboard ] **** ----------------------------------------------------------------------------------------------------------------------
logdir = os.path.join('tensorboard',  datetime.now().strftime("%Y%m%d-%H%M%S"))     # tensorboard를 어디에 저장할지 지정

tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir=logdir, write_graph=True, write_images=True, histogram_freq=1)

# Classifier
n_epoch = 20
model_tensorboard = tf.keras.Sequential([ tf.keras.layers.Dense(1, activation='sigmoid') ])
model_tensorboard.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
result_tensorboard = model_tensorboard.fit(train_ds, validation_data=valid_ds, epochs=n_epoch, verbose=2, callbacks=[tensorboard])     # ****


# https://colab.research.google.com/github/tensorflow/tensorboard/blob/master/docs/r2/image_summaries.ipynb#scrollTo=IJNpyVyxbVtT

# get_ipython().run_line_magic('load_ext', 'tensorboard')
# %tensorboard --logdir tensorboard --port 8008
# tensorboard --logdir=./경로 --port 8008       # Terminal 실행
# http://localhost:8008/
# get_ipython().run_line_magic('tensorboard', '--logdir tensorboard --port 8008')
# 【Terminal】tensorboard --logdir=D:/Python/"강의) [FastCampus] 딥러닝 올인원 패키지"/dataset/tensorboard --port 8008 





# [ EarlyStopping ] **** ----------------------------------------------------------------------------------------------------------------------
earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)

# Classifier
n_epoch = 20
model_earlyStop = tf.keras.Sequential([ tf.keras.layers.Dense(1, activation='sigmoid') ])
model_earlyStop.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
result_earlyStop = model_earlyStop.fit(train_ds, validation_data=valid_ds, epochs=n_epoch, verbose=2, callbacks=[earlystopper])     # ****




# [ Learning_Rate Scheduler ] ****
def scheduler(epoch):
    if epoch < 3:
        return 0.0001
    else:
        return 0.001 * math.exp(0.1 * (10 - epoch))

for epoch in range(5,15):
    print(scheduler(epoch))

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

# Classifier
n_epoch = 20
model_lrSchedule = tf.keras.Sequential([ tf.keras.layers.Dense(1, activation='sigmoid') ])
model_lrSchedule.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
result_lrSchedule = model_lrSchedule.fit(train_ds, validation_data=valid_ds, epochs=n_epoch, verbose=2, callbacks=[lr_scheduler])     # ****



# [ Check Point ] **** ----------------------------------------------------------------------------------------------------------------------
# Weight를 저장
save_path = 'checkpoints'   # 경로지정
checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

n_epoch = 20
model_checkPoint = tf.keras.Sequential([ tf.keras.layers.Dense(1, activation='sigmoid') ])
model_checkPoint.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
result_checkPoint = model_checkPoint.fit(train_ds, validation_data=valid_ds, epochs=n_epoch, verbose=2, callbacks=[checkpoint])     # ****



# 만들어진 체크포인트를 확인해 보고 마지막 체크포인트를 선택해 보겠습니다:
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
os.listdir(checkpoint_path)
checkpoint_dir = os.path.dirname(checkpoint_path)

latest = tf.train.latest_checkpoint(checkpoint_dir)
latest

# 가중치를 저장합니다
model_checkPoint.save_weights('./checkpoints/my_checkpoint')

# 새로운 모델 객체를 만듭니다
model = tf.keras.Sequential([ tf.keras.layers.Dense(1, activation='sigmoid') ])
# model = tf.keras.Model(inputs=inputs, outputs=net, name='Basic_CNN')


# 가중치를 복원합니다
model.load_weights('./checkpoints/my_checkpoint')









# Define the per-epoch callback.
# cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

# def log_confusion_matrix(model, epoch, logs):
#     # Use the model to predict the values from the validation dataset.
#     test_pred_raw = model.predict(test_images)
#     test_pred = np.argmax(test_pred_raw, axis=1)

#     # Calculate the confusion matrix.
#     cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
#     # Log the confusion matrix as an image summary.
#     figure = plot_confusion_matrix(cm, class_names=class_names)
#     cm_image = plot_to_image(figure)

#     # Log the confusion matrix as an image summary.
#     with file_writer_cm.as_default():
#         tf.summary.image("Confusion Matrix", cm_image, step=epoch)












# [ Model_Save_Load ] ======================================================================================================================
# test_dict = {'y1': [2, 12,  6, 19, 5,  5, 14, 8, 12, 20,  1, 10],
#             'y2': [0, 0,  1, 1, 0,  0, 1, 0, 1, 1,  0, 1],
#             'x1': [ 5,  5, 35, 38,  9, 19, 30,  2, 49,  30,  0, 14],
#             'x2': ['a', 'c', 'a', 'b', 'b', 'b', 'a', 'c', 'c', 'a', 'b', 'c'],
#             'x3': [46, 23, 23,  3, 36, 10, 14, 28,  5, 19, 42, 32],
#             'x4': ['g1', 'g2', 'g1', 'g2', 'g1', 'g2', 'g1', 'g2', 'g1', 'g2', 'g2', 'g2']
#             }
# test_df = pd.DataFrame(test_dict)

# y1_col = ['y1']     # Regressor
# y2_col = ['y2']     # Classifier
# x_col = ['x1']

# y = test_df[y1_col].to_numpy()
# X = test_df[x_col].to_numpy()

# modeling
class Test_Model(tf.keras.Model):
    def __init__(self):
        super(Test_Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(1, activation=tf.sigmoid)

    def call(self, x):
        h = self.dense1(x)
        return h


test_model = tf.keras.Sequential([ tf.keras.layers.Dense(1, activation='sigmoid') ])
# test_model = Test_Model()
test_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
test_model.fit(train_ds, validation_data=valid_ds, epochs=n_epoch, verbose=2)
# test_model.weights[0].numpy()[0,0], test_model.weights[1].numpy()[0]





# Save & Load Model (subclass model X) -------------------------------
# 모델 전체를 가져옴
# model_save
folder_path = r'D:\Python\강의) [FastCampus] 딥러닝 올인원 패키지'
save_model_path = folder_path + '/save/model/model_save.h5'
model.save(save_model_path, include_optimizer=True)

# model_load
load_model = tf.keras.models.load_model(save_model_path)
load_model.height


# ※ model.h5 모델 들여다보기 ---------
import h5py
model_file = h5py.File(save_model_path, 'r+')
model_file.keys()




# Save & Load Model_Architecture using JSON (subclass model X) ---------------------
# 모델의 구조만 가져옴
model_json_path = save_model_path + '/model_json.json'

# model_save
with open(model_json_path, 'w') as f:
    f.write(model.to_json())

# model_load
with open(model_json_path, 'r') as f:
    json_load_model = tf.keras.models.model_from_json(f.read())
json_load_model.weights




# Save & Load Weights ----------------------------------------------------
# 모델의 weight만 가져옴
    # save_weights
folder_path = r'D:\Python\강의) [FastCampus] 딥러닝 올인원 패키지'
save_weight_path = folder_path + '/save/weight/my_weight.h5'
test_model.save_weights(save_weight_path)

new_model = Test_Model()
new_model.weights
new_model.predict(X)

    # load_weights
new_model.load_weights(save_weight_path)
new_model.weights[0].numpy()[0,0], new_model.weights[1].numpy()[0]
new_model.predict(X)


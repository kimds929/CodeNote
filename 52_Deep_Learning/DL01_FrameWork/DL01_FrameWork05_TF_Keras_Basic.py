import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import tensorflow as tf
# import torch

from sklearn.linear_model import LinearRegression, LogisticRegression
import time
from IPython.display import clear_output
# clear_output(wait=True)


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

test_df[y1_col + y2_col + x_col]


y1_np = y1.to_numpy()    # Regressor
y2_np = y2.to_numpy()    # Classifier
X_np = X.to_numpy()

np.concatenate([y1_np, y2_np, X_np], axis=1)

plt.figure(figsize=(10,3))
plt.subplot(121)
plt.title('Regressor')
plt.plot(X_np, y1_np, 'o')

plt.subplot(122)
plt.title('Classifier')
plt.plot(X_np, y2_np, 'o')
plt.show()



# [ Sklearn ] ============================================================================
# Regressor
LR_y1 = LinearRegression()
LR_y1.fit(X, y1)
LR_y1_weight = list(np.round((LR_y1.coef_[0,0], LR_y1.intercept_[0]), 5))      # weight

# Classifier
LR_y2 = LogisticRegression(random_state=1)
# LR_y2 = LogisticRegression(penalty='none', tol=0.001, max_iter=10000, random_state=1)
LR_y2.fit(X, y2)
LR_y2_weight = list(np.round((LR_y2.coef_[0,0], LR_y2.intercept_[0]), 5))      # weight


# xp, yp
Xp = np.linspace(np.min(X_np), np.max(X_np), 100).reshape(-1,1)

# Plotting
plt.figure(figsize=(10,3))
plt.subplot(121)
plt.title(f'Regressor: {LR_y1_weight}')
plt.scatter(X, y1, color='steelblue')
plt.plot(Xp, LR_y1.predict(Xp), linestyle='-', color='orange', alpha=0.5)

plt.subplot(122)
plt.title(f'Classifier: {LR_y2_weight}')
plt.scatter(X, y2, color='steelblue')
plt.plot(Xp, LR_y2.predict_proba(Xp)[:,1], linestyle='-', color='orange', alpha=0.5)
plt.show()


def model_plot(X, y, y_pred, weight=None, name='', color='orange', label=None, return_plot=False):
    Xp = np.linspace(np.min(X), np.max(X), 100).reshape(-1,1)

    if return_plot:
        f = plt.figure()
    plt.title(f'{name}: {weight}')
    plt.scatter(X, y, color='steelblue')
    plt.plot(Xp, y_pred, linestyle='-', color=color, alpha=0.5, label=label)
    
    if return_plot:
        plt.close()
        return f

model_plot(X, y1, LR_y1.predict(Xp), LR_y1_weight, '(Sklearn) Regressor', return_plot=True)
model_plot(X, y2, LR_y2.predict_proba(Xp)[:,1], LR_y2_weight, '(Sklearn) Classifier', return_plot=True)



# [ Tensorflow ] ==========================================================================
# (Tensorflow Basic) ---------------------------------------------------------------

# model_01 (Sequential) ****

# 【 Keras Basic (Sequential) 】
    # (Essential Format) -------------------------------------------------------------
    # model = (Sequential or Class)
    # model.compile(...)           → 어떻게 학습할지, option 설정(optimizer, loss, metric)
    # model.fit(x, y)
    
    # (Model Attribute) ---------------------------------------------------------------
    # model.summary()   → Neural_Network Modeling 결과 Display
    # model.evaluate(...)  → Test 데이터 셋에 대해 전체 결과를 보고 싶을때
    # model.predict(...)   → 하나의 입력으로 결과를 보고 싶을때
    # model.weights     → 모델의 현재 Weights

    # (result Attribute) --------------------------------------------------------------
    # result = model.fit(x, y)
    # result.history → model 학습결과를 dictionary내의 list형태로 저장
    
    # ---------------------------------------------------------------------------------



n_epoch = 500
epoch = 0
# ?model_01_1.fit

# Regressor
model_01_1 = tf.keras.Sequential([ tf.keras.layers.Dense(1) ])
model_01_1.compile(optimizer='adam', loss='mse')
                #    , metrics=['mse'])    

# result_01_1 = model_01_1.fit(X_np, y1_np, epochs=n_epoch, verbose=1)
# model_01_1.fit(X_np, y1_np, epochs=n_epoch, verbose=0)
# weight_01_1 = [round(w.numpy().ravel()[0], 5) for w in model_01_1.weights]  # weight
# model_plot(X, y1, LR_y1.predict(Xp), LR_y1_weight, 'Regressor', label='sklearn')
# model_plot(X, y1, model_01_1.predict(Xp), weight_01_1, 'Regressor', label='TF_Keras', color='red')

for i in range(10):
    result_01_1 = model_01_1.fit(X_np, y1_np, epochs=n_epoch, verbose=0)
    weight_01_1 = [round(w.numpy().ravel()[0], 5) for w in model_01_1.weights]  # weight

    clear_output(wait=True)
    epoch += n_epoch
    print(epoch)
    
    weights_reg = {'sklearn':LR_y1_weight, 'TF_keras': weight_01_1}
    
    plt.figure()
    model_plot(X, y1, LR_y1.predict(Xp), weights_reg, 'Regressor', label='sklearn')
    model_plot(X, y1, model_01_1.predict(Xp), weights_reg, 'Regressor', label='TF_Keras', color='red')
    plt.legend()
    plt.show()




# Classifier
n_epoch = 500
epoch = 0

model_01_2 = tf.keras.Sequential([ tf.keras.layers.Dense(1, activation='sigmoid') ])
model_01_2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])    

# result_01_2 = model_01_2.fit(X_np, y2_np, epochs=n_epoch, verbose=0)
# weight_01_2 = np.round((model_01_2.weights[0].numpy()[0,0], model_01_2.weights[1].numpy()[0]),5)  # weight

for i in range(10):
    result_01_2 = model_01_2.fit(X_np, y2_np, epochs=n_epoch, verbose=0)
    weight_01_2 = [round(w.numpy().ravel()[0], 5) for w in model_01_2.weights]  # weight

    clear_output(wait=True)
    epoch += n_epoch
    print(epoch)
    
    weights_clf = {'sklearn':LR_y2_weight, 'TF_keras': weight_01_2}
    
    plt.figure()
    model_plot(X, y2, LR_y2.predict_proba(Xp)[:,1], weights_clf, 'Classifier', label='sklearn')
    model_plot(X, y2, model_01_2.predict(Xp), weights_clf, 'Classifier', label='TF_Keras', color='red')
    plt.legend()
    plt.show()











# (model) **** --------------------------------------------------------
    # Sequential
    # tf.keras.models.Sequential?
    # tf.keras.models.Sequential(layers=None, name=None)
model_02 = tf.keras.Sequential([
    tf.keras.layers.Dense(units=2, activation='relu'),
    tf.keras.layers.Dense(4, activation=None),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

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

model_02 = ModelClass()
# tf.keras.layers.Dense?
# tf.keras.layers.Dense(
#       units, activation=None, use_bias=True,
#       kernel_initializer='glorot_uniform', bias_initializer='zeros',
#       kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
#       kernel_constraint=None, bias_constraint=None, **kwargs,)




# (model.compile(...)) **** --------------------------------------------------------
model_02.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'auc', 'recall', 'precision'])


optimizer_obj = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_obj = tf.keras.losses.BinaryCrossentropy()
metrics_obj = [tf.keras.metrics.Accuracy(), tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]

model_02.compile(optimizer=optimizer_obj, loss=loss_obj, metrics=metrics_obj)


    # compile
# keras_mdl.compile?
# keras_mdl.compile(optimizer='rmsprop', loss=None, metrics=None,
#         loss_weights=None, sample_weight_mode=None, weighted_metrics=None,
#         target_tensors=None, distribute=None, **kwargs,)

# optimizer(tf.keras.optimizers.) : ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Ftrl', 'Nadam', 'Optimizer', 'RMSprop', 'SGD']

# loss(tf.keras.losses.) :
    # (class)
    # ['BinaryCrossentropy', 'CategoricalCrossentropy', 'CategoricalHinge', 'CosineSimilarity',
    #  'Hinge', 'Huber', 'KLDivergence', 'LogCosh', 'Loss', 'MeanAbsoluteError',
    #  'MeanAbsolutePercentageError', 'MeanSquaredError', 'MeanSquaredLogarithmicError',
    #  'Poisson', 'Reduction', 'SparseCategoricalCrossentropy', 'SquaredHinge']

    # (function)
    # ['KLD', 'MAE', 'MAPE', 'MSE', 'MSLE', 'binary_crossentropy',
    #  'categorical_crossentropy', 'categorical_hinge', 'cosine_similarity',
    #  'deserialize', 'get', 'hinge', 'huber', 'kl_divergence', 'kld',
    #  'kullback_leibler_divergence', 'log_cosh']


# metrics(tf.keras.metrics.)
    # (class)
    # ['AUC', 'Accuracy', 'BinaryAccuracy', 'BinaryCrossentropy', 'CategoricalAccuracy',
    #  'CategoricalCrossentropy', 'CategoricalHinge', 'CosineSimilarity', 'FalseNegatives',
    #  'FalsePositives', 'Hinge', 'KLDivergence', 'LogCoshError', 'Mean', 'MeanAbsoluteError',
    #  'MeanAbsolutePercentageError', 'MeanIoU', 'MeanRelativeError', 'MeanSquaredError',
    #  'MeanSquaredLogarithmicError', 'MeanTensor', 'Metric', 'Poisson', 'Precision',
    #  'PrecisionAtRecall', 'Recall', 'RecallAtPrecision', 'RootMeanSquaredError',
    #  'SensitivityAtSpecificity', 'SparseCategoricalAccuracy', 'SparseCategoricalCrossentropy',
    #  'SparseTopKCategoricalAccuracy', 'SpecificityAtSensitivity', 'SquaredHinge',
    #  'Sum', 'TopKCategoricalAccuracy', 'TrueNegatives', 'TruePositives']

    # (function)
    # ['KLD', 'MAE', 'MAPE', 'MSE', 'MSLE', 'binary_accuracy', 'binary_crossentropy',
    #  'categorical_accuracy', 'categorical_crossentropy', 'deserialize', 'get',
    #  'hinge', 'kl_divergence', 'kld', 'kullback_leibler_divergence', 'mae',
    #  'mape', 'mean_absolute_error', 'mean_absolute_percentage_error',
    #  'mean_squared_error', 'mean_squared_logarithmic_error', 'mse', 'msle',
    #  'poisson', 'serialize', 'sparse_categorical_accuracy',
    #  'sparse_categorical_crossentropy', 'sparse_top_k_categorical_accuracy',
    #  'squared_hinge', 'top_k_categorical_accuracy']




# (model.fit(...)) **** --------------------------------------------------------
model_02.fit(X_np, y2_np, batch_size=X_np.shape[0], epochs=10000, verbose=0, shuffle=True)
# keras_mdl.fit?
# keras_mdl.fit(
#       x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
#       validation_split=0.0, validation_data=None, shuffle=True,
#       class_weight=None, sample_weight=None, initial_epoch=0,
#       steps_per_epoch=None, validation_steps=None, validation_freq=1,
#       max_queue_size=10, workers=1, use_multiprocessing=False, **kwargs,)




# (model.summary()) **** --------------------------------------------------------
model_02.summary()          # Neural_Network Modeling 결과 Display
# Model: "model_class_2"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# dense_5 (Dense)              multiple                  4         
# _________________________________________________________________
# dense_6 (Dense)              multiple                  12        
# _________________________________________________________________
# dense_7 (Dense)              multiple                  5         
# =================================================================
# Total params: 21
# Trainable params: 21
# Non-trainable params: 0
# _________________________________________________________________



# (model.evaluate(...)) **** --------------------------------------------------------
model_02.evaluate(X_np, y2_np)
# keras_mdl.evaluate?
# keras_mdl.evaluate(
#       x=None, y=None, batch_size=None, verbose=1,
#       sample_weight=None, steps=None, callbacks=None,
#       max_queue_size=10, workers=1, use_multiprocessing=False,)



# (model.predict(...)) **** --------------------------------------------------------
model_02.predict(X_np)
# ?model_02.predict
# model_02.predict(
#       x, batch_size=None, verbose=0,
#       steps=None, callbacks=None, max_queue_size=10,
#       workers=1, use_multiprocessing=False, )



# (model.weights) **** -------------------------------------------------------------
model_02.weights




# (result.history) **** -------------------------------------------------------------
result = model_02.fit(X_np, y2_np, batch_size=X_np.shape[0], epochs=20000, verbose=0, shuffle=True)
result.history.keys()

for hist in result.history.keys():
    print(f'{hist} : {len(result.history[hist])}')

plt.figure(figsize=(20,9))
for i, hist in enumerate(result.history.keys(),1):
    plt.subplot(2,3,i)
    plt.title(hist)
    plt.plot(np.arange(1,20001), result.history[hist])
plt.show()






# Result_Plotting **** --------------------------------------------------------------
plt.title(f'Classifier: {LR_y2_weight}')
plt.plot(X, y2, 'o')
plt.plot(Xp, LR_y2.predict_proba(Xp)[:,1], linestyle='-', color='orange', alpha=0.5)
plt.plot(Xp, model_02.predict(Xp), linestyle='-', color='red', alpha=0.5)
plt.show()





























import random

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers
import torch

from IPython.display import clear_output
import time


# ○ Overfitting Prevension -------------------------------------------------------------
#   1) Validation_Set: Early_Stoping        # patience: 몇번 연속으로 높은 loss가 나왔는지?
#   2) Regularization
#       . Add Regularization term at Loss_Function
#       . Drop_Out
#       . Batch_Normalization
#       . Learning Rate Schedulers (Step_Decay, Exponential_Decay, Two_Stage_Exponential_Deacay, Linear_Warm_Up)
#   3) Ensemble
#   ※ Parameter에 따라 Model성능 차이가 심함






# 1) Validation_Set: Early_Stoping  ====================================================
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)


# Data Preprocessing ------------------------------------
(x_train_val, y_train_val), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

num_classes = 10
num_features = 784

x_train_val = x_train_val.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784).astype('float32')/255

y_train_val = y_train_val.astype('float32')
y_test = y_test.astype('float32')



# train_validation_split ------------------------------------
indices = np.random.permutation(x_train_val.shape[0])
train_indices = indices[:-10000]
valid_indices = indices[-10000:]

x_train = x_train_val[train_indices]
x_valid = x_train_val[valid_indices]
y_train = y_train_val[train_indices]
y_valid = y_train_val[valid_indices]

print(indices[:10])     # indice Sample
print(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape)





# Modling ---------------------------------------------------
# Loss_Function
def cross_entropy_loss(x, y):
    y = tf.cast(y, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    return tf.reduce_mean(loss)

# Metrics
def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

# Batch Accuracy
def accuracy_batch(model, test_data):
    acc = 0
    for step, (batch_x, batch_y) in enumerate(test_data, 1):
        pred = model(batch_x, is_training=False)
        acc += accuracy(pred, batch_y)
    acc = acc / step * 100
    return acc



# Training Model Overall Run
def train_model(train_data, model, lr, epochs, 
                print_loss=True, plot_graph=True):
    step_l = []
    loss_l = []

    optimizer = tf.optimizers.SGD(lr)
    n_batch = len(list(train_data))
    # Training Run
    # @tf.function
    # def train_running(X, y, model, loss, optimizer):
    #     with tf. GradientTape() as Tape:
    #         y_pred = model(X, is_training=True)
    #         model_loss = loss(y_pred, y)

    #     train_weights = model.trainable_variables
    #     gradients = Tape.gradient(model_loss, train_weights)
    #     optimizer.apply_gradients(zip(gradients, train_weights))
    #     return model_loss

    for epoch in range(1, epochs+1):

        running_loss = 0.0
        for step, (batch_x, batch_y) in enumerate(train_data, 1):

            # model_loss = train_running(batch_x, batch_y, model, cross_entropy_loss, optimizer)
            with tf. GradientTape() as Tape:
                y_pred = model(batch_x, is_training=True)
                model_loss = cross_entropy_loss(y_pred, batch_y)

            train_weights = model.trainable_variables
            gradients = Tape.gradient(model_loss, train_weights)
            optimizer.apply_gradients(zip(gradients, train_weights))

            running_loss += model_loss.numpy()

            if plot_graph:
                if step % 10 == 0:
                    step_l.append(epoch * n_batch + step)
                    loss_l.append(running_loss/10)
                    running_loss = 0.0
        
        if print_loss:
            print(f'epoch: {epoch},  loss: {model_loss.numpy()}')

    if plot_graph:
        plt.plot(step_l, loss_l)
        plt.show()
    
    return model





class NeuralNet(Model):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = layers.Dense(128, activation=tf.nn.relu)
        self.fc2 = layers.Dense(256, activation=tf.nn.relu)
        self.out = layers.Dense(num_classes)
    
    def call(self, x, is_training=False):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)

        if is_training == False:
            x = tf.nn.softmax(x)
        return x

# Dataset ----------------------------------------------------------------------------
batch_size = 200
train_data_01 = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(batch_size).prefetch(1)
valid_data_01 = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).shuffle(60000).batch(batch_size).prefetch(1)
test_data_01 = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(60000).batch(batch_size).prefetch(1)

model = NeuralNet()
# result = train_model(train_data_01, model, 0.05, 5)




import sys
import copy
# Early_stopping ------------------------------------------
def early_stopping(train_data, valid_data, model, lr):
    el = []     # epoch_list
    vll = []    # error_list

    p = 4
    i = 0   # While Loop n_iter (epoch)
    j = 0
    v = sys.float_info.max      # 시스템 최대값
    i_s = i
    # model_s = copy.deepcopy(model)
    weigt_s = model.weights

    while j < p:
        train_model(train_data, model, lr, 1, print_loss=False, plot_graph=False)
        acc = 0
        for step, (batch_x, batch_y) in enumerate(valid_data, 1):
            pred = model(batch_x, is_training=False)
            acc += accuracy(pred, batch_y)

        acc = 100. * acc/step
        error = 100. - acc

        i = i+1
        temp_v = error.numpy()

        el.append(i)
        vll.append(error)

        if temp_v < v:
            j = 0
            model_s = copy.deepcopy(model)
            # weigt_s = copy.deepcopy(model.weights)
            i_s = i
            v = temp_v
        else:
            j = j+1
        print(f'epoch reached: {i}, val_error={error.numpy()}, smallest_error: {v}')

    plt.plot(el, vll)
    plt.show()
    print(f'best_epoch: {i_s}')

    return model_s, i_s
    # return weigt_s, i_s



# train_model(train_data=train_data_01, model=model_01, lr=learning_rate_01, epochs=n_epoch_01, print_loss=True, plot_graph=True)
# cross_entropy_loss(model_01.predict(x_train), y_train)


model_01 = NeuralNet()
learning_rate_01 = 0.01

nn, best_epoch = early_stopping(train_data_01, valid_data_01, model_01, learning_rate_01)
acc = accuracy_batch(nn, test_data_01).numpy()
print(f'Test_Accuracy_of_No_Regularization: {acc}')
# result
# cross_entropy_loss(result[0](x_valid), y_valid)
# 100-accuracy(result[0](x_valid), y_valid)*100






# 2) Regularization ================================================================================================

# (Weight_Decay) --------------------------------
class WD_NeuralNet(Model):
    def __init__(self):
        super(WD_NeuralNet, self).__init__()
        self.fc1 = layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.01)
                                ,activation=tf.nn.relu)
        self.fc2 = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.01)
                                , activation=tf.nn.relu)
        self.out = layers.Dense(num_classes)
    
    def call(self, x, is_training=False):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)

        if is_training == False:
            x = tf.nn.softmax(x)
        return x

wd_nn = WD_NeuralNet()
wd_nn, wd_best_epoch = early_stopping(train_data_01, valid_data_01, wd_nn, 0.01)

wd_acc = accuracy_batch(wd_nn, test_data_01).numpy()
print(f'Test_Accuracy_of_weight_decay: {wd_acc}')



# (Batch_Normalization) --------------------------------
class BN_NeuralNet(Model):
    def __init__(self):
        super(BN_NeuralNet, self).__init__()
        self.fc1 = layers.Dense(128)
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation(tf.nn.relu)

        self.fc2 = layers.Dense(256)
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.Activation(tf.nn.relu)

        self.out = layers.Dense(num_classes)
    
    def call(self, x, is_training=False):
        x = self.fc1(x)
        x = self.bn1(x, training=is_training)
        x = self.act1(x)

        x = self.fc2(x)
        x = self.bn2(x, training=is_training)
        x = self.act2(x)

        x = self.out(x)

        if is_training == False:
            x = tf.nn.softmax(x)
        return x

bn_nn = BN_NeuralNet()
bn, bn_best_epoch = early_stopping(train_data_01, valid_data_01, bn_nn, 0.01)

bn_acc = accuracy_batch(bn_nn, test_data_01).numpy()
print(f'Test_Accuracy_of_batch_normalization: {bn_acc}')




# (Drop_out) --------------------------------
class DO_NeuralNet(Model):
    def __init__(self):
        super(DO_NeuralNet, self).__init__()
        self.fc1 = layers.Dense(128, activation=tf.nn.relu)
        self.do1 = layers.Dropout(rate=0.5)

        self.fc2 = layers.Dense(256, activation=tf.nn.relu)
        self.do2 = layers.Dropout(rate=0.5)
        self.out = layers.Dense(num_classes)
    
    def call(self, x, is_training=False):
        x = self.fc1(x)
        x = self.do1(x, training=is_training)

        x = self.fc2(x)
        x = self.do2(x, training=is_training)

        x = self.out(x)

        if is_training == False:
            x = tf.nn.softmax(x)
        return x

do_nn = DO_NeuralNet()
do, do_best_epoch = early_stopping(train_data_01, valid_data_01, do_nn, 0.01)

do_acc = accuracy_batch(do_nn, test_data_01).numpy()
print(f'Test_Accuracy_of_drop_out: {do_acc}')




# (BN + Drop_out) --------------------------------
class BNDO_NeuralNet(Model):
    def __init__(self):
        super(BNDO_NeuralNet, self).__init__()
        self.fc1 = layers.Dense(128)
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation(tf.nn.relu)
        self.do1 = layers.Dropout(rate=0.5)

        self.fc2 = layers.Dense(256)
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.Activation(tf.nn.relu)
        self.do2 = layers.Dropout(rate=0.5)

        self.out = layers.Dense(num_classes)
    
    def call(self, x, is_training=False):
        x = self.fc1(x)
        x = self.bn1(x, training=is_training)
        x = self.act1(x)
        x = self.do1(x, training=is_training)

        x = self.fc2(x)
        x = self.bn2(x, training=is_training)
        x = self.act2(x)
        x = self.do2(x, training=is_training)

        x = self.out(x)

        if is_training == False:
            x = tf.nn.softmax(x)
        return x

bndo_nn = BNDO_NeuralNet()
bndo, bndo_best_epoch = early_stopping(train_data_01, valid_data_01, bndo_nn, 0.01)

bndo_acc = accuracy_batch(bndo_nn, test_data_01).numpy()
print(f'Test_Accuracy_of_BatchNormalization_Dropout: {bndo_acc}')



# (Learning Rate Schedulers) --------------------------------
# Step_Decay
# Exponential_Decay
# Two_Stage_Exponential_Deacay
# Linear_Warm_Up




# Model Comparison ----------------------------------------------------
print(f'Test_Accuracy_of_No_Regularization: {acc} ({best_epoch})')
print(f'Test_Accuracy_of_weight_decay: {wd_acc} ({wd_best_epoch})')
print(f'Test_Accuracy_of_batch_normalization: {bn_acc}  ({bn_best_epoch})')
print(f'Test_Accuracy_of_drop_out: {do_acc}  ({do_best_epoch})')
print(f'Test_Accuracy_of_BatchNormalization_Dropout: {bndo_acc}  ({bndo_best_epoch})')
#----------------------------------------------------------------------



# 3) Ensemble ======================================================================================================

def ensemble_acc(models, test_data):
    acc = 0
    for step, (batch_x, batch_y) in enumerate(test_data, 1):
        pred = 0
        for m in models:
            pred += m(batch_x, is_training=False).numpy()
        pred /= len(models)
        acc += accuracy(pred, batch_y)
    acc /= step
    return acc

ensemble_acc(models=[nn, bn_nn, do_nn, bndo_nn], test_data=test_data_01)









# Activation Function ================================================================

# ReLU Activations
class ReLU_NeuralNet(Model):
    def __init__(self):
        super(ReLU_NeuralNet, self).__init__()
        self.fc1 = layers.Dense(128, activation=tf.nn.relu)
        self.fc2 = layers.Dense(256, activation=tf.nn.relu)
        self.fc3 = layers.Dense(256, activation=tf.nn.relu)
        self.fc4 = layers.Dense(256, activation=tf.nn.relu)
        self.out = layers.Dense(num_classes)
    
    def call(self, x, is_training=False):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.out(x)

        if is_training == False:
            x = tf.nn.softmax(x)
        return x

# Sigmoid Activations
class Sigmoid_NeuralNet(Model):
    def __init__(self):
        super(Sigmoid_NeuralNet, self).__init__()
        self.fc1 = layers.Dense(128, activation=tf.keras.activations.sigmoid)
        self.fc2 = layers.Dense(256, activation=tf.keras.activations.sigmoid)
        self.fc3 = layers.Dense(256, activation=tf.keras.activations.sigmoid)
        self.fc4 = layers.Dense(256, activation=tf.keras.activations.sigmoid)
        self.out = layers.Dense(num_classes)
    
    def call(self, x, is_training=False):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.out(x)

        if is_training == False:
            x = tf.nn.softmax(x)
        return x



relu_nn = ReLU_NeuralNet()
sigmoid_nn = Sigmoid_NeuralNet()

# optimizer = tf.optimizers.SGD(0.001)
print('Train ReLU Model')
relu_nn_result = train_model(train_data_01, relu_nn, 0.001, 40)
relu_acc = accuracy_batch(relu_nn, test_data_01).numpy()
print(f'Test_Accuracy_of_ReLU_Model: {relu_acc}')

print('Train Sigmoid Model')
sigmoid_nn_result = train_model(train_data_01, sigmoid_nn, 0.001, 40)
sigmoid_acc = accuracy_batch(sigmoid_nn, test_data_01).numpy()
print(f'Test_Accuracy_of_Sigmoid_Model: {sigmoid_acc}')














# Initialization ============================================================

# Standard_Normalization
class SN_NeuralNet(Model):
    def __init__(self):
        super(SN_NeuralNet, self).__init__()
        self.fc1 = layers.Dense(128, 
                    kernel_initializer = tf.keras.initializers.RandomNormal(stddev=1),
                    activation=tf.nn.relu)
        self.fc2 = layers.Dense(256, 
                    kernel_initializer = tf.keras.initializers.RandomNormal(stddev=1),
                    activation=tf.nn.relu)
        self.out = layers.Dense(num_classes,
                    kernel_initializer = tf.keras.initializers.RandomNormal(stddev=1)
                    )
    
    def call(self, x, is_training=False):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)

        if is_training == False:
            x = tf.nn.softmax(x)
        return x

# Xavier
class Xav_NeuralNet(Model):
    def __init__(self):
        super(Xav_NeuralNet, self).__init__()
        self.fc1 = layers.Dense(128, 
                    kernel_initializer = tf.keras.initializers.GlorotNormal(),
                    activation=tf.nn.relu)
        self.fc2 = layers.Dense(256, 
                    kernel_initializer = tf.keras.initializers.GlorotNormal(),
                    activation=tf.nn.relu)
        self.out = layers.Dense(num_classes,
                    kernel_initializer = tf.keras.initializers.GlorotNormal()
                    )
    
    def call(self, x, is_training=False):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)

        if is_training == False:
            x = tf.nn.softmax(x)
        return x

# He
class He_NeuralNet(Model):
    def __init__(self):
        super(He_NeuralNet, self).__init__()
        self.fc1 = layers.Dense(128, 
                    kernel_initializer = tf.keras.initializers.he_normal(),
                    activation=tf.nn.relu)
        self.fc2 = layers.Dense(256, 
                    kernel_initializer = tf.keras.initializers.he_normal(),
                    activation=tf.nn.relu)
        self.out = layers.Dense(num_classes,
                    kernel_initializer = tf.keras.initializers.he_normal()
                    )
    
    def call(self, x, is_training=False):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)

        if is_training == False:
            x = tf.nn.softmax(x)
        return x


sn_nn = SN_NeuralNet()
xav_nn = Xav_NeuralNet()
he_nn = He_NeuralNet()

sn_nn_result = train_model(train_data_01, sn_nn, 0.001, 40)
xav_nn_result = train_model(train_data_01, xav_nn, 0.001, 40)
he_nn_result = train_model(train_data_01, he_nn, 0.001, 40)

sn_acc = accuracy_batch(sn_nn, test_data_01).numpy()
xav_acc = accuracy_batch(xav_nn, test_data_01).numpy()
he_acc = accuracy_batch(he_nn, test_data_01).numpy()
print(f'standard_normalization: {relu_acc}  \n xaxier: {xav_acc} \n he: {he_acc}')




# Optimizer =========================================================

# Training Model Overall Run
def train_model_optimizer(train_data, model, epochs,
                optimizer = tf.optimizers.SGD(0.005),
                print_loss=True, plot_graph=True):
    step_l = []
    loss_l = []

    # optimizer = tf.optimizers.SGD(lr)
    n_batch = len(list(train_data))

    # Training Run
    @tf.function
    def train_running(X, y, model, loss, optimizer):
        with tf. GradientTape() as Tape:
            y_pred = model(X, is_training=True)
            model_loss = loss(y_pred, y)

        train_weights = model.trainable_variables
        gradients = Tape.gradient(model_loss, train_weights)
        optimizer.apply_gradients(zip(gradients, train_weights))
        return model_loss

    for epoch in range(1, epochs+1):

        running_loss = 0.0
        for step, (batch_x, batch_y) in enumerate(train_data, 1):

            model_loss = train_running(batch_x, batch_y, model, cross_entropy_loss, optimizer)
            # with tf. GradientTape() as Tape:
            #     y_pred = model(batch_x, is_training=True)
            #     model_loss = cross_entropy_loss(y_pred, batch_y)

            # train_weights = model.trainable_variables
            # gradients = Tape.gradient(model_loss, train_weights)
            # optimizer.apply_gradients(zip(gradients, train_weights))

            running_loss += model_loss.numpy()

            if plot_graph:
                if step % 10 == 0:
                    step_l.append(epoch * n_batch + step)
                    loss_l.append(running_loss/10)
                    running_loss = 0.0
        
        if print_loss:
            print(f'epoch: {epoch},  loss: {model_loss.numpy()}')

    if plot_graph:
        plt.plot(step_l, loss_l)
        plt.show()
    
    return model


momentum_nn = He_NeuralNet()
adagrad_nn = He_NeuralNet()
rmsprop_nn = He_NeuralNet()
adam_nn = He_NeuralNet()

momentum_optimizer = tf.optimizers.SGD(0.01, momentum=0.9)
adagrad_optimizer = tf.optimizers.Adagrad(0.01)
rmsprop_optimizer = tf.optimizers.RMSprop(0.01)
adam_optimizer = tf.optimizers.Adam(0.01)

momentum_nn_result = train_model_optimizer(train_data_01, momentum_nn, 40, momentum_optimizer)
adagrad_nn_result = train_model_optimizer(train_data_01, adagrad_nn, 40, adagrad_optimizer)
rmsprop_nn_result = train_model_optimizer(train_data_01, rmsprop_nn, 40, rmsprop_optimizer)
adam_nn_result = train_model_optimizer(train_data_01, adam_nn, 40, adam_optimizer)

momentum_acc = accuracy_batch(momentum_nn, test_data_01).numpy()
adagrad_acc = accuracy_batch(adagrad_nn, test_data_01).numpy()
rmsprop_acc = accuracy_batch(rmsprop_nn, test_data_01).numpy()
adam_acc = accuracy_batch(adam_nn, test_data_01).numpy()

print(f'momentum: {momentum_acc}  \n adagrad: {adagrad_acc} \n RMSprop: {rmsprop_acc} \n Adam: {adam_acc}')

dir(momentum_nn)

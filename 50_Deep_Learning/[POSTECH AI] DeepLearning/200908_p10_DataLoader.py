import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import PIL
import PIL.Image
import pathlib

import os

import tensorflow as tf
tf.__version__

# %pip install --upgrade tensorflow


# Essential Function ------------------------------------------------------------------
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
def train_model(train_data, model, optimizer, epochs, 
                print_loss=True, plot_graph=True):
    step_l = []
    loss_l = []

    # optimizer = tf.optimizers.SGD(lr, momentum=0.9)

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

# Early_Stoping
def early_stopping(train_data, valid_data, model, optimizer):
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
        train_model(train_data, model, optimizer, 1, print_loss=False, plot_graph=False)
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




# Image DataLoader  ================================================================================================
# https://www.tensorflow.org/tutorials/load_data/images?hl=ko       # 이미지로드 | 텐서 플로우 코어 | TensorFlow

import pathlib

absolute_path = r'd:\\Python\\★★Python_POSTECH_AI\\Postech_AI 5) Machine_Learning & Deep Learning\\dataset'

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url, 
                                   fname=absolute_path+'/flower_photos',   # 저장명
                                   untar=True)
data_dir = pathlib.Path(data_dir)
data_dir

image_count = len(list(data_dir.glob('*/*.jpg')))
image_count

roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))

batch_size = 10
img_height = 90
img_width = 90

train_data = tf.keras.preprocessing.image_dataset_from_directory(data_dir, validation_split=0.2, 
            subset='training', seed=1, image_size=(img_height, img_width), batch_size=batch_size)

val_data = tf.keras.preprocessing.image_dataset_from_directory(data_dir, validation_split=0.2, 
            subset='validation', seed=1, image_size=(img_height, img_width), batch_size=batch_size)

next(iter(train_data))[0].shape

class_names = train_data.class_names
class_names

plt.figure(figsize=(10,10))
for images, labels in train_data.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.title(class_names[labels[i]])
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.axis('off')

class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.normalization = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(128)
        self.out = tf.keras.layers.Dense(5)

    def call(self, x, is_training=False):
        x = self.normalization(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.out(x)

        if not is_training:
            x = tf.nn.softmax(x)
        return x

optimizer = tf.optimizers.Adam(0.0001)
model = CNN()
train_model(train_data, model, optimizer, 3)

acc = accuracy_batch(model, val_data).numpy()
print(acc)







# DataLoader: CSV data ================================================================================================
# https://www.tensorflow.org/tutorials/load_data/csv    # CSV 데이터로드

absolute_path = r'd:\\Python\\★★Python_POSTECH_AI\\Postech_AI 5) Machine_Learning & Deep Learning\\dataset'

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file(absolute_path + "/train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file(absolute_path + "/eval.csv", TEST_DATA_URL)

np.set_printoptions(precision=3, suppress=True)
print(train_file_path)

def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(file_path, batch_size=5, 
                label_name='survived', na_value='?', num_epochs=1, 
                ignore_errors=True, **kwargs)
    return dataset
# ?tf.data.experimental.make_csv_dataset
# tf.data.experimental.make_csv_dataset(
#     file_pattern, batch_size, column_names=None, column_defaults=None, label_name=None,
#     select_columns=None, field_delim=',', use_quote_delim=True, na_value='', header=True,
#     num_epochs=None, shuffle=True, shuffle_buffer_size=10000, shuffle_seed=None,
#     prefetch_buffer_size=None, num_parallel_reads=None, sloppy=False,
#     num_rows_for_inference=100, compression_type=None, ignore_errors=False,)

raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)


print(type(raw_train_data))

for batch_x, batch_y in raw_train_data:
    for key, value in batch_x.items():
        print(f'{key} : {value.numpy()}')
    break


# Extract only numerical Data
# column_to_use = ['survived', 'age']
# ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 
#     'fare', 'class', 'deck', 'embark_town', 'alone']

column_to_use = [0, 2, 3, 4, 5]

defaults = [tf.int32, tf.float32, tf.float32, tf.float32, tf.float32]
temp_dataset = get_dataset(train_file_path, select_columns=column_to_use, column_defaults=defaults)

example_batch, labels_batch = next(iter(temp_dataset))

print(example_batch)
print()
print(labels_batch)

print(example_batch.values())
print(list(example_batch.values()))
# [list(i.numpy()) for i in example_batch.values()]
print(tf.stack(list(example_batch.values()), axis=1))
# tf.stack(list(example_batch.values()), axis=0)


def pack(features, label):
    return tf.stack(list(features.values()), axis=1), label

packed_dataset = temp_dataset.map(pack)

for batch_x, batch_y in packed_dataset:
    print(batch_x)
    print()
    print(batch_y)
    break


# process numeric and categorical at the same time ---------------------------
# ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 
#     'fare', 'class', 'deck', 'embark_town', 'alone']

# Numerical ***
def MyPackNumericalFeatures(features, labels):
    names = ['age', 'n_siblings_spouses', 'parch', 'fare']
    numerical_features = [features.pop(name) for name in names]
    numerical_features = [tf.cast(feat, tf.float32) for feat in numerical_features]
    numerical_features = tf.stack(numerical_features, axis=1)
    features['numeric'] = numerical_features
    return features, labels

packed_train_data = raw_train_data.map(MyPackNumericalFeatures)
packed_test_data = raw_test_data.map(MyPackNumericalFeatures)

example_batch, labels_batch = next(iter(packed_train_data))
print(example_batch)
print()
print(labels_batch)
print()
print(example_batch['numeric'])


numeric_columns = tf.feature_column.numeric_column('numeric', shape=[4])
numeric_columns = [numeric_columns]

numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)
numeric_layer(example_batch).numpy()


# Categorical ***
CATEGORIES = {
    'sex' : ['male', 'female'],
    'class' : ['First', 'Second', 'Third'],
    'deck' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town': ['Cherbourg', 'Southampton', 'Queenstown'],
    'alone': ['y','n']
}

categorical_columns = []
for feature, vocab in CATEGORIES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(key=feature, vocabulary_list=vocab)
    categorical_columns.append(tf.feature_column.indicator_column(cat_col))

categorical_layer = tf.keras.layers.DenseFeatures(categorical_columns)
print(categorical_layer(example_batch).numpy())



preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns + numeric_columns)
print(preprocessing_layer(example_batch).numpy()[0])


class NueralNet(tf.keras.Model):
    def __init__(self):
        super(NueralNet, self).__init__()
        self.dense1 = tf.keras.layers.Dense(20, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(20, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(2)
    
    def call(self, x, is_training=False):
        x = preprocessing_layer(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.out(x)
        if not is_training:
            x = tf.nn.softmax(x)
        return x

train_data = packed_train_data.shuffle(500)
test_data = packed_test_data

model = NueralNet()

optimizer = tf.optimizers.Adam(1e-3)
model = train_model(train_data, model, optimizer, 10)

acc = accuracy_batch(model, test_data).numpy()
print(acc)






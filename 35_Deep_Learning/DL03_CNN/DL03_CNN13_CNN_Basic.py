import os 
from glob import glob
import time

import tensorflow as tf
import torch

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image   # PIL는 이미지를 load 할 때 이용
import cv2

train_folder_path = r'C:\Users\Admin\Desktop\DataBase\fruits_simple_test\train_set'
test_folder_path = r'C:\Users\Admin\Desktop\DataBase\fruits_simple_test\test_set'
# os.path.basename(train_folder_path)       # last path name

train_img_name = os.listdir(train_folder_path)
train_labels = [(0 if 'apple' in i else 1)for i in train_img_name]
# train_images = [Image.open(f"{train_folder_path}\\{i}") for i in train_img_name]

# train_df = pd.DataFrame([train_img_name,train_labels,train_images], index=['names', 'labels', 'images']).T
train_df = pd.DataFrame([train_img_name,train_labels], index=['names', 'labels']).T
train_df


# [ Generator ]
#  Generator는 나만의 Iterable, Iterator 기능을 만들되, 생성 문법을 기존보다 단순화한 개념 또는 클래스라고 할 수 있다.
# 리스트나 Set과 같은 컬렉션에 대한 iterator는 해당 컬렉션이 이미 모든 값을 가지고 있는 경우이나, 
# Generator는 모든 데이타를 갖지 않은 상태에서 yield에 의해 하나씩만 데이타를 만들어 가져온다는 차이점이 있다. 
# 이러한 Generator는 데이타가 무제한이어서 모든 데이타를 리턴할 수 없는 경우나, 데이타가 대량이어서 일부씩 처리하는 것이 필요한 경우, 
# 혹은 모든 데이타를 미리 계산하면 속도가 느려서 그때 그때 On Demand로 처리하는 것이 좋은 경우 등에 종종 사용된다.

# Image_gen = (Image.open(f"{train_folder_path}\\{p}") for p in train_df['names'] )
# Image_gen = (Image.open(f"{train_folder_path}\\{p}") for p in train_df['names'].sample(len(train_df), random_state=1) )
# Image_gen = (plt.imread(p) for p in train_df['names'].apply(lambda x: f"{train_folder_path}\\{x}") )


# (image generator) --------------------------------------------------------
# a = ( np.stack(train_df['names'].iloc[i:i+group_size].apply(lambda x:plt.imread(f"{train_folder_path}\\{x}")) ) for i in range(0, len(train_df), group_size) )
# b= next(a)
# b.shape

# b                       # tensorflow
# b.transpose(0,1,2,3)    # tensordlow
# b.transpose(0,3,1,2)    # pytorch

# Tensorflow : B, H, W, C (Batch, Height, Width, Channel)
# Pytorch : B, C, H, W (Batch, Channel, Height, Width)
# ----------------------------------------------------------

# --- (numpy random) --------------------------------------
# rng = np.random.default_rng(12345)
# random_seed_gen = np.random.default_rng(1)
# random_seed_gen.permutation(l1)
# np.random.shuffle(l1)       # inplace = False
# np.random.permutation(l1)   # inplace = True
# ----------------------------------------------------------

# (function) make batch --------------------------------------
# def group_list(l, group_size):
#     """
#     :param l:           list
#     :param group_size:  size of each group
#     :return:            Yields successive group-sized lists from l.
#     """
#     for i in range(0, len(l), group_size):
#         yield l[i:i+group_size]
# ----------------------------------------------------------


# (function) make_batch_from_path ------------------------------------------------------------------
def make_batch_from_path(paths, labels=None, batch_size=None, shuffle=False, random_state=None, shape_format="BHWC"):
    batch_size = len(paths) if batch_size is None else batch_size
    
    if shuffle is True:
        index = np.arange(len(paths))
        rng = np.random.default_rng(random_state)
        index_permute = rng.permutation(index)

        loading_paths = np.array(paths)[index_permute]
        if labels is not None:
            loading_labels = np.array(labels)[index_permute]
    else:
        loading_paths = np.array(paths)
        if labels is not None:
            loading_labels = np.array(labels)

    shape_format_dict = {'B':0, 'H':1, 'W':2, 'C':3}
    shape_format_list = [shape_format_dict[c] for c in shape_format]

    for i in range(0, len(loading_paths), batch_size):
        if labels is None:
            yield np.stack([plt.imread(p) for p in loading_paths[i:i+batch_size]]).transpose(*shape_format_list)
        else:
            yield (np.stack([plt.imread(p) for p in loading_paths[i:i+batch_size]]).transpose(*shape_format_list), loading_labels[i:i+batch_size])
# --------------------------------------------------------------------------------------------------------------------



train_img_paths = train_df['names'].apply(lambda x: f"{train_folder_path}\\{x}").values
train_img_paths
train_labels = train_df['labels'].values
train_labels

test_img_paths = [f"{test_folder_path}\\{p}" for p in os.listdir(test_folder_path)]
test_labels = [0,0,0,0, 1,1,1,1]



# train_dataloader = make_batch_from_path(train_img_paths, batch_size=7)
# image_dataloader = make_batch_from_path(train_img_paths, batch_size=7, shape_format='BCHW')
# image_batch = next(train_dataloader)
# image_batch.shape

train_dataloader = make_batch_from_path(train_img_paths, train_labels)
# train_dataloader = make_batch_from_path(train_img_paths, train_labels, batch_size=7)
# train_dataloader = make_batch_from_path(train_img_paths, train_labels, batch_size=7, shuffle=True)
image_batch_X, image_batch_y = next(train_dataloader)
image_batch_X.shape
image_batch_y

test_dataloader = make_batch_from_path(test_img_paths, test_labels)
test_bath_X, test_batch_y = next(test_dataloader)


# (Tensorflow CNN) #####################################################################################################
# tensorflow sigmoid function
# https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=siniphia&logNo=221614161423
# https://www.analyticsvidhya.com/blog/2021/05/guide-for-loss-function-in-tensorflow/

# (Tensorflow Basic) ***
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),  # node: (40 x) 100 x 100 x 32   # weight: 3 × (3 × 3) × 32 + 32
                        # node수, filter_size, padding('same': 영상크기유지, 'valid': 영상사이즈축소 ), activation
                       tf.keras.layers.MaxPool2D(),                                                         # node: (40 x) 50 x 50 x 32
                       tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),               # node: (40 x) 50 x 50 x 64     # weight: 3 × 3 × 32 × 64 + 64
                       tf.keras.layers.MaxPool2D(),                                                         # node: (40 x) 25 x 25 x 64
                       tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),              # node: (40 x) 25 x 25 x 128    # weight: 3 × 3 × 64 × 128 + 128
                       tf.keras.layers.Flatten(),                                                           # node: (40 x) 80000

                       tf.keras.layers.Dense(128, activation='relu'),                                       # node: (40 x) 128              # weight: 80000 × 128 + 128
                    #    tf.keras.layers.BatchNormalization(),
                    #    tf.keras.layers.Dropout(0.3)
                       tf.keras.layers.Dense(128, activation='relu'),                                       # node: (40 x) 128              # weight: 128 × 128 + 128
                    #    tf.keras.layers.BatchNormalization(),
                    #    tf.keras.layers.Dropout(0.3)
                       tf.keras.layers.Dense(2, activation='softmax')])                                     # node: (40 x) 2                # weight: 128 × 2 + 2

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# l1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')

epochs = 10
# result = model.fit(train.data, train.labels, epochs=EPOCHS)
# result = model.fit(image_bath_X/255.0, image_batch_y.astype(float), epochs=epochs)
result = model.fit(image_bath_X/255.0, image_batch_y.astype(float), epochs=epochs, batch_size=6)

model.predict(test_bath_X).argmax(1)
test_batch_y



# (Tensorflow Expert) ***
# np.array(dir(tf.keras.layers)[-100:])
# np.array(dir(tf.keras.losses)[-100:])
# np.array(dir(tf.nn)[-100:])

class TF_CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.layer01_conv = tf.keras.layers.Conv2D(32, (3, 3), padding='same')
        self.layer01_act = tf.keras.layers.Activation('relu')
        self.layer01_maxpool = tf.keras.layers.MaxPool2D()
        
        self.layer02_conv = tf.keras.layers.Conv2D(64, (3, 3), padding='same')
        self.layer02_act = tf.keras.layers.Activation('relu')
        self.layer02_maxpool = tf.keras.layers.MaxPool2D()

        self.layer03_conv = tf.keras.layers.Conv2D(128, (3, 3), padding='same')
        self.layer03_act = tf.keras.layers.Activation('relu')
        self.layer03_maxpool = tf.keras.layers.MaxPool2D()

        self.layer04_flat = tf.keras.layers.Flatten()

        self.layer05_dense = tf.keras.layers.Dense(128)
        self.layer05_act = tf.keras.layers.Activation('relu')
        self.layer06_dense = tf.keras.layers.Dense(128)
        self.layer06_act = tf.keras.layers.Activation('relu')
        self.layer07_dense = tf.keras.layers.Dense(2)
        self.layer07_act = tf.keras.layers.Activation('softmax')

    def call(self, X, training=True):
        self.layer01_r01 = self.layer01_conv(X)
        self.layer01_r02 = self.layer01_act(self.layer01_r01)
        self.layer01_r03 = self.layer01_maxpool(self.layer01_r02)

        self.layer02_r01 = self.layer02_conv(self.layer01_r03)
        self.layer02_r02 = self.layer02_act(self.layer02_r01)
        self.layer02_r03 = self.layer02_maxpool(self.layer02_r02)

        self.layer03_r01 = self.layer03_conv(self.layer02_r03)
        self.layer03_r02 = self.layer03_act(self.layer03_r01)
        self.layer03_r03 = self.layer03_maxpool(self.layer03_r02)

        self.layer04_r01 = self.layer04_flat(self.layer03_r03)

        self.layer05_r01 = self.layer05_dense(self.layer04_r01)
        self.layer05_r02 = self.layer05_act(self.layer05_r01)

        self.layer06_r01 = self.layer06_dense(self.layer05_r02)
        self.layer06_r02 = self.layer06_act(self.layer06_r01)

        self.layer07_r01 = self.layer07_dense(self.layer06_r02)
        self.layer07_r02 = self.layer07_act(self.layer07_r01)
        return self.layer07_r02 


model = TF_CNN()
model(tf.constant(image_batch_X/255))

loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
epochs = 10

# def cross_entropy_loss(y_true, y_pred):
#     y_true = tf.cast(y_true, tf.int64)
#     loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
#     return tf.reduce_mean(loss)

for e in range(epochs):
    losses = []
    train_dataloader = make_batch_from_path(train_img_paths, train_labels)    
    for b, (batch_X, batch_y) in enumerate(train_dataloader):
        batch_X = tf.constant(batch_X/255, dtype=tf.float32)
        batch_y = tf.constant(batch_y, dtype=tf.float32)

        with tf.GradientTape() as tape:
            y_pred = model(batch_X, training=True)
            y_true = batch_y
            loss = loss_function(y_true=y_true, y_pred=y_pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimize = optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        losses.append(loss.numpy())
    print(f"({e+1} epoch) loss: {np.mean(losses)}")
    # print(f"({e+1} epoch) loss: {np.mean(losses)}", end='\r')
    time.sleep(0.3)

test_dataloader = make_batch_from_path(test_img_paths, test_labels)
test_bath_X, test_batch_y = next(test_dataloader)
tf.argmax(model(tf.constant(test_bath_X/255, dtype=tf.float32)), axis=1)
test_batch_y

### 【 Sigmoid, Softmax 】-------------------------------------------------------------
# (class) tf.losses.BinaryCrossentropy()
# (functional) tf.keras.losses.binary_crossentropy(y_true=y_true, y_pred=pred_sigmoid[:,1])
#   . y_true : (label) 0 or 1
#      [0, 1 ...]
#   . y_pred : (sigmoid)
#      [proba, poba ...]

# (class) tf.losses.CategoricalCrossentropy()
# (functional) tf.keras.losses.categorical_crossentropy(y_true=y_true2, y_pred=pred_sigmoid)
#   . y_true : (label) 0 or 1
#      [[0, 1, 0, ...],
#       [0, 0, 1, ...],
#       ...]
#   . y_pred : (sigmoid)
#      [[proba, proba...],
#       [proba, proba...]
#       ...]


# (class) tf.keras.losses.SparseCategoricalCrossentropy()
# (functional) tf.reduce_mean( tf.keras.losses.sparse_categorical_crossentropy(y_true=y_true, y_pred=pred_sigmoid) )
#   . y_true : (label) 0, 1, 2 ...
#      [0, 1 ...]
#   . y_pred : (sigmoid)
#      [[proba, proba...],
#       [proba, proba...]
#       ...]

# ------------------------------------------------------------------------------------

### 【 Logit 】-----------------------------------------------------------------------
# (functional) tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=pred_logit[:,1]) )
#   . y_true : (label) 0 or 1
#      [0, 1 ...]
#   . y_pred : (logit)
#      [proba, poba ...]

# (functional) tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_true2, logits=pred_logit) )
#   . y_true : (label) 0 or 1
#      [[0, 1, 0, ...],
#       [0, 0, 1, ...],
#       ...]
#   . y_pred : (logit)
#      [[proba, proba...],
#       [proba, proba...]
#       ...]

# (functional) tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true(int), logits=pred_logit) )
#   . y_true : (label) 0 or 1
#      [0, 1 ...]
#   . y_pred : (logit)
#      [proba, poba ...]

# ------------------------------------------------------------------------------------








# (Pytorch CNN) #####################################################################################################
# np.array(dir(torch.nn)[-200:])
# np.array(dir(torch.nn.funtional)[-200:])
# np.array(dir(torch.optim)[-100:])

class Torch_CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # torch.nn.Conv2d(3, 32, 3, 1, 1)   # input_channel, output_channel, kernel_size, stride, padding
        self.layer01_conv = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=1, padding=1)
        self.layer01_act = torch.nn.ReLU()
        self.layer01_maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.layer02_conv = torch.nn.Conv2d(32, 64, (3,3), 1, 1)
        self.layer02_act = torch.nn.ReLU()
        self.layer02_maxpool = torch.nn.MaxPool2d(2,2)

        self.layer03_conv = torch.nn.Conv2d(64, 128, (3,3), 1, 1)
        self.layer03_act = torch.nn.ReLU()
        self.layer03_maxpool = torch.nn.MaxPool2d(2,2)

        self.layer04_flat = torch.nn.Flatten()
        # bef_node.view(bef_node.shape[0],-1).shape

        self.layer05_dense = torch.nn.Linear(in_features=128*12*12, out_features=128)
        self.layer05_act = torch.nn.ReLU()
        self.layer06_dense = torch.nn.Linear(128, 128)
        self.layer06_act = torch.nn.ReLU()
        self.layer07_dense = torch.nn.Linear(128, 2)
        self.layer07_act = torch.nn.Softmax()

    def forward(self, X):
        self.layer01_r01 = self.layer01_conv(X)
        self.layer01_r02 = self.layer01_act(self.layer01_r01)
        self.layer01_r03 = self.layer01_maxpool(self.layer01_r02)

        self.layer02_r01 = self.layer02_conv(self.layer01_r03)
        self.layer02_r02 = self.layer02_act(self.layer02_r01)
        self.layer02_r03 = self.layer02_maxpool(self.layer02_r02)

        self.layer03_r01 = self.layer03_conv(self.layer02_r03)
        self.layer03_r02 = self.layer03_act(self.layer03_r01)
        self.layer03_r03 = self.layer03_maxpool(self.layer03_r02)

        self.layer04_r01 = self.layer04_flat(self.layer03_r03)
        
        self.layer05_r01 = self.layer05_dense(self.layer04_r01)
        self.layer05_r02 = self.layer05_act(self.layer05_r01)

        self.layer06_r01 = self.layer06_dense(self.layer05_r02)
        self.layer06_r02 = self.layer06_act(self.layer06_r01)

        self.layer07_r01 = self.layer07_dense(self.layer06_r02)
        self.layer07_r02 = self.layer07_act(self.layer07_r01)
        return self.layer07_r02 

# random_seed = 4332
# torch.manual_seed(random_seed)
# torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed) # if use multi-GPU

# torch.backends.cudnn.deterministic = True   # GPU 사용시 연산결과가 달라질 수 있음.
# torch.backends.cudnn.benchmark = False

# np.random.seed(random_seed)
# random.seed(random_seed)

# use_cuda = False
# # if use_cuda and torch.cuda.is_available():
# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
# else:
#     device = torch.device("cpu")
# print(device)    



model = Torch_CNN()
# model = Torch_CNN().to(device)
# print(net(images.to(device)).shape)
# print(str(net))
# batch_X = torch.tensor(image_batch_X.transpose(0,3,1,2)/255, dtype=torch.float32)
# model(batch_X)

# loss_function = torch.nn.CrossEntropyLoss()
loss_function = torch.nn.BCELoss()
optimizer = torch.optim.Adam(params=model.parameters())
epochs = 10


for e in range(epochs):
    losses = []
    train_dataloader = make_batch_from_path(train_img_paths, train_labels, shape_format='BCHW')
    for b, (batch_X, batch_y) in enumerate(train_dataloader):
        batch_X = torch.tensor(batch_X/255, dtype=torch.float32)
        batch_y = torch.tensor(batch_y.astype(np.int32), dtype=torch.float32)

        # get the inputs
        # batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()                   # weight_initialize
        y_pred = model(batch_X)                 # predict/forward
        y_true = batch_y
        loss = loss_function(y_pred[:,1], y_true)    # loss
        loss.backward()                         # back-propagation/backward
        optimizer.step()                        # update_weight

        with torch.no_grad():
            losses.append(loss.item())
    
    print(f"({e+1} epoch) loss: {np.mean(losses)}")
    # print(f"({e+1} epoch) loss: {np.mean(losses)}", end='\r')
    time.sleep(0.3)


loss_function(y_pred[:,1], y_true)
test_dataloader = make_batch_from_path(test_img_paths, test_labels, shape_format='BCHW')
test_bath_X, test_batch_y = next(test_dataloader)
torch.argmax(model(torch.tensor(test_bath_X/255, dtype=torch.float32)), axis=1)
test_batch_y



#####################################################################################################

loss_obj = torch.nn.CrossEntropyLoss()  
loss_obj(y_pred, torch.tensor(batch_y,dtype=torch.int64))




torch.nn.functional.cross_entropy(y_pred, torch.tensor(batch_y,dtype=torch.int64))



np.array(dir(torch.nn.functional)[-150:])
### 【 Sigmoid, Softmax 】-------------------------------------------------------------
# (class) torch.nn.CrossEntropyLoss(y_pred, y_true)
# (functional) toch.nn.functional.cross_entropy(y_pred, y_true)
#   . y_true : (label) 0, 1, 2 ...        * torch.int64, torch.long
#      [0, 1 ...]
#   . y_pred : (sigmoid)                  * torch.float32, torch.float64
#      [[proba, proba...],
#       [proba, proba...]
#       ...]


### 【 Binary 】-------------------------------------------------------------
# (class) torch.nn.BCELoss(y_pred[:, 1], y_true)
# (functional) torch.nn.functional.binary_cross_entropy(y_pred[:, 1], y_true))
#   . y_true : (label) 0 or 1           * torch.float32, torch.float64
#      [0, 1 ...]
#   . y_pred : (sigmoid)                * torch.float32, torch.float64
#      [proba, poba ...]


# (class) torch.nn.BCELoss(y_pred[:, 1], y_true)
# (functional) torch.nn.functional.binary_cross_entropy_with_logits(y_pred[:, 1], y_true))
#   . y_true : (label) 0 or 1           * torch.float32, torch.float64
#      [0, 1 ...]
#   . y_pred : (logit)                 * torch.float32, torch.float64
#      [proba, poba ...]
# ------------------------------------------------------------------------------------

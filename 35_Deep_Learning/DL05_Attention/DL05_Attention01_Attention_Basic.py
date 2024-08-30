import numpy as np
import pandas as pd
import matplotlib.pylab as plt

import tensorflow as tf
from scipy import io

import math
import time
from six.moves import cPickle

path = r'D:\Python\★★Python_POSTECH_AI\Dataset'
load_data = cPickle.load(open(path + '/attention_mnist_small.pkl', 'rb'))


X_train = load_data['X_train']
X_test = load_data['X_test']
y_train = load_data['y_train']
y_test = load_data['y_test']



# train = io.loadmat('attention_train.mat')
# test = io.loadmat('attention_test.mat')

# X_train = train['X_train'][..., np.newaxis]
# y_train = train['Y_train']
# X_test = test['X_test'][..., np.newaxis]
# y_test = test['Y_test']

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# data_dict = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
# data_dict_small = {'X_train': X_train[:1000], 'y_train': y_train[:1000], 'X_test': X_test[:200], 'y_test': y_test[:200]}

# with open('attention_mnist_small.plk', 'wb') as f:
#     cPickle.dump(data_dict_small, f)




# CNN Model ----------------------------------------------------------------------
class CNN(tf.keras.Model):
    def __init__(self, model_type=1):
        super(CNN, self).__init__()
        self.model_type = model_type

        self.conv1 = tf.keras.layers.Conv2D(8, kernel_size=3, strides=2, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(16, kernel_size=3, strides=2, padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')

        self.conv4 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu')
        self.conv5 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu')

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.model_type == 2:
            x = self.conv4(x)
            x = self.conv5(x)

        x = self.flatten(x)
        x = self.dense(x)
        return x

model = CNN(model_type=1)
# model.build(input_shape=(None, 112,112,1))
# model.summary()
# model(X_train[:5]).shape
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
result = model.fit(X_train, y_train, validation_split=0.2, batch_size=128, epochs=5)
model.evaluate(X_test, y_test)



# ReshapeLayer
class Reshape(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
    def call(self, x):
        shape = np.array(x).shape
        return tf.reshape(x, (shape[0], shape[1]*shape[2], shape[3]))  # B, H*W, C

#### Attention_01 ##############################################################################################
class Attention_01(tf.keras.Model):
    def __init__(self, model_type=1):
        super().__init__()

        self.model_type = model_type

        self.conv1 = tf.keras.layers.Conv2D(8, kernel_size=3, strides=2, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(16, kernel_size=3, strides=2, padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')
        self.reshape = Reshape()
        
        # Key(Wk) : channel × summary_size
        self.w_k = self.add_weight(shape=(32,8), initializer='random_normal', trainable=True)
        # Query(Wq) : summary_size × Attention_type(종류)
        self.w_q = self.add_weight(shape=(8,1), initializer='random_normal', trainable=True)

        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x, training=False):
        # x                 # B, H, W, C
        x = self.conv1(x)   # B, H/2, W/2, 8C
        x = self.conv2(x)   # B, H/4, W/4, 16C
        x = self.conv3(x)   # B, H/8, W/8, 32C

        x = self.reshape(x)    # (value)    # B, H/8*W/8, 32C
        self.value = x
        
        # self.flat = x
        key = tf.matmul(x, self.w_k)      # (key)  # B, H/8*W/8, 8C
        score = tf.nn.softmax(tf.matmul(key, self.w_q), axis=1)     # (score) # softmax( B, H/8*W/8, 1C )
        self.key = key
        self.score = score
        
        if self.model_type == 1:
            x = tf.reduce_sum(x * score, axis=1)
        elif self.model_type == 2:
            x = tf.keras.layers.Flatten()(x * score)
        x = self.dense(x)
        return x

model = Attention_01()
model(sample_img).shape
model.value.shape
model.key.shape
model.score.shape

#### Attention_02 ##############################################################################################
# AttentionLayer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.key_layer = tf.keras.layers.Dense(8, use_bias=False)
        self.query_layer = tf.keras.layers.Dense(1, use_bias=False)
        
    def call(self, x, return_element=False):
        self.value = x
        self.key = self.key_layer(self.value)
        self.score = tf.keras.activations.softmax( self.query_layer(self.key) )
        
        if return_element is True:
            return self.value, self.key, self.score
        else:
            return self.score

class Attention_02(tf.keras.Model):
    def __init__(self, model_type=1):
        super().__init__()

        self.model_type = model_type

        self.conv1 = tf.keras.layers.Conv2D(8, kernel_size=3, strides=2, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(16, kernel_size=3, strides=2, padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')
        self.reshape = Reshape()
        self.attention = AttentionLayer()
        
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x, training=False):
        # x                 # B, H, W, C
        x = self.conv1(x)   # B, H/2, W/2, 8C
        x = self.conv2(x)   # B, H/4, W/4, 16C
        x = self.conv3(x)   # B, H/8, W/8, 32C

        x = self.reshape(x)    # (value)    # B, H/8*W/8, 32C
        
        value, key, score = self.attention(x, return_element=True)
        self.value = value
        self.key = key
        # score = self.attention(x)
        self.score = score
        
        if self.model_type == 1:
            x = tf.reduce_sum(x * score, axis=1)
        elif self.model_type == 2:
            x = tf.keras.layers.Flatten()(x * score)
        x = self.dense(x)
        return x

model = Attention_02()
model(sample_img).shape
model.value.shape
model.key.shape
model.score.shape

# ----------------------------------------------------------------------
model2 = Attention_01()
# model2(X_train[:5]).shape

# sc = model2.score
# flat = model2.flat
# tf.reduce_mean(sc, axis=1).shape

# sc.shape
# flat.shape

# (sc * flat).shape
# tf.reduce_mean(sc * flat, axis=1).shape



# model2.build(input_shape=(None, 112,112,1))
# model2.summary()
# model2(X_train[:5]).shape
model2.compile(loss='categorical_crossentropy', metrics=['accuracy'])
result2 = model2.fit(X_train, y_train, validation_split=0.2, batch_size=128, epochs=5)
model2.evaluate(X_test, y_test)





# Model Comparison ----------------------------------------------
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',  mode='min', patience=4, restore_best_weights=True, verbose=2)

# cnn_1
cnn1_start = time.time()
model_cnn1 = CNN(model_type=1)  # Conv32
model_cnn1.build(input_shape=(None, 112,112,1))
# model_cnn1.summary()
model_cnn1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
result_cnn1 = model_cnn1.fit(X_train, y_train, validation_split=0.2, batch_size=128, epochs=50, callbacks=[es], verbose=2)
cnn1_end = time.time()


# cnn_2
cnn2_start = time.time()
model_cnn2 = CNN(model_type=2)  # Conv128
model_cnn2.build(input_shape=(None, 112,112,1))
# model_cnn2.summary()
model_cnn2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
result_cnn2 = model_cnn2.fit(X_train, y_train, validation_split=0.2, batch_size=128, epochs=50, callbacks=[es], verbose=2)
cnn2_end = time.time()


# attention1
att1_start = time.time()
model_att1 = Attention(model_type=1)    # tf.reduce_sum
model_att1.build(input_shape=(None, 112,112,1))
# model_att1.summary()
model_att1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
result_att1 = model_att1.fit(X_train, y_train, validation_split=0.2, batch_size=128, epochs=50, callbacks=[es], verbose=2)
att1_end = time.time()

# attention2
att2_start = time.time()
model_att2 = Attention(model_type=2)    # flatten
model_att2.build(input_shape=(None, 112,112,1))
# model_att2.summary()
model_att2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
result_att2 = model_att2.fit(X_train, y_train, validation_split=0.2, batch_size=128, epochs=50, callbacks=[es], verbose=2)
att2_end = time.time()


# # attention3
# att3_start = time.time()
# model_att3 = Attention(model_type=3)
# model_att3.build(input_shape=(None, 112,112,1))
# # model_att3.summary()
# model_att3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# result_att3 = model_att3.fit(X_train, y_train, validation_split=0.2, batch_size=128, epochs=50, callbacks=[es], verbose=2)
# model_att3.evaluate(X_test, y_test)
# att3_end = time.time()


print(f'cnn1 (32): {format(cnn1_end - cnn1_start, ".2f")}')
model_cnn1.evaluate(X_test, y_test)
print()
print(f'cnn2 (128): {format(cnn2_end - cnn2_start, ".2f")}')
model_cnn2.evaluate(X_test, y_test)
print()
print(f'att1: {format(att1_end - att1_start, ".2f")}')
model_att1.evaluate(X_test, y_test)
print()
print(f'att2: {format(att2_end - att2_start, ".2f")}')
model_att2.evaluate(X_test, y_test)
# print()
# print(f'att3: {format(att3_end - att3_start, "2.f")}')
# model_att3.evaluate(X_test, y_test)







model_att1(X_test[:5])  # test_data input
model_att1.score.shape  # attention_score





# CNN + Attention2 Model ----------------------------------------------------------------------
class Attention_02(tf.keras.Model):
    def __init__(self):
        super(Attention_02, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(16, kernel_size=3, strides=2, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu')

        self.w_q = self.add_weight(shape=(64, 1), initializer='random_normal', trainable=True)

        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x, training=False, return_score=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = tf.reshape(x, shape=(-1, 196, 64))
        self.score = tf.nn.softmax(tf.matmul(x, self.w_q), axis=1)
        x = tf.reduce_sum(x * self.score, axis=1)
        x = self.dense(x)

        if return_score:
            return x, self.score
        else:
            return x

attention_model02 = Attention_02()
# attention_model02(X_train[:10]).shape
# attention_model02.score.shape

# attention_model02.build(input_shape=(None, 112,112,1))
# attention_model02.summary()
# es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',  mode='min', patience=4, restore_best_weights=True, verbose=2)

attention_model02.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# result_attention02 = attention_model02.fit(X_train, y_train, validation_split=0.2, batch_size=64, epochs=50, callbacks=[es], verbose=2)
result_attention02 = attention_model02.fit(X_train, y_train, validation_split=0.2, batch_size=64, epochs=10, verbose=2)

# attention_model02.evaluate(X_test, y_test)


test_image = X_test[35]
plt.imshow(test_image[:,:,0], 'gray_r')
plt.show()

# pred, score = attention_model02(test_image[np.newaxis,...], return_score=True)
# score_map = score.numpy().reshape(14,14)
attention_model02(test_image[np.newaxis,...])
score_map = attention_model02.score.numpy().reshape(14,14)

import skimage.transform
score_map_upscaling = skimage.transform.pyramid_expand(score_map, upscale=8, sigma=15)

plt.imshow(score_map_upscaling, 'gray')
plt.imshow(test_image[:,:,0], 'gray_r')
plt.show()

test_image[:,:,0].shape[0]


# # attention_score
# wh_ratio = score_map.shape[0] / score_map.shape[1]
# wh_length = (np.sqrt(np.product(attention_model02.score.shape) / wh_ratio)
# np.array(attention_model02.score).shape

# np.product(test_image.shape) / np.product(attention_model02.score.shape)

def attention_plot(image, score_map, highlight_color='white', figsize=(4,4), alpha=0.03, return_plot=False):
    w_ratio = image.shape[0] / score_map.shape[0]
    h_ratio = image.shape[1] / score_map.shape[1]
    scale_mean = int(np.mean([w_ratio, h_ratio]))

    score_map_upscaling = skimage.transform.pyramid_expand(score_map, upscale=scale_mean, sigma=scale_mean*2)

    x_mesh = np.arange(1, image.shape[0]+1, 1)
    # y_mesh = np.arange(1, image.shape[1]+1, 1)
    y_mesh = np.arange(image.shape[1]+1, 1, -1)

    xs, ys = np.meshgrid(x_mesh, y_mesh)
    k = np.hstack([xs.reshape(-1,1), ys.reshape(-1,1)])

    attention_alpha = (score_map_upscaling*1/np.max(score_map_upscaling)).ravel()
    if highlight_color.lower() == 'white':
        attention_colors = plt.cm.gray(attention_alpha)
    elif highlight_color.lower() == 'black':
        attention_colors = plt.cm.binary(attention_alpha)        
    elif highlight_color.lower() == 'yellow':
        attention_colors = plt.cm.YlOrBr(attention_alpha)
    elif highlight_color.lower() == 'blue':
        attention_colors = plt.cm.Blues(attention_alpha)
    elif highlight_color.lower() == 'red':
        attention_colors = plt.cm.Reds(attention_alpha)

    image_alpha = 1 - image.ravel()
    image_colors = plt.cm.gray(image_alpha)

    fig = plt.figure(figsize=figsize)
    plt.scatter(k[:,0], k[:,1], color=image_colors)
    plt.scatter(k[:,0], k[:,1], color=attention_colors, alpha=alpha)
    plt.show()

    if return_plot:
        return fig


attention_plot(image=test_image[:,:,0], score_map=score_map)























### torch Attention ###########################################################################################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

import torch
import tensorflow as tf
from scipy import io

import math
import time
from six.moves import cPickle

path =r'D:\Python\★★Python_POSTECH_AI\Dataset\attention_mnist_small.pkl'
load_data = cPickle.load(open(path, 'rb'))

X_train = load_data['X_train']
X_test = load_data['X_test']
y_train = load_data['y_train']
y_test = load_data['y_test']
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)



use_cuda = False
# if use_cuda and torch.cuda.is_available():
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(device)    

# train, set
X_torch = torch.tensor(X_train.transpose(0,3,1,2), dtype=torch.float32)
y_torch = torch.tensor(y_train, dtype=torch.int32)

# test_set
X_torch_test = torch.tensor(X_test.transpose(0,3,1,2), dtype=torch.float32)
y_torch_test = torch.tensor(y_test, dtype=torch.int32)

# sample
sample_torch = torch.tensor(X_train[[0]].transpose(0,3,1,2), dtype=torch.float32)


# dataset, dataloader
train_dataset = torch.utils.data.TensorDataset(X_torch, y_torch)
test_dataset = torch.utils.data.TensorDataset(X_torch, y_torch)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)



class Reshape(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        with torch.no_grad():
            shape = np.array(x.to('cpu').detach()).shape
        return x.view(shape[0], shape[1], shape[2]*shape[3]).transpose(1,2)

class AttentionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key_layer = torch.nn.Linear(32, 8, bias=False)
        self.query_layer = torch.nn.Linear(8, 1, bias=False)
    
    def forward(self, x, return_element=False):
        self.value = x
        self.key = self.key_layer(self.value)
        self.score = torch.nn.functional.softmax( self.query_layer(self.key), dim=1 )
        
        if return_element is True:
            return self.value, self.key, self.score
        else:
            return self.score


class Attention_torch(torch.nn.Module):
    def __init__(self, model_type=1):
        super().__init__()
        
        self.model_type = model_type
        
        self.conv_layer = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, (3,3), 2, 1)
            ,torch.nn.ReLU()
            ,torch.nn.Conv2d(8, 16, (3,3), 2, 1)
            ,torch.nn.ReLU()
            ,torch.nn.Conv2d(16, 32, (3,3), 2, 1)
            ,torch.nn.ReLU()
        )
        self.reshape = Reshape()
        self.attention = AttentionLayer()
        self.dense_layer = torch.nn.Sequential(
            torch.nn.Linear(32, 10)
            ,torch.nn.Softmax()
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.reshape(x)
        
        value, key, score = self.attention(x, return_element=True)
        self.value = value
        self.key = key
        # score = self.attention(x)
        self.score = score
        
        x = torch.sum(x * score, axis=1)
        x = self.dense_layer(x)
        return x

model = Attention_torch().to(device)
model(sample_torch.to(device)).shape


loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
epochs = 10

# training 
losses = []
for e in range(epochs):
    epoch_loss = []
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()                   # wegiht initialize
        pred = model(batch_X.to(device))                   # predict
        loss = loss_function(pred, torch.argmax(batch_y, axis=1).to(device))     # loss
        loss.backward()                         # backward
        optimizer.step()                        # update_weight

        with torch.no_grad():
            epoch_loss.append( loss.to('cpu').detach().numpy() )
    losses.append(np.mean(epoch_loss))
    # print(f"{e+1} epochs) loss: {losses[-1]}")
    print(f"{e+1} epochs) loss: {losses[-1]}", end='\r')
    
plt.figure()
plt.plot(losses)
plt.show()



############################################################################################################################################################################

















































class MultiAttention(tf.keras.Model):
    def __init__(self):
        super(MultiAttention, self).__init__()
        
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 3, 2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(32, 3, 2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, 3, 2, padding='same', activation='relu'),
        ])  # 13 * 25 * 64

        # self.w_k = self.add_weight(shape)
        self.w_q = self.add_weight(shape=(64, 10), initializer='random_normal', trainable=True)
        # self.dense = tf.keras.layers.Dense(10, activation='sigmoid')
        
        self.dense1 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.dense3 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.dense4 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.dense5 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.dense6 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.dense7 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.dense8 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.dense9 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.dense10 = tf.keras.layers.Dense(1, activation='sigmoid')

    
    def call(self, x, training=False):
        x = self.conv(x)
        x = tf.reshape(x, (-1, 13*25, 64))

        score = tf.nn.softmax(tf.matmul(x, self.w_q), axis=1)[..., tf.newaxis]
        self.score = score
        
        x = tf.expand_dims(x, axis=-2)
        x = tf.reduce_sum(x * score, axis=1)        # 10, 64
        # x = tf.keras.layers.Flatten()(x)
        # x = self.dense(x)
        x1 = self.dense1(x[:,0,:])
        x2 = self.dense2(x[:,1,:])
        x3 = self.dense3(x[:,2,:])
        x4 = self.dense4(x[:,3,:])
        x5 = self.dense5(x[:,4,:])
        x6 = self.dense6(x[:,5,:])
        x7 = self.dense7(x[:,6,:])
        x8 = self.dense8(x[:,7,:])
        x9 = self.dense9(x[:,8,:])
        x10 = self.dense10(x[:,9,:])

        x = tf.concat([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], axis=-1)
        return x
        

attention_multi = MultiAttention()
attention_multi.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.05), metrics=['accuracy'])
result_multi = attention_multi.fit(images, labels, epochs=10, batch_size=64)


attention_multi(images[10][np.newaxis,...])
score_maps = attention_multi.score.numpy()

plt.imshow(images[10])
plt.show()


for i in range(10):
    plt.title(i)
    plt.imshow(score_maps[0,:,i,0].reshape(13, 25))
    plt.show()
















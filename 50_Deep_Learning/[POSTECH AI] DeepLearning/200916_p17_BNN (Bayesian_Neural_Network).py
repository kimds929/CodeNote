import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp
# search wight distribution


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = (x_train.astype('float32')/255.0)[..., np.newaxis]
x_test = (x_test.astype('float32')/255.0)[..., np.newaxis]

y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

plt.imshow(np.squeeze(x_train[7]), cmap='gray')



# Deterministic_CNN Model ---------------------------------------------------------------------------------------------------
class Deterministic_CNN(tf.keras.Model):
    def __init__(self):
        super(Deterministic_CNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.do1 = tf.keras.layers.Dropout(0.25)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.do2 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(10)
    
    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.do1(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.do2(x, training=training)
        x = self.dense2(x)
        return x



# Model Learning ----------------------
cnn_model = Deterministic_CNN()

cnn_optimizer = tf.keras.optimizers.Adam(0.01)
cnn_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
cnn_model.compile(optimizer=cnn_optimizer, loss=cnn_loss, metrics=['accuracy'])
result = cnn_model.fit(x_train, y_train, epochs=10, validation_split=0.1)

cnn_model.evaluate(x_test, y_test, verbose=2) # deterministic


# Prediction ---------------------
np.set_printoptions(suppress=True)   # science 표기법끄기

def proba_plot(proba_list):
    plt.bar(x=range(len(proba_list)), height=proba_list)
    # plt.yscale('symlog')
    plt.xlabel('calsses')
    plt.ylabel('proba')
    plt.ylim(bottom=0, top=1)
    plt.show()


img_nums = 7
plt.imshow(np.squeeze(x_train[img_nums]), cmap='gray')
plt.show()

confirm_x = np.array(x_train[img_nums])[np.newaxis,...]
pred_cnn = tf.nn.softmax(cnn_model(confirm_x)).numpy()[0]
print(pred_cnn)
print('predicted', np.argmax(pred_cnn, axis=-1))
proba_plot(pred_cnn)

# Noise Prediction ----------------------
noise = np.random.random((1,28,28,1))
plt.imshow(np.squeeze(noise), cmap='gray')
plt.show()

pred_noise = tf.nn.softmax(cnn_model(noise)).numpy()[0]
print(pred_noise)
print('noise_predicted', np.argmax(pred_noise, axis=-1))
proba_plot(pred_noise)



# Baysian_CNN Model ---------------------------------------------------------------------------------------------------


# ?tfp.python.layers.Convolution2DReparameterization
# ?tfp.python.layers.Convolution2DFlipout
# ?tfp.python.layers.DenseReparameterization
class Baysian_CNN(tf.keras.Model):
    def __init__(self):
        super(Baysian_CNN, self).__init__()

        # Convolution Layer가 Distribution으로 나타남
        # W ~ N(μ, σ²)
        # W = μ + σ ⊙ ε
        self.conv_prob1 = tfp.python.layers.Convolution2DReparameterization(
                            filters=32, kernel_size=(3,3), strides=(2,2), padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation('relu')
        self.conv_prob2 = tfp.python.layers.Convolution2DReparameterization(
                            filters=64, kernel_size=(3,3), strides=(2,2), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.Activation('relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense_prob1 = tfp.python.layers.DenseReparameterization(512, activation='relu')
        self.dense_prob2 = tfp.python.layers.DenseReparameterization(10)

        # # Sequential Modeling
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Input(shape=(28,28,1)))
        self.model.add(self.conv_prob1)
        self.model.add(self.bn1)
        self.model.add(self.act1)
        self.model.add(self.conv_prob2)
        self.model.add(self.bn2)
        self.model.add(self.act2)
        self.model.add(self.flatten)
        self.model.add(self.dense_prob1)
        self.model.add(self.dense_prob2)

    def call(self, x, training=False):
        return self.model(x, training=training)
        # x = self.conv_prob1(x)
        # x = self.bn1(x, training=training)
        # x = self.act1(x)
        # x = self.conv_prob2(x)
        # x = self.bn2(x, training=training)
        # x = self.act2(x)
        # x = self.flatten(x)
        # x = self.dense_prob1(x)
        # x = self.dense_prob2(x)
        # return x
    
    def train_step(self, data):     # model.fit 동작을 변경
        images, labels = data
        with tf.GradientTape() as Tape:
            # logits = self.model(images, training=True)
            logits = self.model(images, training=True)
            loss_ce = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
            loss_kld = tf.reduce_sum(self.model.losses) / images.shape[0]
            loss = loss_ce + loss_kld

        gradients = Tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        accuracy = tf.metrics.categorical_accuracy(y_true=labels, y_pred=logits)
        
        return {'loss':loss, 'CrossEntropy': loss_ce, 'KL_Divergence': loss_kld, 'accuracy': accuracy}
            

cnn_baysian = Baysian_CNN()

baysian_optimizer = tf.keras.optimizers.Adam(0.01)
cnn_baysian.compile(optimizer=baysian_optimizer, metrics=['accuracy'])
baysian_result = cnn_baysian.fit(x_train, y_train, epochs=20, batch_size=3000, verbose=2)
# cnn_baysian.evaluate(x_test, y_test, verbose=2) # deterministic



# Prediction ---------------------
np.set_printoptions(suppress=True)   # science 표기법켜기
img_nums = np.random.randint(10)
plt.imshow(np.squeeze(x_train[img_nums]), cmap='gray')
plt.show()

bayisan_x = np.array(x_train[img_nums])[np.newaxis,...]
pred_baysian = tf.nn.softmax(cnn_baysian(bayisan_x)).numpy()[0]
print(pred_baysian)
print('baysian_predicted', np.argmax(pred_baysian, axis=-1))
proba_plot(pred_baysian)

# Noise Prediction ----------------------
noise = np.random.random((1,28,28,1))
plt.imshow(np.squeeze(noise), cmap='gray')
plt.show()

pred_noise = tf.nn.softmax(cnn_baysian(noise)).numpy()[0]
print(pred_noise)
print('noise_predicted', np.argmax(pred_noise, axis=-1))
proba_plot(pred_noise)



def mean_dev(image):
    probability_list=[]
    for i in range(100):
        prob = tf.nn.softmax(cnn_baysian(np.array(image)[np.newaxis,...])).numpy().squeeze()

        probability_list.append(prob)
    return np.mean(probability_list, axis=0), np.std(probability_list, axis=0)

img_nums = np.random.randint(10)
mean, dev = mean_dev(x_train[img_nums])
plt.imshow(np.squeeze(x_train[img_nums]), cmap='gray')
plt.show()

print('mean', mean)
print('dev:', dev)
predicted_mean = np.argmax(mean)
print('predicted_min:', predicted_mean,'/ prob:', mean[predicted_mean])
predicted_dev = np.argmax(dev)
print('predicted_dev:', predicted_dev,'/ prob:', dev[predicted_mean])









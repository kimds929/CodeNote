# Import Library
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf
# %matplotlib inline


# Load Data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

pd.DataFrame(x_train[5]).to_clipboard()


print("The training data set is:\n")
print(x_train.shape)
print(y_train.shape)

x_train

x_train.max()

# flatten
x_train = x_train.reshape((x_train.shape[0],28*28)) / 255
x_test = x_test.reshape((x_test.shape[0],28*28)) / 255

print('Pixel max : ', x_train.max())
print('Pixel min : ', x_train.min())

print("The training data set is:\n")
print(x_train.shape)
print(y_train.shape)


x_train[5]
x_train[5].shape    # well, that's not a picture (or image), it's an array.

img = np.reshape(x_train[5], [28,28])
img = x_train[5].reshape([28,28])

# So now we have a 28x28 matrix, where each element is an intensity level from 0 to 1.  
img.shape

# image show
plt.figure(figsize = (6,6))
plt.imshow(img, 'gray')
plt.xticks([])
plt.yticks([])
plt.show()


y_train[5]




# 2.1. Import Library ----------------------------------------
# Import Library
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 2.2. Load MNIST Data
# Load Data

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], 28*28)) / 255
x_test = x_test.reshape((x_test.shape[0],28*28)) / 255



img = x_train[5].reshape(28,28)

plt.figure(figsize = (6,6))
plt.imshow(img, 'gray')
plt.xticks([])
plt.yticks([])
plt.show()

print ('Train labels : {}'.format(y_train[5]))



# Define Structure
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, activation=tf.nn.relu, input_shape=(28*28,)),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax, input_shape=(100,))
    ])


model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

n_batch = 50 # batch size
n_epoch = 10 # epoch

# Train Model & Evaluate Test Data
training_records = model.fit(x_train, y_train, batch_size=n_batch, epochs=n_epoch)

training_records.history    # 학습결과


plt.figure(figsize=(10,8))
plt.plot((np.arange(n_epoch)+1), training_records.history['loss'], label = 'training')
plt.xlabel('Iteration', fontsize = 15)
plt.ylabel('Loss', fontsize = 15)
plt.legend(fontsize = 12)
plt.ylim([np.min(training_records.history['loss'])-0.05, np.max(training_records.history['loss'])])
plt.show()

test_scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss: {}'.format(test_scores[0]))
print('Test accuracy: {} %'.format(test_scores[1]*100))


# 2.8. Test or Evaluate ---------------------------------------------
rand_idx = np.random.randint(1, x_test.shape[0]+1)

test_x = x_test[rand_idx]
test_y = y_test[rand_idx]
test_x = test_x.reshape(1,28*28)

logits = model.predict(test_x)
predict = np.argmax(logits)
np.set_printoptions(precision = 2, suppress = True)
print('Probability : {}'.format(logits.ravel()))
print('Prediction : {}'.format(predict))
print('True Label : {}'.format(test_y))

if predict == test_y:
    print('Prediction is Correct')
else:
    print('Prediction is Incorrect')

plt.figure(figsize = (6,6))
plt.imshow(test_x.reshape(28,28), 'gray')
plt.xticks([])
plt.yticks([])
plt.show()
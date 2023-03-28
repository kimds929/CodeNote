import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# %matplotlib inline


# 3.2. Load MNIST Data
# Load Data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((x_train.shape[0],28*28)) / 255
x_test = x_test.reshape((x_test.shape[0],28*28)) / 255

print(x_train.shape, x_test.shape)


# Use Only 1,5,6 Digits to Visualize
train_idx1 = np.array(np.where(y_train == 1))
train_idx5 = np.array(np.where(y_train == 5))
train_idx6 = np.array(np.where(y_train == 6))
train_idx = np.sort(np.concatenate((train_idx1, train_idx5, train_idx6), axis= None))

test_idx1 = np.array(np.where(y_test == 1))
test_idx5 = np.array(np.where(y_test == 5))
test_idx6 = np.array(np.where(y_test == 6))
test_idx = np.sort(np.concatenate((test_idx1, test_idx5, test_idx6), axis= None))

train_imgs = x_train[train_idx]
train_labels = y_train[train_idx]
test_imgs = x_test[test_idx]
test_labels = y_test[test_idx]

n_train = train_imgs.shape[0]
n_test = test_imgs.shape[0]

print ("The number of training images : {}, shape : {}".format(n_train, train_imgs.shape))
print ("The number of testing images : {}, shape : {}".format(n_test, test_imgs.shape))



# Define Structure

# Encoder Structure
encoder = tf.keras.models.Sequential([
    tf.keras.layers.Dense(500, activation='relu', input_shape=(28*28,)),
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(2, activation=None)
    ])

# Decoder Structure
decoder = tf.keras.models.Sequential([
    tf.keras.layers.Dense(300, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(28*28, activation=None)
    ])

# Autoencoder = Encoder + Decoder
autoencoder = tf.keras.models.Sequential([encoder, decoder])

autoencoder.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error',
              metrics=['mse'])

# Train Model & Evaluate Test Data
n_batch = 50
n_iter = train_imgs.shape[0]/n_batch
n_epoch = 10

training = autoencoder.fit(train_imgs, train_imgs, batch_size=n_batch, epochs=n_epoch)

# autoencoder.evaluate(test_imgs, test_imgs)



# Image display
plt.figure(figsize=(10,8))
plt.plot((np.arange(n_epoch)+1)*n_iter, training.history['loss'], label = 'training')
plt.xlabel('Iteration', fontsize = 15)
plt.ylabel('Loss', fontsize = 15)
plt.legend(fontsize = 12)
plt.ylim([np.min(training.history['loss']), np.max(training.history['loss'])])
plt.show()

test_scores = autoencoder.evaluate(test_imgs, test_imgs, verbose=2)
print('Test loss: {}'.format(test_scores[0]))
print('Mean Squared Error: {} %'.format(test_scores[1]*100))



# Visualize Evaluation on Test Data
rand_idx = np.random.randint(0, test_imgs.shape[0])
print(rand_idx)
# rand_idx = 1168
test_img = test_imgs[rand_idx]
reconst_img = autoencoder.predict(test_img.reshape(1,28*28))

plt.figure(figsize = (10, 8))
plt.subplot(1,2,1)
plt.imshow(test_img.reshape(28,28), 'gray')
plt.title('Input Image', fontsize = 12)
plt.xticks([])
plt.yticks([])
plt.subplot(1,2,2)
plt.imshow(reconst_img.reshape(28,28), 'gray')
plt.title('Reconstructed Image', fontsize = 12)
plt.xticks([])
plt.yticks([])

plt.show()






# Visualize Latent Space
latent = encoder.predict(test_imgs)
reconst = decoder.predict(latent)

print(latent.shape, reconst.shape)

# Initialize Canvas
nx = 20
ny = 20
x_values = np.linspace(-5, 5, nx)
y_values = np.linspace(-5, 8, ny)
canvas = np.empty((28*ny, 28*nx))

for i, yi in enumerate(y_values):
        for j, xi in enumerate(x_values):
            latent_coordinates_ = np.array([[xi, yi]])
            reconst_from_latent_ = decoder.predict(latent_coordinates_)
            canvas[(nx-i-1)*28:(nx-i)*28,j*28:(j+1)*28] = reconst_from_latent_.reshape(28, 28)


plt.figure(figsize = (16, 7))
plt.subplot(1,2,1)
plt.scatter(latent[test_labels == 1,0], latent[test_labels == 1,1], label = '1')
plt.scatter(latent[test_labels == 5,0], latent[test_labels == 5,1], label = '5')
plt.scatter(latent[test_labels == 6,0], latent[test_labels == 6,1], label = '6')
plt.title('Latent Space', fontsize = 12)
plt.xlabel('Z1', fontsize = 12)
plt.ylabel('Z2', fontsize = 12)
plt.legend(fontsize = 12)
plt.axis('equal')
plt.subplot(1,2,2)
plt.imshow(canvas, 'gray')
plt.title('Manifold', fontsize = 12)
plt.xlabel('Z1', fontsize = 12)
plt.ylabel('Z2', fontsize = 12)
plt.xticks([])
plt.yticks([])
plt.show()







# rand_latent
rand_latent = np.array([np.random.rand(2)*16-8])
# rand_latent = np.array([[0, 0]])
# rand_latent = np.array([[-6, -2]])
print(rand_latent)

def flatten_to_image(flat_array, height=-1, width=-1):
    canvas = []

    if width == -1 and height > 0:
        width = int(flat_array.shape[1] / height)
    elif height == -1 and width > 0:
        height = int(flat_array.shape[1] / width)

    for k in range(height):
        canvas.append( flat_array[0, k*width:(k+1)*width] )

    return np.array(canvas)


gen_number = decoder.predict(rand_latent)
plt.imshow(flatten_to_image(gen_number, height=28), 'gray')










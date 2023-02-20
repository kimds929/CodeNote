import random

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

import tensorflow as tf
# from tensorflow.keras import Model, layers
# import torch

from glob import glob
import imageio
import PIL

import os
import sys
import copy
import time
import IPython
from IPython.display import clear_output
from IPython import display
# from ipywidgets import interact

# GPU 사용여부 확인방법
# [console] nvidia-smi -lms 3000 (miliseconds)
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# sess = tf.compat.v1.Session(config=config)


(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images[..., np.newaxis].astype('float32')
print(train_images.shape, np.min(train_images), np.max(train_images))

train_images = (train_images - 127.5) / 127.5    # [0, 255] to [-1, 1]
print(train_images.shape, np.min(train_images), np.max(train_images))

buffer_size = 60000
batch_size = 256

train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
# print(next(iter(train_dataset)).shape)




# Generator -------------------------------------------------------------------------------
def make_generator_model():
    model = tf.keras.Sequential()
    # input_shape: (batch, 100)
    model.add(tf.keras.layers.Dense(units=7*7*256, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    # shape: (batch, 7*7*256)

    model.add(tf.keras.layers.Reshape((7,7,256)))
    # shape: (batch, 7, 7, 256)

    model.add(tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(5,5), strides=(1,1), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())      # shape: (batch, 7, 7, 128)

    model.add(tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(5,5), strides=(2,2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())      # shape: (batch, 14, 14, 64)

    model.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(5,5), strides=(2,2), 
                padding='same', use_bias=False, activation='tanh'))  # shape : (batch, 28, 28, 1)

    return model
# tf.keras.layers.BatchNormalization?

generator = make_generator_model()

# Noise data 확인
noise = tf.random.normal((1, 100))
gen_image = generator(noise, training=False)
print(gen_image.shape)

# 이미지 확인
plt.imshow(gen_image[0,:,:,0], 'gray')
plt.colorbar()
plt.show()



# Discriminator -------------------------------------------------------------------------------
def make_discriminator_model():
    model = tf.keras.Sequential()
    # Input shape: (batch_size, 28, 28, 1)
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5), strides=(2,2), padding='same', input_shape=(28,28,1)))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))     # shape: (batch_size, 14, 14, 64)

    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(5,5), strides=(2,2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))     # shape: (batch_size, 7, 7, 128)

    model.add(tf.keras.layers.Flatten())        # shape: (batch_size, 7 * 7 *128)

    model.add(tf.keras.layers.Dense(1))
    return model


# Test untrainined discriminator    
discriminator = make_discriminator_model()
decision = discriminator(gen_image)

print(decision.numpy())                 # result_value
print(tf.sigmoid(decision).numpy())     # result_proba


# Model Learning -------------------------------------------------------------------------------
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(real_output), fake_output)

    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
    # return -cross_entropy(tf.zeros_like(fake_output), fake_output)


generator_opt = tf.keras.optimizers.Adam(1e-4)
discriminator_opt = tf.keras.optimizers.Adam(1e-4)

# epochs = 70
# epochs = 5
noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal((num_examples_to_generate, noise_dim))

@tf.function
def train_step(train_batch_x):
    # noise = tf.random.normal([batch_size, noise_dim])        # train_data만큼의 가짜 이미지도 만들어야 함
    noise = tf.random.normal([train_batch_x.shape[0], noise_dim])        # train_data만큼의 가짜 이미지도 만들어야 함

    with tf.GradientTape() as GenTape, tf.GradientTape() as DiscTape:
        genrated_images = generator(noise, training=True)   # generates (batch_size, 28, 28, 1) images.

        real_output = discriminator(train_batch_x, training=True)
        fake_output = discriminator(genrated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        

    gradients_of_generator = GenTape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = DiscTape.gradient(disc_loss, discriminator.trainable_variables)

    generator_opt.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_opt.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    # return


def generated_and_save_image(model, epochs, test_input):
    predictions = model(test_input, training=False)      # generated images

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epochs))
    plt.show()


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)
        
        clear_output(wait=True)
        generated_and_save_image(model=generator, epochs=epoch+1, test_input=seed)

        print(f'Time for epoch {epoch+1} is {time.time()-start} sec')
    clear_output(wait=True)
    generated_and_save_image(model=generator, epochs=epochs, test_input=seed)

epochs = 70
noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal((num_examples_to_generate, noise_dim))

generator = make_generator_model()
discriminator = make_discriminator_model()


train(train_dataset, epochs)


batch = next(iter(train_dataset))
batch.shape

train_step(batch)
discriminator(batch, training=True)



# def display_image(epoch_no):
#     return PIL.image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


anim_file = 'gan.gif'

with imageio.get_writer(uri=anim_file, mode='I') as writer:
    filenames = glob('image*.png')    
    filenames = sorted(filenames)
    last = -1

    for i, filename in enumerate(filenames):
        frame = 2*(i**0.5)

        if round(frame) > round(last):
            last = frame
        
        else:
            continue
    
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)


if IPython.version_info > (6,2,0, ''):
    display_image(filename = anim_file)


try:
    from google.colab import files
except ImportError:
    pass
else:
    files.download(anim_file)

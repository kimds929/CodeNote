import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
# pip install tensorflow_probability
# conda install cloudpickle
# import tensorflow_probability as tfp


(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32')/255
x_train = x_train[..., np.newaxis]
print(x_train.shape, x_test.shape)


class VAE(tf.keras.Model):
    def __init__(self, latent_dim, beta=1):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim        # latent_dim 저장
        self.beta = beta                    # beta저장

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same',
                        activation='relu', input_shape=(28, 28, 1)),   # shape: (batch_size, 14, 14, 32)
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same',
                        activation='relu'),     # (batch_size, 7, 7, 64),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=latent_dim * 2)     # μ, σ  # shape: (batch_size, latent_dim * 2)
                                                            # σ: -∞ ~ ∞
        ])

        self.decoder = tf.keras.Sequential([
            # input_shape: (batch_size, latent_dim)
            tf.keras.layers.Dense(units=7*7*32, activation='relu', input_shape=(latent_dim,)),   # (batch, 7*7*32)
            tf.keras.layers.Reshape((7, 7, 32)), # shape: (batch_size, 7, 7, 32)
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', 
                        activation='relu'),  # shape: (batch_size, 14, 14, 64)
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', 
                        activation='relu'),   # shape: (batch_size, 28, 28, 32)
            tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')    # logit
        ])


    def encode(self, x):    # x_shape: (batch.size
        mu, logvar = tf.split(value=self.encoder(x), num_or_size_splits=2, axis=1)  # tf.split vector를 주면 잘라줌
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        epsilon = tf.random.normal(mu.shape)
        sigma = tf.exp(0.5 * logvar)
        return epsilon * sigma + mu

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    @tf.function
    def generate(self, epsilon=None):
        if epsilon is None:
            epsilon = tf.random.normal((100, self.latent_dim))
        return self.decode(epsilon, apply_sigmoid=True)

    def train_step(self, x):
        with tf.GradientTape() as Tape:
            mu, logvar = self.encode(x) # (batch_size, latent_dim), (batch_size, latent_dim)
            z = self.reparameterize(mu, logvar)
            x_logit = self.decode(z)

            # reconstruction
            recon = tf.reduce_sum(tf.losses.binary_crossentropy(y_true=x, y_pred=x_logit, from_logits=True), axis=[1,2])  # shape: (batch_size, ) : batch 단위로 loss를 구함
            kl = 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(logvar) -logvar - 1., axis=1)   # shape: (batch_size, )
            loss = tf.reduce_mean(self.beta * kl + recon)

        gradients = Tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {'elbo': loss, 'kl': kl, 'recon':recon}

# axis = sum
# a = tf.constant([[[1,2], [3,4]], [[5,6],[7,8]], [[9,10], [11,12]]])
# a
# tf.reduce_sum(a)
# tf.reduce_sum(a, axis=[1,2])


vae = VAE(latent_dim=2, beta=1)
generated_image1 = vae.generate()
plt.imshow(generated_image1[0, :, :, 0], cmap='gray')

optimizer = tf.keras.optimizers.Adam(1e-4)
vae.compile(optimizer)
vae.fit(x_train, batch_size=100, epochs=30)

generated_image_after = vae.generate()
plt.imshow(generated_image_after[3, :, :, 0], 'gray')


def plot_latent_images(model, n, digit_size=28):
    norm = tfp.distributions.Norm(0, 1)
    grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
    grid_y = norm.quantile(np.linspace(0.05, 0.95, n))

    image_width = digit_size * n
    image_height = image_width
    image = np.zeros((image_height, image_width))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z = np.array([[xi, yi]])
            x_decoded = model.generate(z)
            digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
            image[i * digit_size : (i+1)*digit_size, j * digit_size : (j+1)*digit_size] = digit.numpy()

    plt.figure(figsize=(10,10))
    plt.imshow(image, cmap='gray')
    plt.aixs('off')
    plt.show()




# vae_beta10 = VAE(latent_dim=2, beta=10)
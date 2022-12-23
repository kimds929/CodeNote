import tensorflow as tf


class UNet(tf.keras.Model):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.conv1_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=1, padding='same', activation='relu')
        self.conv1_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=1, padding='same', activation='relu')
        
        self.conv2_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='same', activation='relu')
        self.conv2_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='same', activation='relu')
        
        self.conv3_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu')
        self.conv3_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu')
        
        self.conv4_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same', activation='relu')
        self.conv4_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same', activation='relu')
        
        self.conv5_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same', activation='relu')
        self.conv5_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same', activation='relu')
        
        self.maxpool = tf.keras.layers.MaxPool2D()   # 나눠써도됨 (weight가 없기때문에)
        
        self.convT6_0 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(2,2), strides=2, padding='same', activation='relu')
        self.conv6_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same', activation='relu')
        self.conv6_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same', activation='relu')
        
        self.convT7_0 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(2,2), strides=2, padding='same', activation='relu')
        self.conv7_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu')
        self.conv7_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu')
        
        self.convT8_0 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(2,2), strides=2, padding='same', activation='relu')
        self.conv8_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='same', activation='relu')
        self.conv8_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='same', activation='relu')
        
        self.convT9_0 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(2,2), strides=2, padding='same', activation='relu')
        self.conv9_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=1, padding='same', activation='relu')
        self.conv9_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=1, padding='same', activation='relu')
        
        self.conv9_3 = tf.keras.layers.Conv2D(filters=5, kernel_size=(1,1), strides=1, padding='same', activation='softmax')
    
    def call(self, x, training=False):
        # x = tf.multiply(x, 1/255)
        x = x/255
        
        x = self.conv1_1(x)
        x = x_1 = self.conv1_2(x)
        x = self.maxpool(x)
        
        x = self.conv2_1(x)
        x = x_2 = self.conv2_2(x)
        x = self.maxpool(x)
        
        x = self.conv3_1(x)
        x = x_3 = self.conv3_2(x)
        x = self.maxpool(x)
        
        x = self.conv4_1(x)
        x = x_4 = self.conv4_2(x)
        x = self.maxpool(x)
        
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        
        x = self.convT6_0(x)
        x = tf.concat([x, x_4], axis=-1)
        x = self.conv6_1(x)
        x = self.conv6_2(x)
        
        x = self.convT7_0(x)
        x = tf.concat([x, x_3], axis=-1)
        x = self.conv7_1(x)
        x = self.conv7_2(x)
        
        x = self.convT8_0(x)
        x = tf.concat([x, x_2], axis=-1)
        x = self.conv8_1(x)
        x = self.conv8_2(x)
        
        x = self.convT9_0(x)
        x = tf.concat([x, x_1], axis=-1)
        x = self.conv9_1(x)
        x = self.conv9_2(x)
        
        x = self.conv9_3(x)
        
        return x
"""
Michael Patel
April 2020

Project description:
    Build a GAN using Cifar-10 data set

File description:
    For model definitions
"""
################################################################################
# Imports
import tensorflow as tf

from parameters import *


################################################################################
# Discriminator
def build_discriminator():
    m = tf.keras.Sequential()

    # Layer 1: Conv: 32x32x64
    m.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(5, 5),
        strides=1,
        padding="same",
        input_shape=(32, 32, 3)
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Layer 2: Conv: 16x16x128
    m.add(tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(5, 5),
        strides=2,
        padding="same"
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Layer : Dropout
    m.add(tf.keras.layers.Dropout(rate=DROPOUT_RATE))

    # Layer 3: Conv: 8x8x128
    m.add(tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(5, 5),
        strides=2,
        padding="same"
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Layer 4: Conv: 4x4x256
    m.add(tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=(5, 5),
        strides=2,
        padding="same"
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Layer : Dropout
    m.add(tf.keras.layers.Dropout(rate=DROPOUT_RATE))

    # Layer 5: Flatten
    m.add(tf.keras.layers.Flatten())

    # Layer 6: Output
    m.add(tf.keras.layers.Dense(
        units=1
    ))

    return m


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Conv 1: 32x32x64
        self.conv1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same"
        )

        # Batchnorm 1
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        # Leaky 1
        self.leaky1 = tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA)

        # Conv 2: 16x16x128
        self.conv2 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=2,
            padding="same"
        )

        # Batchnorm 2
        self.batchnorm2 = tf.keras.layers.BatchNormalization()

        # Leaky 2
        self.leaky2 = tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA)

        # Conv 3: 8x8x128
        self.conv3 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=2,
            padding="same"
        )

        # Batchnorm 3
        self.batchnorm3 = tf.keras.layers.BatchNormalization()

        # Leaky 3
        self.leaky3 = tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA)

        # Conv 4: 4x4x256
        self.conv4 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=2,
            padding="same"
        )

        # Batchnorm 4
        self.batchnorm4 = tf.keras.layers.BatchNormalization()

        # Leaky 4
        self.leaky4 = tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA)

        # Flatten
        self.flatten = tf.keras.layers.Flatten()

        # Dropout

        # Dense
        self.fc = tf.keras.layers.Dense(
            units=1,
            activation=tf.keras.activations.sigmoid
        )

    # forward call
    def call(self, x):
        # Layer 1: Conv 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.leaky1(x)

        # Layer 2: Conv 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.leaky2(x)

        # Layer 3: Conv 3
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.leaky3(x)

        # Layer 4: Conv 4
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.leaky4(x)

        # Layer 5: Flatten
        x = self.flatten(x)

        # Layer 6: Output
        x = self.fc(x)

        return x


# Generator
def build_generator():
    m = tf.keras.Sequential()

    # Layer 1: Fully connected
    m.add(tf.keras.layers.Dense(
        units=4*4*256,
        use_bias=False,
        input_shape=(100, )
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Layer 2: Reshape
    m.add(tf.keras.layers.Reshape(
        target_shape=(4, 4, 256)
    ))

    # Layer 3: Conv: 8x8x128
    m.add(tf.keras.layers.Conv2DTranspose(
        filters=128,
        kernel_size=(5, 5),
        strides=2,
        padding="same"
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Layer 4: Conv: 16x16x128
    m.add(tf.keras.layers.Conv2DTranspose(
        filters=128,
        kernel_size=(5, 5),
        strides=2,
        padding="same"
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Layer 5: Conv: 32x32x128
    m.add(tf.keras.layers.Conv2DTranspose(
        filters=128,
        kernel_size=(5, 5),
        strides=2,
        padding="same"
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Layer 6: Conv: 32x32x3
    m.add(tf.keras.layers.Conv2DTranspose(
        filters=3,  # RGB
        kernel_size=(5, 5),
        strides=1,
        padding="same",
        activation=tf.keras.activations.tanh
    ))

    return m


# Generator
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        # Fully connected
        self.fc = tf.keras.layers.Dense(
            units=4*4*256
        )

        # Batchnorm 1
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        # Leaky 1
        self.leaky1 = tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA)

        # Reshape
        self.reshape = tf.keras.layers.Reshape(
            target_shape=(4, 4, 256)
        )

        # Conv 1: 8x8x128
        self.conv1 = tf.keras.layers.Conv2DTranspose(
            filters=128,
            kernel_size=(4, 4),
            strides=2,
            padding="same"
        )

        # Batchnorm 2
        self.batchnorm2 = tf.keras.layers.BatchNormalization()

        # Leaky 2
        self.leaky2 = tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA)

        # Conv 2: 16x16x128
        self.conv2 = tf.keras.layers.Conv2DTranspose(
            filters=128,
            kernel_size=(4, 4),
            strides=2,
            padding="same"
        )

        # Batchnorm 3
        self.batchnorm3 = tf.keras.layers.BatchNormalization()

        # Leaky 3
        self.leaky3 = tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA)

        # Conv 3: 32x32x128
        self.conv3 = tf.keras.layers.Conv2DTranspose(
            filters=128,
            kernel_size=(4, 4),
            strides=2,
            padding="same"
        )

        # Batchnorm 4
        self.batchnorm4 = tf.keras.layers.BatchNormalization()

        # Leaky 4
        self.leaky4 = tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA)

        # Conv 4: 32x32x3
        self.conv4 = tf.keras.layers.Conv2DTranspose(
            filters=3,  # RGB
            kernel_size=(3, 3),
            padding="same",
            activation=tf.keras.activations.tanh
        )

    # forward call
    def call(self, x, training=True):
        # Layer 1: Fully connected
        x = self.fc(x)
        x = self.batchnorm1(x, training=training)
        x = self.leaky1(x)

        # Layer 2: Reshape
        x = self.reshape(x)

        # Layer 3: Conv 1
        x = self.conv1(x)
        x = self.batchnorm2(x, training=training)
        x = self.leaky2(x)

        # Layer 4: Conv 2
        x = self.conv2(x)
        x = self.batchnorm3(x, training=training)
        x = self.leaky3(x)

        # Layer 5: Conv 3
        x = self.conv3(x)
        x = self.batchnorm4(x, training=training)
        x = self.leaky4(x)

        # Layer 6: Conv 4
        x = self.conv4(x)

        return x

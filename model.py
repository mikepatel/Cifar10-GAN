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


################################################################################
# Discriminator
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Conv 1
        self.conv1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same"
        )

        # Batchnorm 1
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        # Leaky 1
        self.leaky1 = tf.keras.layers.LeakyReLU()

        # Conv 2
        self.conv2 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=2,
            padding="same"
        )

        # Batchnorm 2
        self.batchnorm2 = tf.keras.layers.BatchNormalization()

        # Leaky 2
        self.leaky2 = tf.keras.layers.LeakyReLU()

        # Conv 3
        self.conv3 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=2,
            padding="same"
        )

        # Batchnorm 3
        self.batchnorm3 = tf.keras.layers.BatchNormalization()

        # Leaky 3
        self.leaky3 = tf.keras.layers.LeakyReLU()

        # Conv 4
        self.conv4 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=2,
            padding="same"
        )

        # Batchnorm 4
        self.batchnorm4 = tf.keras.layers.BatchNormalization()

        # Leaky 4
        self.leaky4 = tf.keras.layers.LeakyReLU()

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
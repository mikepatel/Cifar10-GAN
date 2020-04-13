"""
Michael Patel
April 2020

Project description:
    Build a GAN using Cifar-10 data set

File description:
    For model preprocessing and training
"""
################################################################################
# Imports
import os
import numpy as np
from datetime import datetime
import glob
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf

from parameters import *
from model import Discriminator, Generator


################################################################################
# discriminaor loss function
def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = cross_entropy(
        tf.ones_like(real_output),
        real_output
    )

    fake_loss = cross_entropy(
        tf.zeros_like(fake_output),
        fake_output
    )

    total_loss = real_loss + fake_loss
    return total_loss


# generator loss function
def generator_loss(generated_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generated_loss = cross_entropy(
        tf.ones_like(generated_output),
        generated_output
    )

    return generated_loss


# generate and save images
def generate_images(model, epoch, z_input, save_dir):
    predictions = model(z_input, training=False)

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
        plt.axis("off")

    fig_name = os.path.join(save_dir, f'Epoch {epoch:06d}')
    plt.savefig(fig_name)
    plt.close()


# training loop
def train(dataset, d, g, d_optimizer, g_optimizer, z_input, save_dir):
    for e in range(NUM_EPOCHS+1):
        # get real images
        idx = np.random.randint(0, len(dataset), size=BATCH_SIZE)
        real_images = dataset[idx]

        # generate noise
        noise = tf.random.normal(shape=(BATCH_SIZE, NOISE_DIM))

        # GradientTape
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            # generator
            generated_image = g(noise)

            # discriminator
            real_output = d(real_images)
            fake_output = d(generated_image)

            # loss functions
            g_loss = generator_loss(fake_output)
            d_loss = discriminator_loss(real_output, fake_output)

        # compute gradients recorded on "tape"
        g_gradients = g_tape.gradient(g_loss, g.trainable_variables)
        d_gradients = d_tape.gradient(d_loss, d.trainable_variables)

        # apply gradients to model variables to minimize loss function
        g_optimizer.apply_gradients(zip(g_gradients, g.trainable_variables))
        d_optimizer.apply_gradients(zip(d_gradients, d.trainable_variables))

        generate_images(
            model=g,
            epoch=e,
            z_input=z_input,
            save_dir=save_dir
        )


################################################################################
# Main
if __name__ == "__main__":
    # print TF version
    print(f'TF version: {tf.__version__}')

    # create output directory for results
    output_dir = "results\\" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ----- ETL ----- #
    # ETL = Extraction, Transformation, Load
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # rescale images from [0, 255] to [-1, 1]
    train_images = (train_images.astype(np.float32) - 127.5) / 127.5
    test_images = (test_images.astype(np.float32) - 127.5) / 127.5

    # augment data set

    # create validation set
    midpoint = int(len(test_images) / 2)
    val_images = test_images[:midpoint]
    val_labels = test_labels[:midpoint]
    test_images = test_images[midpoint:]
    test_labels = test_labels[midpoint:]

    print(f'Shape of training images: {train_images.shape}')
    print(f'Shape of training labels: {train_labels.shape}')
    print(f'Shape of validation images: {val_images.shape}')
    print(f'Shape of validation labels: {val_labels.shape}')
    print(f'Shape of test images: {test_images.shape}')
    print(f'Shape of test labels: {test_labels.shape}')

    # ----- MODEL ----- #
    discriminator = Discriminator()
    discriminator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        beta_1=BETA_1
    )

    generator = Generator()
    generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        beta_1=BETA_1
    )

    # ----- TRAINING ----- #
    z_input_gen = tf.random.normal(shape=(BATCH_SIZE, NOISE_DIM))

    train(
        dataset=train_images,
        d=discriminator,
        g=generator,
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        z_input=z_input_gen,
        save_dir=output_dir
    )

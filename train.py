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
from datetime import datetime
import glob
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf

from parameters import *
from model import Discriminator


################################################################################
# Main
if __name__ == "__main__":
    # print TF version
    print(f'TF version: {tf.__version__}')

    # create output directory for results
    output_dir = "results\\" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

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

    #
    discriminator = Discriminator()

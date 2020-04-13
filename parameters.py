"""
Michael Patel
April 2020

Project description:
    Build a GAN using Cifar-10 data set

File description:
    For model and training parameters

CIFAR-10 classes:
1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck
"""
################################################################################
NUM_EPOCHS = 1
BATCH_SIZE = 16

LEARNING_RATE = 0.0002
BETA_1 = 0.5

LEAKY_RELU = 0.2  # default is 0.3
DROPOUT = 0.3

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
NUM_CHANNELS = 3

NUM_CLASSES = 10

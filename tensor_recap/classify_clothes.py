# import the necessary libraries

import tensorflow as tf
# secondary libraries

import numpy as np
import matplotlib.pyplot as plt
import os
import random
import cv2
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(f' Image Training Data: {train_images.shape}')
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)
print(train_labels)



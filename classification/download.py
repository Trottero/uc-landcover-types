# This script trains a ResNet50 model on the Eurosat dataset and saves it to disk for later use

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns

import tensorflow_datasets as tfds

devices = tf.config.list_physical_devices('GPU')
print(devices)


# Download tensorflow eurosat dataset
ds = tfds.load('eurosat/all', split='train', as_supervised=True)
assert isinstance(ds, tf.data.Dataset)
print(ds)

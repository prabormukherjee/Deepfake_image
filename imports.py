# Import Libraries

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython import display

import plot_utils
from gan import gan_compile
from train import train_dcgan

print('Tensorflow version:', tf.__version__)


batch_size = 32

num_features = 100

seed = tf.random.normal(shape=[batch_size, 100])
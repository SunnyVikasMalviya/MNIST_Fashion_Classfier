#Tensorflow and keras libraries and supporting libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


#importing mnist clothing dataset
fash = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fash.load_data()
class_names = ['T-shirt/Top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneakers', 'Bag', 'Ankle Boot'] 


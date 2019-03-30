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
'''
The dataset is loaded into 4 arrays of images and labels.
Images are 28*28 numpy arrays with pixel values ranging from  0 to 255.
Labels are arrays of integers ranging from 0 to 9 corresponding to the class
of clothing the images represent.
The class names are not included in the labels, hence a new list is created
that contains the class names of all the clothes.
'''

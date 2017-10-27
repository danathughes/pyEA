## create_mnist.py
##
## Script for creating a pickle file containing the MNIST dataset

import numpy as np
import cPickle as pickle

from tensorflow.examples.tutorials.mnist import input_data

# Load the mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Extract training, validation and test images
X_train = np.reshape(mnist.train.images, (55000, 28, 28, 1))
X_validate = np.reshape(mnist.validation.images, (5000, 28, 28, 1))
X_test = np.reshape(mnist.test.images, (10000, 28, 28, 1))

y_train = mnist.train.labels
y_validate = mnist.validation.labels
y_test = mnist.test.labels

# Build a dictionary with these values
dataset = {'train': (X_train, y_train),
           'validate': (X_validate, y_validate),
           'test': (X_test, y_test)}

# Save this to a pickle file
with open('mnist.pkl', 'wb') as pickle_file:
	pickle.dump(dataset, pickle_file)
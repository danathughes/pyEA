## create_cifar10.py
##
## Script for creating a pickle file containing the CIFAR-10 dataset.
##
## NOTE: Download the CIFAR-10 dataset from https://www.cs.toronto.edu/~kriz/cifar.html

import numpy as np
import cPickle as pickle

def convertData(batch):
	"""
	Reshape the data in the batch to a numpy array of shape (10000,32,32,3)
	"""

	data = np.zeros((10000,32,32,3), np.float32)

	# Extract each channel
	r = batch['data'][:,:1024]
	g = batch['data'][:,1024:2048]
	b = batch['data'][:,2048:]

	# Reshape and cast as floats, scale to 0.0--1.0
	r = np.reshape(r, (10000,32,32)).astype(np.float32) / 255.0
	g = np.reshape(g, (10000,32,32)).astype(np.float32) / 255.0
	b = np.reshape(b, (10000,32,32)).astype(np.float32) / 255.0

	# Make the images
	data[:,:,:,0] = r
	data[:,:,:,1] = g
	data[:,:,:,2] = b

	return data


def convertLabels(batch):
	"""
	Reshape the labels in the batch to a numpy array of shape (10000,10) [One-Hot]
	"""

	one_hot = np.zeros((10000,10), np.float32)

	for i in range(10000):
		one_hot[i,batch['labels'][i]] = 1.0

	return one_hot


# Load the datasets from the cifar-10 directory
with open('cifar-10-batches-py/data_batch_1', 'rb') as fo:
	data_batch_1=pickle.load(fo)
with open('cifar-10-batches-py/data_batch_2', 'rb') as fo:
	data_batch_2=pickle.load(fo)
with open('cifar-10-batches-py/data_batch_3', 'rb') as fo:
	data_batch_3=pickle.load(fo)
with open('cifar-10-batches-py/data_batch_4', 'rb') as fo:
	data_batch_4=pickle.load(fo)
with open('cifar-10-batches-py/data_batch_5', 'rb') as fo:
	data_batch_5=pickle.load(fo)
with open('cifar-10-batches-py/test_batch', 'rb') as fo:
	test_batch=pickle.load(fo)

train_batches = [data_batch_1, data_batch_2, data_batch_3, data_batch_4]

X_train_batches = [convertData(x) for x in train_batches]
y_train_batches = [convertLabels(x) for x in train_batches]

X_train = np.concatenate(X_train_batches, axis=0)
y_train = np.concatenate(y_train_batches, axis=0)

X_validate = convertData(data_batch_5)
y_validate = convertLabels(data_batch_5)

X_test = convertData(test_batch)
y_test = convertData(test_batch)

# Build a dictionary with these values
dataset = {'train': (X_train, y_train),
           'validate': (X_validate, y_validate),
           'test': (X_test, y_test)}

# Save this to a pickle file
with open('cifar_10.pkl', 'wb') as pickle_file:
	pickle.dump(dataset, pickle_file)
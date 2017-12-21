## eval_cnn.py
##
##

import tensorflow as tf
import cPickle as pickle
import numpy as np

import sys

from cnn_NSGAII import CNN_Individual

from sklearn.model_selection import KFold

BATCH_SIZE = 1000

def make_batches(X, y, batch_size=BATCH_SIZE, shuffle=True):
	"""
	Get some batches
	"""

	X_batches = []
	y_batches = []

	# Randomize the indices
	if shuffle:
		idx = np.random.permutation(len(X))
	else:
		idx = np.array(range(len(X)))

	cur_idx = 0

	while cur_idx < len(X):
		end_idx = min(cur_idx + batch_size, len(X))
		X_batches.append(X[idx[cur_idx:end_idx]])
		y_batches.append(y[idx[cur_idx:end_idx]])

		cur_idx += batch_size

	return X_batches, y_batches


class Models:
	"""
	"""

	def __init__(self, population):
		"""
		Create the actual models in tensorflow
		"""

		self.population = population

		# Create a tensorflow session
		self.sess = tf.InteractiveSession()

		# How big is the input and output?
		input_shape = population[0].input_shape
		output_size = population[0].output_size

		# Create a universal input and output
		self.input = tf.placeholder(tf.float32, (None,) + input_shape)
		self.target = tf.placeholder(tf.float32, (None, output_size))

		# Create an optimizer
		self.optimizer = tf.train.AdamOptimizer(0.01)

		# Collection of tensors
		self.outputs = []
		self.losses = []
		self.accuracies = []
		self.train_steps = []

		# Build all the models
		for i in range(len(self.population)):
			individual = self.population[i]
			namespace = 'Individual_%d' % i
			output_tensor, loss, accuracy, train_step = self.build_model(individual, namespace)

			self.outputs.append(output_tensor)
			self.losses.append(loss)
			self.accuracies.append(accuracy)
			self.train_steps.append(train_step)

		# Create tensorboard and initialize the variables
		self.tensorboard = tf.summary.FileWriter('./tensorboard', self.sess.graph)
		self.sess.run(tf.global_variables_initializer())


	def build_model(self, individual, namespace=""):
		"""
		Build the actual model
		"""

		# Build everything within the provided namespace
		with tf.variable_scope(namespace):
			input_tensor, output_tensor = individual.generate_model(self.input)

		loss = tf.losses.softmax_cross_entropy(self.target, output_tensor)

		target_label = tf.argmax(self.target, 1)
		pred_label = tf.argmax(output_tensor, 1)
		equality = tf.equal(target_label, pred_label)
		accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

		train_step = self.optimizer.minimize(loss)

		return output_tensor, loss, accuracy, train_step


	def train(self, X, y):
		"""
		Run a train step on all models
		"""

		x_batch, y_batch = make_batches(X, y)

		for _x, _y in zip(x_batch, y_batch):

			fd = {self.input: _x, self.target: _y}

			self.sess.run(self.train_steps, feed_dict=fd)


	def loss(self, X, y):
		"""
		Calculate the losses
		"""

		total_loss = [0.0] * len(self.population)

		x_batch, y_batch = make_batches(X,y)

		for _x, _y in zip(x_batch, y_batch):

			fd = {self.input: _x, self.target: _y}
			loss = self.sess.run(self.losses, feed_dict=fd)

			for i in range(len(total_loss)):
				total_loss[i] += float(len(_x)*loss[i]) / len(X)

		return total_loss


	def accuracy(self, X, y):
		"""
		Calculate the accuracies
		"""

		total_accuracy = [0.0] * len(self.population)

		x_batch, y_batch = make_batches(X,y)

		for _x, _y in zip(x_batch, y_batch):

			fd = {self.input: _x, self.target: _y}

			correct = self.sess.run(self.accuracies, feed_dict=fd)

			for i in range(len(total_accuracy)):
				total_accuracy[i] += float(len(_x) * correct[i]) / len(X)

		return total_accuracy


	def param_count(self):
		"""
		Calculate the number of parameters in each model
		"""

		param_counts = []

		for i in range(len(self.population)):
			# Get all the trainable variables in the namespace
			namespace = 'Individual_%d/' % i
			model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=namespace)

			# Get the shape of each variable
			var_shapes = [v.get_shape().as_list() for v in model_vars]

			# Count the total number of variables
			var_counts = [reduce(lambda x,y: x*y, v, 1) for v in var_shapes]
			total_vars = reduce(lambda x,y: x+y, var_counts, 0)

			param_counts.append(total_vars)

		return param_counts



if __name__ == '__main__':
	"""
	Extract population and data filenames from the command line, load both and evaluate
	"""

	if len(sys.argv) != 4:
		print "USAGE: python eval_cnn.py <population_path> <dataset_path> <output_path>"
		sys.exit(-1)

	population_path = sys.argv[1]
	data_path = sys.argv[2]
	output_path = sys.argv[3]

	# Load the population
	pickle_file = open(population_path)
	population = pickle.load(pickle_file)
	pickle_file.close()

	# Load the datafile
	pickle_file = open(data_path)
	dataset = pickle.load(pickle_file)
	pickle_file.close()

	X, y = dataset['train']
	X_valid, y_valid = dataset['validate']

	# Since we're doing cross-validation, merge the two sets
	X = np.concatenate([X,X_valid], axis=0)
	y = np.concatenate([y,y_valid], axis=0)

	# Create the tensorflow models
	models = Models(population)

	# Create a k-Fold object
	kf = KFold(n_splits=10, shuffle=True)

	fold_num = 1

	total_losses = [0.0] * len(population)
	total_accuracies = [0.0] * len(population)

	for train_idx, valid_idx in kf.split(X):
		train_X, train_y = X[train_idx], y[train_idx]
		valid_X, valid_y = X[valid_idx], y[valid_idx]

		print "Fold Number %d" % fold_num
		fold_num += 1

		for i in range(20):
			models.train(train_X, train_y)

		losses = models.loss(valid_X, valid_y)
		accuracies = models.accuracy(valid_X, valid_y)

		for i in range(len(population)):
			total_losses[i] += 0.1*losses[i]
			total_accuracies[i] += 0.1*accuracies[i]

	# All done training, get the objectives
	num_params = models.param_count()

	# Release resources
	models.sess.close()

	# Apply the objectives to each individual in the population
	objectives = []
	for i in range(len(population)):
		objectives.append([1.0 - total_accuracies[i], num_params[i]])

	# Save the population to the original file
	pickle_file = open(output_path, 'wb')
	pickle.dump(objectives, pickle_file)
	pickle_file.close()



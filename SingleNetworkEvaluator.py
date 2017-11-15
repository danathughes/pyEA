## SingleNetworkEvaluator.py
##
## Class to evaluate networks as they are added

import os
import cPickle as pickle

import random
import tensorflow as tf
import numpy as np

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


class SingleNetworkEvaluator:
	"""
	"""

	def __init__(self, dataset_filename, population_path='./population', train_steps=100):
		"""
		Create an object with the dataset loaded, and a path to store individuals and results
		"""

		self.population_path = population_path

		self.verbose = True

		self.individual_num = 0

		# Load the dataset
		if self.verbose:
			print "Loading dataset:", dataset_filename

		with open(dataset_filename) as pickle_file:
			self.dataset = pickle.load(pickle_file)

		# Input and target shapes
		self.train_x, self.train_y = self.dataset['train']
		self.validate_x, self.validate_y = self.dataset['validate']

		self.input_shape = self.train_x.shape[1:]
		self.target_shape = self.train_y.shape[1:]

		# Create a session
#		self.sess = tf.InteractiveSession()
		self.sess = None

		# Input and output tensors
#		self.input = tf.placeholder(tf.float32, (None,) + self.input_shape)
#		self.target = tf.placeholder(tf.float32, (None,) + self.target_shape)
		self.input = None
		self.target = None

		# Create an optimizer
#		self.optimizer = tf.train.AdamOptimizer(0.01)
		self.optimizer = None

		self.output = None
		self.loss = None
		self.accuracy = None
		self.train_step = None

		self.has_model = False


		self.num_train_steps = train_steps


	def __build_model(self, individual):
		"""
		Build the actual model
		"""

		try:
			input_tensor, self.output_tensor = individual.generate_model(self.input)
			self.loss = tf.losses.softmax_cross_entropy(self.target, self.output_tensor)

			target_label = tf.argmax(self.target, 1)
			pred_label = tf.argmax(self.output_tensor, 1)
			equality = tf.equal(target_label, pred_label)
			self.accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

			self.train_step = self.optimizer.minimize(self.loss)
			self.has_model = True

			if self.verbose:
				print "Model #%d Built" % self.individual_num

		except:
			if self.verbose:
				print "Couldn't create model!"
			self.output_tensor = None
			self.loss = None
			self.accuracy = None
			self.train_step = None
			self.has_model = False


	def __train(self, x, y):
		"""
		Run a train step on the model
		"""

		if not self.has_model:
			return


		for i in range(self.num_train_steps):
			# Make some batches
			x_batch, y_batch = make_batches(x, y)

			for _x, _y in zip(x_batch, y_batch):
				fd = {self.input: _x, self.target: _y}
				self.sess.run(self.train_step, feed_dict=fd)
		
		if self.verbose:
			loss, accuracy = self.__loss_and_accuracy(x,y)
			print "Loss: %f, Accuracy: %f" % (loss, accuracy)


	def __param_count(self):
		"""
		How many parameters in the model?
		"""

		# Get all the trainable variables in the namespace
		model_vars = tf.trainable_variables()

		# Get the shape of each variable
		var_shapes = [v.get_shape().as_list() for v in model_vars]

		# Count the total number of variables
		var_counts = [reduce(lambda x,y: x*y, v, 1) for v in var_shapes]
		total_vars = reduce(lambda x,y: x+y, var_counts, 0)

		return total_vars


	def __loss_and_accuracy(self, x, y):
		"""
		Calculate the accuracy of the current model
		"""

		if not self.has_model:
			return 1.0e9, 0.0


		total_loss = 0.0
		total_accuracy = 0.0

		x_batch, y_batch = make_batches(x,y)

		for _x, _y in zip(x_batch, y_batch):

			fd = {self.input: _x, self.target: _y}
			loss, correct = self.sess.run([self.loss, self.accuracy], feed_dict=fd)

			total_loss += float(len(_x) * loss) / len(x)
			total_accuracy += float(len(_x) * correct) / len(x)

		return total_loss, total_accuracy


	def add(self, individual):
		"""
		Evaluate the provided individual
		"""

		print "====================="
		if self.verbose:
			print individual
			print str(individual.gene)

		# Save the individual
		filename = self.population_path + '/individual_%d.pkl' % self.individual_num
		results_name = self.population_path + '/objectives_%d.pkl' % self.individual_num

		with open(filename, 'wb') as pickle_file:
			pickle.dump(str(individual.gene), pickle_file)

		# Delete whatever is in the current graph
		tf.reset_default_graph()
		self.sess = tf.InteractiveSession()

		self.input = tf.placeholder(tf.float32, (None,) + self.input_shape)
		self.target = tf.placeholder(tf.float32, (None,) + self.target_shape)
		self.optimizer = tf.train.AdamOptimizer(0.01)

		# Try to make the model
		self.has_model = False
	#	with tf.variable_scope('model'):
		self.__build_model(individual)
		self.sess.run(tf.global_variables_initializer())

		# Train the model
		self.__train(self.train_x, self.train_y)

		# Get the results
		loss, accuracy = self.__loss_and_accuracy(self.validate_x, self.validate_y)
		num_params = self.__param_count()

		# All done!
		self.sess.close()

		if self.verbose:
			print "Model #%d: Num Params: %d; Validation Loss: %f; Accuracy: %f" % (self.individual_num, num_params, loss, accuracy)

		# Save the results
		with open(results_name, 'wb') as pickle_file:
			pickle.dump([1.0 - accuracy, num_params], pickle_file)

		# Update the individual's objective
		individual.objective = [1.0 - accuracy, num_params]

		self.individual_num += 1


	def evaluate(self):
		"""
		Save the current set of individuals to a pickle file and call the evaluation program
		"""

		pass


	def reset(self):
		"""
		Empty the list of individuals to be evaluated
		"""

		pass
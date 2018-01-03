## SingleNetworkEvaluator.py
##
## Class to evaluate networks as they are added

import os
import cPickle as pickle

import random
import tensorflow as tf
import numpy as np

from sklearn.model_selection import KFold

BATCH_SIZE = 100
NUM_SPLITS = 3


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

	def __init__(self, dataset_filename, population_path='./population', train_steps=500, gpu_id='/device:GPU:0'):
		"""
		Create an object with the dataset loaded, and a path to store individuals and results
		"""

		self.population_path = population_path

		self.verbose = True

		self.individual_num = 0

		self.gpu_id = gpu_id

		# Load the dataset
		if self.verbose:
			print "Loading dataset:", dataset_filename

		with open(dataset_filename) as pickle_file:
			self.dataset = pickle.load(pickle_file)

		# Input and target shapes
		train_x, train_y = self.dataset['train']
		validate_x, validate_y = self.dataset['validate']

		self.X = np.concatenate([train_x, validate_x], axis=0)
		self.y = np.concatenate([train_y, validate_y], axis=0)

		self.kfold = KFold(n_splits=NUM_SPLITS, shuffle=True)

		self.input_shape = self.X.shape[1:]
		self.target_shape = self.y.shape[1:]

		# Create a session
		self.sess = None

		# Input and output tensors
		self.input = None
		self.target = None

		# Create an optimizer
		self.optimizer = None

		self.output = None
		self.loss = None
		self.accuracy = None
		self.train_step = None

		self.has_model = False

		self.num_train_steps = train_steps

		self.sess_config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=True)
#		self.sess_config = tf.ConfigProto(allow_soft_placement=True)
		self.sess_config.gpu_options.allocator_type='BFC'
		self.sess_config.gpu_options.per_process_gpu_memory_fraction = 0.20
		self.sess_config.gpu_options.allow_growth = True


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
				print "\tStep %d: Loss: %f, Accuracy: %f" % (i, loss, accuracy)
			else:
				print


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
			print "--------------------"

		# Save the individual
		filename = self.population_path + '/individual_%d.pkl' % self.individual_num
		results_filename = self.population_path + '/objectives_%d.pkl' % self.individual_num

		pickle_file = open(filename, 'wb')
		pickle.dump(individual.gene, pickle_file)
		pickle_file.close()

		# Delete whatever is in the current graph
		tf.reset_default_graph()
		self.sess = tf.Session(config = self.sess_config)

		with tf.device(self.gpu_id):
			self.input = tf.placeholder(tf.float32, (None,) + self.input_shape)
			self.target = tf.placeholder(tf.float32, (None,) + self.target_shape)
			self.optimizer = tf.train.AdamOptimizer(0.0001)

			# Try to make the model
			self.has_model = False

			self.__build_model(individual)
		self.sess.run(tf.global_variables_initializer())

		# Split the training data into 10 folds
		loss, accuracy = 0.0, 0.0

		fold_num = 1

		# Train the model
		for train_idx, valid_idx in self.kfold.split(self.X):
			print "  Fold %d: " % fold_num
			fold_num += 1

			train_x, train_y = self.X[train_idx], self.y[train_idx]
			valid_x, valid_y = self.X[valid_idx], self.y[valid_idx]

			self.__train(train_x, train_y)

			# Get the results
			fold_loss, fold_accuracy = self.__loss_and_accuracy(valid_x, valid_y)
			loss += float(fold_loss) / NUM_SPLITS
			accuracy += float(fold_accuracy) / NUM_SPLITS

			# Reset the parameters
			self.sess.run(tf.global_variables_initializer())

		num_params = self.__param_count()

		# All done!
		self.sess.close()

		if self.verbose:
			print "Model #%d: Num Params: %d; Validation Loss: %f; Accuracy: %f" % (self.individual_num, num_params, loss, accuracy)

		# Save the results
		pickle_file = open(results_filename, 'wb')
		pickle.dump([1.0 - accuracy, num_params], pickle_file)
		pickle_file.close()

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
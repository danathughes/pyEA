## SingleNetworkEvaluator.py
##
## Class to evaluate networks as they are added

import os
import cPickle as pickle

import random
import tensorflow as tf
import numpy as np

BATCH_SIZE = 100

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

	def __init__(self, dataset_filename, population_path='./population', gpu_id='/device:GPU:0', **kwargs):
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
		self.train_x, self.train_y = self.dataset['train']
		self.validate_x, self.validate_y = self.dataset['validate']

		self.input_shape = self.train_x.shape[1:]
		self.target_shape = self.train_y.shape[1:]

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

		self.sess_config = tf.ConfigProto(allow_soft_placement=False)
#		self.sess_config = tf.ConfigProto(allow_soft_placement=True)
		self.sess_config.gpu_options.allocator_type='BFC'
		self.sess_config.gpu_options.per_process_gpu_memory_fraction = 0.60
		self.sess_config.gpu_options.allow_growth = True

		# When to stop training
		self.max_train_steps = kwargs.get('max_train_steps', 5000)
		self.min_train_steps = kwargs.get('min_train_steps', 20)
		self.filter_lambda_1 = kwargs.get('filter_lambda_1', 0.05)
		self.filter_lambda_2 = kwargs.get('filter_lambda_2', 0.05)
		self.filter_lambda_3 = kwargs.get('filter_lambda_3', 0.05)

		self.R_crit = kwargs.get('R_crit', 1.5)
		self.num_R_crit = kwargs.get('num_R_crit', 10)

		self.X_prev = 0.0
		self.X_filter = 0.0
		self.var_est = 0.0
		self.var_est_data = 0.0


	def __variance_ratio(self, loss):
		"""
		Check if the training has converged

		Uses the formula from S. Natarajan and R.R. Rhinehart, "Automated Stopping Criteria for Neural Network Training"
		"""

		self.var_est = (0.05 * (loss - self.X_filter)**2) + (0.95 * self.var_est)
		self.X_filter = (0.05 * loss) + (0.95 * self.X_filter)
		self.var_est_data = (0.05 * (loss - self.X_prev)**2) + (0.95 * self.var_est_data)

		R = (2.0 - self.filter_lambda_1) * self.var_est / (self.var_est_data + 1.0e-8)

		self.X_prev = loss

		return R


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


	def __train(self, x, y, validation_x=None, validation_y=None):
		"""
		Run a train step on the model
		"""

	
		if not self.has_model:
			return

		# Reset steady state ID stuff
		self.X_prev = 0.0
		self.X_filter = 0.0
		self.var_est = 0.0
		self.var_est_data = 0.0
		R_crit_count = 0

		# Maintain a
		done = False
		i = 0

		while not done:
#		for i in range(self.max_train_steps):
			# Make some batches
			x_batch, y_batch = make_batches(x, y)

			for _x, _y in zip(x_batch, y_batch):
				fd = {self.input: _x, self.target: _y}
				self.sess.run(self.train_step, feed_dict=fd)

			# Check if the training is done
			i += 1
			loss, accuracy = self.__loss_and_accuracy(x,y)
			validation_loss, validation_accuracy = self.__loss_and_accuracy(validation_x, validation_y)
			R = self.__variance_ratio(validation_loss)

			# Is the loss stable yet?
			if R < self.R_crit:
				R_crit_count += 1
			else:
				R_crit_count = 0
			if R_crit_count > self.num_R_crit:
				done = True

			# Has enough training been done yet?
			if i < self.min_train_steps:
				done = False

			# Has the maximum amount of training been finished?
			if i > self.max_train_steps:
				done = True

			if self.verbose:
				print "\tStep %d: Loss: %f, Accuracy: %f, Validation Loss: %f, Validation Accuracy: %f, R: %f" % (i, loss, accuracy, validation_loss, validation_accuracy, R)
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


	def evaluate(self, individual):
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
		pickle.dump(individual.genotype, pickle_file)
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

		self.__train(self.train_x, self.train_y, self.validate_x, self.validate_y)

		# Get the results
		loss, accuracy = self.__loss_and_accuracy(self.validate_x, self.validate_y)
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

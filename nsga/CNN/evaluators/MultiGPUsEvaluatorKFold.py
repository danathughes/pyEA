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


class MultiGPUsEvaluatorKFold:
	"""
	"""

	def __init__(self, dataset_filename, num_models=1, population_path='./population', gpu_id='/device:GPU:0', **kwargs):
		"""
		Create an object with the dataset loaded, and a path to store individuals and results
		"""

		self.population_path = population_path
		self.num_models = num_models

		self.verbose = True

		self.model_num = 0
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

		self.num_folds = kwargs.get('num_folds', 5)

		self.kfold = KFold(n_splits=self.num_folds, shuffle=True)

		self.input_shape = self.X.shape[1:]
		self.target_shape = self.y.shape[1:]

		self.outputs = [None] * self.num_models
		self.losses = [None] * self.num_models
		self.accuracies = [None] * self.num_models
		self.train_steps = [None] * self.num_models

		self.individuals = [None] * self.num_models
		self.namespaces = [None] * self.num_models

		self.filenames = [None] * self.num_models
		self.results_filenames = [None] * self.num_models


		self.sess_config = tf.ConfigProto(allow_soft_placement=False)
#		self.sess_config = tf.ConfigProto(allow_soft_placement=True)
		self.sess_config.gpu_options.allocator_type='BFC'
		self.sess_config.gpu_options.per_process_gpu_memory_fraction = 0.60
		self.sess_config.gpu_options.allow_growth = True


		self.sess = tf.Session(config = self.sess_config)

		self.input = tf.placeholder(tf.float32, (None,) + self.input_shape)
		self.target = tf.placeholder(tf.float32, (None,) + self.target_shape)
		self.optimizer = tf.train.AdamOptimizer(0.0001)


		# When to stop training
		self.max_train_steps = kwargs.get('max_train_steps', 5000)
		self.min_train_steps = kwargs.get('min_train_steps', 20)
		self.filter_lambda_1 = kwargs.get('filter_lambda_1', 0.05)
		self.filter_lambda_2 = kwargs.get('filter_lambda_2', 0.05)
		self.filter_lambda_3 = kwargs.get('filter_lambda_3', 0.05)

		self.R_crit = kwargs.get('R_crit', 1.0)
		self.num_R_crit = kwargs.get('num_R_crit', 10)

		self.X_prev = [0.0] * self.num_models
		self.X_filter = [0.0] * self.num_models
		self.var_est = [0.0] * self.num_models
		self.var_est_data = [0.0] * self.num_models


	def __variance_ratio(self, losses):
		"""
		Check if the training has converged

		Uses the formula from S. Natarajan and R.R. Rhinehart, "Automated Stopping Criteria for Neural Network Training"
		"""

		R = [0.0] * self.num_models

		for i in range(self.num_models):

			self.var_est[i] = (0.05 * (losses[i] - self.X_filter[i])**2) + (0.95 * self.var_est[i])
			self.X_filter[i] = (0.05 * losses[i]) + (0.95 * self.X_filter[i])
			self.var_est_data[i] = (0.05 * (losses[i] - self.X_prev[i])**2) + (0.95 * self.var_est_data[i])

			R[i] = (2.0 - self.filter_lambda_1) * self.var_est[i] / (self.var_est_data[i] + 1.0e-8)

			self.X_prev[i] = losses[i]

		return R


	def __build_model(self, individual):
		"""
		Build the actual model
		"""

		namespace = 'Individual%d' % self.individual_num
		self.namespaces[self.model_num] = namespace

		if True:
#		try:
			with tf.variable_scope(namespace):
				input_tensor, output_tensor = individual.generate_model(self.input)

			loss = tf.losses.softmax_cross_entropy(self.target, output_tensor)

			target_label = tf.argmax(self.target, 1)
			pred_label = tf.argmax(output_tensor, 1)
			equality = tf.equal(target_label, pred_label)
			accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

			train_step = self.optimizer.minimize(loss)

			# Success, add this to the list of tensors / operators
			self.outputs[self.model_num] = output_tensor
			self.losses[self.model_num] = loss
			self.accuracies[self.model_num] = accuracy
			self.train_steps[self.model_num] = train_step

			self.individuals[self.model_num] = individual

			if self.verbose:
				print "Model #%d Built" % self.individual_num
			return True
		else:
#		except:
			if self.verbose:
				print "Couldn't create model!"
			return False


	def __train(self, x, y):
		"""
		Run a train step on the model
		"""

		# Reset steady state ID stuff
		self.X_prev = [0.0] * self.num_models
		self.X_filter = [0.0] * self.num_models
		self.var_est = [0.0] * self.num_models
		self.var_est_data = [0.0] * self.num_models
		R_crit_count = [0] * self.num_models

		# Maintain a
		done = False
		i = 0

		while not done:
#		for i in range(self.max_train_steps):
			# Make some batches
			x_batch, y_batch = make_batches(x, y)

			for _x, _y in zip(x_batch, y_batch):
				fd = {self.input: _x, self.target: _y}
				self.sess.run(self.train_steps, feed_dict=fd)

			# Check if the training is done
			i += 1
			loss, accuracy = self.__loss_and_accuracy(x,y)
			R = self.__variance_ratio(loss)

			done = True

			for j in range(self.num_models):
				# Is the loss stable yet?
				if R[j] < self.R_crit:
					R_crit_count[j] += 1
				else:
					R_crit_count[j] = 0

				if R_crit_count[j] < self.num_R_crit:
					done = False

			# Has enough training been done yet?
			if i < self.min_train_steps:
				done = False

			# Has the maximum amount of training been finished?
			if i > self.max_train_steps:
				done = True

#			if self.verbose:
#				print "\tStep %d: Loss: %f, Accuracy: %f, R: %f" % (i, loss, accuracy, R)
#			else:
#				print


	def __param_count(self):
		"""
		How many parameters in the model?
		"""

		total_vars = [0] * self.num_models

		for i in range(self.num_models):
			namespace = self.namespaces[i]

			# Get all the trainable variables in the namespace
			model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=namespace)

			# Get the shape of each variable
			var_shapes = [v.get_shape().as_list() for v in model_vars]

			# Count the total number of variables
			var_counts = [reduce(lambda x,y: x*y, v, 1) for v in var_shapes]
			total_vars[i] = reduce(lambda x,y: x+y, var_counts, 0)

		return total_vars


	def __loss_and_accuracy(self, x, y):
		"""
		Calculate the accuracy of the current model
		"""

		total_losses = [0.0] * self.num_models
		total_accuracy = [0.0] * self.num_models

		x_batch, y_batch = make_batches(x,y)

		for _x, _y in zip(x_batch, y_batch):

			fd = {self.input: _x, self.target: _y}
			results = self.sess.run(self.losses + self.accuracies, feed_dict=fd)

			for i in range(self.num_models):
				total_losses[i] += float(len(_x) * results[i]) / len(x)
				total_accuracy[i] += float(len(_x) * results[self.num_models + i]) / len(x)

		return total_losses, total_accuracy


	def add(self, individual):
		"""
		Evaluate the provided individual
		"""

		print "====================="
		if self.verbose:
			print individual
			print "--------------------"

		# Save the individual
		self.filenames[self.model_num] = self.population_path + '/individual_%d.pkl' % self.individual_num
		self.results_filenames[self.model_num] = self.population_path + '/objectives_%d.pkl' % self.individual_num

		pickle_file = open(self.filenames[self.model_num], 'wb')
		pickle.dump(individual.gene, pickle_file)
		pickle_file.close()


		# Build this particular model -- if successful, increment the model number
		device_gpu = '/device:GPU:%d' % self.model_num
		with tf.device(device_gpu):
			if self.__build_model(individual):
				self.model_num += 1
			else:
				pass

		self.individual_num += 1

		if self.model_num == self.num_models:
			self.evaluate()
			self.reset()


	def evaluate(self):
		"""
		Evaluate the individuals
		"""

		if self.verbose:
			print "===Evaluating==="


		# Split the training data into 10 folds
		model_loss = [0.0] * self.num_models
		model_accuracy = [0.0] * self.num_models

		fold_num = 1

		# Train the model
		for train_idx, valid_idx in self.kfold.split(self.X):
			print "  Fold %d: " % fold_num
			fold_num += 1

			train_x, train_y = self.X[train_idx], self.y[train_idx]
			valid_x, valid_y = self.X[valid_idx], self.y[valid_idx]


			self.sess.run(tf.global_variables_initializer())
			self.__train(train_x, train_y)

			# Get the results
			fold_losses, fold_accuracies = self.__loss_and_accuracy(valid_x, valid_y)

			for i in range(self.num_models):
				model_loss[i] += float(fold_losses[i]) / self.num_folds
				model_accuracy[i] += float(fold_accuracies[i]) / self.num_folds

		num_params = self.__param_count()

		# All done!
		self.sess.close()

		# Save the results
		for i in range(self.num_models):
			pickle_file = open(self.results_filenames[i], 'wb')
			pickle.dump([1.0 - model_accuracy[i], num_params[i]], pickle_file)
			pickle_file.close()

			# Update the individual's objective
			self.individuals[i].objective = [1.0 - model_accuracy[i], num_params[i]]


	def reset(self):
		"""
		Empty the list of individuals to be evaluated
		"""

		self.outputs = [None] * self.num_models
		self.losses = [None] * self.num_models
		self.accuracies = [None] * self.num_models
		self.train_steps = [None] * self.num_models

		self.individuals = [None] * self.num_models

		self.filenames = [None] * self.num_models
		self.results_filenames = [None] * self.num_models

		self.model_num = 0

		self.sess.close()
		tf.reset_default_graph()

		self.sess = tf.Session(config = self.sess_config)

		self.input = tf.placeholder(tf.float32, (None,) + self.input_shape)
		self.target = tf.placeholder(tf.float32, (None,) + self.target_shape)
		self.optimizer = tf.train.AdamOptimizer(0.0001)


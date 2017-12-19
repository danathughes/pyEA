## Gene.py
##
##

"""
1. What is the dimensionality for each gene type?

INPUT - [ m * n * k] - This is the case for a 2D gene (i.e., image)
		[ m * k ] - This is the case for a 1D gene (i.e., accelerometer)

2. Constraints on kernel_size, stripe or num_kernels when generating random Genes?

3. Can we sample on a Gaussion distribution to get the number of conv and pooling?
	Also number of fully connected?

"""

import random

from AbstractGene import *
from DummyGene import *


class FullyConnectedGene(AbstractGene):
	"""
	"""
	def __init__(self, size, **kwargs):
		"""
		size				- number of neurons (integer)
		activation_function - e.g., tf.sigmoid
		"""

		AbstractGene.__init__(self)


		self.size = size
		self.activation = kwargs.get('activation', tf.nn.relu)

		self.type = FULLY_CONNECTED
		self.dimension = size

		# Make sure that the diemsnionality is expressed as a tuple
		if type(size) == int or type(size) == float:
			self.dimension = (size,)

		# Mutation parameters - on average, add or subtract 10 units
		lambda_size = kwargs.get('lambda_size', 5)
		n_min_size = kwargs.get('n_min_size', 5)

		self.size_prob_params = (lambda_size, n_min_size)

		self.max_size = kwargs.get('max_size', 250)


	def clone(self):
		"""
		"""

		return FullyConnectedGene(self.size, activation=self.activation,
			                      lambda_size=self.size_prob_params[0], n_min_size=self.size_prob_params[1])


	def equals(self, other):
		"""
		Type and meta-parameters should all match
		"""

		isSame = other.type == self.type

		# If this is a fully connected unit, 
		if isSame:
			isSame = isSame and self.size == other.size
			isSame = isSame and self.activation == other.activation

		return isSame


	def canFollow(self, prevGene):
		"""
		A FullyConnectedGene can follow any of the other types of genes
		"""

		return True


	def outputDimension(self):
		"""
		The output dimensions is just the number of units
		"""

		return self.dimension


	def minInputDimension(self):
		"""
		Recurse through next gene to figure out minimum valid input size
		"""

		return (1,)


	def mutate(self):
		"""
		Modify either the size or the activation function
		"""

		# Pich a random mutation to perform
		mutation = np.random.choice([self._mutateSize, self._mutateActivation])

		return mutation()


	def _mutateSize(self):
		"""
		Change the number of units
		"""

		# How much should the size change?
		size_diff = self.__modifiedPoisson(self.size_prob_params)

		old_size = self.dimension[0]

		# Should the size or increase or decrease
		if old_size == 1 or np.random.random() < 0.5:
			new_size = old_size + size_diff
		else:
			new_size = old_size - size_diff

		new_size = max(new_size, 1)
		new_size = min(new_size, self.max_size)

		self.dimension = (new_size,)

		# Did this mutate?
		return new_size != old_size


	def _mutateActivation(self):
		"""
		Change the activation function
		"""

		new_activation = random.choice(ACTIVATION_FUNCTIONS)

		# Keep picking activations until something different occurs
		while self.activation == new_activation:
			new_activation = random.choice(ACTIVATION_FUNCTIONS)

		self.activation = new_activation

		# The previous while loop ensures mutation
		return True


	def __modifiedPoisson(self, prob_params):
		"""
		Sample from the modified Poisson distribution
		"""

		return np.random.poisson(prob_params[0]) + prob_params[1]		


	def generateLayer(self, input_tensor):
		"""
		The output is a Fully Connected layer.
		"""

		# Get the previous tensor
		input_shape = input_tensor.get_shape().as_list()[1:]

		# Is the input tensor flat?  If not, flatten
		if len(input_shape) > 1:
			input_tensor = tf.contrib.layers.flatten(input_tensor)
			input_shape = input_tensor.get_shape().as_list()[1:]

		# Create the weights and bias to the softmax layer
		weights = tf.Variable(tf.truncated_normal(tuple(input_shape) + self.dimension, stddev=0.05))
		bias = tf.Variable(tf.constant(0.0, shape=self.dimension))

		self.tensor = tf.nn.relu(tf.matmul(input_tensor, weights) + bias)

		return self.tensor


	def __str__(self):
		"""
		"""

		return "F.CONN:\tOutput Dimensions: " + str(self.outputDimension())


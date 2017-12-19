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

import tensorflow as tf
import numpy as np


class Conv1DGene(AbstractGene):
	"""
	"""
	def __init__(self, kernel_shape, stride, num_kernels, **kwargs):
		"""
		kernel_shape - should be a 1-tuple, e.g, (20,)
		stride	   - should be a 1-tuple, e.g, (2,)
		num_kernels  - should be an integer
		"""

		AbstractGene.__init__(self)


		self.kernel_shape = kernel_shape
		self.stride = stride
		self.num_kernels = num_kernels
		self.activation = kwargs.get('activation', tf.nn.relu)

		self.type = CONV1D
		self.dimension = None

		# What should the mutation parameters be (lambda, n_min)?

		# Add 2 to the shape on average (n_min=1, lambda=1)
		lambda_shape = kwargs.get('lambda_shape', 1)
		n_min_shape = kwargs.get('n_min_shape', 1)

		# Add 1 to the stride on average (n_min=1, lambda=0)
		lambda_stride = kwargs.get('lambda_stride', 0)
		n_min_stride = kwargs.get('n_min_stride', 1)

		# Add 5 on average to the number of kernels (n_min=1, lambda=4)
		lambda_kernels = kwargs.get('lambda_kernels', 4)
		n_min_kernels = kwargs.get('n_min_kernels', 1)

		self.shape_prob_params = (lambda_shape, n_min_shape)
		self.stride_prob_params = (lambda_stride, n_min_stride)
		self.kernel_prob_params = (lambda_kernels, n_min_kernels)


	def clone(self):
		"""
		"""

		return Conv1DGene(self.kernel_shape, self.stride, self.num_kernels, activation=self.activation, 
			              lambda_shape=self.shape_prob_params[0], n_min_shape=self.shape_prob_params[1],
			              lambda_stride=self.stride_prob_params[0], n_min_stride=self.stride_prob_params[1],
			              lambda_kernels=self.kernel_prob_params[0], n_min_kernels=self.kernel_prob_params[1])


	def equals(self, other):
		"""
		Type and meta-parameters should all match
		"""

		isSame = other.type == self.type

		# Check if the other parameters are also the same, if the type is the same
		if isSame:
			isSame = isSame and self.kernel_shape == other.kernel_shape
			isSame = isSame and self.stride == other.stride
			isSame = isSame and self.num_kernels == other.num_kernels
			isSame = isSame and self.activation == other.activation

			# NOTE:  Should mutation probabilities also be compared?

		return isSame


	def canFollow(self, prevGene):
		"""
		A Conv1Dgene can follow an 'InputGene' or an 'Pool1DGene'
		The constraints are kernel_size should not larger than prevGene.output_size
		"""

		# Is the previous gene a valid type?
		if not prevGene.type in [INPUT, CONV1D, POOL1D]:
			return False

		# Is the dimensionality 2? (length x channels)
		if len(prevGene.outputDimension()) != 2:
			return False

		# Get the output dimension of the previous gene
		prevLength, prevChannels  = prevGene.outputDimension()

		# Is the kernel larger than the previous length?
		if self.kernel_shape[0] > prevLength:
			return False

		# So far, no problem with dimensionality.  Check if further down the genotype is valid

		# Is there another gene down the line?
		if not self.next_gene:
			return False

		# What would the shape of the output be?
		out_length = (prevLength - self.kernel_shape[0]) / self.stride[0] + 1

		dummy = DummyGene((out_length, self.num_kernels))
		dummy.type = self.type

		return self.next_gene.canFollow(dummy)


	def outputDimension(self):
		"""
		Calculate the output dimension based on the input dimension, kernel_shape, and stride
		"""

		# Is this connected to some prior gene?
		prevLength, channels = self.prev_gene.outputDimension()
		myLength = (prevLength - self.kernel_shape[0]) / self.stride[0] + 1

		self.dimension = (myLength, self.num_kernels)
		return self.dimension


	def minInputDimension(self):
		"""
		Recurse through next gene to figure out minimum valid input size
		"""

		next_min_dimension = self.next_gene.minInputDimension()

		if len(next_min_dimension) == 1:
			# Next gene is Fully Connected or Output
			# next_min_dimension = (1,)
			min_dimension = (self.kernel_shape[0], 1)
		else:
			# Next gene is Conv1D or Pool1D
			# next_min_dimenaion = (12, 1)
			min_dimension = (self.stride[0]*(next_min_dimension[0] - 1) + self.kernel_shape[0], next_min_dimension[1])

		return min_dimension



	def mutate(self):
		"""
		Modify one of the parameters to make a new gene
		"""

		# Pick a random mutation to perform
		mutation = random.choice([self._mutateKernelShape, self._mutateStride, self._mutateNumKernels, self._mutateActivation])

		return mutation()


	def _mutateKernelShape(self):
		"""
		Change the shape of kernels
		"""

		# How much should the kernel shape change?
		width_diff = self.__modifiedPoisson(self.shape_prob_params)

		# What's the largest the kernel can be?
		if len(self.next_gene.minInputDimension()) == 1:			# Not a pool or convolution layer 
			min_output_size = 1
		else:
			min_output_size = self.next_gene.minInputDimension()[0]

		input_size = self.prev_gene.outputDimension()[0]

		# The kernel width cannot be so large as to be smaller than the minimum output size
		# Also, this covers the kernel not exceeding the input size:
		# given min_output_size=1, max_kernel_width = input_size
		max_kernel_width = np.floor( input_size - self.stride[0] * (min_output_size - 1) )

		# The kernel cannot be less than the stride (otherwise, the stride skips some data)
		min_kernel_width = self.stride[0]

		# Add or subtract to the current (old) kernel width
		old_kernel_width = self.kernel_shape[0]

		# If the kernel width is one, we have to add
		if old_kernel_width == min_kernel_width:
			new_kernel_width = old_kernel_width + width_diff
		# If the kernel width is the max width, then we must subtract
		elif old_kernel_width == max_kernel_width:
			new_kernel_width = old_kernel_width - width_diff
		# Otherwise, randomly add or subtract
		elif np.random.random() < 0.5:
			new_kernel_width = old_kernel_width + width_diff
		else:
			new_kernel_width = old_kernel_width - width_diff

		# Don't exceed the two limits
		new_kernel_width = min(new_kernel_width, max_kernel_width)
		new_kernel_width = max(new_kernel_width, min_kernel_width)

		self.kernel_shape = (int(new_kernel_width),)

		# Did it mutate?  Only if the new width is different than the old width
		return new_kernel_width != old_kernel_width


	def _mutateStride(self):
		"""
		Change the stride size
		"""

		# How much should the stride be changed?
		stride_diff = self.__modifiedPoisson(self.stride_prob_params)

		# What's the largest the stride can be
		if len(self.next_gene.minInputDimension()) == 1:			# Not a pool or convolution layer 
			min_output_size = 1
		else:
			min_output_size = self.next_gene.minInputDimension()[0]
			
		input_size = self.prev_gene.outputDimension()[0]

		# The stride can be at most the kernel shape
		max_stride = self.kernel_shape[0]

		# If the minimum output size is greater than 1, then there is an additional constraint on the stride
		if min_output_size > 1:
			max_stride = np.floor((input_size - self.kernel_shape[0]) / (min_output_size - 1))

		# The stride cannot be larger than the kernel size
		max_stride = min(max_stride, self.kernel_shape[0])

		# Add or subtract the stride?
		old_stride = self.stride[0]

		# If the stride is one, we have to add
		if old_stride == 1:
			new_stride = old_stride + stride_diff
		# If the stride is the max stride, then we must subtract
		elif old_stride == max_stride:
			new_stride = old_stride - stride_diff
		# Otherwise, randomly add or subtract
		elif np.random.random() < 0.5:
			new_stride = old_stride + stride_diff
		else:
			new_stride = old_stride - stride_diff

		# Don't exceed the limits
		new_stride = min(new_stride, max_stride)
		new_stride = max(new_stride, 1)

		self.stride = (int(new_stride),)

		# Did it mutate?
		return new_stride != old_stride


	def _mutateNumKernels(self):
		"""
		Change the number of kernels
		"""

		# How many kernels to add / subtract?
		kernel_diff = self.__modifiedPoisson(self.kernel_prob_params)

		# Should this be added to or substracted from the current number of kernels?
		if self.num_kernels == 1 or np.random.random() < 0.5:
			# Add
			self.num_kernels += kernel_diff
		else:
			# Substract
			self.num_kernels -= kernel_diff
			self.num_kernels = max(self.num_kernels, 1)       # Don't go below 1 kernel!

		self.num_kernels = int(self.num_kernels)

		# Regardless, it definitely mutated
		return True


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
		Create a 1D convolutional layer
		"""

		# Get the previous tensor shape
		input_shape = input_tensor.get_shape().as_list()[1:]

		# Create the convolution weights and bias
		filter_shape = (self.kernel_shape[0], input_shape[1], self.num_kernels)
		weights = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05))
		bias = tf.Variable(tf.constant(0.0, shape=(self.num_kernels,)))

		self.tensor = self.activation(tf.nn.conv1d(input_tensor, weights, self.stride[0], 'VALID') + bias)

		return self.tensor


	def __str__(self):
		"""
		"""

		return "CONV1D:\tKernel: " + str((self.kernel_shape) + (self.num_kernels,)) + ";\tStride: " + str(self.stride) + ";\tOutput Dimensions: " + str(self.outputDimension())


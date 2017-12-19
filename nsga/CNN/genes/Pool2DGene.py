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

from AbstractGene import *
from DummyGene import *

import random

class Pool2DGene(AbstractGene):
	"""
	"""
	def __init__(self, pool_shape, stride, **kwargs):
		"""
		pool_size	- should be a 2-tuple, e.g, (2,2)
		stride	   - should be a 2-tuple, e.g, (2,2)
		"""

		AbstractGene.__init__(self)

		self.pool_shape = pool_shape
		self.stride = stride

		self.type = POOL2D
		self.dimension = None

		# Mutation parameters
		lambda_shape = kwargs.get('lambda_shape', 0)
		n_min_shape = kwargs.get('n_min_shape', 1)

		lambda_stride = kwargs.get('lambda_stride', 0)
		n_min_stride = kwargs.get('n_min_stride', 1)

		self.shape_prob_params = (lambda_shape, n_min_shape)
		self.stride_prob_params = (lambda_stride, n_min_stride)


	def clone(self):
		"""
		"""

		return Pool2DGene(self.pool_shape, self.stride,
						  lambda_shape=self.shape_prob_params[0], n_min_shape=self.shape_prob_params[1],
						  lambda_stride=self.stride_prob_params[0], n_min_stride=self.shape_prob_params[1])


	def equals(self, other):
		"""
		Type and meta-parameters should all match
		"""

		# Are they the same type?
		isSame = other.type == self.type

		# Check if the other parameters are also the same
		if isSame:
			isSame = isSame and self.pool_shape == other.pool_shape
			isSame = isSame and self.stride == other.stride

		return isSame


	def canFollow(self, prevGene):
		"""
		A Conv1Dgene can follow an 'InputGene' or an 'Pool1DGene'
		The constraints are kernel_size should not larger than prevGene.output_size
		"""

		# Is the previous gene a valid type?
		if not prevGene.type in [INPUT, CONV2D, POOL2D]:
			return False

		# Is the dimensionality 2? (length x channels)
		if len(prevGene.outputDimension()) != 3:
			return False

		# Get the output dimension of the previous gene
		prevHeight, prevWidth, prevChannels  = prevGene.outputDimension()

		# Is the kernel larger than the previous length?
		if self.pool_shape[0] > prevHeight or self.pool_shape[1] > prevWidth:
			return False

		# So far, no problem with dimensionality.  Check if further down the genotype is valid

		# Is there another gene down the line?
		if not self.next_gene:
			return False

		# What would the shape of the output be?
		out_height = (prevHeight - self.pool_shape[0]) / self.stride[0] + 1
		out_width = (prevWidth - self.pool_shape[1]) / self.stride[1] + 1

		dummy = DummyGene((out_height, out_width, prevChannels))
		dummy.type = self.type

		return self.next_gene.canFollow(dummy)


	def outputDimension(self):
		"""
		Calculate the output dimension based on the input dimension, kernel_size, and stride
		"""

		prevHeight, prevWidth, channels = self.prev_gene.outputDimension()
		myHeight = (prevHeight - self.pool_shape[0]) / self.stride[0] + 1
		myWidth = (prevWidth - self.pool_shape[1]) / self.stride[1] + 1

		self.dimension = (myHeight, myWidth, channels)
		return self.dimension


	def minInputDimension(self):
		"""
		Recurse through next gene to figure out minimum valid input size
		"""

		next_min_dimension = self.next_gene.minInputDimension()
		if len(next_min_dimension) == 1:
			# Next gene is Fully Connected or Output
			# next_min_dimension = (1,)
			min_dimension = (self.pool_shape[0], self.pool_shape[1], 1)
		else:
			# Next gene is Conv1D or Pool1D
			# next_min_dimenaion = (12, 1)
			size1 = self.stride[0]*(next_min_dimension[0]-1) + self.pool_shape[0]
			size2 = self.stride[1]*(next_min_dimension[1]-1) + self.pool_shape[1]
			min_dimension = (size1, size2, next_min_dimension[2])

		return min_dimension


	def mutate(self):
		"""
		Modify one of the parameters to make a new gene
		"""

		# Pick a random mutation to perform
		mutation = random.choice([self._mutatePoolShape, self._mutateStride])

		return mutation()


	def _mutatePoolShape(self):
		"""
		Change the size of the pool
		"""

		# Pick a random pool dimension to mutate
		_dim = random.choice([0,1])

		# How much should the shape change?
		shape_diff = self.__modifiedPoisson(self.shape_prob_params)

		# What's the largest the pooling size can be?
		if len(self.next_gene.minInputDimension()) == 1:			# Not a pool or convolution layer 
			min_output_size = 1
		else:
			min_output_size = self.next_gene.minInputDimension()[_dim]

		input_size = self.prev_gene.outputDimension()[_dim]

		# The pool width cannot be so large as to be smaller than the minimum output size
		max_pool_size = np.floor( input_size - self.stride[_dim] * (min_output_size - 1))

		# If the other dimension is not 1, it's okay to have a pool size of 1, otherwise, it cannot be
		# less than 2
		if self.pool_shape[1 - _dim] > 1:
			min_pool_size = max(self.stride[_dim], 1)
		else:
			min_pool_size = max(self.stride[_dim], 2)

		# Add or subtract to the pool shape
		old_pool_size = self.pool_shape[_dim]

		# If the pool size is 2, we have to add
		if old_pool_size == min_pool_size:
			new_pool_size = old_pool_size + shape_diff
		elif old_pool_size == max_pool_size:
			new_pool_size = old_pool_size - shape_diff
		elif np.random.random() < 0.5:
			new_pool_size = old_pool_size + shape_diff
		else:
			new_pool_size = old_pool_size - shape_diff

		# Don't exceed the two limits
		new_pool_size = max(new_pool_size, min_pool_size)
		new_pool_size = min(new_pool_size, max_pool_size)

		self.pool_shape = list(self.pool_shape)
		self.pool_shape[_dim] = new_pool_size
		self.pool_shape = tuple(self.pool_shape)

		# Did it mutate?
		return new_pool_size != old_pool_size


	def _mutateStride(self):
		"""
		Change the stride size
		"""

		# Pick a random pool dimension to mutate
		_dim = random.choice([0,1])

		# How much should the stride be changes?
		stride_diff = self.__modifiedPoisson(self.stride_prob_params)

		# What's the largest the stride can be?
		if len(self.next_gene.minInputDimension()) == 1:			# Not a pool or convolution layer 
			min_output_size = 1
		else:
			min_output_size = self.next_gene.minInputDimension()[_dim]

		input_size = self.prev_gene.outputDimension()[_dim]

		# The stride cannot exceed the pool shape
		max_stride = self.pool_shape[_dim]

		# If the minimum output size is greater than 1, then there is an aditional constraint on the stride
		if min_output_size > 1:
			max_stride = np.floor( (input_size - self.pool_shape[_dim]) / (min_output_size - 1) )

		# The stride cannot be larger than the pool size
		max_stride = min(max_stride, self.pool_shape[_dim])

		# Add or subtract the stride?
		old_stride = self.stride[_dim]

		# If the stride is one, we need to add
		if old_stride == 1:
			new_stride = old_stride + stride_diff
		# If the stride is the max stride, then we must subtract
		elif old_stride == max_stride:
			new_stride = old_stride - stride_diff
			# Otherwise, just add or subtract at random
		elif np.random.random() < 0.5:
			new_stride = old_stride + stride_diff
		else:
			new_stride = old_stride - stride_diff

		# Constrain by the limits
		new_stride = min(new_stride, max_stride)
		new_stride = max(new_stride, 1)

		self.stride = list(self.stride)
		self.stride[_dim] = new_stride
		self.stride = tuple(self.stride)

		# Did it mutate?
		return new_stride != old_stride


	def __modifiedPoisson(self, prob_params):
		"""
		Sample from the modified Poisson distribution
		"""

		return np.random.poisson(prob_params[0]) + prob_params[1]


	def generateLayer(self, input_tensor):
		"""
		Create a 1D pooling layer
		"""

		self.tensor = tf.layers.max_pooling2d(input_tensor, self.pool_shape, self.stride)

		return self.tensor


	def __str__(self):
		"""
		"""

		return "POOL2D:\tPool Shape: " + str(self.pool_shape) + ";\tStride: " + str(self.stride) + ";\tOutput Dimensions: " + str(self.outputDimension())


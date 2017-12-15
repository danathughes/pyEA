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


class Pool1DGene(AbstractGene):
	"""
	"""
	def __init__(self, pool_shape, stride):
		"""
		pool_size	- should be a 1-tuple, e.g, (2,)
		stride	   - should be a 1-tuple, e.g, (2,)
		"""

		AbstractGene.__init__(self)


		self.pool_shape = pool_shape
		self.stride = stride

		self.type = POOL1D
		self.dimension = None

	def clone(self):
		"""
		"""

		return Pool1DGene(self.pool_shape, self.stride)

	def equals(self, other):
		"""
		Type and meta-parameters should all match
		"""
		if other.type != POOL1D:
			return False

		return ( self.pool_shape == other.pool_shape and
	 			(self.stride == other.stride) )

	def canFollow(self, prevGene):
		"""
		A Conv1Dgene can follow an 'InputGene' or an 'Pool1DGene'
		The constraints are kernel_size should not larger than prevGene.output_size
		"""

		# Is the previous gene a valid type?
		if not prevGene.type in [INPUT, CONV1D]:
			return False

		# Is the dimensionality 2? (length x channels)
		if len(prevGene.outputDimension()) != 2:
			return False

		# Get the output dimension of the previous gene
		prevLength, prevChannels  = prevGene.outputDimension()

		# Is the kernel larger than the previous length?
		if self.pool_shape[0] > prevLength:
			return False

		# So far, no problem with dimensionality.  Check if further down the genotype is valid

		# Is there another gene down the line?
		if not self.next_gene:
			return False

		# What would the shape of the output be?
		out_length = (prevLength - self.pool_shape[0]) / self.stride[0] + 1

		dummy = DummyGene((out_length, prevChannels))
		dummy.type = self.type

		return self.next_gene.canFollow(dummy)


	def outputDimension(self):
		"""
		Calculate the output dimension based on the input dimension, kernel_size, and stride
		"""

		# Is this connected to some prior gene?
		prevLength, channels = self.prev_gene.outputDimension()
		myLength = (prevLength - self.pool_shape[0]) / self.stride[0] + 1

		self.dimension = (myLength, channels)
		return self.dimension


	def minInputDimension(self):
		"""
		Recurse through next gene to figure out minimum valid input size
		"""

		next_min_dimension = self.next_gene.minInputDimension()
		if len(next_min_dimension) == 1:
			# Next gene is Fully Connected or Output
			# next_min_dimension = (1,)
			min_dimension = (self.pool_shape[0], 1)
		else:
			# Next gene is Conv1D or Pool1D
			# next_min_dimenaion = (12, 1)
			min_dimension = (self.stride[0]*(next_min_dimension[0] - 1)
				+ self.pool_shape[0], next_min_dimension[1])

		return min_dimension


	def mutate(self):
		"""
		kernel_size, stride should be mutated based on the constraints from prevGene and nextGene

		constraints:
			kernel_shape = (20,)
			stride = (2,)
		"""

		# keep the original values
		size = self.pool_shape[0]
		stride = self.stride[0]

		#
		size_pre_out = self.prev_gene.outputDimension()[0]
		size_next_in = self.next_gene.minInputDimension()[0]

		# Mutate the kernel
		min_pool_size = MIN_POOL_SIZE
		max_pool_size = size_pre_out-(size_next_in-1)*stride

		min_stride_size = MIN_POOL_STRIDE
		if size_next_in > 1:
			temp = int((size_pre_out-size)/(size_next_in-1))
			max_stride_size = temp if temp<size else size
		else:
			max_stride_size = size

		# What to mutate
		mutation = random.choice(['kernel_size', 'stride_size'])

		"""
		1. Make a list of possible values for a variable, list_values
		2. Sample values on a distribution and make a list by normalizing them, list_probs
		3. t_value = random.choice(list_values, list_probs)
		"""
		if mutation == 'kernel_size':
			size_list = list(range(min_pool_size, max_pool_size+1))
			if size in size_list:
				size_list.remove(size)
			if len(size_list) > 0:
				self.pool_shape = (random.choice(size_list), )
			else:
				mutation = 'stride_size'
		else:
			stride_list = list(range(min_stride_size, max_stride_size+1))
			if stride in stride_list:
				stride_list.remove(stride)
			if len(stride_list) > 0:
				self.stride = (random.choice(stride_list), )
			else:
				return False

		# after some change, check validity of the new gene
		if self.canFollow(self.prev_gene):
			return True
		else:
			self.pool_shape = (size,)
			self.stride = (stride,)
			self.num_kernels = num_kernels
			return False


	def generateLayer(self, input_tensor):
		"""
		Create a 1D pooling layer
		"""

		self.tensor = tf.layers.max_pooling1d(input_tensor, self.pool_shape, self.stride)

		return self.tensor


	def __str__(self):
		"""
		"""

		return "POOL1D:\tPool Shape: " + str(self.pool_shape) + ";\tStride: " + str(self.stride) + ";\tOutput Dimensions: " + str(self.outputDimension())


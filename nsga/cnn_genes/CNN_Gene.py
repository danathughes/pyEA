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


# Enumerate the different gene types
INPUT = "INPUT"
CONV1D = "CONV1D"
CONV2D = "CONV2D"
POOL1D = "POOL1D"
POOL2D = "POOL2D"
FULLY_CONNECTED = "FULLYCONNECTED"
OUTPUT = "OUTPUT"

import random
import numpy as np
#from models.parts import *

import tensorflow as tf


MIN_CNN_WIDTH = 1
MAX_CNN_WIDTH = 28
MIN_CNN_KERNELS = 5
MAX_CNN_KERNELS = 100
MIN_CNN_STRIDE = 1
MAX_CNN_STRIDE = 28
MIN_POOL_SIZE = 2
MAX_POOL_SIZE = 28
MIN_POOL_STRIDE = 1
MAX_POOL_STRIDE = 28
MIN_FULL_CONNECTION = 5
MAX_FULL_CONNECTION = 1000

MUTATE_ADD_PROB = 0.5
MUTATE_REMOVE_PROB = 0.5
MUTATE_MODIFY_PROB = 0.8





class Gene:
	"""
	An abstract gene.
	"""

	def __init__(self):
		"""
		Create a new gene.  This shouldn't do anything for this abstract class
		"""


		# What type of gene is this?  Since this is abstract, it isn't anything
		self.type = None

		self.prev_gene = None
		self.next_gene = None

		self.tensor = None


	def canFollow(self, prevGene):
		"""
		Can this gene follow the previous gene?  I.e., are all constraints satisfied?
		"""

		pass


	#### Why was outputDimension hidden (two underlines makes it 'private' in Python)?  --Dana
	def outputDimension(self, prevGene):
		"""
		What is the dimensionality of the output of this gene?
		"""

		return None


	def minInputDimension(self):
		"""
		Recurse through next gene to figure out minimum valid input size
		"""

		pass


	def clone(self):
		"""
		Make a copy of this gene
		"""

		pass

	def equals(self, other):
		"""
		If two genes have the same type and data
		"""

		pass

	def mutate(self):
		"""
		Alter this gene, ensuring that the constraints from the previous and next gene are satisfied
		"""

		pass


	def generateLayer(self, input_tensor):
		"""
		Create the CNN part(s) (tuple of objects) used to construct this particular layer in the CNN
		"""

		pass

	def __str__(self):
		"""
		String representation of this gene
		"""

		return "Abstract Gene"


	def equals(self, other):
		"""
		Is this gene the same as another gene?
		"""

		return False


class InputGene(Gene):
	"""
	"""
	def __init__(self, input_shape):
		"""
		Placeholder gene for the input dimensionality of the problem set
		input_shape = (height, width, num_channels) (2D data)
						  (length, num_channels) (1D data)
		"""

		Gene.__init__(self)

		self.dimension = input_shape
		self.type = INPUT

		self.prev_gene = None
		self.next_gene = None


	def clone(self):
		"""
		"""

		return InputGene(self.dimension)

	def equals(self, other):
		"""
		The input gene is the same if the other is an INPUT gene type, and
		the dimensionality is the same
		"""

		if other.type != INPUT:
			return False

		return self.dimension == other.dimension

	def canFollow(self, prevGene=None):
		"""
		This never follows a gene, it's the input
		"""

		return False


	def outputDimension(self):
		"""
		"""

		return self.dimension

	def minInputDimension(self):
		"""
		Recurse through next gene to figure out minimum valid input size
		"""
		return None

	def mutate(self):
		"""
		"""

		return False 		# This cannot mutate!

	def generateLayer(self, input_tensor=None):
		"""
		Input layer is simply a placeholder tensor, with dimensionality given
		"""

		if input_tensor is not None:
			self.tensor = input_tensor
		else:
			self.tensor = tf.placeholder(tf.float32, (None,) + self.dimension)

		return self.tensor


	def __str__(self):
		"""
		"""

		return "INPUT: \tOutput Dimensions: " + str(self.outputDimension())


class OutputGene(Gene):
	"""
	"""

	def __init__(self, output_size):
		"""
		Placeholder for an output gene
		"""

		Gene.__init__(self)

		self.dimension = output_size
		self.type = OUTPUT


	def clone(self):
		"""
		"""

		return OutputGene(self.dimension)

	def equals(self, other):
		"""
		The output gene is the same if the other is an OUTPUT gene type, and
		the dimensionality is the same
		"""

		if other.type != OUTPUT:
			return False

		return self.dimension == other.dimension

	def canFollow(self, prevGene):
		"""
		"""

		# Don't let this follow another output gene
		return prevGene.type != OUTPUT

	def outputDimension(self, prevGene=None):
		"""
		"""

		return self.dimension

	def minInputDimension(self):
		"""
		Recurse through next gene to figure out minimum valid input size
		"""

		return (1,)

	def mutate(self):
		"""
		"""

		return False       # This cannot mutate


	def generateLayer(self, input_tensor):
		"""
		The output is a Softmax layer (for now).  Need to make different outputs?
		"""

		# Is the input tensor flat?  If not, flatten
		input_shape = input_tensor.get_shape().as_list()[1:]

		if len(input_shape) > 1:
			input_tensor = tf.contrib.layers.flatten(input_tensor)
			input_shape = input_tensor.get_shape().as_list()[1:]

		# Create the weights and bias to the softmax layer
		weights = tf.Variable(tf.truncated_normal((input_shape[0], self.dimension), stddev=0.05))
		bias = tf.Variable(tf.constant(0.0, shape=(self.dimension,)))

		self.tensor = tf.nn.softmax(tf.matmul(input_tensor, weights) + bias)

		return self.tensor

	def __str__(self):
		"""
		"""

		return "OUTPUT:\tOutput Dimensions: " + str(self.outputDimension())


class DummyGene(Gene):
	"""
	A Gene used just to propagate a dimensionality through a genotype
	"""

	def __init__(self, shape):

		Gene.__init__(self)

		self.shape = shape
		self.type = INPUT

	def clone(self):
		dummy =  DummyGene(self.shape)
		dummy.type = self.type

		return dummy

	def equals(self, other):
		"""
		"""

		return False

	def canFollow(self, prevGene):
		pass

	def outputDimension(self):
		return self.shape

	def minInputDimension(self):
		"""
		Recurse through next gene to figure out minimum valid input size
		"""

	def mutate(self):
		pass

	def __str__(self):
		"""
		"""

		return "DUMMY: \tOutput Dimensions: " + str(self.outputDimension())


class Conv1DGene(Gene):
	"""
	"""
	def __init__(self, kernel_shape, stride, num_kernels, activation_function, **kwargs):
		"""
		kernel_shape - should be a 1-tuple, e.g, (20,)
		stride	   - should be a 1-tuple, e.g, (2,)
		num_kernels  - should be an integer
		activation_function - a Tensorflow activation tensor (e.g., tf.sigmoid)
		"""

		Gene.__init__(self)


		self.kernel_shape = kernel_shape
		self.stride = stride
		self.num_kernels = num_kernels
		self.activation = activation_function

		self.type = CONV1D

		# What should the mutation parameters be (lambda, n_min)?

		# Add 2 to the shape on average (n_min=1, lambda=1)
		lambda_shape = kwargs.get('lambda_size', 1)
		n_min_shape = kwargs.get('n_min_shape', 1)

		# Add 1 to the stride on average (n_min=1, lambda=0)
		lambda_stride = kwargs.get('lambda_stride', 0)
		n_min_stride = kwargs.get('n_min_stride', 1)

		self.shape_prob_params = (lambda_shape, n_min_shape)
		self.stride_prob_params = (lambda_stride, n_min_stride)


	def clone(self):
		"""
		"""

		return Conv1DGene(self.kernel_shape, self.stride, self.num_kernels, self.activation, 
			              lambda_shape=self.shape_prob_params[0], n_min_shape=self.shape_prob_params[1],
			              lambda_stride=self.stride_prob_params[0], n_min_stride=self.shape_prob_params[1])


	def equals(self, other):
		"""
		Type and meta-parameters should all match
		"""

		if other.type != CONV1D:
			return False

		return ((self.kernel_shape == other.kernel_shape) and
	 			(self.stride == other.stride) and
				(self.num_kernels == other.num_kernels) and
				(self.activation == other.activation))

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
		kernel_shape, stride and num_kernels should be mutated based on the constraints from
		self.prev_gene and self.next_gene which are two members of the genotype.

		constraints:
			kernel_shape = (20,)
			stride = (2,)
			num_kernels = 8
		"""

		# keep the original values
		size = self.kernel_shape[0]
		stride = self.stride[0]
		num_kernels = self.num_kernels

		#
		size_pre_out = self.prev_gene.outputDimension()[0]
		size_next_in = self.next_gene.minInputDimension()[0]

		# Mutate the kernel
		min_kernel_size = MIN_CNN_WIDTH
		max_kernel_size = size_pre_out-(size_next_in-1)*stride

		min_stride_size = MIN_CNN_STRIDE
		if size_next_in > 1:
			temp = int((size_pre_out-size)/(size_next_in-1))
			max_stride_size = temp if temp<size else size
		else:
			max_stride_size = size

		# What to mutate
		mutation = random.choice(['kernel_size', 'stride_size', 'num_kernels'])

		"""
		1. Make a list of possible values for a variable, list_values
		2. Sample values on a distribution and make a list by normalizing them, list_probs
		3. t_value = random.choice(list_values, list_probs)
		"""
		if mutation == 'kernel_size':
			size_list = list(range(min_kernel_size, max_kernel_size+1))
			if size in size_list:
				size_list.remove(size)
			if len(size_list) > 0:
				self.kernel_shape = (random.choice(size_list), )
			else:
				mutation = 'stride_size'
		elif mutation == 'stride_size':
			stride_list = list(range(min_stride_size, max_stride_size+1))
			if stride in stride_list:
				stride_list.remove(stride)
			if len(stride_list) > 0:
				self.stride = (random.choice(stride_list), )
			else:
				mutation = 'num_kernels'
		else: # mutation == 'num_kernels'
			factor = 0.5
			min_size = int(num_kernels*(1-factor))
			min_size = 1 if min_size<1 else min_size
			max_size = int(num_kernels*(1+factor))

			size_list = list(range(min_size, max_size+1))
			if num_kernels in size_list:
				size_list.remove(num_kernels)
			if len(size_list)>4:         # What is with the number 4?  Comments would be nice
				self.num_kernels = random.choice(size_list)
			else:
				self.num_kernels = np.random.randint(MIN_CNN_KERNELS, MAX_CNN_KERNELS+1)

		# after some change, check validity of the new gene
		if self.canFollow(self.prev_gene):
			return True
		else:
			self.kernel_shape = (size,)
			self.stride = (stride,)
			self.num_kernels = num_kernels
			return False


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

		self.tensor = tf.nn.relu(tf.nn.conv1d(input_tensor, weights, self.stride[0], 'VALID') + bias)

		return self.tensor


	def __str__(self):
		"""
		"""

		return "CONV1D:\tKernel: " + str((self.kernel_shape) + (self.num_kernels,)) + ";\tStride: " + str(self.stride) + ";\tOutput Dimensions: " + str(self.outputDimension())


class Conv2DGene(Gene):
	"""
	"""
	def __init__(self, kernel_shape, stride, num_kernels, activation_function):
		"""
		kernel_shape - should be a 2-tuple, e.g, (20,20)
		stride	   - should be a 2-tuple, e.g, (2,2)
		num_kernels  - should be an integer
		activation_function - a Tensorflow activation tensor (e.g., tf.sigmoid)
		"""

		Gene.__init__(self)

		self.kernel_shape = kernel_shape
		self.stride = stride
		self.num_kernels = num_kernels
		self.activation = activation_function

		self.type = CONV2D
		self.dimension = None


	def clone(self):
		"""
		"""

		return Conv2DGene(self.kernel_shape, self.stride, self.num_kernels, self.activation)

	def equals(self, other):
		"""
		Type and meta-parameters should all match
		"""

		if other.type != CONV2D:
			return False

		return ( (self.kernel_shape == other.kernel_shape) and
	 			(self.stride == other.stride) and
				(self.num_kernels == other.num_kernels) and
				(self.activation == other.activation) )

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
		if self.kernel_shape[0] > prevHeight or self.kernel_shape[1] > prevWidth:
			return False

		# So far, no problem with dimensionality.  Check if further down the genotype is valid

		# Is there another gene down the line?
		if not self.next_gene:
			return False

		# What would the shape of the output be?
		out_height = (prevHeight - self.kernel_shape[0]) / self.stride[0] + 1
		out_width = (prevWidth - self.kernel_shape[1]) / self.stride[1] + 1

		dummy = DummyGene((out_height, out_width, self.num_kernels))
		dummy.type = self.type

		return self.next_gene.canFollow(dummy)


	def outputDimension(self):
		"""
		Calculate the output dimension based on the input dimension, kernel_size, and stride
		"""

		prevHeight, prevWidth, channels = self.prev_gene.outputDimension()
		myHeight = (prevHeight - self.kernel_shape[0]) / self.stride[0] + 1
		myWidth = (prevWidth - self.kernel_shape[1]) / self.stride[1] + 1

		self.dimension = (myHeight, myWidth, self.num_kernels)
		return self.dimension

	def minInputDimension(self):
		"""
		Recurse through next gene to figure out minimum valid input size
		"""

		next_min_dimension = self.next_gene.minInputDimension()
		if len(next_min_dimension) == 1:
			# Next gene is Fully Connected or Output
			# next_min_dimension = (1,)
			min_dimension = (self.kernel_shape[0], self.kernel_shape[1], 1)
		else:
			# Next gene is Conv1D or Pool1D
			# next_min_dimenaion = (12, 1)
			size1 = self.stride[0]*(next_min_dimension[0] - 1) + self.kernel_shape[0]
			size2 = self.stride[1]*(next_min_dimension[1] - 1) + self.kernel_shape[1]
			min_dimension = (size1, size2, next_min_dimension[2])

		return min_dimension

	def mutate(self):
		"""
		kernel_shape, stride and num_kernels should be mutated based on the constraints from
		self.prev_gene and self.next_gene which are two members of the genotype.

		constraints:
			kernel_shape = (20,)
			stride = (2,)
			num_kernels = 8
		"""

		# keep the original values
		size = self.kernel_shape
		stride = self.stride
		num_kernels = self.num_kernels

		t_size = list(size)
		t_stride = list(stride)
		t_num_kernels = num_kernels

		#
		size_pre_out = self.prev_gene.outputDimension()[0:2]

		next_gene_minIn = self.next_gene.minInputDimension()
		if len(next_gene_minIn) == 1:
			# next_gene is a FC or Output
			size_next_in = (1, 1)
		else:
			size_next_in = next_gene_minIn[0:2]

		# Mutate the kernel
		min_kernel_size = [MIN_CNN_WIDTH, MIN_CNN_WIDTH]
		max_kernel_size = [size_pre_out[0]-(size_next_in[0]-1)*stride[0],
					 size_pre_out[1]-(size_next_in[1]-1)*stride[1]]

		min_stride_size = [MIN_CNN_STRIDE, MIN_CNN_STRIDE]
		max_stride_size = [MIN_CNN_STRIDE, MIN_CNN_STRIDE]
		for i in [0, 1]:
			if size_next_in[i] > 1:
				temp = int((size_pre_out[i]-size[i])/(size_next_in[i]-1))
				max_stride_size[i] = temp if temp<size[i] else size[i]
			else:
				max_stride_size[i] = size[i]

		# What to mutate
		mutation = random.choice(['kernel_size', 'stride_size', 'num_kernels'])

		"""
		1. Make a list of possible values for a variable, list_values
		2. Sample values on a distribution and make a list by normalizing them, list_probs
		3. t_value = random.choice(list_values, list_probs)
		"""
		if mutation == 'kernel_size':
			for i in [0, 1]:
				size_list = list(range(min_kernel_size[i], max_kernel_size[i]+1))
				if size[i] in size_list:
					size_list.remove(size[i])
				if len(size_list) > 0:
					t_size[i] = random.choice(size_list)
				else:
					mutation = 'stride_size'
					break
		elif mutation == 'stride_size':
			for i in [0, 1]:
				stride_list = list(range(min_stride_size[i], max_stride_size[i]+1))
				if stride[i] in stride_list:
					stride_list.remove(stride[i])
				if len(stride_list) > 0:
					t_stride[i] = random.choice(stride_list)
				else:
					mutation = 'num_kernels'
					break
		else: # mutation == 'num_kernels'
			factor = 0.5
			min_size = int(num_kernels*(1-factor))
			min_size = 1 if min_size<1 else min_size
			max_size = int(num_kernels*(1+factor))

			size_list = list(range(min_size, max_size+1))
			if num_kernels in size_list:
				size_list.remove(num_kernels)

			if len(size_list)>4:
				t_num_kernels = random.choice(size_list)
			else:
				t_num_kernels = np.random.randint(MIN_CNN_KERNELS, MAX_CNN_KERNELS+1)

		if mutation == 'kernel_size':
			self.kernel_shape = tuple(t_size)
		elif mutation == 'stride_size':
			self.stride = tuple(t_stride)
		elif mutation == 'num_kernels':
			self.num_kernels = t_num_kernels
		# after some change, check validity of the new gene
		if self.canFollow(self.prev_gene):
			return True
		else:
			self.kernel_shape = size
			self.stride = stride
			self.num_kernels = num_kernels
			return False


	def generateLayer(self, input_tensor):
		"""
		Create a 2D convolutional layer
		"""

		# Get the previous tensor shape
		input_shape = input_tensor.get_shape().as_list()[1:]

		# Create the convolution weights and bias
		filter_shape = self.kernel_shape + (input_shape[2], self.num_kernels)
		weights = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05))
		bias = tf.Variable(tf.constant(0.0, shape=(self.num_kernels,)))

		try:
			self.tensor = tf.nn.relu(tf.nn.conv2d(input_tensor, weights, (1,) + self.stride + (1,), 'VALID') + bias)
		except:
			print "Error!"
			print "Input Shape:", input_tensor.get_shape()
			print "Kernel Shape:", self.kernel_shape
			print "Num Kernels:", self.num_kernels


		return self.tensor


	def __str__(self):
		"""
		"""

		return "CONV2D:\tKernel: " + str((self.kernel_shape) + (self.num_kernels,)) + ";\tStride: " + str(self.stride) + ";\tOutput Dimensions: " + str(self.outputDimension())


class Pool1DGene(Gene):
	"""
	"""
	def __init__(self, pool_shape, stride):
		"""
		pool_size	- should be a 1-tuple, e.g, (2,)
		stride	   - should be a 1-tuple, e.g, (2,)
		"""

		Gene.__init__(self)


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


class Pool2DGene(Gene):
	"""
	"""
	def __init__(self, pool_shape, stride):
		"""
		pool_size	- should be a 2-tuple, e.g, (2,2)
		stride	   - should be a 2-tuple, e.g, (2,2)
		"""

		Gene.__init__(self)

		self.pool_shape = pool_shape
		self.stride = stride

		self.type = POOL2D
		self.dimension = None


	def clone(self):
		"""
		"""

		return Pool2DGene(self.pool_shape, self.stride)

	def equals(self, other):
		"""
		Type and meta-parameters should all match
		"""
		if other.type != POOL2D:
			return False

		return ( self.pool_shape == other.pool_shape and
	 			(self.stride == other.stride) )

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
		kernel_shape, stride and num_kernels should be mutated based on the constraints from
		self.prev_gene and self.next_gene which are two members of the genotype.

		constraints:
			kernel_shape = (20,)
			stride = (2,)
			num_kernels = 8
		"""

		# keep the original values
		size = self.pool_shape
		stride = self.stride

		t_size = list(size)
		t_stride = list(stride)

		#
		size_pre_out = self.prev_gene.outputDimension()[0:2]

		next_gene_minIn = self.next_gene.minInputDimension()
		if len(next_gene_minIn) == 1:
			# next_gene is a FC or Output
			size_next_in = (1, 1)
		else:
			size_next_in = next_gene_minIn[0:2]

		# Mutate the kernel
		min_pool_size = [MIN_POOL_SIZE, MIN_POOL_SIZE]
		max_pool_size = [size_pre_out[0]-(size_next_in[0]-1)*stride[0],
					 size_pre_out[1]-(size_next_in[1]-1)*stride[1]]

		min_stride_size = [MIN_POOL_STRIDE, MIN_POOL_STRIDE]
		max_stride_size = [MIN_POOL_STRIDE, MIN_POOL_STRIDE]
		for i in [0, 1]:
			if size_next_in[i] > 1:
				temp = int((size_pre_out[i]-size[i])/(size_next_in[i]-1))
				max_stride_size[i] = temp if temp<size[i] else size[i]
			else:
				max_stride_size[i] = size[i]

		# What to mutate
		mutation = random.choice(['pool_size', 'stride_size'])

		"""
		1. Make a list of possible values for a variable, list_values
		2. Sample values on a distribution and make a list by normalizing them, list_probs
		3. t_value = random.choice(list_values, list_probs)
		"""
		if mutation == 'pool_size':
			for i in [0, 1]:
				size_list = list(range(min_pool_size[i], max_pool_size[i]+1))
				if size[i] in size_list:
					size_list.remove(size[i])
				if len(size_list) > 0:
					t_size[i] = random.choice(size_list)
				else:
					mutation = 'stride_size'
					break
		else: # mutation == 'stride_size':
			for i in [0, 1]:
				stride_list = list(range(min_stride_size[i], max_stride_size[i]+1))
				if stride[i] in stride_list:
					stride_list.remove(stride[i])
				if len(stride_list) > 0:
					t_stride[i] = random.choice(stride_list)
				else:
					return False
		if mutation == 'pool_size':
			self.pool_shape = tuple(t_size)
		elif mutation == 'stride_size':
			self.stride = tuple(t_stride)
		# after some change, check validity of the new gene
		if self.canFollow(self.prev_gene):
			return True
		else:
			self.kernel_shape = size
			self.stride = stride
			return False


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


class FullyConnectedGene(Gene):
	"""
	"""
	def __init__(self, size, activation_function):
		"""
		size				- number of neurons (integer)
		activation_function - e.g., tf.sigmoid
		"""

		Gene.__init__(self)

		self.size = size
		self.activation = activation_function

		self.type = FULLY_CONNECTED
		self.dimension = size


	def clone(self):
		"""
		"""

		return FullyConnectedGene(self.size, self.activation)

	def equals(self, other):
		"""
		Type and meta-parameters should all match
		"""
		if other.type != FULLY_CONNECTED:
			return False

		return ( self.size == other.size and
	 			(self.activation == other.activation) )


	def canFollow(self, prevGene):
		"""
		A FullyConnectedGene can follow any of the other types of genes
		"""

		return True


	def outputDimension(self):
		"""
		Calculate the output dimension based on the input dimension, kernel_size, and stride
		"""

		return self.dimension

	def minInputDimension(self):
		"""
		Recurse through next gene to figure out minimum valid input size
		"""

		return (1,)

	def mutate(self):
		"""
		size should be mutated based on the constraints from prevGene and nextGene
		"""
		factor = 0.5
		min_size = int(self.size*(1-factor))
		min_size = 1 if min_size<1 else min_size
		max_size = int(self.size*(1+factor))

		size_list = list(range(min_size, max_size+1))
		if self.size in size_list:
			size_list.remove(self.size)

		if len(size_list)>0:
			self.size = random.choice(size_list)
			self.dimension = self.size
			return True
		else:
			self.size = np.random.randint(MIN_FULL_CONNECTION, MAX_FULL_CONNECTION+1)
			self.dimension = self.size


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
		weights = tf.Variable(tf.truncated_normal((input_shape[0], self.dimension), stddev=0.05))
		bias = tf.Variable(tf.constant(0.0, shape=(self.dimension,)))

		self.tensor = tf.nn.relu(tf.matmul(input_tensor, weights) + bias)

		return self.tensor


	def __str__(self):
		"""
		"""

		return "F.CONN:\tOutput Dimensions: " + str(self.outputDimension())


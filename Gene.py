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
from models.parts import *


MIN_CNN_WIDTH = 2
MAX_CNN_WIDTH = 75
MIN_CNN_KERNELS = 5
MAX_CNN_KERNELS = 30
MIN_CNN_STRIDE = 1
MAX_CNN_STRIDE = 5
MIN_POOL_SIZE = 2
MAX_POOL_SIZE = 5
MIN_POOL_STRIDE = 1
MAX_POOL_STRIDE = 5
MIN_FULL_CONNECTION = 5
MAX_FULL_CONNECTION = 200


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


	def mutate(self, prevGene, nextGene):
		"""
		Alter this gene, ensuring that the constraints from the previous and next gene are satisfied
		"""

		pass


	def generateLayer(self, name_suffix):
		"""
		Create the CNN part(s) (tuple of objects) used to construct this particular layer in the CNN
		"""

		pass


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


	def canFollow(self, prevGene=None):
		"""
		This never follows a gene, it's the input
		"""

		return False


	def outputDimension(self):
		"""
		"""

		return self.dimension


	def mutate(self, prevGene, nextGene):
		"""
		"""
		assert prevGene is None, "The input should not have previous gene!"
		print "You are mutating an input, not allowed!"


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

	def canFollow(self, prevGene=None):
		"""
		"""

		return True

	def outputDimension(self, prevGene=None):
		"""
		"""

		return self.dimension


	def mutate(self, prevGene, nextGene):
		"""
		"""

		pass


class DummyGene(Gene):
	"""
	A Gene used just to propagate a dimensionality through a genotype
	"""

	def __init__(self, shape):
	
		Gene.__init__(self)

		self.shape = shape
		self.type = INPUT

	def canFollow(self, prevGene):
		pass

	def outputDimension(self):
		return self.shape

	def mutate(self, prevGene, nextGene):
		pass


class Conv1DGene(Gene):
	"""
	"""
	def __init__(self, kernel_shape, stride, num_kernels, activation_function):
		"""
		kernel_shape - should be a 1-tuple, e.g, (20,)
		stride       - should be a 1-tuple, e.g, (2,)
		num_kernels  - should be an integer
		activation_function - a Tensorflow activation tensor (e.g., tf.sigmoid)
		"""
	
		Gene.__init__(self)


		self.kernel_shape = kernel_shape
		self.stride = stride
		self.num_kernels = num_kernels
		self.activation = activation_function

		self.type = CONV1D


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
		Calculate the output dimension based on the input dimension, kernel_size, and stride
		"""

		# Is this connected to some prior gene?
		prevLength, channels = self.prev_gene.outputDimension()
		myLength = (prevLength - self.kernel_shape[0]) / self.stride[0] + 1

		self.dimension = (myLength, self.num_kernels)
		return self.dimension


	def mutate(self, prevGene, nextGene):
		"""
		kernel_size, stride and num_kernels should be mutated based on the constraints from prevGene and nextGene
		"""
		pass


class Conv2DGene(Gene):
	"""
	"""
	def __init__(self, kernel_shape, stride, num_kernels, activation_function):
		"""
		kernel_shape - should be a 2-tuple, e.g, (20,20)
		stride       - should be a 2-tuple, e.g, (2,2)
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


	def canFollow(self, prevGene):
		"""
		A Conv1Dgene can follow an 'InputGene' or an 'Pool1DGene'
		The constraints are kernel_size should not larger than prevGene.output_size
		"""

		# Is the previous gene a valid type?
		if not prevGene.type in [INPUT, CONV2D, POOL2D]:
			print "blah"
			return False

		# Is the dimensionality 2? (length x channels)
		if len(prevGene.outputDimension()) != 3:
			print "blah1"
			return False

		# Get the output dimension of the previous gene
		prevHeight, prevWidth, prevChannels  = prevGene.outputDimension()

		# Is the kernel larger than the previous length?
		if self.kernel_shape[0] > prevHeight or self.kernel_shape[1] > prevWidth:
			print "blah2"
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


	def mutate(self, prevGene, nextGene):
		"""
		kernel_size, stride and num_kernels should be mutated based on the constraints from prevGene and nextGene
		"""
		pass


class Pool1DGene(Gene):
	"""
	"""
	def __init__(self, pool_shape, stride):
		"""
		pool_size    - should be a 1-tuple, e.g, (2,)
		stride       - should be a 1-tuple, e.g, (2,)
		"""
	
		Gene.__init__(self)


		self.pool_shape = pool_shape
		self.stride = stride

		self.type = POOL1D
		self.dimension = None


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


	def mutate(self, prevGene, nextGene):
		"""
		kernel_size, stride and num_kernels should be mutated based on the constraints from prevGene and nextGene		
		"""

		pass


class Pool2DGene(Gene):
	"""
	"""
	def __init__(self, pool_shape, stride):
		"""
		pool_size    - should be a 2-tuple, e.g, (2,2)
		stride       - should be a 2-tuple, e.g, (2,2)
		"""
	
		Gene.__init__(self)

		self.pool_shape = pool_shape
		self.stride = stride

		self.type = POOL2D
		self.dimension = None


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


	def mutate(self, prevGene, nextGene):
		"""
		kernel_size, stride and num_kernels should be mutated based on the constraints from prevGene and nextGene		
		"""
		pass


class FullyConnectedGene(Gene):
	"""
	"""
	def __init__(self, size, activation_function):
		"""
		size                - number of neurons (integer)
		activation_function - e.g., tf.sigmoid
		"""
	
		Gene.__init__(self)

		self.size = size
		self.activation = activation_function

		self.type = FULLY_CONNECTED
		self.dimension = size

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


	def mutate(self, prevGene, nextGene):
		"""
		kernel_size, stride and num_kernels should be mutated based on the constraints from prevGene and nextGene		
		"""
		pass

"""
# Helper function
# randomly generate a ConvGene based on the lastGene's output dimension
"""
def generate1DConvGene(lastGene):
	## specify the min and max for each random functions
	max_size = min(MAX_CNN_WIDTH, lastGene.outputDimension()[0])
	kernel_size = np.random.randint(MIN_CNN_WIDTH, max_size+1)
	conv_stride = np.random.randint(MIN_CNN_STRIDE, MAX_CNN_STRIDE+1)
	num_kernels = np.random.randint(MIN_CNN_KERNELS, MAX_CNN_KERNELS+1)

	# activation_function ???
	return Conv1DGene((kernel_size,), (conv_stride,), num_kernels, activation_function=None)

def generate2DConvGene(lastGene):
	## specify the min and max for each random functions
	max_height = min(MAX_CNN_WIDTH, lastGene.outputDimension()[0])
	max_width = min(MAX_CNN_WIDTH, lastGene.outputDimension()[1])
	kernel_height = np.random.randint(MIN_CNN_WIDTH, max_height+1)
	kernel_width = np.random.randint(MIN_CNN_WIDTH, max_width+1)
	
	conv_stride_height = np.random.randint(MIN_CNN_STRIDE, MAX_CNN_STRIDE+1)
	conv_stride_width = np.random.randint(MIN_CNN_STRIDE, MAX_CNN_STRIDE+1)

	num_kernels = np.random.randint(MIN_CNN_KERNELS, MAX_CNN_KERNELS+1)

	# activation_function ???
	return Conv2DGene((kernel_height, kernel_width), (conv_stride_height, conv_stride_width), num_kernels, activation_function=None)

"""
# Helper function
# randomly generate a PoolGene based on the lastGene's output dimension
"""
def generate1DPoolGene(lastGene):
	## specify the min and max for each random functions
	max_size = min(MAX_POOL_SIZE, lastGene.outputDimension()[0])
	pool_size = np.random.randint(MIN_POOL_SIZE, max_size+1)
	pool_stride = np.random.randint(MIN_POOL_STRIDE, MAX_POOL_STRIDE+1)

	# activation_function ???
	return Pool1DGene((pool_size,), (pool_stride,))


def generate2DPoolGene(lastGene):
	## specify the min and max for each random functions
	max_height = min(MAX_POOL_SIZE, lastGene.outputDimension()[0])
	max_width = min(MAX_POOL_SIZE, lastGene.outputDimension()[1])

	pool_height = np.random.randint(MIN_POOL_SIZE, max_height+1)
	pool_width = np.random.randint(MIN_POOL_SIZE, max_width+1)

	pool_stride_height = np.random.randint(MIN_POOL_STRIDE, MAX_POOL_STRIDE+1)
	pool_stride_width = np.random.randint(MIN_POOL_STRIDE, MAX_POOL_STRIDE+1)

	# activation_function ???
	return Pool2DGene((pool_height, pool_width), (pool_stride_height, pool_stride_width))


"""
# Helper function
# randomly generate a FullyConnectedGene based on the lastGene's output dimension
"""
def generateFullConnectedGene(FullyConnectedGene, lastGene):
	## specify the min and max for each random functions
	size = np.random.randint(MIN_FULL_CONNECTION, MAX_FULL_CONNECTION+1)

	# activation_function ???
	return FullyConnectedGene(size, activation_function=None)

"""
Create a list of genes that describes a random, valid CNN
"""
def generateGenotypeProb(input_size, output_size, ConvProb, PoolProb=1.0, FullConnectProb = 0.5):

	# Is this a 1D or 2D input shape?
	if len(input_size) == 2:
		is2D = False
	else:
		is2D = True

	# Pick out the appropriate Gene types
	if is2D:
		generateConvGene = generate2DConvGene
		generatePoolGene = generate2DPoolGene
	else:
		generateConvGene = generate1DConvGene
		generatePoolGene = generate1DPoolGene

	lastGene = InputGene(input_size)
	outGene = OutputGene(output_size)

	lastGene.next_gene = outGene

	genotype = [lastGene]
	print(lastGene.outputDimension())

	#### NOTE: May need to have two generateConvGene and generatePoolGene, for each possible shape (1D and 2D)

	# Add convolution layers (and possibly pooling layers) until a random check fails
	while random.random() < ConvProb:
		if MIN_CNN_WIDTH > lastGene.outputDimension()[0]:
			break
		if is2D and MIN_CNN_WIDTH > lastGene.outputDimension()[1]:
			break

		# Add the Convolution layer, with random arguments...
		tmpGene = generateConvGene(lastGene)
		tmpGene.next_gene = outGene
		print('kernel_size: {}, conv_stride: {}, num_kernels: {}'.format(tmpGene.kernel_shape, tmpGene.stride, tmpGene.num_kernels))
		if tmpGene.canFollow(lastGene):
			lastGene.next_gene = tmpGene
			tmpGene.prev_gene = lastGene
			lastGene = tmpGene
			genotype.append(lastGene)
			print(lastGene.outputDimension())
			print("ConvGene added!")
		else:
			print("ConvGene can not follow lastGene - %s!" % lastGene.type)
			print('Failed to create a Genotype!')
			print('=======================')
			return

		# Should a pooling layer be added?
		if random.random() < PoolProb:
			if MIN_POOL_SIZE > lastGene.outputDimension()[0] :
				break
			if is2D and MIN_POOL_SIZE > lastGene.outputDimension()[1]:
				break
			tmpGene = generatePoolGene(lastGene)
			tmpGene.next_gene = outGene
			print('kernel_size: {}, pool_stride: {}'.format(tmpGene.pool_shape, tmpGene.stride))
			if tmpGene.canFollow(lastGene):
				lastGene.next_gene = tmpGene
				tmpGene.prev_gene = lastGene
				lastGene = tmpGene
				genotype.append(lastGene)
				print(lastGene.outputDimension())
				print("PoolGene added!")
			else:
				print("PoolGene can not follow lastGene - %s!" % lastGene.type)
				print('Failed to create a Genotype!')
				print('=======================')
				return

	# Added all the Convolution layers, now add FC layers
	while random.random() < FullConnectProb:
		# Add a fully connected layer
		tmpGene = generateFullConnectedGene(FullyConnectedGene, lastGene)
		tmpGene.next_gene = outGene
		if tmpGene.canFollow(lastGene):
			lastGene.next_gene = tmpGene
			tmpGene.prev_gene = lastGene
			lastGene = tmpGene
			genotype.append(lastGene)
			print(lastGene.outputDimension())
			print("FullyConnectedGene added!")
		else:
			print("FullyConnectedGene can not follow lastGene - %s!" % lastGene.type)
			print('Failed to create a Genotype!')
			print('=======================')
			return

	print('Successfuly Created a Genotype!')
	print('=======================')

	genotype.append(outGene)

	return genotype

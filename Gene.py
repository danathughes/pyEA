## Gene.py
##
##

"""
1. What is the dimensionality for each gene type?

INPUT - [ m * n * k] ???

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

import random
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


 
	def canFollow(self, prevGene):
		"""
		Can this gene follow the previous gene?  I.e., are all constraints satisfied?
		"""

		pass


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

	def __init__(self, input_size):
		"""
		Placeholder gene for the input dimensionality of the problem set
		"""

		self.dimension = input_size
		self.type = INPUT


	def canFollow(self, prevGene=None):
		"""
		This never follows a gene, it's the input
		"""
		if prevGene is not None:
			return False
		else:
			return True


	def outputDimension(self, prevGene=None):
		"""
		"""
		assert prevGene is None, "There shouldn't be prevGene for InputGene!"
		return self.dimension


	def mutate(self, prevGene, nextGene):
		"""
		"""
		assert prevGene is None, "The input should not have previous gene!"
		print "You are mutating an input, not allowed!"


class Conv1DGene(Gene):
	"""
	"""

	def __init__(self, kernel_size, stride, num_kernels, activation_function):
		"""
		"""

		self.kernel_size = kernel_size
		self.stride = stride
		self.num_kernels = num_kernels
		self.activation = activation_function

		self.type = CONV1D


	def canFollow(self, prevGene):
		"""
		A Conv1Dgene can follow an 'InputGene' or an 'Pool1DGene'
		The constraints are kernel_size should not larger than prevGene.output_size
		"""
		if prevGene.type == INPUT or prevGene.type == Pool1DGene:
			## next step is to see if 
			output_size = outputDimension(prevGene)		## calculate output dimension
			if self.kernel_size > output_size:
				return False
			else:
				return True
		else:
			return False


	def outputDimension(self, prevGene):
		"""
		Calculate the output dimension based on the input dimension, kernel_size, and stride
		"""

		input_size = prevGene.dimensionality
		output_size = (input_size-self.kernel_size)/stride + 1

		self.dimension = output_size
		return self.dimension


	def mutate(self, prevGene, nextGene):
		"""
		kernel_size, stride and num_kernels should be mutated based on the constraints from prevGene and nextGene
		"""

		pass



class Conv2DGene(Gene):
	"""
	"""

	def __init__(self, kernel_size, stride, num_kernels, activation_function):
		"""
		"""

		self.kernel_size = kernel_size
		self.stride = stride
		self.num_kernels = num_kernels
		self.activation = activation_function

		self.type = CONV2D


	def canFollow(self, prevGene):
		"""
		A Conv2Dgene can follow an 'InputGene' or an 'Pool2DGene'
		The constraints are kernel_size should not larger than prevGene.output_size
		"""
		if prevGene.type == INPUT or prevGene.type == Pool2DGene:
			## next step is to see if 
			output_size = outputDimension(prevGene)		## calculate output dimension
			if self.kernel_size > output_size:
				return False
			else: 
				return True
		else:
			return False


	def outputDimension(self, prevGene):
		"""
		Calculate the output dimension based on the input dimension, kernel_size, and stride
		"""

		input_size = prevGene.dimensionality
		output_size = (input_size-self.kernel_size)/stride + 1

		self.dimensionality = output_size
		return self.dimensionality


	def mutate(self, prevGene, nextGene):
		"""
		kernel_size, stride and num_kernels should be mutated based on the constraints from prevGene and nextGene
		"""

		pass


class Pool1DGene(Gene):
	"""
	"""

	def __init__(self, kernel_size, stride, activation_function):
		"""
		"""

		self.kernel_size = kernel_size
		self.stride = stride
		self.activation = activation_function

		self.type = POOL1D


	def canFollow(self, prevGene):
		"""
		A Pool1DGene can only follow an 'Conv1DGene'
		"""
		if prevGene.type == Conv1DGene:
			## next step is to see if 
			output_size = outputDimension(prevGene)		## calculate output dimension
			if self.kernel_size > output_size:
				return False
			else: 
				return True
		else:
			return False


	def outputDimension(self, prevGene):
		"""
		Calculate the output dimension based on the input dimension, kernel_size, and stride
		"""

		input_size = prevGene.dimensionality
		output_size = (input_size-self.kernel_size)/stride + 1

		self.dimensionality = output_size
		return self.dimensionality


	def mutate(self, prevGene, nextGene):
		"""
		kernel_size, stride and num_kernels should be mutated based on the constraints from prevGene and nextGene		
		"""

		pass


class Pool2DGene(Gene):
	"""
	"""

	def __init__(self, kernel_size, stride, activation_function):
		"""
		"""

		self.kernel_size = kernel_size
		self.stride = stride
		self.activation = activation_function

		self.type = POOL2D


	def canFollow(self, prevGene):
		"""
		A Pool2DGene can only follow an 'Conv2DGene'
		"""
		if prevGene.type == Conv2DGene:
			## next step is to see if 
			output_size = outputDimension(prevGene)		## calculate output dimension
			if self.kernel_size > output_size:
				return False
			else return True
		else
			return False


	def outputDimension(self, prevGene):
		"""
		Calculate the output dimension based on the input dimension, kernel_size, and stride
		"""

		input_size = prevGene.dimensionality
		output_size = (input_size-self.kernel_size)/stride + 1

		self.dimensionality = output_size
		return self.dimensionality


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
		"""

		self.size = size
		self.activation = activation_function

		self.type = FULLY_CONNECTED


	def canFollow(self, prevGene):
		"""
		A FullyConnectedGene can follow an 'Pool1DGene', an 'Pool2DGene' or another 'FullyConnectedGene'
		"""
		if prevGene.type == Conv1DGene or prevGene.type == Conv2DGene:
			return True
		else if prevGene.type == FullyConnectedGene:
			## Should the num of nodes of the following fully-connected layer be smaller???
			if prevGene.dimensionality < self.dimensionality:
				return False
			else:
				return True
		else:
			return False


	def outputDimension(self, prevGene):
		"""
		Calculate the output dimension based on the input dimension, kernel_size, and stride
		"""

		input_size = prevGene.dimensionality
		output_size = (input_size-self.kernel_size)/stride + 1

		self.dimensionality = output_size
		return self.dimensionality


	def mutate(self, prevGene, nextGene):
		"""
		kernel_size, stride and num_kernels should be mutated based on the constraints from prevGene and nextGene		
		"""

		pass



"""
# Helper function
# randomly generate a ConvGene based on the lastGene's output dimension
"""
def generateConvGene(ConvGene, lastGene):
	## specify the min and max for each random functions
	kernel_size = np.random.randint(MIN_CNN_WIDTH, max_width+1)
	conv_stride = np.random.randint(MIN_CNN_STRIDE, MAX_CNN_STRIDE+1)
	num_kernels = np.random.randint(MIN_CNN_KERNELS, MAX_CNN_KERNELS+1)

	# activation_function ???
	return ConvGene(kernel_size, conv_stride, num_kernels, activation_function=None)

"""
# Helper function
# randomly generate a PoolGene based on the lastGene's output dimension
"""
def generatePoolGene(PoolGene, lastGene):
	## specify the min and max for each random functions
	pool_size = np.random.randint(MIN_POOL_SIZE, MAX_POOL_SIZE+1)
	pool_stride = np.random.randint(MIN_POOL_STRIDE, MAX_POOL_STRIDE+1)

	# activation_function ???
	return PoolGene(kernel_size, pool_stride, activation_function=None)


"""
# Helper function
# randomly generate a FullyConnectedGene based on the lastGene's output dimension
"""
def generateConvGene(FullyConnectedGene, lastGene):
	## specify the min and max for each random functions
	size = np.random.randint(MIN_FULL_CONNECTION, MAX_FULL_CONNECTION+1)

	# activation_function ???
	return FullyConnectedGene(size, activation_function=None)


"""
Create a list of genes that describes a random, valid CNN
"""
def generateGenotypeProb(input_size, output_size, ConvProb=0.5, PoolProb=1.0, FullConnectProb = 0.5, is2D=False):
	# Pick out the appropriate Gene types
	if is2D:
		ConvGene = Conv2DGene
		PoolGene = Pool2DGene
	else:
		ConvGene = Conv1DGene
		PoolGene = Pool1DGene

	lastGene = InputGene(input_size)
	genotype = [lastGene]

	# Add convolution layers (and possibly pooling layers) until a random check fails
	while random.random() < ConvProb:
		# Add the Convolution layer, with random arguments...
		lastGene = generateConvGene(ConvGene, lastGene)
		genotype.append(lastGene)

		# Should a pooling layer be added?
		if random.random() < PoolProb:
			lastGene = generatePoolGene(PoolGene, lastGene)
			genotype.append(lastGene)

	# Added all the Convolution layers, now add FC layers
	while random.random() < FullConnectProb:
		# Add a fully connected layer
		lastGene = generateFullConnectedGene(FullyConnectedGene, lastGene)
		genotype.append(lastGene)

	return genotype



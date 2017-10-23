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


	def clone(self):
		"""
		Make a copy of this gene
		"""

		pass


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

	def __str__(self):
		"""
		String representation of this gene
		"""

		return "Abstract Gene"


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


	def canFollow(self, prevGene):
		"""
		"""

		# Don't let this follow another output gene
		return prevGene.type != OUTPUT

	def outputDimension(self, prevGene=None):
		"""
		"""

		return self.dimension


	def mutate(self, prevGene, nextGene):
		"""
		"""

		pass

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

	def canFollow(self, prevGene):
		pass

	def outputDimension(self):
		return self.shape

	def mutate(self, prevGene, nextGene):
		pass

	def __str__(self):
		"""
		"""

		return "DUMMY: \tOutput Dimensions: " + str(self.outputDimension())


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


	def clone(self):
		"""
		"""

		return Conv1DGene(self.kernel_shape, self.stride, self.num_kernels, self.activation)


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


	def clone(self):
		"""
		"""

		return Conv2DGene(self.kernel_shape, self.stride, self.num_kernels, self.activation)


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


	def mutate(self, prevGene, nextGene):
		"""
		kernel_size, stride and num_kernels should be mutated based on the constraints from prevGene and nextGene
		"""
		pass

	def __str__(self):
		"""
		"""

		return "CONV2D:\tKernel: " + str((self.kernel_shape) + (self.num_kernels,)) + ";\tStride: " + str(self.stride) + ";\tOutput Dimensions: " + str(self.outputDimension())


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


	def clone(self):
		"""
		"""

		return Pool1DGene(self.pool_shape, self.stride)


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


	def __str__(self):
		"""
		"""

		return "POOL1D:\tPool Shape: " + str(self.pool_shape) + ";\tStride: " + str(self.stride) + ";\tOutput Dimensions: " + str(self.outputDimension())


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


	def clone(self):
		"""
		"""

		return Pool2DGene(self.pool_shape, self.stride)		


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


	def __str__(self):
		"""
		"""

		return "POOL2D:\tPool Shape: " + str(self.pool_shape) + ";\tStride: " + str(self.stride) + ";\tOutput Dimensions: " + str(self.outputDimension())


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


	def clone(self):
		"""
		"""

		return FullyConnectedGene(self.size, self.activation)

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


	def __str__(self):
		"""
		"""

		return "F.CONN:\tOutput Dimensions: " + str(self.outputDimension())


"""
# Helper functions --- Genotypes will implement one set of generators
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


class Genotype:
	"""
	A representation of a full network
	"""

	def __init__(self, input_shape, output_size, conv_prob = 1.0, pooling_prob = 1.0, fullConnection_prob = 0.5):
		"""
		Create a random genotype
		"""

		self.input_shape = input_shape
		self.output_size = output_size

		# Probabilities of generating convolution, pooling and fully connected layers
		self.convolutionProb = conv_prob
		self.poolingProb = pooling_prob
		self.fullConnectionProb = fullConnection_prob

		# Is this a 1D or 2D CNN?
		if len(input_shape) == 3:
			self.is2D = True
			self.__generateConvGene = generate2DConvGene
			self.__generatePoolGene = generate2DPoolGene
		elif len(input_shape) == 2:
			self.is2D = False
			self.__generateConvGene = generate1DConvGene
			self.__generatePoolGene = generate1DPoolGene
		else:
			# Not a valid input shape
			print "Invalid input dimensionality: %d" % len(input_shape)

		# Create a gentopye
		self.genotype = self.generateGenotype()


	def clone(self):
		"""
		Make a copy of this genotype
		"""

		clone = Genotype(self.input_shape, self.output_shape, self.convolutionProb, self.poolingProb, self.fullConnectionProb)
		clone.genotype = [gene.clone() for gene in self.genotype]
		for i in range(len(clone.genotype) - 1):
			clone.genotype[i].next_gene = clone.genotype[i+1]
			clone.genotype[i+1].prev_gene = clone.genotype[i]

		return clone


	def __generateFullConnection(self, lastGene):
		## specify the min and max for each random functions
		size = np.random.randint(MIN_FULL_CONNECTION, MAX_FULL_CONNECTION+1)

		# activation_function ???
		return FullyConnectedGene(size, activation_function=None)


	def generateGenotype(self):
		"""
		Create the actual genotype
		"""

		lastGene = InputGene(self.input_shape)
		outGene = OutputGene(self.output_size)

		lastGene.next_gene = outGene

		genotype = [lastGene]

		# Add the Convolutional Layers (and pooling layers)
		while random.random() < self.convolutionProb:
			if MIN_CNN_WIDTH > lastGene.outputDimension()[0]:
				break
			if self.is2D and MIN_CNN_WIDTH > lastGene.outputDimension()[1]:
				break

			# Add the convolution layer, with random genes
			tempGene = self.__generateConvGene(lastGene)
			tempGene.next_gene = outGene

			if tempGene.canFollow(lastGene):
				lastGene.next_gene = tempGene
				tempGene.prev_gene = lastGene
				lastGene = tempGene
				genotype.append(lastGene)
			else:
				# Failed to create a genotype
				return None

			# Should this be followed by a pooling layer?
			if random.random() < self.poolingProb:
				if MIN_POOL_SIZE > lastGene.outputDimension()[0]:
					break
				if self.is2D and MIN_POOL_SIZE > lastGene.outputDimension()[1]:
					break

				tempGene = self.__generatePoolGene(lastGene)
				tempGene.next_gene = outGene

				if tempGene.canFollow(lastGene):
					lastGene.next_gene = tempGene
					tempGene.prev_gene = lastGene
					lastGene = tempGene
					genotype.append(lastGene)

				else:
					# Failed to create a genotype
					return None

		# Fill in fully connected layers
		while random.random() < self.fullConnectionProb:
			tempGene = self.__generateFullConnection(lastGene)
			tempGene.next_gene = outGene

			if tempGene.canFollow(lastGene):
				lastGene.next_gene = tempGene
				tempGene.prev_gene = lastGene
				lastGene = tempGene
				genotype.append(lastGene)

			else:
				# Failed to create a genotype
				return None

		# Genotype successfully created
		genotype.append(outGene)

		return genotype


	def __str__(self):
		"""
		"""

		string = ""

		for gene in self.genotype[:-1]:
			string += str(gene) + '\n'
		string += str(self.genotype[-1])

		return string


	def crossover(self, other):
		"""
		"""

		gene1 = self.genotype
		gene2 = other.genotype

		# What are valid crossover points?
		crossover_points = []

		for i in range(1, len(gene1)):
			for j in range(1, len(gene2)):
				if gene1[i].canFollow(gene2[j-1]) and gene2[j].canFollow(gene1[i-1]):
					# Are we just swapping inputs or outputs?
					if i==1 and j==1:
						pass
					elif i==len(gene1) and j==len(gene2):
						pass
					else:
						crossover_points.append((i,j))

		# if the list is empty, cannot do anything (force a mutation)
		if len(crossover_points) == 0:
			return None, None

		# Make two new genotypes
		crossover_point = random.choice(crossover_points)

		child_gene1 = []
		child_gene2 = []

		# Populate the first half of each children
		for i in range(crossover_point[0]):
			child_gene1.append(gene1[i].clone())
		for j in range(crossover_point[1]):
			child_gene2.append(gene2[j].clone())

		# Populate the second half of each child
		for i in range(crossover_point[0], len(gene1)):
			child_gene2.append(gene1[i].clone())
		for j in range(crossover_point[1], len(gene2)):
			child_gene1.append(gene2[j].clone())

		# Link the previous and next genes in each child
		for i in range(len(child_gene1) - 1):
			child_gene1[i].next_gene = child_gene1[i+1]
		for i in range(1, len(child_gene1)):
			child_gene1[i].prev_gene = child_gene1[i-1]

		for i in range(len(child_gene2) - 1):
			child_gene2[i].next_gene = child_gene2[i+1]
		for i in range(1, len(child_gene2)):
			child_gene2[i].prev_gene = child_gene2[i-1]

		# Done!
		child1 = Genotype(self.input_shape, self.output_size, self.convolutionProb, self.poolingProb, self.fullConnectionProb)
		child2 = Genotype(self.input_shape, self.output_size, self.convolutionProb, self.poolingProb, self.fullConnectionProb)

		child1.genotype = child_gene1
		child2.genotype = child_gene2

		return child1, child2		

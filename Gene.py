## Gene.py
##
##

# Enumerate the different gene types
INPUT = "INPUT"
CONV1D = "CONV1D"
CONV2D = "CONV2D"
POOL1D = "POOL1D"
POOL2D = "POOL2D"
FULLY_CONNECTED = "FULLYCONNECTED"

import random
from models.parts import *

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

	def __init__(self, dimensionality):
		"""
		Placeholder gene for the input dimensionality of the problem set
		"""

		self.dimensionality = dimensionality
		self.type = INPUT


	def canFollow(self, prevGene):
		"""
		This never follows a gene, it's the input
		"""

		return False


	def outputDimension(self, prevGene):
		"""
		"""

		return self.dimensionality


	def mutate(self, prevGene, nextGene):
		"""
		"""

		pass




class Conv1DGene(Gene):
	"""
	"""

	def __init__(self, kernel_size, stride, num_filters, activation_function):
		"""
		"""

		self.kernel_size = kernel_size
		self.stride = stride
		self.num_filters = num_filters
		self.activation = activation_function

		self.type = CONV1D


	def canFollow(self, prevGene):
		"""
		"""

		pass


	def outputDimension(self, prevGene):
		"""
		"""

		pass


	def mutate(self, prevGene, nextGene):
		"""
		"""

		pass



class Conv2DGene(Gene):
	"""
	"""

	def __init__(self, kernel_size, stride, num_filters, activation_function):
		"""
		"""

		self.kernel_size = kernel_size
		self.stride = stride
		self.num_filters = num_filters
		self.activation = activation_function

		self.type = CONV2D


	def canFollow(self, prevGene):
		"""
		"""

		pass


	def outputDimension(self, prevGene):
		"""
		"""

		pass


	def mutate(self, prevGene, nextGene):
		"""
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
		"""

		pass


	def outputDimension(self, prevGene):
		"""
		"""

		pass


	def mutate(self, prevGene, nextGene):
		"""
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
		"""

		pass


	def outputDimension(self, prevGene):
		"""
		"""

		pass


	def mutate(self, prevGene, nextGene):
		"""
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
		"""

		pass


	def outputDimension(self, prevGene):
		"""
		"""

		pass


	def mutate(self, prevGene, nextGene):
		"""
		"""

		pass



def generateGenotype(inputGene, ConvProb=0.5, PoolProb=1.0, FullConnectProb = 0.5, is2D=False):
	"""
	Create a list of genes that describes a random, valid CNN
	"""

	# Pick out the appropriate Gene types
	if is2D:
		Conv = Conv2DGene
		Pool = Pool2DGene
	else:
		Conv = Conv1DGene
		Pool = Pool1DGene

	genotype = [inputGene]

	# Add convolution layers (and possibly pooling layers) until a random check fails
	while random.random() < ConvProb:
		# Add the Convolution layer, with random arguments...
		genotype.append(Conv())

		# Should a pooling layer be added?
		if random.random() < PoolProb:
			genotype.append(Pool())


	# Added all the Convolution layers, now add FC layers
	while random.random() < FullConnectProb:
		# Add a fully connected layer
		genotype.append(FullConnectProb())


	return genotype


def generateGenotypeA(inputGene, numConv, numFullConnected, is2D=False):
	"""
	Create a list of genes that describes a random, valid CNN
	"""

	# Pick out the appropriate Gene types
	if is2D:
		Conv = Conv2DGene
		Pool = Pool2DGene
	else:
		Conv = Conv1DGene
		Pool = Pool1DGene

	genotype = [inputGene]

	# Add convolution layers (and possibly pooling layers) until a random check fails
	for i in range(numConv):
		genotype.append(Conv())
		genotype.append(Pool())

	# Added all the Convolution layers, now add FC layers
	for i in range(numFullConnected):
		genotype.append(FullConnectProb())

	return genotype



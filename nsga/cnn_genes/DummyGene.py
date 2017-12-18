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


class DummyGene(AbstractGene):
	"""
	A Gene used just to propagate a dimensionality through a genotype
	"""

	def __init__(self, shape):

		AbstractGene.__init__(self)

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
		
		pass


	def mutate(self):
		pass


	def __str__(self):
		"""
		"""

		return "DUMMY: \tOutput Dimensions: " + str(self.outputDimension())


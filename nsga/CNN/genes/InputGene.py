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

class InputGene(AbstractGene):
	"""
	"""
	def __init__(self, input_shape):
		"""
		Placeholder gene for the input dimensionality of the problem set
		input_shape = (height, width, num_channels) (2D data)
						  (length, num_channels) (1D data)
		"""

		AbstractGene.__init__(self)

		self.dimension = input_shape
		self.type = INPUT


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
		The output dimensionality of the input layer; i.e., the shape of the input to be accepted
		"""

		return self.dimension


	def minInputDimension(self):
		"""
		Recurse through next gene to figure out minimum valid input size
		"""

		return None


	def mutate(self):
		"""
		The input layer cannot mutate; this function does nothing and always returns False
		"""

		return False


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
		Pretty string containing output dimensionality
		"""

		return "INPUT: \tOutput Dimensions: " + str(self.outputDimension())


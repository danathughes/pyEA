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


class OutputGene(AbstractGene):
	"""
	"""

	def __init__(self, output_size):
		"""
		Placeholder for an output gene
		"""

		AbstractGene.__init__(self)

		self.dimension = output_size
		self.type = OUTPUT

		if type(output_size) == int:
			self.dimension = (output_size,)


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
		weights = tf.Variable(tf.truncated_normal(tuple(input_shape) + self.dimension, stddev=0.05))
		bias = tf.Variable(tf.constant(0.0, shape=self.dimension,))

		self.tensor = tf.nn.softmax(tf.matmul(input_tensor, weights) + bias)

		return self.tensor


	def __str__(self):
		"""
		"""

		return "OUTPUT:\tOutput Dimensions: " + str(self.outputDimension())
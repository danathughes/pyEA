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


class FullyConnectedGene(AbstractGene):
	"""
	"""
	def __init__(self, size, activation_function):
		"""
		size				- number of neurons (integer)
		activation_function - e.g., tf.sigmoid
		"""

		AbstractGene.__init__(self)

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


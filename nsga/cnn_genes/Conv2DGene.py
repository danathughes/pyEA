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


class Conv2DGene(AbstractGene):
	"""
	"""
	def __init__(self, kernel_shape, stride, num_kernels, activation_function):
		"""
		kernel_shape - should be a 2-tuple, e.g, (20,20)
		stride	   - should be a 2-tuple, e.g, (2,2)
		num_kernels  - should be an integer
		activation_function - a Tensorflow activation tensor (e.g., tf.sigmoid)
		"""

		AbstractGene.__init__(self)

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

## generator.py
##
## Functions to generate genes

from genes import *

import numpy as np

# The modified Poisson distribution used for generating parameters
def modifiedPoisson(prob_params):
	"""
	Sample from the modified Poisson distribution
	"""

	return np.random.poisson(prob_params[0]) + prob_params[1]

"""
# Helper functions --- Genotypes will implement one set of generators
# randomly generate a ConvGene based on the lastGene's output dimension
"""
def createConvGeneGenerator(dimension, **kwargs):
	"""
	Build a generator for creating convolutional layers
	"""

	# Pull out arguments, or use defaults
	# Initial creation parameters
	lambda_shape_create = kwargs.get('lambda_shape_create', 3)
	n_min_shape_create = kwargs.get('n_min_shape_create', 2)

	lambda_stride_create = kwargs.get('lambda_stride_create', 0)
	n_min_stride_create = kwargs.get('n_min_stride_create', 1)

	lambda_kernels_create = kwargs.get('lambda_kernels_create', 10)
	n_min_kernels_create = kwargs.get('n_min_kernels_create', 5)

	shape_prob_params = (lambda_shape_create, n_min_shape_create)
	stride_prob_params = (lambda_stride_create, n_min_stride_create)
	kernels_prob_param = (lambda_kernels_create, n_min_kernels_create)

	def generate1DConvGene(lastGene, nextGene):
		"""
		Create a new gene which will fit between the previous and next gene
		"""

		# What are the boundaries of this gene (input and output size)
		input_size = lastGene.outputDimension()[0]
		min_output_size = nextGene.minInputDimension()[0]

		# If the next layer only has a dimensionality of one, it is F.C. or OUTPUT
		if len(nextGene.minInputDimension()) == 1:
			min_output_size = 1

		# Determine what the minimum and maximum size is, assuming a stride of 1
		min_kernel_size = 1
		max_kernel_size = np.floor( input_size - (min_output_size - 1) )

		# It's possible to not create a Gene? (Maybe...but that would imply incompatible parameters)
		if max_kernel_size < min_kernel_size: 
			return None

		# Determine the kernel size
		kernel_size = modifiedPoisson(shape_prob_params)
		kernel_size = max(kernel_size, min_kernel_size)
		kernel_size = min(kernel_size, max_kernel_size)

		# Now determine limits on stride
		min_stride = 1
		max_stride = kernel_size 		# Cannot be bigger than the kernel

		# Possible there is a second constraint based on the output size
		if min_output_size > 1:
			max_stride = np.floor((input_size - kernel_size) / (min_output_size - 1))

		max_stride = min(max_stride, kernel_size)

		stride = modifiedPoisson(stride_prob_params)
		stride = max(stride, min_stride)
		stride = min(stride, max_stride)

		# Finally, determine the number of kernels to have.  There is no limit on this
		num_kernels = modifiedPoisson(kernels_prob_param)

		# Build the Conv1DGene
		kernel_size = (int(kernel_size),)
		stride = (int(stride),)

		gene = Conv1DGene(kernel_size, stride, int(num_kernels), **kwargs)

		return gene


	def generate2DConvGene(lastGene, nextGene):
		"""
		Create a new gene which will fit between the previous and next gene
		"""

		# What are the boundaries of this gene (input and output size)
		input_height, input_width, _ = lastGene.outputDimension()
		min_output_size = nextGene.minInputDimension()

		# If the next layer only has a dimensionality of one, it is F.C. or OUTPUT
		if len(min_output_size) == 1:
			min_output_height, min_output_width = 1, 1
		else:
			min_output_height = min_output_size[0]
			min_output_width = min_output_size[1]

		# Determine what the minimum and maximum size is, assuming a stride of 1
		min_kernel_height, min_kernel_width = 1, 1
		max_kernel_height = np.floor( input_height - (min_output_height - 1) )
		max_kernel_width = np.floor( input_width - (min_output_width - 1) )

		# It's possible to not create a Gene? (Maybe...but that would imply incompatible parameters)
		if max_kernel_height < min_kernel_height: 
			return None
		if max_kernel_width < min_kernel_width: 
			return None

		# Determine the kernel size
		kernel_height = modifiedPoisson(shape_prob_params)
		kernel_height = max(kernel_height, min_kernel_height)
		kernel_height = min(kernel_height, max_kernel_height)

		kernel_width = modifiedPoisson(shape_prob_params)
		kernel_width = max(kernel_width, min_kernel_width)
		kernel_width = min(kernel_width, max_kernel_width)

		# Now determine limits on stride
		min_stride_height, min_stride_width = 1, 1
		max_stride_height, max_stride_width = kernel_height, kernel_width 		# Cannot be bigger than the kernel

		# Possible there is a second constraint based on the output size
		if min_output_height > 1:
			max_stride_height = np.floor((input_height - kernel_height) / (min_output_height - 1))
		if min_output_width > 1:
			max_stride_width = np.floor((input_width - kernel_width) / (min_output_width - 1))

		max_stride_height = min(max_stride_height, kernel_height)
		max_stride_width = min(max_stride_width, kernel_width)

		stride_height = modifiedPoisson(stride_prob_params)
		stride_height = max(stride_height, min_stride_height)
		stride_height = min(stride_height, max_stride_height)

		stride_width = modifiedPoisson(stride_prob_params)
		stride_width = max(stride_width, min_stride_width)
		stride_width = min(stride_width, max_stride_width)

		# Finally, determine the number of kernels to have.  There is no limit on this
		num_kernels = modifiedPoisson(kernels_prob_param)

		# Build the Conv1DGene
		kernel_size = (int(kernel_height), int(kernel_width))
		stride = (int(stride_height), int(stride_width))

		gene = Conv2DGene(kernel_size, stride, int(num_kernels), **kwargs)

		return gene

	if dimension == 1:
		return generate1DConvGene
	else:
		return generate2DConvGene



def createPoolGeneGenerator(dimension, **kwargs):
	"""
	Build a generator for creating pooling layers
	"""

	# Pull out arguments for creation parameters, or use defaults
	lambda_shape_create = kwargs.get('lambda_shape_create', 0)
	n_min_shape_create = kwargs.get('n_min_shape_create', 2)

	lambda_stride_create = kwargs.get('lambda_stride_create', 1)
	n_min_stride_create = kwargs.get('n_min_stride_create', 1)

	shape_prob_params = (lambda_shape_create, n_min_shape_create)
	stride_prob_params = (lambda_stride_create, n_min_stride_create)

	# Two options -- 1D or 2D
	def generate1DPoolGene(lastGene, nextGene):
		"""
		Create a new gene which will fit between the last and next gene
		"""

		# What are the boundaries of this gene (input and output size)
		input_size = lastGene.outputDimension()[0]
		min_output_size = nextGene.minInputDimension()[0]

		# If the next layer only has a dimensionality of one, it is F.C. or output
		if len(nextGene.minInputDimension()) == 1:
			min_output_size = 1

		# Determine what the minimum and maximum pooling size is, assuming a stride of 1
		min_pool_size = 1
		max_pool_size = np.floor( input_size - (min_output_size - 1))

		# Sanity check -- the pool size must be feasible
		if max_pool_size < min_pool_size:
			return None

		# Determine the pool size
		pool_size = modifiedPoisson(shape_prob_params)
		pool_size = max(pool_size, min_pool_size)
		pool_size = min(pool_size, max_pool_size)

		# Now determine the limits on stride
		min_stride = 1
		max_stride = pool_size

		# Possible second constraint based on the output size requirements
		if min_output_size > 1:
			max_stride = np.floor((input_size - pool_size) / (min_output_size - 1))

		max_stride = min(max_stride, pool_size)

		stride = modifiedPoisson(stride_prob_params)
		stride = max(stride, min_stride)
		stride = min(stride, max_stride)

		# Build the Pool1DGene
		pool_size = (int(pool_size),)
		stride = (int(stride),)

		gene = Pool1DGene(pool_size, stride, **kwargs)

		return gene


	def generate2DPoolGene(lastGene, nextGene):
		"""
		Create a new gene which will fit between the last and next gene
		"""

		# What are the boundaries of this gene (input and output size)
		input_height, input_width, _ = lastGene.outputDimension()
		min_output_size = nextGene.minInputDimension()

		# If the next layer only has a dimensionality of one, it is F.C. or output
		if len(nextGene.minInputDimension()) == 1:
			min_output_height, min_output_width = 1, 1
		else:
			min_output_height = min_output_size[0]
			min_output_width = min_output_size[1]


		# Determine what the minimum and maximum pooling size is, assuming a stride of 1
		min_pool_height, min_pool_width = 1, 1
		max_pool_height = np.floor( input_height - (min_output_height - 1) )
		max_pool_width = np.floor( input_width - (min_output_width - 1) )

		# Sanity check -- the pool size must be feasible
		if max_pool_height < min_pool_height:
			return None
		if max_pool_width < min_pool_width:
			return None

		# Determine the pool size
		pool_height = modifiedPoisson(shape_prob_params)
		pool_height = max(pool_height, min_pool_height)
		pool_height = min(pool_height, max_pool_height)

		pool_width = modifiedPoisson(shape_prob_params)
		pool_width = max(pool_width, min_pool_width)
		pool_width = min(pool_width, max_pool_width)

		# Now determine the limits on stride
		min_stride_height, min_stride_width = 1, 1
		max_stride_height, max_stride_width = pool_height, pool_width

		# Possible second constraint based on the output size requirements
		if min_output_height > 1:
			max_stride_height = np.floor((input_height - pool_height) / (min_output_height - 1))
		if min_output_width > 1:
			max_stride_width = np.floor((input_width - pool_width) / (min_output_width - 1))

		max_stride_height = min(max_stride_height, pool_height)
		max_stride_width = min(max_stride_width, pool_width)

		stride_height = modifiedPoisson(stride_prob_params)
		stride_height = max(stride_height, min_stride_height)
		stride_height = min(stride_height, max_stride_height)

		stride_width = modifiedPoisson(stride_prob_params)
		stride_width = max(stride_width, min_stride_width)
		stride_width = min(stride_width, max_stride_width)

		# Build the Pool1DGene
		pool_size = (int(pool_height), int(pool_width))
		stride = (int(stride_height), int(stride_width))

		gene = Pool2DGene(pool_size, stride, **kwargs)

		return gene

	if dimension == 1:
		return generate1DPoolGene
	else:
		return generate2DPoolGene


def createFullConnectionGeneGenerator(**kwargs):
	"""
	Build a generator for creating fully connected layers
	"""

	# Pull out arguments for creation parameters, or use defaults
	lambda_size_create = kwargs.get('lambda_size_create', 10)
	n_min_size_create = kwargs.get('n_min_size_create', 10)

	size_prob_params = (lambda_size_create, n_min_size_create)

	def generateFullConnection(lastGene, nextGene=None):
		"""
		"""

		size = modifiedPoisson(size_prob_params)

		return FullyConnectedGene(size, **kwargs)

	return generateFullConnection
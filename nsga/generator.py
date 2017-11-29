## generator.py
##
## Functions to generate genes

from CNN_Gene import *

import numpy as np

"""
# Helper functions --- Genotypes will implement one set of generators
# randomly generate a ConvGene based on the lastGene's output dimension
"""
def generate1DConvGene(lastGene, nextGene):
	## specify the min and max for each random functions

	# What are the boundaries of this gene (input and output size)
	input_size = lastGene.outputDimension()[0]
	min_output_size = nextGene.minInputDimension()[0]

	# If the next layer is FC or output, then the output can be length of 1
	if len(nextGene.minInputDimension()) == 1:
		min_output_size = 1

	# Figure out the range of sizes for the kernel
	min_size = MIN_CNN_WIDTH
	max_size = MAX_CNN_WIDTH

	# The maximum the kernel can be is
	# input_size - min_output_size + 1
	max_size = min(MAX_CNN_WIDTH, input_size - min_output_size + 1)

	if max_size < MIN_CNN_WIDTH:
		return None

	kernel_size = np.random.randint(MIN_CNN_WIDTH, max_size+1)

	# The stride can be up to
	# ((input_size - kernel_size + 1) / min_output_size) + 1
	max_stride = ((input_size - kernel_size + 1) / min_output_size) + 1
	max_stride = min(MAX_CNN_STRIDE, max_stride)

	# Stride can also not exceed the kernel size
	max_stride = min(max_stride, kernel_size)

	if max_stride < MIN_CNN_STRIDE:
		return None

	conv_stride = np.random.randint(MIN_CNN_STRIDE, max_stride+1)

	# Can have any number of kernels
	num_kernels = np.random.randint(MIN_CNN_KERNELS, MAX_CNN_KERNELS+1)

	# activation_function ???
	return Conv1DGene((kernel_size,), (conv_stride,), num_kernels, activation_function=None)


def generate2DConvGene(lastGene, nextGene):
	## specify the min and max for each random functions

	# What are the boundaries of this gene (input and output size)

	input_height, input_width, _ = lastGene.outputDimension()
	min_output_size = nextGene.minInputDimension()

	min_output_height = 1
	min_output_width = 1

	if len(min_output_size) > 1:
		min_output_height = min_output_size[0]
		min_output_width = min_output_size[1]

	min_height = MIN_CNN_WIDTH
	min_width = MIN_CNN_WIDTH

	# The maximum the kernel can be in either direction is
	# input_size - min_output_size + 1
	max_height = min(MAX_CNN_WIDTH, input_height - min_output_height + 1)
	max_width = min(MAX_CNN_WIDTH, input_width - min_output_width + 1)

	if max_height < MIN_CNN_WIDTH or max_width < MIN_CNN_WIDTH:
		return None

	kernel_height = np.random.randint(MIN_CNN_WIDTH, max_height+1)
	kernel_width = np.random.randint(MIN_CNN_WIDTH, max_width+1)

	# The stride can be up to
	# ((input_size - kernel_size + 1) / min_output_size) + 1
	max_stride_height = ((input_height - kernel_height + 1) / min_output_height) + 1
	max_stride_width = ((input_width - kernel_width + 1) / min_output_width) + 1

	max_stride_height = min(MAX_CNN_STRIDE, max_stride_height)
	max_stride_width = min(MAX_CNN_STRIDE, max_stride_width)

	# Stride cannot exced kernel size
	max_stride_height = min(max_stride_height, kernel_height)
	max_stride_width = min(max_stride_width, kernel_width)

	if max_stride_height < MIN_CNN_STRIDE or max_stride_width < MIN_CNN_STRIDE:
		return None

	conv_stride_height = np.random.randint(MIN_CNN_STRIDE, max_stride_height+1)
	conv_stride_width = np.random.randint(MIN_CNN_STRIDE, max_stride_width+1)

	num_kernels = np.random.randint(MIN_CNN_KERNELS, MAX_CNN_KERNELS+1)

	# activation_function ???
	return Conv2DGene((kernel_height, kernel_width), (conv_stride_height, conv_stride_width), num_kernels, activation_function=None)

"""
# Helper function
# randomly generate a PoolGene based on the lastGene's output dimension
"""
def generate1DPoolGene(lastGene, nextGene):
	## specify the min and max for each random functions
	input_size = lastGene.outputDimension()[0]
	min_output_size = nextGene.minInputDimension()[0]

	# The largest the pooling size can be is
	# input_size - min_output_size + 1
	max_size = min(MAX_POOL_SIZE, input_size - min_output_size + 1)
	if max_size < MIN_POOL_SIZE:
		return None

	pool_size = np.random.randint(MIN_POOL_SIZE, max_size+1)

	# The largest the strice can be is
	# ((input_size - pool_size + 1) / min_output_size) + 1
	max_stride = ((input_size - pool_size + 1) / min_output_size) + 1

	# Stride cannot exceed pool size
	max_stride = min(max_stride, pool_size)
	max_stride = min(max_stride, MAX_POOL_STRIDE)

	if max_strice < MIN_POOL_STRIDE:
		return None

	pool_stride = np.random.randint(MIN_POOL_STRIDE, max_stride + 1)

	# activation_function ???
	return Pool1DGene((pool_size,), (pool_stride,))


def generate2DPoolGene(lastGene, nextGene):

	input_height, input_width, _ = lastGene.outputDimension()
	min_output_size = nextGene.minInputDimension()

	min_output_height = 1
	min_output_width = 1

	if len(min_output_size) > 1:
		min_output_height = min_output_size[0]
		min_output_width = min_output_size[1]

	min_height = MIN_POOL_SIZE
	min_width = MIN_POOL_SIZE

	## specify the min and max for each random functions
	max_height = min(MAX_POOL_SIZE, input_height - min_output_height + 1)
	max_width = min(MAX_POOL_SIZE, input_width - min_output_width + 1)

	if max_height < min_height or max_width < min_width:
		return None

	pool_height = np.random.randint(min_height, max_height+1)
	pool_width = np.random.randint(min_width, max_width+1)

	# Determine maximimum stride
	max_stride_height = ((input_height - pool_height + 1) / min_output_height) + 1
	max_stride_width = ((input_width - pool_width + 1) / min_output_width) + 1

	max_stride_height = min(max_stride_height, pool_height)
	max_stride_height = min(max_stride_height, MAX_POOL_STRIDE)
	max_stride_width = min(max_stride_width, pool_width)
	max_stride_width = min(max_stride_width, MAX_POOL_STRIDE)

	if max_stride_height < MIN_POOL_STRIDE or max_stride_width < MIN_POOL_STRIDE:
		return None

	pool_stride_height = np.random.randint(MIN_POOL_STRIDE, max_stride_height + 1)
	pool_stride_width = np.random.randint(MIN_POOL_STRIDE, max_stride_width + 1)

	# activation_function ???
	return Pool2DGene((pool_height, pool_width), (pool_stride_height, pool_stride_width))



def generateFullConnection(lastGene, nextGene=None):
	## specify the min and max for each random functions
	size = np.random.randint(MIN_FULL_CONNECTION, MAX_FULL_CONNECTION+1)

	# activation_function ???
	return FullyConnectedGene(size, activation_function=None)
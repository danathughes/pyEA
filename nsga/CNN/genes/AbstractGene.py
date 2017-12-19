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
#from models.parts import *

import tensorflow as tf

ACTIVATION_FUNCTIONS = [tf.nn.relu, tf.sigmoid, tf.tanh]


class AbstractGene:
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

		self.tensor = None


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


	def minInputDimension(self):
		"""
		Recurse through next gene to figure out minimum valid input size
		"""

		pass


	def clone(self):
		"""
		Make a copy of this gene
		"""

		pass

	def equals(self, other):
		"""
		If two genes have the same type and data
		"""

		pass

	def mutate(self):
		"""
		Alter this gene, ensuring that the constraints from the previous and next gene are satisfied
		"""

		pass


	def generateLayer(self, input_tensor):
		"""
		Create the CNN part(s) (tuple of objects) used to construct this particular layer in the CNN
		"""

		pass

	def __str__(self):
		"""
		String representation of this gene
		"""

		return "Abstract Gene"


	def equals(self, other):
		"""
		Is this gene the same as another gene?
		"""

		return False


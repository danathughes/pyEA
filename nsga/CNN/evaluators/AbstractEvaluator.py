## SingleNetworkEvaluator.py
##
## Class to evaluate networks as they are added

import os
import cPickle as pickle
import numpy as np

import random

class AbstractEvaluator:
	"""
	"""

	def __init__(self, dataset, **kwargs):
		"""
		Create an object with the dataset loaded, and a path to store individuals and results
		"""

		pass


	def evaluate(self, individual):
		"""
		Evaluate the provided individual
		"""

		pass

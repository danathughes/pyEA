## SingleNetworkEvaluator.py
##
## Class to evaluate networks as they are added

import os
import cPickle as pickle
import numpy as np

import random

class DummyEvaluator:
	"""
	"""

	def __init__(self, dataset_filename, population_path='./population', train_steps=250):
		"""
		Create an object with the dataset loaded, and a path to store individuals and results
		"""

		pass


	def evaluate(self, individual):
		"""
		Evaluate the provided individual
		"""

		# Update the individual's objective
		x = individual.objective[0]
		y = individual.objective[1]

		if x == np.inf:
			x = 100000.
		if y == np.inf:
			y = 100000.

		individual.objective = [np.random.normal(x, x/10), np.random.normal(y, y/10)]


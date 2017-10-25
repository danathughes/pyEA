## ProxyEvaluator.py
##
## Class to hold individuals until a large population can be evaluated at once

import random

class ProxyEvaluator:
	"""
	"""

	def __init__(self):
		"""
		"""

		self.individuals = []


	def add(self, individual):
		"""
		Add an individual to be evaluated
		"""

		self.individuals.append(individual)


	def evaluate(self):
		"""
		"""

		for individual in self.individuals:
			individual.objective[0] += 10*random.random()
			individual.objective[1] += 10*random.random()


		print "Evaluating %d individuals" % len(self.individuals)


	def reset(self):
		"""
		Empty the list of individuals to be evaluated
		"""

		print "Resetting Evaluator"

		self.individuals = []
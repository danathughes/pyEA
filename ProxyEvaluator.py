## ProxyEvaluator.py
##
## Class to hold individuals until a large population can be evaluated at once

import os
import cPickle as pickle

import random

class ProxyEvaluator:
	"""
	"""

	def __init__(self):
		"""
		"""

		self.individuals = []
		self.population_path = './population'

		self.population = 0


	def add(self, individual):
		"""
		Add an individual to be evaluated
		"""

		self.individuals.append(individual)


	def evaluate(self):
		"""
		Save the current set of individuals to a pickle file and call the evaluation program
		"""

		# Save the individuals to a pickle file
		filename = self.population_path + '/generation_%d.pkl' % self.population
		results_name = self.population_path + '/objectives_%d.pkl' % self.population

		with open(filename, 'wb') as pickle_file:
			pickle.dump(self.individuals, pickle_file)

		# Run the evaluation program
		cmd = 'python eval_cnn.py ' + filename + ' mnist.pkl ' + results_name
		os.system(cmd)

		# Copy the objective back into the individuals
		with open(results_name, 'rb') as pickle_file:
			objectives = pickle.load(pickle_file)

		for i in range(len(self.individuals)):
			self.individuals[i].objective = objectives[i]


		print "Evaluating %d individuals:" % len(self.individuals)
		for obj in objectives:
			print '\t', obj


	def reset(self):
		"""
		Empty the list of individuals to be evaluated
		"""

		print "Resetting Evaluator"

		self.individuals = []
		self.population += 1
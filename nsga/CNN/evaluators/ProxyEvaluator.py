## ProxyEvaluator.py
##
## Class to hold individuals until a large population can be evaluated at once

import os
import cPickle as pickle

import random

class MinCNNIndividual():
	"""
	"""

	def __init__(self, individual):
		"""
		"""

		self.input_shape = individual.input_shape
		self.output_size = individual.output_size
		self.genotype = individual.genotype

	def generate_model(self, input_tensor):
		"""
		"""

		if input_tensor == None:
			input_tensor = self.genotype[0].generateLayer(input_tensor)

		prev_tensor = input_tensor

		tensors = []

		for gene in self.genotype:
			prev_tensor = gene.generateLayer(prev_tensor)
			if prev_tensor == None:
				print "AAARG!"
				print gene
				print
			tensors.append(prev_tensor)

		# Return the input and output tensor
		return tensors[0], tensors[-1]


class ProxyEvaluator:
	"""
	"""

	def __init__(self):
		"""
		"""

		self.individuals = []
		self.population_path = './population'

		self.population = 0


	def evaluate(self, individual):
		"""
		Add an individual to be evaluated
		"""

		self.individuals.append(individual)


	def run(self):
		"""
		Save the current set of individuals to a pickle file and call the evaluation program
		"""

		# Save the individuals to a pickle file
#		filename = self.population_path + '/generation_%d.pkl' % self.population
#		results_name = self.population_path + '/objectives_%d.pkl' % self.population



#		for ind in self.individuals:
#			min_pop.append(MinCNNIndividual(ind))

#		for k in range(len(min_pop)/5):

		for k in range(len(self.individuals)):
			filename = self.population_path + '/generation_%d_%d.pkl' % (self.population,k)
			results_name = self.population_path + '/objectives_%d_%d.pkl' % (self.population,k)

			with open(filename, 'wb') as pickle_file:
#				pickle.dump(self.individuals, pickle_file)
				pickle.dump([MinCNNIndividual(self.individuals[k])], pickle_file)

			# Run the evaluation program
			cmd = 'python eval_cnn_xval.py ' + filename + ' mnist.pkl ' + results_name
			os.system(cmd)

			# Copy the objective back into the individuals
			with open(results_name, 'rb') as pickle_file:
				objectives = pickle.load(pickle_file)

			self.individuals[k].objective = objectives[0]


	def reset(self):
		"""
		Empty the list of individuals to be evaluated
		"""

		print "Resetting Evaluator"

		self.individuals = []
		self.population += 1
## CNN_Individual.py
##
##

import random

from CNN_Gene import Genotype
from pyNSGAII import AbstractIndividual


class CNN_Individual(AbstractIndividual):
	"""
	An individual encoding a CNN
	"""

	def __init__(self, input_shape, output_size, evaluator):
		"""
		"""

		AbstractIndividual.__init__(self)

		self.input_shape = input_shape
		self.output_size = output_size

		self.gene = Genotype(input_shape, output_size)
		self.objective = [1000*random.random(), 1000*random.random()]

		self.evaluator = evaluator


	def calculateObjective(self):
		"""
		"""

		self.evaluator.add(self)


	def clone(self):
		"""
		Make a copy of me!
		"""

		clone = CNN_Individual()
		clone.gene = self.gene.clone()
		clone.objective = self.objective


	def crossover(self, other):
		"""
		Perform crossover between these two genes
		"""

		child1 = CNN_Individual(self.input_shape, self.output_size, self.evaluator)
		child2 = CNN_Individual(self.input_shape, self.output_size, self.evaluator)

		gene1, gene2 = self.gene.crossover(other.gene)
		child1.gene = gene1
		child2.gene = gene2

		if gene1 == None:
			child1.gene = self.gene.clone()
			child1.mutate()
		if gene2 == None:
			child2.gene = other.gene.clone()
			child2.mutate()

		return child1, child2


	# Test Conv1D mutate
	def mutate(self):
		"""
		Mutate this individual
		"""
		# Randomly select a gene in the Genotype, [Input, Conv, ..., FC, Output]
		# i_mutateGene = random.randrange(1, len(self.genotype))
		self.gene.mutate()
		# 


	def generate_model(self, namespace=None, input_tensor=None):
		"""
		Build the tensorflow model
		"""

		prev_tensor = input_tensor

		tensors = []

		for gene in self.gene.genotype:
			prev_tensor = gene.generateLayer(prev_tensor)
			tensors.append(prev_tensor)

		# Return the input and output tensor
		return tensors[0], tensors[-1]


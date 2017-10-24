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

	def __init__(self, input_shape, output_size):
		"""
		"""

		AbstractIndividual.__init__(self)

		self.input_shape = input_shape
		self.output_size = output_size

		self.gene = Genotype(input_shape, output_size)
		self.objective = [1000*random.random(), 1000*random.random()]


	def calculateObjective(self):
		"""
		"""

		# Gotta implement this by feeding the gene to a tensorflow evaluator
		self.objective[0] += 10*random.random() - 5
		self.objective[1] += 10*random.random() - 5


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

		child1 = CNN_Individual(self.input_shape, self.output_size)
		child2 = CNN_Individual(self.input_shape, self.output_size)

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


	def mutate(self):
		"""
		Mutate this individual
		"""
		self.gene.mutate()
		# 

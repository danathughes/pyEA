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

	def __init__(self, input_shape, output_size, evaluator, mutation_rate = 0.25):
		"""
		"""

		AbstractIndividual.__init__(self)

		self.input_shape = input_shape
		self.output_size = output_size

		self.gene = Genotype(input_shape, output_size)
		self.objective = [100000.0, 100000.0]

		self.mutation_rate = mutation_rate

		self.evaluator = evaluator


	def calculateObjective(self):
		"""
		"""

		self.evaluator.add(self)


	def clone(self):
		"""
		Make a copy of me!
		"""

		clone = CNN_Individual(self.input_shape, self.output_shape, self.evaluator)
		clone.gene = self.gene.clone()
		clone.objective = self.objective


	def isEqual(self, other):
		"""
		Check if 'I' have the same genotype as 'other' does
		"""

		if len(self.gene.genotype) == len(other.gene.genotype):
			for i in range(len(self.gene.genotype)):
				if str(self.gene.genotype[i]) != str(other.gene.genotype[i]):
					return False
			return True
		else:
			return False

	def crossover(self, other):
		"""
		Perform crossover between these two genes
		"""

		child1 = CNN_Individual(self.input_shape, self.output_size, self.evaluator)
		child2 = CNN_Individual(self.input_shape, self.output_size, self.evaluator)

		gene1, gene2 = self.gene.crossover(other.gene)
		child1.gene = gene1
		child2.gene = gene2

		# If the genes were not created, try mutating the parent genes.  If that doesn't work,
		# just create a whole new gene
		if gene1 == None:
			child1.gene = self.gene.clone()
			mutated = child1.gene.mutate()
			if not mutated:
				child1.gene = Genotype(self.input_shape, self.output_size)
		if gene2 == None:
			child2.gene = other.gene.clone()
			mutated = child2.gene.mutate()
			if not mutated:
				child2.gene = Genotype(self.input_shape, self.output_size)

		return child1, child2


	# Test Conv1D mutate
	def mutate(self):
		"""
		Mutate this individual
		"""

		# The NSGA-II algorithm automatically calls mutate.  Would like to
		# have mutation actually be a rare occurance, due to crossover

		if random.random() < self.mutation_rate:
			self.gene.mutate()


	def generate_model(self, input_tensor):
		"""
		Build the tensorflow model
		"""

		if input_tensor == None:
			print "ARGH!"
			print self
			print
			print self.gene
			print
			return None

		prev_tensor = input_tensor

		tensors = []

		for gene in self.gene.genotype:
			prev_tensor = gene.generateLayer(prev_tensor)
			if prev_tensor == None:
				print "AAARG!"
				print gene
				print
			tensors.append(prev_tensor)

		# Return the input and output tensor
		return tensors[0], tensors[-1]


## CNN_Individual.py
##
##

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


	def calculateObjective(self):
		"""
		"""

		# Gotta implement this by feeding the gene to a tensorflow evaluator
		pass


	def crossover(self, other):
		"""
		Perform crossover between these two genes
		"""

		# What are valid crossover points?
		crossover_points = []

		for i in range(1, len(self.gene)):
			for j in range(1, len(other.gene)):
				if self.gene[i].canFollow(other.gene[j-1]) and other.gene[j].canFollow(self.gene[i-1]):
					# Are we just swapping inputs or outputs?
					if i==1 and j==1:
						pass
					elif i==len(self.gene) and j==len(other.gene):
						pass
					else:
						crossover_points.append((i,j))

		# if the list is empty, cannot do anything
		if len(crossover_points) == 0:
			return None, None

		# Make two new genotypes
		child1 = []
		child2 = []

		crossover_point = random.choice(crossover_points)

		# Populate the first half of each children
		for i in range(crossover_point[0]):
			child1.append(self.gene[i].clone())
		for j in range(crossover_point[1]):
			child2.append(other.gene[j].clone())

		# Populate the second half of each child
		for i in range(crossover_point[0], len(self.gene)):
			child2.append(self.gene[i].clone())
		for j in range(crossover_point[1], len(other.gene)):
			child1.append(other.gene[j].clone())

		# Link the previous and next genes in each child
		for i in range(len(child1) - 1):
			child1[i].next_gene = child1[i+1]
		for i in range(1, len(child1)):
			child1[i].prev_gene = child1[i-1]

		for i in range(len(child2) - 1):
			child2[i].next_gene = child2[i+1]
		for i in range(1, len(child2)):
			child2[i].prev_gene = child2[i-1]

		# Done!
		return child1, child2



	def mutate(self):
		"""
		Mutate this individual
		"""

		pass
## CNN_Individual.py
##
##


"""
3. Evaluate individual

"""

from Gene import *


class CNN_Individual:
	"""
	"""

	def __init__(self, generateGenotype=generateGenotypeProb):
		"""
		"""

		self.genotype = self.generateGenotype(inputGene, ConvProb=0.5, PoolProb=1.0, FullConnectProb = 0.5, is2D=False)
		self.objective = None


	def crossover(self, otherIndividual, crossover_rate):
		"""
		Return two children using the other individual as a second parent
		"""

		pass


	def mutate(self, mutation_rate):
		"""
		"""

		pass


	def generatePhenotype(self):
		"""
		"""

		pass
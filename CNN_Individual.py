## CNN_Individual.py
##
##


"""
1. Evaluate individual, Write to files in cnn_NSGAII.py and then load by eval_XXX.py as the previous program?


"""

from Gene import *


class CNN_Individual:
	"""
	"""

	def __init__(self, generateGenotype=generateGenotypeProb):
		"""
		"""
		if generateGenotype is not None:
			self.genotype = self.generateGenotype(inputGene, ConvProb=0.5, PoolProb=1.0, FullConnectProb = 0.5, is2D=False)
			## How can we correct the genotype if it is not valid ???
		else:
			self.genotype = None
		self.objective = (1.0e8, 1.0e8)


	def __checkValidity(self):
		"""
		"""

		for i in ragne(1,len(self.genotype)):
			preGene = self.genotype(i-1)
			curGene = self.genotype(i)
			if not curGene.canFollow(preGene):
				return False

		return True

	def __correctInvalidGenotype(self):
		"""
		"""

		pass


	def crossover(self, otherInd, crossover_rate=None):
		"""
		Return two children using the other individual as a second parent
		"""

		offspring_A = Individual(generateGenotype=None)
		offspring_B = Individual(generateGenotype=None)

		# one-point crossover
		# select one position 'i_A' in self.genotype, and select one feasible position 'i_B' in otherInd.genotype

		genotype_A = []
		genotype_B = []
		for i in range(0, i_A):
			genotype_A.append(self.genotype[i])
		for i in range(i_B, len(otherInd.genotype)):
			genotype_A.append(otherInd.genotype[i])

		for i in range(0, i_B-1):
			genotype_B.append(otherInd.genotype[i])
		for i in range(i_A+1, len(self.genotype)):
			genotype_B.append(self.genotype[i])

		offspring_A.genotype = genotype_A
		offspring_B.genotype = genotype_B

		return offspring_A, offspring_B


	def mutate(self, mutation_rate):
		"""
		"""
		for i in range(len(self.genotype)):
			if random.random() < mutation_rate:
				pos = i
				if 0 < pos < len(self.genotype)-1:
					preGene = self.genotype[pos-1]
					curGene = self.genotype[pos+1]
				else if pos == 0:
					preGene = None
					curGene = self.genotype[pos+1]
				else:
					preGene = self.genotype[pos-1]
					curGene = None				

				self.genotype[pos].mutate(preGene, curGene)


	def generatePhenotype(self):
		"""
		Do we need this?
		"""

		pass



"""

	def crossover(self, other, crossover_prob=0.5, calc_objectives=True):
		"""
		# Spawn two offspring
		"""

		offspring_A = Individual(self.generatingFunction, self.objectiveFunction)
		offspring_B = Individual(self.generatingFunction, self.objectiveFunction)

		# Go through each chromosome and swap
		for i in range(len(self.gene)):
			c1 = self.gene[i]
			c2 = other.gene[i]
			# Should these chromosomes swap in the offspring?
			if np.random.random() < crossover_prob:
				offspring_A.gene[i] = c1
				offspring_B.gene[i] = c2
			else:
				offspring_B.gene[i] = c1
				offspring_A.gene[i] = c2

		# Calculate the objectives of these offspring
		if calc_objectives:
			offspring_A.calculateObjective()
			offspring_B.calculateObjective()

		return offspring_A, offspring_B


	def mutate(self, mutation_prob=0.1, calc_objectives=True):
		"""
		# Mutate each chromosome with a random probability
		"""

		# Create a dummy gene to pull random ("mutated") chromosomes from
		dummy_gene = self.generatingFunction()

		# Mutate individual chromosomes as needed
		for i in range(len(self.gene)):
			if np.random.random() < mutation_prob:
				self.gene[i] = dummy_gene[i]

		# Calculate the new objective
		if calc_objectives:
			self.calculateObjective()

		return

"""				
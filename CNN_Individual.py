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

	def __init__(self, input_size, output_size, generateGenotype):
		"""
		input_size = (height, width, num_channels)
		"""
		self.input_size = input_size
		self.output_size = output_size
		self.generateGenotype = generateGenotype
		if self.generateGenotype is not None:
			self.genotype = self.generateGenotype(input_size, output_size, ConvProb=0.9, PoolProb=1.0, FullConnectProb = 0.5, is2D=False)
			## How can we correct the genotype if it is not valid ???
		else:
			self.genotype = None
		


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


	def crossover(self, otherIndividual, crossover_rate=None):
		"""
		Return two children using the other individual as a second parent
		"""

		offspring_A = CNN_Individual(self.input_size, self.output_size, generateGenotype=None)
		offspring_B = CNN_Individual(self.input_size, self.output_size, generateGenotype=None)

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
				elif pos == 0:
					preGene = None
					curGene = self.genotype[pos+1]
				else:
					preGene = self.genotype[pos-1]
					curGene = None				

				self.genotype[pos].mutate(preGene, curGene)
			
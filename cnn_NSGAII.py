## pyNSGAII
##
## Simple program implementing the Non-dominated Sorting Genetic Algorithm II
## (NSGA-II) on a test problem.
##
## Reference:
## Deb, Kalyanmoy and Pratap, Amrit and Agarwal, Sameer and Meyarivan, T., 
## "A Fast and Elistist Multiobjective Genetic Algorithm: NSGA-II," 
## IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, April 2002.

import numpy as np
import matplotlib.pyplot as plt

import os, sys

def createCNNgenotype(N=2):
	"""
	Create an N layer (conv-pool) CNN encoding
	"""

	MIN_CNN_WIDTH = 2
	MAX_CNN_WIDTH = 75
	MIN_CNN_KERNELS = 5
	MAX_CNN_KERNELS = 30
	MIN_CNN_STRIDE = 1
	MAX_CNN_STRIDE = 5
	MIN_POOL_SIZE = 2
	MAX_POOL_SIZE = 5
	MIN_POOL_STRIDE = 1
	MAX_POOL_STRIDE = 5
	MIN_FULL_CONNECTION = 5
	MAX_FULL_CONNECTION = 200

	# Genotype is [CNN_width, num_CNN_kernels, CNN_stride, pool_size, pool_stride] * N
	# followed by number of full connections.  In total, an integer string of length 
	# 5*N+1.

	genotype = [0]*(5*N+1)

	# Fill in the CNN and pooling layer encodings
	for i in range(N):
		genotype[5*i+0] = np.random.randint(MIN_CNN_WIDTH, MAX_CNN_WIDTH+1)
		genotype[5*i+1] = np.random.randint(MIN_CNN_KERNELS, MAX_CNN_KERNELS+1)
		genotype[5*i+2] = np.random.randint(MIN_CNN_STRIDE, MAX_CNN_STRIDE+1)
		genotype[5*i+3] = np.random.randint(MIN_POOL_SIZE, MAX_POOL_SIZE+1)
		genotype[5*i+4] = np.random.randint(MIN_POOL_STRIDE, MAX_POOL_STRIDE+1)

	# Fill in the fully connected layer bit
	genotype[5*N] = np.random.randint(MIN_FULL_CONNECTION, MAX_FULL_CONNECTION+1)

	return genotype


def dummyObjective(gene):
	"""
	"""

	return 0


# Keep track of how many CNNs were created
class Counter:
	"""
	"""

	def __init__(self):
		self.count = 0

	def increment(self):
		self.count += 1


counter = Counter()


class Individual:
	"""
	An individual consists of a gene and an objective
	"""

	def __init__(self, generatingFunction=createCNNgenotype, objectiveFunction=dummyObjective, counter=counter):
		"""
		Create a new individual using the gene generating function
		"""

		self.generatingFunction = generatingFunction
		self.objectiveFunction = objectiveFunction

		self.gene = generatingFunction()
		self.objective = objectiveFunction(self.gene)

		self.dominationSet = set()
		self.numDominated = 0
		self.rank = 0

		self.crowdingDistance = 0

		self.name = 'cnn_%d' % counter.count
		counter.increment()

	def calculateObjective(self):
		"""
		"""

		self.objective = self.objectiveFunction(self.gene)


	def dominates(self, other):
		"""
		Return True if this Individual dominates Individual other
		"""

		dominates = True
		dominate_one = False

		for o1,o2 in zip(self.objective, other.objective):
			# To dominate, all of this Individual's objectives must be at least
			# as good as the other's
			dominates = dominates and o1 <= o2

			# To dominate, at least one of this Individua's objectives must be
			# better than the other's
			dominate_one = dominate_one or o1 < o2

		return dominates and dominate_one


	def crossover(self, other, crossover_prob=0.5, calc_objectives=True):
		"""
		Spawn two offspring
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
		Mutate each chromosome with a random probability
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


### Routine GA functions
def tournamentSelection(population, k=2, p=0.5):
	"""
	Perform tournament selection on the (assumed sorted) population.  Size of
	tournament is k, selection probability is p
	"""

	assert k <= len(population), "Tournament size cannot exceed population size"

	# Select k indices at random from the list
	indices = np.random.choice(range(len(population)), k, False).tolist()
	indices.sort()

	# Select an individual
	selection = None

	for i in indices:
		# Select this index?
		if np.random.random() < p and not selection:
			selection = population[i]

	# Was an individual selected from the population?  If not, assign the final
	# index 
	if not selection:
		selection = population[indices[-1]]

	return selection


### NSGA Algorithm

def write_file(filename, population):
	"""
	"""

	f = open(filename, 'w')
	for p in population:
		name = p.name
		data = ','.join([str(g) for g in p.gene])
		f.write(name + ',' + data + '\n')
	f.close()


def assign_objectives(filename, population):
	f = open(filename)
	lines = f.readlines()
	f.close()

	results = {}
	for l in lines:
		l = l.strip().split(',')
		results[l[0]] = [float(l[1]), float(l[2])]

	for p in population:
		p.objective = results[p.name]


class NSGA_II:
	"""
	"""

	def __init__(self, population_size):
		"""
		Create a new population
		"""

		assert population_size > 1, "Need a population of at least two to perform GA."

		self.population_size = population_size
		self.population = [Individual() for i in range(self.population_size)]
		self.generation = 0

		self.selection = tournamentSelection

		# Evaluate the first population
#		write_file('./gene_eval/Generation0_genes.dat', self.population)
#		cmd = 'python eval_cnn_phenotypes.py ./gene_eval/Generation0_genes.dat ./gene_eval/Generation0_results.dat'
#		os.system(cmd)
#		assign_objectives('./gene_eval/Generation0_results.dat', self.population)


	def generate_children(self):
		"""
		Perform an GA step
		"""

		# Create a new population
		children = []

		for i in range(0,len(self.population),2):
			# Select two individuals using tournament selection
			parent1 = self.selection(self.population)
			parent2 = self.selection(self.population)

			# Perform crossover and mutation
			offspring1, offspring2 = parent1.crossover(parent2)
			offspring1.mutate()
			offspring2.mutate()

			# Add these to the new population
			children.append(offspring1)
			children.append(offspring2)

		# There may be an extra individual if the population had an odd number
		return children[:self.population_size]


	def step(self):
		"""
		Perform an NSGA-II step
		"""

		parents = self.population
		children = self.generate_children()

		# Evaluate the children
		prefix = './gene_eval/Children%d_' % self.generation
		write_file(prefix + 'genes.dat', children)
		cmd = 'python eval_cnn_phenotypes.py ' + prefix + 'genes.dat ' + prefix + 'results.dat'
		os.system(cmd)
		assign_objectives(prefix+'results.dat', children)


		# Temporarily merge the two populations to create a new one
		self.population = parents + children

		# Sort the population
		self.sortPopulation()

		# Trim the rejected half of the population
		self.population = self.population[:self.population_size]

		# Increment the generation count
		self.generation += 1


	def sortPopulation(self):
		"""
		Sort the population
		"""

		# Determine the non-dominated fronts
		fronts = self.__fastNonDominatedSort()

		# Assign a crowding distance to each individual in each front
		for f in fronts:
			self.__crowdingDistanceAssignment(f)

		# Sort the population
		self.__crowdingDistanceSort()


	def __fastNonDominatedSort(self):
		"""
		Sort the population accoring to non-dominated fronts

		Assumes population consists of tuples - (gene, objective)
		"""

		fronts = [[]]
		ranks = {}

		# Part 1 - find the domination front and domination number of individuals
		for p in self.population:
			p.dominationSet = set()
			p.numDominated = 0

			for q in self.population:
				if p != q:
					if p.dominates(q):
						p.dominationSet.add(q)     # q is dominated by p
					elif q.dominates(p):
						p.numDominated += 1		# p dominated by another

			if p.numDominated == 0:				# p belongs to the first front
				p.rank = 0
				fronts[0].append(p)

		# Part 2 - sort the other solutions
		i = 0

		while fronts[i]:
			Q = []									# The next front

			for p in fronts[i]:
				for q in p.dominationSet:
					q.numDominated -= 1

					if q.numDominated == 0:		# q belongs to the next front
						q.rank = i + 1
						Q.append(q)

			i+= 1
			fronts.append(Q)

		return fronts[:-1]


	def __crowdingDistanceAssignment(self, front):
		"""
		Assigns a distance metric to each individual in the front based on how many
		solutions are nearby in the front
		"""

		# Don't do anything if the front is empty
		if len(front) == 0:
			return

		# How many objectives does an individual have?
		numObjectives = len(front[0].objective)

		for individual in front:
			individual.crowdingDistance = 0.0

			for m in range(numObjectives):
				# Sort by the objective
				obj_and_front = [(ind.objective[m], ind) for ind in front]
				obj_and_front.sort()

				# What is the minimum and maximum objective value?
				f_min = obj_and_front[0][0]
				f_max = obj_and_front[-1][0]

				# If f_max == f_min, there will be a divide by zero error.
				# This is a temporary (?) fix
				if f_max == f_min:
					f_max += 1.0

				# Assign the distance at the endpoints to infinity
				obj_and_front[0][1].crowdingDistance = np.inf
				obj_and_front[-1][1].crowdingDistance = np.inf

				# Update the remaining instances
				for i in range(1,len(obj_and_front)-1):
					ind = obj_and_front[i][1]
					prev_m = obj_and_front[i-1][0]
					next_m = obj_and_front[i+1][0]
					ind.crowdingDistance += (next_m - prev_m) / (f_max - f_min)

		# All done!
		return


	def __crowdingDistanceSort(self):
		"""
		Sort the population according to the ranking and crowding distance.

		NOTE:  It is assumed that rank and crowdingDistance is calculated for each
		       individual in the population.
		"""

		# Create a tuple which allows for sorting primarily on rank, followed by
		# crowding distance.  The negative of the crowding distance is used to 
		# ensure that less crowded regions are prefered.
		rank_and_individual = [((ind.rank, -ind.crowdingDistance), ind) for ind in self.population]
		rank_and_individual.sort()

		# Assign the sorted individuals to the population
		self.population = [x[1] for x in rank_and_individual]

		return


def get_pts(pop):
	"""
	"""

	x = []
	y = []

	for p in pop:
		x.append(p.objective[0])
		y.append(p.objective[1])

	return x,y


def test(pop_size, num_steps, delay=0.5, ax_range=[0,0.001,0,25000]):
	"""
	Run a test
	"""

	import time

	# Build the NSGA-II test
	nsga = NSGA_II(pop_size)

	# Create a plot to show how things are going
	plt.ion()

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.axis(ax_range)

	x,y = get_pts(nsga.population)
	plt_data, = ax.plot(x, y, 'o')

	fig.canvas.draw()

	for i in range(num_steps):
		nsga.step()

		print "Step #%d" % (i+1)
		x,y = get_pts(nsga.population)
		plt_data.set_xdata(x)
		plt_data.set_ydata(y)
		fig.canvas.draw()

	print "Done.  10 best solutions are:"

	nsga.sortPopulation()

	for i in range(10):
		print "Solution #%d" % i
		print "  Variable: ", nsga.population[i].gene
		print "  Objective:", nsga.population[i].objective

	print "Press a key"
	a = input()


if __name__=='__main__':
	test(10, 100)
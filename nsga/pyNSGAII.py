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

import os
import re
import cPickle as pickle


class AbstractIndividual:
	"""
	An individual consists of a gene and an objective
	"""

	def __init__(self):
		"""
		Create a new individual using the gene generating function
		"""

		self.gene = None
		self.objective = None

		self.dominationSet = set()
		self.numDominated = 0
		self.rank = 0

		self.crowdingDistance = 0


	def calculateObjective(self):
		"""
		"""

		pass


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

		pass


	def mutate(self, mutation_prob=0.1, calc_objectives=True):
		"""
		Mutate each chromosome with a random probability
		"""

		pass


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

class NSGA_II:
	"""
	"""

	def __init__(self, population_size, IndividualClass, **kwargs):
		"""
		Create a new population
		"""

		self.Individual = IndividualClass

		assert population_size > 1, "Need a population of at least two to perform GA."

		self.population_size = population_size
		self.population = [self.Individual() for i in range(self.population_size)]
		self.generation = 0
		self.population = []
		self.selection = tournamentSelection

		# Possible callback function(s)
		self.callbacks = []
		self.step_callback = kwargs.get('step_callback', None)
		self.sort_callback = kwargs.get('sort_callback', None)
		# self.restore = restoreClass

	def initialize(self):
		self.population = [self.Individual() for i in range(self.population_size)]

		for individual in self.population:
			individual.calculateObjective()


	def restore(self, restore_path):
		"""
		Collect all files' paths in the given restore_path;
		Read the individuals and objectives into self.population
		Sort the population
		Trunk population and select the top population_size individuals
		"""

		# individual includes genotype and its objectives
		individual_list = []
		# for i in os.listdir(restore_path):
		# 	if os.path.isfile(os.path.join(restore_path, i)):
		# 		print os.path.join(restore_path, i)

		for dirpath, directories, filenames in os.walk(restore_path):
			# for directory in directories:
			# 	print os.path.join(dirpath, directory)
			for filename in filenames:
				if 'objectives' in filename:
					# name_list = re.split('[_  .]',filename)
					name_list = re.findall(r'\d+', filename)
					nums = [int(s) for s in name_list if s.isdigit()]
					num = nums[0]
					individual_name = 'individual_%d.pkl' % num
					individual_path = os.path.join(dirpath, individual_name)
					objective_path = os.path.join(dirpath, filename)
					# print (individual_path, objectives_path)
					if os.path.isfile(individual_path) and os.path.isfile(objective_path):
						individual_list.append((individual_path, objective_path))

		num_individual = len(individual_list)

		# initialize a new population
		self.population = [self.Individual() for i in range(num_individual)]

		# read the data into the new population
		for i in range(num_individual):
			with open(individual_list[i][0]) as pickle_file:
				genotype = pickle.load(pickle_file)
			with open(individual_list[i][1]) as pickle_file:
				objective = pickle.load(pickle_file)

			self.population[i].genotype = genotype
			self.population[i].objective = objective

		self.sortPopulation()

		# print "All Objectives:"
		# for individual in self.population:
		# 	print individual.objective

		if len(self.population) < self.population_size:
			print "The individuals loaded are not enough to form a population, reinitialize..."
			print self.initialize()
		else:
			self.population = self.population[0:self.population_size]
			print "Done restoring the population!"


	def add_callback(self, callback, trigger):
		"""
		Add a callback function
		"""

		self.callbacks.append(callback)


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

			# Evaluate each child's objective function
			offspring1.calculateObjective()
			offspring2.calculateObjective()

		# There may be an extra individual if the population had an odd number
		return children[:self.population_size]


	def step(self):
		"""
		Perform an NSGA-II step
		"""

		# Call the step callback function
		if self.step_callback:
			self.step_callback()

		parents = self.population
		children = self.generate_children()

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

		if self.sort_callback:
			self.sort_callback()

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

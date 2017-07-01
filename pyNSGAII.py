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


## The SCH problem

# Problem definitions
def SCH(x):
	"""
	n = 1
	Variable bounds = [-10^3, 10^3]
	Objectives:
	  f1(x) = x^2
	  f2(x) = (x-2)^2
	Optimal solutions:
	  x in [0,2]
	convex
	"""

	assert x >= -1e3, "x is out of bounds: %f" % x
	assert x <= 1e3, "x is out of bounds: %f" % x

	return [x**2, (x-2)**2]

# Generating function
def generateRandomGenotypeSCH(n=1, bounds=[-1e3, 1e3]):
	"""
	Create a new gene with the desired dimensionality and witning the 
	bounds given
	"""

	return np.random.random()*2e1 - 1e1

###

class Individual:
	"""
	An individual consists of a gene and an objective
	"""

	def __init__(self, generatingFunction=generateRandomGenotypeSCH, objectiveFunction=SCH):
		"""
		Create a new individual using the gene generating function
		"""

		self.gene = generatingFunction()
		self.objective = objectiveFunction(self.gene)


def createPopulation(size):
	"""
	Create a population of the given size using the function provided
	to generate new genes
	"""

	return [Individual() for i in range(size)]


def dominates(x, y):
	"""
	Return True if Individual x dominates Individual y.
	"""

	dominates = True
	dominate_one = False

	for i,j in zip(x.objective, y.objective):
		dominates = dominates and i <= j
		dominate_one = dominate_one or i < j

	return dominates and dominate_one


def fastNonDominatedSort(population):
	"""
	Sort the population accoring to non-dominated fronts

	Assumes population consists of tuples - (gene, objective)
	"""

	dominationSet = {}
	numDominated = {}
	fronts = [[]]
	ranks = {}

	# Part 1 - find the domination front and domination number of individuals
	for p in population:
		dominationSet[p] = set()
		numDominated[p] = 0

		for q in population:
			if p != q:
				if dominates(p,q):
					dominationSet[p].add(q)     # q is dominated by p
				elif dominates(q,p):
					numDominated[p] += 1		# p dominated by another

		if numDominated[p] == 0:				# p belongs to the first front
			ranks[p] = 0
			fronts[0].append(p)

	# Part 2 - sort the other solutions
	i = 0

	while fronts[i]:
		Q = []									# The next front

		for p in fronts[i]:
			for q in dominationSet[p]:
				numDominated[q] -= 1

				if numDominated[q] == 0:		# q belongs to the next front
					ranks[q] = i + 1
					Q.append(q)

		i+= 1
		fronts.append(Q)

	return fronts[:-1]










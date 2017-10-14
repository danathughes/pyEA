## operations.py
##
## 

import random

from Gene import *

def crossover(gene1, gene2):
	"""
	Create two new genes
	"""

	# What are valid crossover points?
	crossover_points = []

	for i in range(1, len(gene1)):
		for j in range(1, len(gene2)):
			if gene1[i].canFollow(gene2[j-1]) and gene2[j].canFollow(gene1[i-1]):
				# Are we just swapping inputs or outputs?
				if i==1 and j==1:
					pass
				elif i==len(gene1) and j==len(gene2):
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
		child1.append(gene1[i].clone())
	for j in range(crossover_point[1]):
		child2.append(gene2[j].clone())

	# Populate the second half of each child
	for i in range(crossover_point[0], len(gene1)):
		child2.append(gene1[i].clone())
	for j in range(crossover_point[1], len(gene2)):
		child1.append(gene2[j].clone())

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

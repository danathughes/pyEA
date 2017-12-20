# population_tracker.py
#
# Keeps track of individuals, so that other individuals can check to see if they have been
# evaluated before

class PopulationTracker:
	"""
	Keep track of individuals
	"""

	def __init__(self):
		"""
		"""

		self.population = []


	def add(self, individual):
		"""
		"""

		self.population.append(individual)


	def contains(self, individual):
		"""
		"""

		for other in self.population:
			if individual.equals(other):
				return True

		return False
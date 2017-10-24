## cnn_ga.py
##
##

from pyNSGAII import NSGA_II
from CNN_Individual import CNN_Individual as TmpIndividual
from visualizer import *


# Parameters
POPULATION_SIZE = 200
INPUT_SHAPE = (100,100,3)
OUTPUT_SIZE = 10

class CNN_Individual(TmpIndividual):
	"""
	A CNN individual with the input shape and output size defined
	"""

	def __init__(self):
		"""
		"""

		TmpIndividual.__init__(self, INPUT_SHAPE, OUTPUT_SIZE)



if __name__ == '__main__':
	ga = NSGA_II(POPULATION_SIZE, CNN_Individual)
	vis = Visualizer()

	vis.plot(ga.population)

	for i in range(1000):
		ga.step()
		vis.plot(ga.population)

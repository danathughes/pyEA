## cnn_ga.py
##
##


from __future__ import print_function

from pyNSGAII import NSGA_II
from CNN_Individual import CNN_Individual as TmpIndividual
from visualizer import *

from ProxyEvaluator import *

# Parameters
POPULATION_SIZE = 200
INPUT_SHAPE = (100,3)
OUTPUT_SIZE = 10

TENSORFLOW_EVALUATOR = ProxyEvaluator()


class CNN_Individual(TmpIndividual):
	"""
	A CNN individual with the input shape and output size defined
	"""

	def __init__(self):
		"""
		"""

		TmpIndividual.__init__(self, INPUT_SHAPE, OUTPUT_SIZE, TENSORFLOW_EVALUATOR)



if __name__ == '__main__':
	ga = NSGA_II(POPULATION_SIZE, CNN_Individual, 
		          sort_callback=TENSORFLOW_EVALUATOR.evaluate, 
		          step_callback=TENSORFLOW_EVALUATOR.reset)
	vis = Visualizer()

	vis.plot(ga.population)

	for i in range(1000):
		ga.step()
		vis.plot(ga.population)

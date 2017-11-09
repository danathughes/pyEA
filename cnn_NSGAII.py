## cnn_ga.py
##
##


# from __future__ import print_function

from pyNSGAII import NSGA_II
from CNN_Individual import CNN_Individual as TmpIndividual
from visualizer import *

from ProxyEvaluator import *

import config_loader


TENSORFLOW_EVALUATOR = ProxyEvaluator()

config = config_loader.load('NSGA_II.cfg')
INPUT_SHAPE = config['input_shape']
OUTPUT_SIZE = config['output_size']

class CNN_Individual(TmpIndividual):
	"""
	A CNN individual with the input shape and output size defined
	"""

	def __init__(self):
		"""
		"""

		TmpIndividual.__init__(self, INPUT_SHAPE, OUTPUT_SIZE, TENSORFLOW_EVALUATOR)



if __name__ == '__main__':

	ga = NSGA_II(config['population_size'], CNN_Individual,
		          sort_callback=TENSORFLOW_EVALUATOR.evaluate,
		          step_callback=TENSORFLOW_EVALUATOR.reset)

	vis = Visualizer()

	# Evaluate the initial population
	for individual in ga.population:
		individual.calculateObjective()

	TENSORFLOW_EVALUATOR.evaluate()

	vis.plot(ga.population)

	for i in range(100):
		ga.step()
		vis.plot(ga.population)

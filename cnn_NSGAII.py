## cnn_ga.py
##
##


# from __future__ import print_function

from pyNSGAII import NSGA_II
from CNN_Individual import CNN_Individual as TmpIndividual
from visualizer import *
import ConfigParser # Python3: configparser

from ProxyEvaluator import *

# Read parameters from Congfig file

config = ConfigParser.ConfigParser()
config.read('NSGA_II.cfg')

POPULATION_SIZE = config.getint('NSGA_II', 'population_size')
OUTPUT_SIZE = config.getint('NSGA_II', 'output_size')

input_shape1 = config.getint('NSGA_II', 'input_shape1')
input_shape2 = config.getint('NSGA_II', 'input_shape2')
input_shape3 = config.getint('NSGA_II', 'input_shape3')

if input_shape3 == 0:
	INPUT_SHAPE = (input_shape1, input_shape2)
else:
	INPUT_SHAPE = (input_shape1, input_shape2, input_shape3)

print 'INPUT_SHAPE:\t\t', INPUT_SHAPE
print 'OUTPUT_SIZE:\t\t', OUTPUT_SIZE
print 'POPULATION_SIZE:\t', POPULATION_SIZE

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

## cnn_ga.py
##
##


# from __future__ import print_function

from nsga.pyNSGAII import NSGA_II
from nsga import CNN
from nsga.utils.visualizer import *
from nsga.utils import config_loader
from nsga.utils.population_tracker import PopulationTracker

from nsga.CNN.evaluators import *

#TENSORFLOW_EVALUATOR = ProxyEvaluator()
#TENSORFLOW_EVALUATOR = SingleNetworkEvaluator('cifar_10.pkl')
#TENSORFLOW_EVALUATOR = SingleNetworkEvaluator('mnist.pkl')
#TENSORFLOW_EVALUATOR = MultiNetworkEvaluator('mnist.pkl', 5, population_path='./experiments/mnist/population', max_train_steps=5, min_train_steps=1)
#TENSORFLOW_EVALUATOR = MultiNetworkEvaluatorKFold('mnist.pkl', 5, population_path='./experiments/mnist/population', max_train_steps=5, min_train_steps=1, num_folds=2)
#TENSORFLOW_EVALUATOR = SingleNetworkEvaluator('mnist.pkl', population_path='./experiments/mnist/population', max_train_steps=5, min_train_steps=1)
TENSORFLOW_EVALUATOR = SingleNetworkEvaluatorKFold('mnist.pkl', population_path='./experiments/mnist/population', max_train_steps=5, min_train_steps=1, num_folds=2)
#TENSORFLOW_EVALUATOR = DummyEvaluator('mnist.pkl')

POPULATION_TRACKER = PopulationTracker()

config = config_loader.load('NSGA_II.cfg')
INPUT_SHAPE = config['input_shape']
OUTPUT_SIZE = config['output_size']

class CNN_Individual(CNN.Individual):
	"""
	A CNN individual with the input shape and output size defined
	"""

	def __init__(self):
		"""
		"""

		CNN.Individual.__init__(self, INPUT_SHAPE, OUTPUT_SIZE, TENSORFLOW_EVALUATOR, population_tracker=POPULATION_TRACKER)



if __name__ == '__main__':

#	ga = NSGA_II(config['population_size'], CNN_Individual,
#		          sort_callback=TENSORFLOW_EVALUATOR.evaluate,
#		          step_callback=TENSORFLOW_EVALUATOR.reset)

	ga = NSGA_II(config['population_size'], CNN_Individual)

	vis = Visualizer([0,1], [0,1.00000,0,100000])

	# Evaluate the initial population
	for individual in ga.population:
		individual.calculateObjective()

	vis.plot(ga.population)

	print "=== Initial Population"
	print "Current Population Objectives:"
#	for p in ga.population:
#		print "  ", p.objective


	for i in range(100):
		ga.step()
		vis.plot(ga.population)
		print "=== Population %d" % (i+1)
#		print "Current Population Objectives:"
#		for p in ga.population:
#			print "  ", p.objective

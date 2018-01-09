# PoolEvaluator.py

from SingleNetworkEvaluator import *

import sys
IS_PY2 = sys.version_info < (3, 0)

if IS_PY2:
    from Queue import Queue
else:
    from queue import Queue

from threading import Thread


class Worker(Thread):
	""" Thread evaluating individuals from a given individuals queue """
	def __init__(self, evaluator, individuals):
		Thread.__init__(self)
		self.individuals = individuals
		self.evaluator_id = id
		self.daemon = True
		self.evaluator = evaluator
		self.start()

	def run(self):
		while True:
			individual = self.individuals.get()
			try:
				self.evaluator.evaluate(individual)

			except Exception as e:
				""" An exception happened in this thread """
				print(e)

			finally:
				""" Mark this individual evaluation as done, whether an exception happened or not """
				self.individuals.task_done()


class ThreadPool:
	""" Pool of threads evaluating individuals from a queue """
	def __init__(self, evaluator_list, population_size):
		self.individuals = Queue(population_size)
		for evaluator in evaluator_list:
			Worker(evaluator, self.individuals)

	def add_individual(self, individual):
		""" Add a individual to the queue """
		self.individuals.put(individual)

	def map(self, individual_list):
		""" Add a list of tasks to the queue """
		for individual in individual_list:
			self.add_individual(individual)

	def wait_completion(self):
		""" Wait for completion of all the tasks in the queue """
		self.individuals.join()


class ThreadPoolEvaluator:
	def __init__(self, dataset_filename, population_path, num_threads, population_size):
		self.population_size = population_size
		self.num_individual = 0

		""" Creat num_threads evaluators on different GPUs """
		evaluator_list = []
		for i in range(num_threads):
			device_id = '/device:GPU:%d' % i
			population_path_device = population_path + '/gpu_%d' % i
			evaluator_list.append( SingleNetworkEvaluator(dataset_filename, population_path_device, gpu_id=device_id) )

		""" Instantiate a thread pool with NUM_THREADS evaluator threads """
		self.pool = ThreadPool(evaluator_list, self.population_size)


	def evaluate(self, individual):
		self.pool.add_individual(individual)
		self.num_individual += 1

		""" Once a whole population is added to the queue,
		the program pauses untill the population is evaluated """
		if self.num_individual >= self.population_size:
			self.pool.wait_completion()
			# reset the counter
			self.num_individual = 0

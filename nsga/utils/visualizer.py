## visualizer.py
##
##

import matplotlib.pyplot as plt


class Visualizer:
	"""
	Plots Pareto front of a population
	"""

	def __init__(self, indices = [0,1], axis_range = [0.0, 1.0, 0.0, 50000.0], save_old = True):
		"""
		"""

		# Should old objectives be saved?
		if save_old:
			plt.ion()

		self.fig = plt.figure()

		self.ax = self.fig.add_subplot(111)
		self.ax.axis(axis_range)

		self.x_ind = indices[0]
		self.y_ind = indices[1]

		self.plt_data, = self.ax.plot([], [], 'o')


	def plot(self, population):
		"""
		Plot the objective function of the population
		"""

		x = [ind.objective[self.x_ind] for ind in population]
		y = [ind.objective[self.y_ind] for ind in population]

		self.plt_data.set_xdata(x)
		self.plt_data.set_ydata(y)

		self.fig.canvas.draw()

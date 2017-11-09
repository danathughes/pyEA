# config_loader.py
#
# Loads configuration files


import ConfigParser # Python3: configparser


# Read parameters from Congfig file
def load(path):
	"""
	"""

	config = {}

	parser = ConfigParser.ConfigParser()
	parser.read('NSGA_II.cfg')

	config['population_size'] = parser.getint('NSGA_II', 'population_size')
	config['output_size'] = parser.getint('NSGA_II', 'output_size')

	input_shape1 = parser.getint('NSGA_II', 'input_shape1')
	input_shape2 = parser.getint('NSGA_II', 'input_shape2')
	input_shape3 = parser.getint('NSGA_II', 'input_shape3')

	if input_shape3 == 0:
		config['input_shape'] = (input_shape1, input_shape2)
	else:
		config['input_shape'] = (input_shape1, input_shape2, input_shape3)

	return config
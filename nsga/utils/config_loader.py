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
	config['num_threads'] = parser.getint('NSGA_II', 'num_threads')
	config['restore_path'] = parser.get('NSGA_II', 'restore_path')
	config['population_path'] = parser.get('NSGA_II', 'population_path')
	config['dataset_filename'] = parser.get('NSGA_II', 'dataset_filename')

	input_shape1 = parser.getint('NSGA_II', 'input_shape1')
	input_shape2 = parser.getint('NSGA_II', 'input_shape2')
	input_shape3 = parser.getint('NSGA_II', 'input_shape3')

	if input_shape3 == 0:
		config['input_shape'] = (input_shape1, input_shape2)
	else:
		config['input_shape'] = (input_shape1, input_shape2, input_shape3)

	return config

## opportunity_full_cnn_trainer.py
##
## This script trains a CNN to perform classification on individual frames of data  

import tensorflow as tf

from datasets.firstTireData import *
from batch_generator import BatchGenerator

import cPickle as pickle
import random

from models.parts import *
from models.CNN1D_GA import CNN1D
from models.parts_GA import *

import cPickle as pickle

import sys

# Windowing information
WINDOW_SIZE = 200
NUM_SENSORS = 1
NUM_CLASSES = 4

BATCH_SIZE = 50
WEIGHT_DECAY_RATE = 1e-8
LEARNING_RATE = 1e-4

SPLIT_RATIO = (0.8, 0.1, 0.1)

NUM_EPOCHS = 500

DATA_FILE = 'maro_dataset.pkl'

def decode_layers(_genotype):
  """
  ['cnn_xxx', ('INPUT', kernel_size, stride, num_kernels), ('CONV1D', kernel_size, stride), ('FULLY_CONNECTED', size)] to layers
  """
  layers = []

  # Disregard the layer name
  genotype = _genotype[1:]

  for gene in genotype:
    if gene[0] in [INPUT, CONV1D, CONV2D]:
      layers.append(Convolutional1D(gene[1], gene[3], stride=gene[2]))
    else if gene[0] in [POOL1D, POOL2D]:
      layers.append(Pool1D(gene[1], stride=gene[2]))
      layers.append(ReLU())
    else:
      # Do the fully connected layer
      layers.append(Flatten())
      layers.append(FullConnection(gene[1]))

  return layers


def load_cnn_layers(filename):
  """
  convert  ['cnn_xxx', 'INPUT', kernel_size, stride, num_kernels, 'CONV1D', kernel_size, stride, 'FULLY_CONNECTED', size] to
  ['cnn_xxx', ('INPUT', kernel_size, stride, num_kernels), ('CONV1D', kernel_size, stride), ('FULLY_CONNECTED', size)]
  """

  f = open(filename)
  lines = f.readlines()
  f.close()

  cnn_layer_list = []

  for line in lines:
    line = line.strip().split(',')
    data = [line[0]]
    i = 1
    while i< len(line):
      if line[i] in [INPUT, CONV1D, CONV2D]:
        gene_type = line[i]
        kernel_size = int(line[i+1])
        stride = int(line[i+2])
        num_kernels = int(line[i+3])
        data.append( (gene_type, kernel_size, stride, num_kernels) )
        i = i+4
      else if line[i] in [POOL1D, POOL2D]:
        gene_type = line[i]
        kernel_size = int(line[i+1])
        stride = int(line[i+2])
        data.append( (gene_type, kernel_size, stride) )
        i = i+3     
      else: # fully_connected
        gene_type = line[i]
        size = int(line[i+1])
        data.append( (gene_type, size) )
        i = i+2

    cnn_layer_list.append(data)

  return cnn_layer_list


def write_results(filename, results):

  f = open(filename, 'w')

  for name, result in results.items():
    f.write('%s,%f,%d\n'%(name, result[0], result[1]))

  f.close()


def train(train_steps, sess, data_set, input_tensor, target_tensor):
  data_set.reset()

  _data = data_set.get_batch(BATCH_SIZE)
  while _data['batch_size'] > 0:
    fd = {input_tensor: _data['input'], target_tensor: _data['target']}
    sess.run(train_steps, feed_dict=fd)
    _data = data_set.get_batch(BATCH_SIZE)

def get_costs(cost_ops, sess, data_set, input_tensor, target_tensor):
  data_set.reset()

  costs = [0.0] * len(cost_ops)

  num_samples = data_set.num_samples()

  _data = data_set.get_batch(BATCH_SIZE)
  while _data['batch_size'] > 0:
    fd = {input_tensor: _data['input'], target_tensor: _data['target']}
    _cost = sess.run(cost_ops, feed_dict = fd)

    for i in range(len(costs)):
      costs[i] += _cost[i] / num_samples

    _data = data_set.get_batch(BATCH_SIZE)

    return costs

if __name__ == '__main__':

  # Load the datasets
  f = open('maro_dataset.pkl')
  training_set, validation_set, test_set = pickle.load(f)
  f.close()

  # Load the list of CNNs
  filename = sys.argv[1]
  out_filename = sys.argv[2]
  cnn_layer_list = load_cnn_layers(filename)

  print "Number of Samples"
  print "  Training Set:   %d" % training_set.num_samples()
  print "  Validation Set: %d" % validation_set.num_samples()
  print "  Test Set:       %d" % test_set.num_samples()
  print

  # Create the input, target and weight tensors
  input_tensor = tf.placeholder(tf.float32, (None, WINDOW_SIZE, NUM_SENSORS))
  target_tensor = tf.placeholder(tf.float32, (None, NUM_CLASSES))

  print "Building CNN Models...",
  cnns = {}
  cnn_names = []
  results = {}

  for layers in cnn_layer_list:
    cnn_name = layers[0]
    hidden_layers = decode_layers(layers)
    try:
      with tf.variable_scope(cnn_name):
        cnns[cnn_name] = CNN1D(input_tensor, hidden_layers, target_tensor)
        cnn_names.append(cnn_name)
    except:
      # Some sort of error building the thing.  Make the resulting objectives HUGE!
      print cnn_name + " cannot be built!"
      results[cnn_name] = (100000000., 100000000.)

  print "Done"
  print

  # Create the optimizer and initialize TensorFlow session
  optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
  train_steps = [optimizer.minimize(cnns[m].objective) for m in cnn_names]
  costs = [cnns[m].objective for m in cnn_names]

  sess = tf.InteractiveSession()

  writer = tf.summary.FileWriter('./log', sess.graph)

  sess.run(tf.global_variables_initializer())

  for i in range(NUM_EPOCHS):
    print "Training Step", i

    train(train_steps, sess, training_set, input_tensor, target_tensor)
    
  cnn_results = get_costs(costs, sess, validation_set, input_tensor, target_tensor)

  for i in range(len(cnn_names)):
    name = cnn_names[i]
    r = cnn_results[i]
    results[name] = (r, cnns[name].total_params)

  write_results(out_filename, results)
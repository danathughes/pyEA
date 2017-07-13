## CNN1D.py
##
## A simple 1D Convolution Neural Network classifier model.

import tensorflow as tf
import numpy as np

from parts_GA import Softmax

BATCH_SIZE = 100


def cross_entropy(prediction, target):
   "Creates a tensor calculating the cross-entropy between the prediction and target tensor"
   return tf.reduce_mean(-tf.reduce_sum(target * tf.log(prediction + 1e-10), reduction_indices=[1]))
 

class CNN1D(object):

   def __init__(self, input_tensor, hidden_layers, target_tensor, **kwargs):
      """
      Build a deep, 1D convolutional neural network network

      inputShape    - shape of the input window: (time steps x number sensors)
      hidden_layers - A list of hidden layers, which take the form of a tuple, which depends
                      on the type (first element of the tuple)
      num_labels    - the number of unique classes to identify
      """

      num_time_steps = kwargs.get('num_time_steps', 200)
      num_sensors = kwargs.get('num_sensors', 1)
      num_labels = kwargs.get('num_labels', 4)

      # Input and target placeholders
      self.input = input_tensor
      self.target = target_tensor
      self.total_params = 0

      # Build up the hidden layers for the network
      # Start by reshaping the input to work with the 2D convolutional tensors
      current_layer = tf.reshape(self.input, [-1, num_time_steps, 1, num_sensors])

      for layer in hidden_layers:
         current_layer, num_params = layer.build(current_layer)
         self.total_params += num_params

      # Create the output layer by creating a fully connected softmax layer
      softmax = Softmax(num_labels, name='softmax')

      self.output, num_params = softmax.build(current_layer)
      self.total_params += num_params

      # Set the objective to the cross entropy of the output and target
      self.objective = cross_entropy(self.output, self.target)



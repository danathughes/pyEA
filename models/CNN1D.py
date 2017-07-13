## CNN1D.py
##
## A simple 1D Convolution Neural Network classifier model.

import tensorflow as tf
import numpy as np

from parts import Softmax

BATCH_SIZE = 100


def cross_entropy(prediction, target, weights):
   "Creates a tensor calculating the cross-entropy between the prediction and target tensor"
   return tf.reduce_mean(-weights * tf.reduce_sum(target * tf.log(prediction + 1e-10), reduction_indices=[1]))
 

class CNN1D(object):

   def __init__(self, inputShape, hidden_layers, num_labels, **kwargs):
      """
      Build a deep, 1D convolutional neural network network

      inputShape    - shape of the input window: (time steps x number sensors)
      hidden_layers - A list of hidden layers, which take the form of a tuple, which depends
                      on the type (first element of the tuple)
      num_labels    - the number of unique classes to identify
      """

      # Unpack the input frame
      num_time_steps, num_sensors = inputShape

      self._num_targets = num_labels

      # Input and target placeholders
      self.input = tf.placeholder(tf.float32, shape=[None, num_time_steps, num_sensors])
      self.target = tf.placeholder(tf.float32, shape=[None, num_labels])
      self.weights = tf.placeholder(tf.float32, shape=[None, 1])

      self._weight_decay = kwargs.get('weight_decay', 0.0)

      # Build up the hidden layers for the network
      # Start by reshaping the input to work with the 2D convolutional tensors
      current_layer = tf.reshape(self.input, [-1, num_time_steps, 1, num_sensors])

      for layer in hidden_layers:
         current_layer = layer.build(current_layer)

      # Create the output layer by creating a fully connected softmax layer
      softmax = Softmax(num_labels)
      self.output = softmax.build(current_layer)

      # Set the objective to the cross entropy of the output and target
      self._cross_entropy = cross_entropy(self.output, self.target, self.weights)

      self._weight_cost = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

      self._objective = self._cross_entropy #+ self._weight_decay * self._weight_cost

      # And also be able to predict the accuracy
      correct_prediction = tf.equal(tf.argmax(self.target, 1), tf.argmax(self.output, 1))
      self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


   def objective(self):
      """
      Return the objective tensor of this network
      """

      return self._objective


   def accuracy(self):
      """
      Return the accuracy tensor of this network
      """

      return self._accuracy


   def get_feed_dict(self, data):
      """
      Create a feed dictionary for this model
      """

      return {self.input: data['input'], self.target: data['target'], self.weights: data['weight']}


   def train(self, train_step, data):
      """
      Train on the input data (x) and the target (y).  The train step is some optimizer
      """

      train_step.run(feed_dict=self.get_feed_dict(data))


   def train_batch(self, train_step, data_set, batch_size = BATCH_SIZE):
      """
      Train on the a full dataset
      """

      # Run them all, and then set the index back
      index_snapshot = data_set.get_current_index()
      num_samples = data_set.num_samples()
      data_set.reset()

      _data = data_set.get_batch(batch_size)

      while _data['batch_size'] > 0:

         self.train(train_step, _data)
         _data = data_set.get_batch(batch_size)


   def get_accuracy(self, data_set, batch_size = BATCH_SIZE):
      """
      Determine the accuracy of the provided data set
      """

      # Run them all, and then set the index back
      index_snapshot = data_set.get_current_index()
      num_samples = data_set.num_samples()
      data_set.reset()

      total_correct = 0.0
      total_count = 0.0

      _data = data_set.get_batch(batch_size)

      while _data['batch_size'] > 0:

         total_count += _data['batch_size']
         acc = self._accuracy.eval(feed_dict = self.get_feed_dict(_data))
         total_correct += acc * _data['batch_size']

         _data = data_set.get_batch(batch_size)

      data_set.set_index(index_snapshot)

      return total_correct / total_count


   def get_cost(self, data_set, batch_size = BATCH_SIZE):
      """
      Determine the cost of the provided data set
      """

      # Run them all, and then set the index back
      index_snapshot = data_set.get_current_index()
      num_samples = data_set.num_samples()
      data_set.reset()

      total_cost = 0.0

      _data = data_set.get_batch(batch_size)

      while _data['batch_size'] > 0:
         cost = self._objective.eval(feed_dict = self.get_feed_dict(_data))
         total_cost += cost

         _data = data_set.get_batch(batch_size)

      data_set.set_index(index_snapshot)

      return total_cost / data_set.num_samples()


   def get_class_pdf(self, data_set, batch_size = BATCH_SIZE):
      """
      """

      # Run them all, and then set the index back
      index_snapshot = data_set.get_current_index()
      num_samples = data_set.num_samples()
      data_set.reset()

      batch = data_set.get_batch(batch_size)

      classification = np.zeros((num_samples, self._num_targets))
      targets = np.zeros((num_samples, self._num_targets))

      cur_idx = 0

      while batch['input'].shape[0] > 0:
         batch_size = batch['input'].shape[0]

         prev_idx = cur_idx
         cur_idx += batch_size
         class_pdf = self.output.eval(feed_dict = self.get_feed_dict(batch))

         classification[prev_idx:cur_idx,:] = class_pdf
         targets[prev_idx:cur_idx,:] = batch['target']

         batch = data_set.get_batch(batch_size)

      return classification, targets




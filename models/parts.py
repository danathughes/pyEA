## parts.py
##
## Objects which represent parts (e.g., activation function, weights, etc.) of a neural network.  For
## constructing more complex neural networks

import tensorflow as tf

# Constants for type of pooling layer to use
MAX_POOL = "MAX"
AVG_POOL = "AVG"


def weight_variable(shape):
   """
   Create a weight matrix
   """

   return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
   """
   Create a bias variable
   """

   return tf.Variable(tf.constant(0.1, shape=shape))


class Convolutional1D:
   """
   A 1-D Convolutional Layer
   """

   def __init__(self, kernel_width, num_kernels, **kwargs):
      """
      Create a holder for the convolutional layer

      Arguments:
        kernel_width - Size of each kernel
        num_kernels  - Number of kernels (feature maps) to use

      Optional Arguments:
        name    - A name for the layer.  Default is None
        stride  - Stride of the kernel.  Default is 1
        padding - Padding type of the kernel, default is "VALID"
                  One of "SAME" or "VALID"
      """

      # Simply hold on to the parameters for now
      self.kernel_width = kernel_width
      self.num_kernels = num_kernels

      self.name = kwargs.get("name", None)
      self.stride = kwargs.get("stride", 1)
      self.padding = kwargs.get("padding", "VALID")

      # Placeholder for the weight variable and this layer
      self.weights = None
      self.layer = None

      
   def build(self, input_layer):
      """
      Construct the layer in tensorflow
      """

      # Get the number of input channels
      input_shape = input_layer.get_shape()
      num_input_channels = input_shape[-1].value

      # Create the weights and convolutional layer
      weight_shape = [self.kernel_width, 1, num_input_channels, self.num_kernels]
      self.weights = weight_variable(weight_shape)
      self.layer = tf.nn.conv2d(input_layer, self.weights, strides=[1,self.stride,1,1], padding=self.padding)

      return self.layer
      
      
class Pool1D:
   """
   A 1-D Pooling Layer
   """

   def __init__(self, pool_width, **kwargs):
      """
      Create a holder for the pooling layer

      Arguments:
        pool_width  - Number of values to pool over
        pool_stride - Size of the stride

      Optional Arguments:
        name      - A name for the layer.  Default is None
        stride    - Stride of the pool.  Default is same as pool width
        padding   - Padding type of the kernel, default is "VALID"
                    One of "SAME" or "VALID"
        pool_type - Type of pooling layer to use.  Default is "MAX"
                    One of "MAX" or "AVG"
      """

      # Simply hold on to the parameters for now
      self.pool_width = pool_width

      self.name = kwargs.get("name", None)
      self.stride = kwargs.get("stride", self.pool_width)
      self.padding = kwargs.get("padding", "VALID")
      self.pool_type = kwargs.get("pool_type", MAX_POOL)

      # Placeholder for the resulting layer
      self.layer = None

      
   def build(self, input_layer):
      """
      Construct the layer in tensorflow
      """

      # Create the pooling layer
      if self.pool_type == MAX_POOL:
         self.layer = tf.nn.max_pool(input_layer, ksize=[1,self.pool_width,1,1], strides=[1,self.stride,1,1], padding=self.padding)
      elif self.pool_type == AVG_POOL:
         self.layer = tf.nn.avg_pool(input_layer, ksize=[1,self.pool_width,1,1], strides=[1,self.stride,1,1], padding=self.padding)
      else:
         # Invalid pooling type
         print "INVALID POOLING TYPE! ", self.pool_type

      return self.layer
      
      
class ReLU:
   """
   A Rectified Linear Unit Layer
   """

   def __init__(self, **kwargs):
      """
      Create a rectified linear unit layer

      Optional Arguments:
        name      - A name for the layer.  Default is None
      """

      # Simply hold on to the parameters for now
      self.name = kwargs.get("name", None)

      # Placeholder for the resulting layer
      self.bias = None
      self.layer = None

      
   def build(self, input_layer):
      """
      Construct the layer in tensorflow
      """

      # Create a bias
      bias_shape = [input_layer.get_shape()[-1].value]
      self.bias = bias_variable(bias_shape)

      # Create the ReLU layer
      self.layer = tf.nn.relu(input_layer + self.bias)

      return self.layer
      

class Sigmoid:
   """
   A sigmoid activation function
   """

   def __init__(self, **kwargs):
      """
      Create a sigmoid layer

      Optional Arguments:
        name        - A name for the layer.  Default is None
      """

      # Simply hold on to the parameters for now
      self.name = kwargs.get("name", None)

      # Placeholders for the resulting layer
      self.bias = None
      self.layer = None


   def build(self, input_layer):
      """
      Construct the layer in tensorflow
      """

      # Create a bias
      bias_shape = [input_layer.get_shape()[-1].value]
      self.bias = bias_variable(bias_shape)

      # Create the Sigmoid layer
      self.layer = tf.sigmoid(input_layer + self.bias)

      return self.layer

      
class FullConnection:
   """
   A Fully Connected Layer
   """

   def __init__(self, output_size, **kwargs):
      """
      Create a fully connected weight matrix

      Arguments:
        output_size - The output size of the weight matrix

      Optional Arguments:
        name      - A name for the layer.  Default is None
      """

      # Simply hold on to the parameters for now
      self.output_size = output_size
      self.name = kwargs.get("name", None)

      # Placeholder for the resulting layer
      self.weights = None
      self.layer = None

      
   def build(self, input_layer):
      """
      Construct the layer in tensorflow
      """

      # Create a weight matrix
      input_size = input_layer.get_shape()[-1].value
      self.weights = weight_variable([input_size, self.output_size])

      # Create the ReLU layer
      self.layer = tf.matmul(input_layer, self.weights)

      return self.layer


class Flatten:
   """
   A Flattening Layer
   """

   def __init__(self, **kwargs):
      """
      Create a layer which flattens the input

      Optional Arguments:
        name      - A name for the layer.  Default is None
      """

      # Simply hold on to the parameters for now
      self.name = kwargs.get("name", None)

      # Placeholder for the resulting layer
      self.layer = None

      
   def build(self, input_layer):
      """
      Construct the layer in tensorflow
      """

      # Determine the size of the input when flattened
      input_layer_shape = input_layer.get_shape()[1:].dims
      flattened_dimension = reduce(lambda x,y: x*y, input_layer_shape, tf.Dimension(1))

      # Create the ReLU layer
      self.layer = tf.reshape(input_layer, [-1, flattened_dimension.value])

      return self.layer


class Softmax:
   """
   A Softmax Layer
   """

   def __init__(self, output_size, **kwargs):
      """
      Create a softmax layer

      Arguments:
        output_size - The size of the output of the softmax layer / number of labels

      Optional Arguments:
        name      - A name for the layer.  Default is None
      """

      # Simply hold on to the parameters for now
      self.name = kwargs.get("name", None)
      self.output_size = output_size

      # Placeholder for the weights, bias and resulting layer
      self.weights = None
      self.bias = None
      self.layer = None

      
   def build(self, input_layer):
      """
      Construct the layer in tensorflow
      """

      # Determine the size of the input
      input_size = input_layer.get_shape()[-1].value

      self.weights = weight_variable([input_size, self.output_size])
      self.bias = bias_variable([self.output_size])
      self.layer = tf.nn.softmax(tf.matmul(input_layer, self.weights) + self.bias)

      return self.layer


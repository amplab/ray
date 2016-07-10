import tensorflow as tf
import numpy as np
import ray

@ray.remote(16 * [np.ndarray], [])
def update_weights(*weight):
  """
  Updates the weights on a worker

  :param weight: Variable number of weights to be applied to the network
  :rtype: None
  """
  feed_dict = dict(zip(placeholders, weight))
  sess.run(assignment, feed_dict=feed_dict)

@ray.remote([np.ndarray, np.ndarray], 16 * [np.ndarray])
def compute_grad(X, Y):
  """
  Computes the gradient of the network.

  :param X:Numpy array of images in the form of [224,224,3]
  :param Y:Labels corresponding to each image
  :rtype: List of gradients for each variable
  """
  return sess.run([g for (g,v) in compgrads], feed_dict={images:X, y_true:Y, dropout:0.5})

def print_accuracy(X, Y):
  """
  Prints the accuracy of the network
  :param X: Numpy array for input images
  :param Y: Numpy array for labels
  :rtype: None
  """
  print sess.run(accuracy, feed_dict={images:X, y_true:Y, dropout:1.0})

def setup_variables(params, placeholders, assigns, kernelshape, biasshape):
  """
  Creates the variables for each layer and adds the variables and the components needed to feed them to various lists
  
  :param params: List of network parameters used for creating feed_dicts
  :param placeholders: List of placeholders used for feeding weights into
  :param assigns: List of assignments used for actually setting variables
  :param kernelshape:Shape of the kernel used for the conv layer
  :param biasshape:Shape of the bias used
  :rtype: None
  """
  kernel = tf.Variable(tf.zeros(kernelshape, dtype=tf.float32))
  biases = tf.Variable(tf.constant(0.0, shape=biasshape, dtype=tf.float32),
                       trainable=True, name='biases')
  kernel_new = tf.placeholder(tf.float32, shape=kernel.get_shape())
  biases_new = tf.placeholder(tf.float32, shape=biases.get_shape())
  update_kernel = kernel.assign(kernel_new)
  update_biases = biases.assign(biases_new)
  params += [kernel, biases]
  placeholders += [kernel_new, biases_new]
  assigns += [update_kernel, update_biases]

def conv_layer(prevlayer, shape, scope):
  """
  Constructs a convolutional layer for the network.
 
  :param prevlayer: The previous layer to connect the network together.
  :param shape: The strides used for convolution
  :param scope: Current scope of tensorflow
  :rtype: Tensor, Activation of layer
  """
  kernel = parameters[-2]
  bias = parameters[-1]
  conv = tf.nn.conv2d(prevlayer, kernel, shape, padding='SAME')
  addbias = tf.nn.bias_add(conv, bias)
  return tf.nn.relu(addbias, name=scope)

images = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
y_true = tf.placeholder(tf.float32, shape=[None, 1000])
parameters = []
assignment = []
placeholders = []
# conv1
with tf.name_scope('conv1') as scope:
  setup_variables(parameters, placeholders, assignment, [11, 11, 3, 96], [96])
  conv1 = conv_layer(images, [1, 4, 4, 1], scope)
# pool1
pool1 = tf.nn.max_pool(conv1,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1],
                       padding='VALID',
                       name='pool1')


# lrn1
pool1lrn = tf.nn.lrn(pool1, depth_radius=5, bias=1.0,
                            alpha=0.0001, beta=0.75,
                            name="LocalResponseNormalization")
# conv2
with tf.name_scope('conv2') as scope:
  setup_variables(parameters, placeholders, assignment, [5, 5, 96, 256], [256])
  conv2 = conv_layer(pool1lrn, [1, 1, 1, 1], scope)

pool2 = tf.nn.max_pool(conv2,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1],
                       padding='VALID',
                       name='pool2')

# lrn2
pool2lrn = tf.nn.lrn(pool2, depth_radius=5, bias=1.0,
                            alpha=0.0001, beta=0.75,
                            name="LocalResponseNormalization")
# conv3
with tf.name_scope('conv3') as scope:
  setup_variables(parameters, placeholders, assignment, [3, 3, 256, 384], [384])
  conv3 = conv_layer(pool2lrn, [1, 1, 1, 1], scope)
# conv4
with tf.name_scope('conv4') as scope:
  setup_variables(parameters, placeholders, assignment, [3, 3, 384, 384], [384])
  conv4 = conv_layer(conv3, [1, 1, 1, 1], scope)
# conv5
with tf.name_scope('conv5') as scope:
  setup_variables(parameters, placeholders, assignment, [3, 3, 384, 256], [256])
  conv5 = conv_layer(conv4, [1, 1, 1, 1], scope)

# pool5
pool5 = tf.nn.max_pool(conv5,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1],
                       padding='VALID',
                       name='pool5') 

# lrn5
pool5lrn = tf.nn.lrn(pool5, depth_radius=5, bias=1.0,
                            alpha=0.0001, beta=0.75,
                            name="LocalResponseNormalization")
dropout = tf.placeholder(tf.float32)

with tf.name_scope('fc1') as scope:
  n_input = int(np.prod(pool5lrn.get_shape().as_list()[1:]))
  setup_variables(parameters, placeholders, assignment, [n_input, 4096], [4096])
  fc_in = tf.reshape(pool5lrn, [-1, n_input])
  fc_layer1 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(fc_in, parameters[-2]), parameters[-1]))
  fc_out1 = tf.nn.dropout(fc_layer1, dropout)

with tf.name_scope('fc2') as scope:
  n_input = int(np.prod(fc_out1.get_shape().as_list()[1:]))
  setup_variables(parameters, placeholders, assignment, [n_input, 4096], [4096])
  fc_in = tf.reshape(fc_out1, [-1, n_input])
  fc_layer2 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(fc_in, parameters[-2]), parameters[-1]))
  fc_out2 = tf.nn.dropout(fc_layer2, dropout)

with tf.name_scope('fc3') as scope:
  n_input = int(np.prod(fc_out2.get_shape().as_list()[1:]))
  setup_variables(parameters, placeholders, assignment, [n_input, 1000], [1000])
  fc_in = tf.reshape(fc_out2, [-1, n_input])
  fc_layer3 = tf.nn.softmax(tf.nn.bias_add(tf.matmul(fc_in, parameters[-2]), parameters[-1]))

y_pred = fc_layer3 / tf.reduce_sum(fc_layer3,
                        reduction_indices=len(fc_layer3.get_shape()) - 1,
                        keep_dims=True)
# manual computation of crossentropy
y_pred = tf.clip_by_value(y_pred, tf.cast(1e-10, dtype=tf.float32),
                          tf.cast(1. - 1e-10, dtype=tf.float32))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_pred),
                                reduction_indices=len(y_pred.get_shape()) - 1))
#opt =  tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
opt = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9) # aNY OTHER OPTIMIZIER CAN BE PLACED HERE
correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

compgrads = opt.compute_gradients(cross_entropy, parameters)

application = opt.apply_gradients(zip(placeholders,parameters))
sess = tf.Session()
sess.run(tf.initialize_all_variables())

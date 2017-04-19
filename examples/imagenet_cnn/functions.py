import tensorflow as tf
import numpy as np
import ray
import ray.array.remote as ra

from typing import Tuple, List

@ray.remote([List[Tuple[ray.ObjRef, ray.ObjRef]]], [int])
def num_images(batches):
  """Counts number of images in batches.
  
  Args:
    batches (List): Collection of batches of images and labels.
  
  Returns:
    int: The number of images
  """
  shape_refs = [ra.shape(batch[0]) for batch in batches]
  return sum([ray.get(shape_ref)[0] for shape_ref in shape_refs])

@ray.remote([List[Tuple[ray.ObjRef, ray.ObjRef]]], [np.ndarray])
def compute_mean_image(batches):
  """Computes the mean of images in batches.
  
  Args:
    batches (List): Collection of batches of images and labels.
  
  Returns:
    ndarray: The mean image
  """

  if len(batches) == 0:
    raise Exception("No images were passed into `compute_mean_image`.")
  sum_image_refs = [ra.sum(batch[0], axis=0) for batch in batches]
  sum_images = [ray.get(ref) for ref in sum_image_refs]
  n_images = num_images(batches)
  return np.sum(sum_images, axis=0).astype("float64") / ray.get(n_images)

@ray.remote([np.ndarray, np.ndarray, np.ndarray, np.ndarray], [np.ndarray, np.ndarray, np.ndarray, np.ndarray])
def shuffle_arrays(images1, labels1, images2, labels2):
  """Shuffles the data of two batches.
  
  Args:
    images1 (ndarray): Images of first batch
    labels1 (ndarray): Labels of first batch
    images2 (ndarray): Images of second batch
    labels2 (ndarray): Labels of second batch
  
  Returns:
    ndarray: Shuffled images for first batch
    ndarray: Shuffled labels for first batch
    ndarray: Shuffled images for second batch
    ndarray: Shuffled labels for second batch
  """
  to_shuffle = zip(images1, labels1) + zip(images2, labels2)
  np.random.shuffle(to_shuffle)
  length = len(images1)
  first_batch = zip(*to_shuffle[0:length])
  first_array = map(np.asarray, first_batch)
  second_batch = zip(*to_shuffle[length:len(to_shuffle)])
  second_array = map(np.asarray, second_batch)
  return first_array[0], first_array[1], second_array[0], second_array[1]

@ray.remote([Tuple, Tuple], [Tuple])
def shuffles_tuples(first_batch, second_batch):
  """Calls shufflearrays in order to properly serialize the data.

  Args:
    first_batch (tuple): First batch to be shuffled.
    second_batch (tuple): Second batch to be shuffled
  
  Returns:
    Tuple: Two batches of images and labels.
  """
  if (first_batch == None) or (second_batch == None):
    return (first_batch, second_batch)
  shuffled_batches = shuffle_arrays(first_batch[0], first_batch[1], second_batch[0], second_batch[1])
  return ((shuffled_batches[0], shuffled_batches[1]), (shuffled_batches[2], shuffled_batches[3]))

@ray.remote([list, dict], [np.ndarray])
def convert(img_list, image_pairs):
  """Converts filename strings to integer labels.
  
  Args:
    imglist (List): Filenames
    imagepairs (dict): Lookup table for filenames
  Returns:
    ndarray: Integer labels
  """
  return np.asarray(map(lambda img_name:int(image_pairs[img_name]), img_list))

def one_hot(x):
  """Converts integer labels to one hot vectors.
  
  Args:
    x (int): Index to be set to one

  Returns:
    ndarray: One hot vector.
  """
  zero = np.zeros([1000])
  zero[x] = 1.0
  return zero

def crop_image(img):
  """Crops an input image to prove more training for the network.
  
  Args:
    img (ndarray): Image to be cropped

  Returns:
    ndarray: Cropped image
  """
  cropx = np.random.randint(0,31)
  cropy = np.random.randint(0,31)
  return img[cropx:(cropx+224),cropy:(cropy+224)]

def shuffle_imagenet(batches):
  """Shuffles the entirety of the provided imagenet.
  
  Args:
    batches: Either a subset of imagenet or the entire imagenet
 
  Returns:
    List: A shuffled imagenet or subset
  """
  permuted_batches = map(tuple, np.random.permutation(batches))
  grouped_up_batches = zip(permuted_batches[0::2], permuted_batches[1::2])
  grouped_up_batches = zip(*map(lambda tup:ray.get(shuffles_tuples(tup[0], tup[1])), grouped_up_batches)) # We shuffle by swapp
  return grouped_up_batches[0] + grouped_up_batches[1]

@ray.remote(16 * [np.ndarray], [])
def update_weights(*weight):
  """Updates the weights on a worker

  Args: 
    weight: Variable number of weights to be applied to the network
  
  Returns: 
    None
  """
  feed_dict = dict(zip(placeholders, weight))
  sess.run(assignment, feed_dict=feed_dict)

@ray.remote([np.ndarray, np.ndarray, np.ndarray], 16 * [np.ndarray])
def compute_grad(X, Y, mean):
  """Computes the gradient of the network.

  Args:
    X (ndarray): Numpy array of images in the form of [224,224,3]
    Y (ndarray): Labels corresponding to each image
    mean (ndarray): Mean image to subtract from images

  Returns: 
    List of gradients for each variable
  """
  randindices = np.random.randint(0, len(X), size=[128])
  subset_X = map(lambda ind:X[ind], randindices) - mean
  subset_Y = np.asarray(map(lambda ind:one_hot(Y[ind]), randindices))
  cropped_X = np.asarray(map(crop_image, subset_X))
  return sess.run([g for (g,v) in comp_grads], feed_dict={images:cropped_X, y_true:subset_Y, dropout:0.5})

@ray.remote([np.ndarray, np.ndarray], [np.float32])
def print_accuracy(X, Y):
  """Prints the accuracy of the network
  
  Args:
    X: Numpy array for input images
    Y: Numpy array for labels
  
  Returns: 
    None
  """
  one_hot_Y = np.asarray(map(one_hot, Y))
  cropped_X = np.asarray(map(crop_image, X))
  return sess.run(accuracy, feed_dict={images:cropped_X, y_true:one_hot_Y, dropout:1.0})

def setup_variables(params, placeholders, assigns, kernelshape, biasshape):
  """Creates the variables for each layer and adds the variables and the components needed to feed them to various lists
  
  Args:
    params (List): Network parameters used for creating feed_dicts
    placeholders (List): Placeholders used for feeding weights into
    assigns (List): Assignments used for actually setting variables
    kernelshape (List): Shape of the kernel used for the conv layer
    biasshape (List):Shape of the bias used
  
  Returns: 
    None
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

def conv_layer(prev_layer, shape, scope):
  """Constructs a convolutional layer for the network.

  Args:
    prevlayer (Tensor): The previous layer to connect the network together.
    shape (List): The strides used for convolution
    scope (Scope): Current scope of tensorflow

  Returns:
    Tensor: Activation of layer
  """
  kernel = parameters[-2]
  bias = parameters[-1]
  conv = tf.nn.conv2d(prev_layer, kernel, shape, padding='SAME')
  add_bias = tf.nn.bias_add(conv, bias)
  return tf.nn.relu(add_bias, name=scope)

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
pool1_lrn = tf.nn.lrn(pool1, depth_radius=5, bias=1.0,
                             alpha=0.0001, beta=0.75,
                             name="LocalResponseNormalization")

# conv2
with tf.name_scope('conv2') as scope:
  setup_variables(parameters, placeholders, assignment, [5, 5, 96, 256], [256])
  conv2 = conv_layer(pool1_lrn, [1, 1, 1, 1], scope)

pool2 = tf.nn.max_pool(conv2,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1],
                       padding='VALID',
                       name='pool2')

# lrn2
pool2_lrn = tf.nn.lrn(pool2, depth_radius=5, bias=1.0,
                             alpha=0.0001, beta=0.75,
                             name="LocalResponseNormalization")

# conv3
with tf.name_scope('conv3') as scope:
  setup_variables(parameters, placeholders, assignment, [3, 3, 256, 384], [384])
  conv3 = conv_layer(pool2_lrn, [1, 1, 1, 1], scope)

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
pool5_lrn = tf.nn.lrn(pool5, depth_radius=5, bias=1.0,
                             alpha=0.0001, beta=0.75,
                             name="LocalResponseNormalization")

dropout = tf.placeholder(tf.float32)

with tf.name_scope('fc1') as scope:
  n_input = int(np.prod(pool5_lrn.get_shape().as_list()[1:]))
  setup_variables(parameters, placeholders, assignment, [n_input, 4096], [4096])
  fc_in = tf.reshape(pool5_lrn, [-1, n_input])
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
opt = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9) # Any other optimizier can be placed here
correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

comp_grads = opt.compute_gradients(cross_entropy, parameters)

application = opt.apply_gradients(zip(placeholders,parameters))
sess = tf.Session()
sess.run(tf.initialize_all_variables())

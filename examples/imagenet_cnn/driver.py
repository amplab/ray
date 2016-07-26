from typing import List, Tuple
import numpy as np
import ray
import ray.services as services
import ray.array.remote as ra
import os
import tensorflow as tf
import imagenet
import argparse
import boto3
import random

import tarfile, io
import Image

num_workers = 3
worker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../scripts/default_worker.py")
services.start_ray_local(num_workers=num_workers, worker_path=worker_path)

def load_chunk(tarfile, size=None):
  """Load a number of images from a single imagenet .tar file.

  This function also converts the image from grayscale to RGB if neccessary.

  Args:
    tarfile (tarfile.TarFile): The archive from which the files get loaded.
    size (Optional[Tuple[int, int]]): Resize the image to this size if provided.

  Returns:
    numpy.ndarray: Contains the image data in format [batch, w, h, c]
  """
  result = []
  filenames = []
  for member in tarfile.getmembers():
    filename = member.path
    content = tarfile.extractfile(member)
    img = Image.open(content)
    rgbimg = Image.new("RGB", img.size)
    rgbimg.paste(img)
    if size != None:
      rgbimg = rgbimg.resize(size, Image.ANTIALIAS)
    result.append(np.array(rgbimg).reshape(1, rgbimg.size[0], rgbimg.size[1], 3))
    filenames.append(filename)
  return np.concatenate(result), filenames

@ray.remote([str, str, List], [np.ndarray, List])
def load_tarfile_from_s3(bucket, s3_key, size=[]):
  """Load an imagenet .tar file.

  Args:
    bucket (str): Bucket holding the imagenet .tar.
    s3_key (str): s3 key from which the .tar file is loaded.
    size (List[int]): Resize the image to this size if size != []; len(size) == 2 required.

  Returns:
    np.ndarray: The image data (see load_chunk).
  """
  s3 = boto3.client("s3")
  response = s3.get_object(Bucket=bucket, Key=s3_key)
  output = io.BytesIO()
  chunk = response["Body"].read(1024 * 8)
  while chunk:
    output.write(chunk)
    chunk = response["Body"].read(1024 * 8)
  output.seek(0) # go to the beginning of the .tar file
  tar = tarfile.open(mode="r", fileobj=output)
  return load_chunk(tar, size=size if size != [] else None)

@ray.remote([str, List, List], [List])
def load_tarfiles_from_s3(bucket, s3_keys, size=[]):
  """Load a number of imagenet .tar files.

  Args:
    bucket (str): Bucket holding the imagenet .tars.
    s3_keys (List[str]): List of s3 keys from which the .tar files are being loaded.
    size (List[int]): Resize the image to this size if size != []; len(size) == 2 required.

  Returns:
    np.ndarray: Contains object references to the chunks of the images (see load_chunk).
  """

  return [load_tarfile_from_s3(bucket, s3_key, size) for s3_key in s3_keys]

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

def conv_layer(parameters, prev_layer, shape, scope):
  """Constructs a convolutional layer for the network.

  Args:
    parameters (List): Parameters used in constructing layer.
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

def net_initialization():
  images = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
  y_true = tf.placeholder(tf.float32, shape=[None, 1000])
  parameters = []
  assignment = []
  placeholders = []
  # conv1
  with tf.name_scope('conv1') as scope:
    setup_variables(parameters, placeholders, assignment, [11, 11, 3, 96], [96])
    conv1 = conv_layer(parameters, images, [1, 4, 4, 1], scope)
  
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
    conv2 = conv_layer(parameters, pool1_lrn, [1, 1, 1, 1], scope)
  
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
    conv3 = conv_layer(parameters, pool2_lrn, [1, 1, 1, 1], scope)
  
  # conv4
  with tf.name_scope('conv4') as scope:
    setup_variables(parameters, placeholders, assignment, [3, 3, 384, 384], [384])
    conv4 = conv_layer(parameters, conv3, [1, 1, 1, 1], scope)
  
  # conv5
  with tf.name_scope('conv5') as scope:
    setup_variables(parameters, placeholders, assignment, [3, 3, 384, 256], [256])
    conv5 = conv_layer(parameters, conv4, [1, 1, 1, 1], scope)
  
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
  return comp_grads, sess, application, accuracy, images, y_true, dropout, placeholders, parameters, assignment


def net_reinitialization(net_vars):
  return net_vars

ray.reusables.net_vars = ray.Reusable(net_initialization, net_reinitialization)

@ray.remote([List], [int])
def num_images(batches):
  """Counts number of images in batches.
  
  Args:
    batches (List): Collection of batches of images and labels.
  
  Returns:
    int: The number of images
  """
  shape_refs = [ra.shape(batch[0]) for batch in batches]
  return sum([ray.get(shape_ref)[0] for shape_ref in shape_refs])

@ray.remote([List], [np.ndarray])
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
  cropx = np.random.randint(0, 31)
  cropy = np.random.randint(0, 31)
  return img[cropx:(cropx + 224), cropy:(cropy + 224)]

def shuffle_imagenet(batches):
  """Shuffles the entirety of the provided imagenet.
  
  Args:
    batches: Either a subset of imagenet or the entire imagenet
 
  Returns:
    List: A shuffled imagenet or subset
  """
  permuted_batches = map(tuple, np.random.permutation(batches))
  grouped_up_batches = zip(permuted_batches[0::2], permuted_batches[1::2])
  grouped_up_batches = zip(*map(lambda tup: ray.get(shuffles_tuples(tup[0], tup[1])), grouped_up_batches)) # We shuffle by swapp
  return grouped_up_batches[0] + grouped_up_batches[1]

@ray.remote(16 * [np.ndarray], [])
def update_weights(*weight):
  """Updates the weights on a worker

  Args: 
    weight: Variable number of weights to be applied to the network
  
  Returns: 
    None
  """
  _, sess, _, _, _, _, _, placeholders, _, assignment = ray.reusables.net_vars
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
  comp_grads, sess, _, _, images, y_true, dropout, _, _, _ = ray.reusables.net_vars
  randindices = np.random.randint(0, len(X), size=[128])
  subset_X = map(lambda ind: X[ind], randindices) - mean
  subset_Y = np.asarray(map(lambda ind: one_hot(Y[ind]), randindices))
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
  _, sess, _, accuracy, images, y_true, dropout, _, _, _ = ray.reusables.net_vars
  one_hot_Y = np.asarray(map(one_hot, Y))
  cropped_X = np.asarray(map(crop_image, X))
  return sess.run(accuracy, feed_dict={images:cropped_X, y_true:one_hot_Y, dropout:1.0})

num_of_shuffles = 5 # Number of times batches are swaped with each other
batch_num = 0 # Tracker for accuracy printing and for user to see which epoch training is on
weights = [] # List to add weights of 

# Arguments to specify where the imagenet data is stored.
parser = argparse.ArgumentParser(description="Parse information for data loading.") 
parser.add_argument("--s3-bucket", required=True, type=str, help="Name of the bucket that contains the image data.")
parser.add_argument("--key-prefix", default="ILSVRC2012_img_train/n015", type=str, help="Prefix for files to fetch.")
parser.add_argument("--label-file", default="train.txt", type=str, help="File containing labels")

# Preparing keys for the data to be downloaded from
args = parser.parse_args()
s3 = boto3.resource("s3")
imagenet_bucket = s3.Bucket(args.s3_bucket)
objects = imagenet_bucket.objects.filter(Prefix=args.key_prefix)
images = [obj.key for obj in objects.all()]
print("Keys created")

# Downloading the label file, converting each line to key:values
s3 = boto3.client("s3")
label_file = s3.get_object(Bucket=args.s3_bucket, Key=args.label_file)
name_label_pairs = label_file["Body"].read().split("\n")
image_pairs = map(lambda line: line.split(" ", 2), name_label_pairs)
image_pairs = ray.put(dict(map(lambda tup: (os.path.basename(tup[0]), tup[-1]), image_pairs)))
print("Label dictionary created")

# Downloading imagenet and computing mean image of entire set for processing batches
imagenet = ray.get(load_tarfiles_from_s3(args.s3_bucket, map(str, images), [256, 256]))
mean_ref = compute_mean_image(imagenet)
print("Imagenet downloaded and mean computed")

# Converted the parsed filenames to integer labels, creating our batches
batches = map(lambda tup: (tup[0], convert(tup[-1], image_pairs)), imagenet)
print("Batches created")

# Imagenet is typically not preshuffled, so this loop does that.
if len(batches) % 2 == 0:
  batches.append(None)
print batches
for i in range(num_of_shuffles):
  batches = shuffle_imagenet(batches)
  print("{}".format(ray.get(batches[0][1])))
batches = filter(lambda tup: tup != None, batches)
print("Batches shuffled")

_, sess, application, _, _, _, _, placeholders, parameters, assignment = ray.reusables.net_vars

# Initialize a matrix using a normal distribution for each weight and bias in the network 
# and assign them to the local weights. 
for placeholder in placeholders:
  weights.append(np.random.normal(scale = 1e-1, size=placeholder.get_shape()))
sess.run(assignment, feed_dict=dict(zip(placeholders, weights)))
print("Weights passed")

while True:
  print("Start of loop")
  results = []

  # Get weights from local network and 
  weights = sess.run(parameters) # Retrieve weights from local network
  weight_refs = map(ray.put, weights) #Place weights into objstore
  for i in range(num_workers):
    update_weights(*weight_refs) # Update the weights on each worker.
  print("Weights sent")

  #Print accuracy
  if (batch_num % 100 == 0):
    x_ref, y_ref = random.choice(batches)
    print ray.get(print_accuracy(x_ref, y_ref))

  #Send the requests to compute the gradients to the workers
  for i in range(num_workers):
    x_ref,y_ref = random.choice(batches)
    results.append(compute_grad(x_ref, y_ref, mean_ref))
  print("Grad references recieved")

  # Take the mean across each set of gradients and apply to the local network.
  gotten_gradients = map(lambda x: map(ray.get, x), results) # Get the actual gradients, halting the program until all are available.
  gradients = [np.asarray([grad_set[i] for grad_set in gotten_gradients]) for i in range(16)] # 16 gradients, one for each variable
  gradient_mean = map(lambda x: np.mean(x, axis=0), gradients) # Taking mean over all the samples
  sess.run(application, feed_dict=dict(zip(placeholders, gradient_mean))) # Feeding the new values in
  print("End of batch {}".format(batch_num))
  batch_num += 1

services.cleanup()

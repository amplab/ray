# Most of the tensorflow code is adapted from Tensorflow's tutorial on using CNNs to train MNIST
# https://www.tensorflow.org/versions/r0.9/tutorials/mnist/pros/index.html#build-a-multilayer-convolutional-network
import numpy as np
import ray
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import functions

if __name__ == "__main__":
  ray.services.start_ray_local(num_workers=3)

  # The number of sets of random hyperparameters to try.
  trials = 2
  # The number of training passes over the dataset to use for network.
  epochs = 10

  # Load the mnist data and turn the data into remote objects.
  print "Downloading the MNIST dataset. This may take a minute."
  mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
  train_images = ray.put(mnist.train.images)
  train_labels = ray.put(mnist.train.labels)
  validation_images = ray.put(mnist.validation.images)
  validation_labels = ray.put(mnist.validation.labels)

  # Define a remote function that takes a set of hyperparameters as well as the
  # data, consructs and trains a network, and returns the validation accuracy.
  @ray.remote([dict, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray], [float])
  def train_cnn_and_compute_accuracy(params, epochs, train_images, train_labels, validation_images, validation_labels):
    # Extract the hyperparameters from the params dictionary.
    learning_rate = params["learning_rate"]
    batch_size = params["batch_size"]
    keep = 1 - params["dropout"]
    stddev = params["stddev"]
    # Create the input placeholders for the network.
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)
    # Create the network.
    train_step, accuracy = functions.cnn_setup(x, y, keep_prob, learning_rate, stddev)
    # Do the training and evaluation.
    with tf.Session() as sess:
      # Initialize the network weights.
      sess.run(tf.initialize_all_variables())
      for i in range(1, epochs):
        # Fetch the next batch of data.
        image_batch = functions.get_batch(train_images, i, batch_size)
        label_batch = functions.get_batch(train_labels, i, batch_size)
        # Do one step of training.
        sess.run(train_step, feed_dict={x: image_batch, y: label_batch, keep_prob: keep})
        if i % 100 == 0:
          # Estimate the training accuracy every once in a while.
          train_ac = accuracy.eval(feed_dict={x: image_batch, y: label_batch, keep_prob: 1.0})
          # If the training accuracy is too low, stop early in order to avoid
          # wasting computation.
          if train_ac < 0.25:
            # Compute the validation accuracy and return.
            totalacc = accuracy.eval(feed_dict={x: validation_images, y: validation_labels, keep_prob: 1.0})
            return totalacc
      # Training is done, compute the validation accuracy and return.
      totalacc = accuracy.eval(feed_dict={x: validation_images, y: validation_labels, keep_prob: 1.0})
    return float(totalacc)

  # Do the hyperparameter optimization

  best_params = None
  best_accuracy = 0
  results = []

  # Randomly generate some hyperparameters, and launch a task for each set.
  for i in range(trials):
    learning_rate = 10 ** np.random.uniform(-5, 5)
    batch_size = np.random.randint(1, 100)
    dropout = np.random.uniform(0, 1)
    stddev = 10 ** np.random.uniform(-5, 5)
    params = {"learning_rate": learning_rate, "batch_size": batch_size, "dropout": dropout, "stddev": stddev}
    results.append((params, train_cnn_and_compute_accuracy(params, epochs, train_images, train_labels, validation_images, validation_labels)))

  # Fetch the results of the tasks and print the results.
  for i in range(trials):
    params, ref = results[i]
    accuracy = ray.get(ref)
    print """We achieve accuracy {:.3}% with
        learning_rate: {:.2}
        batch_size: {}
        dropout: {:.2}
        stddev: {:.2}
      """.format(100 * accuracy, params["learning_rate"], params["batch_size"], params["dropout"], params["stddev"])
    if accuracy > best_accuracy:
      best_params = params
      best_accuracy = accuracy

  # Record the best performing set of hyperparameters.
  print """Best accuracy over {} trials was {:.3} with
        learning_rate: {:.2}
        batch_size: {}
        dropout: {:.2}
        stddev: {:.2}
    """.format(trials, 100 * best_accuracy, best_params["learning_rate"], best_params["batch_size"], best_params["dropout"], best_params["stddev"])

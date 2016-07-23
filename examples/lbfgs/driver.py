import os
import ray

import numpy as np
import scipy.optimize
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

if __name__ == "__main__":
  ray.services.start_ray_local(num_workers=16)

  print "Downloading and loading MNIST data..."
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

  image_dimension = 784
  label_dimension = 10
  w_shape = [image_dimension, label_dimension]
  w_size = np.prod(w_shape)
  b_shape = [label_dimension]
  b_size = np.prod(b_shape)
  dim = w_size + b_size

  def net_initialization():
    x = tf.placeholder(tf.float32, [None, image_dimension])
    w = tf.Variable(tf.zeros(w_shape))
    b = tf.Variable(tf.zeros(b_shape))
    y = tf.nn.softmax(tf.matmul(x, w) + b)
    y_ = tf.placeholder(tf.float32, [None, label_dimension])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    cross_entropy_grads = tf.gradients(cross_entropy, [w, b])

    w_new = tf.placeholder(tf.float32, w_shape)
    b_new = tf.placeholder(tf.float32, b_shape)
    update_w = w.assign(w_new)
    update_b = b.assign(b_new)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    return sess, update_w, update_b, cross_entropy, cross_entropy_grads, x, y_, w_new, b_new

  def net_reinitialization(net_vars):
    return net_vars

  ray.reusables.net_vars = ray.Reusable(net_initialization, net_reinitialization)

  def load_weights(theta):
    sess, update_w, update_b, _, _, _, _, w_new, b_new = ray.reusables.net_vars
    sess.run([update_w, update_b], feed_dict={w_new: theta[:w_size].reshape(w_shape), b_new: theta[w_size:]})

  @ray.remote([np.ndarray, np.ndarray, np.ndarray], [float])
  def loss(theta, xs, ys):
    sess, _, _, cross_entropy, _, x, y_, _, _ = ray.reusables.net_vars
    load_weights(theta)
    return float(sess.run(cross_entropy, feed_dict={x: xs, y_: ys}))

  @ray.remote([np.ndarray, np.ndarray, np.ndarray], [np.ndarray])
  def grad(theta, xs, ys):
    sess, _, _, _, cross_entropy_grads, x, y_, _, _ = ray.reusables.net_vars
    load_weights(theta)
    gradients = sess.run(cross_entropy_grads, feed_dict={x: xs, y_: ys})
    return np.concatenate([g.flatten() for g in gradients])

  batch_size = 100
  num_batches = mnist.train.num_examples / batch_size
  batches = [mnist.train.next_batch(batch_size) for _ in range(num_batches)]

  batch_refs = [(ray.put(xs), ray.put(ys)) for (xs, ys) in batches]

  # From the perspective of scipy.optimize.fmin_l_bfgs_b, full_loss is simply a
  # function which takes some parameters theta, and computes a loss. Similarly,
  # full_grad is a function which takes some parameters theta, and computes the
  # gradient of the loss. Internally, these functions use Ray to distribute the
  # computation of the loss and the gradient over the data that is represented
  # by the remote object references is x_batches and y_batches and which is
  # potentially distributed over a cluster. However, these details are hidden
  # from scipy.optimize.fmin_l_bfgs_b, which simply uses it to run the L-BFGS
  # algorithm.
  def full_loss(theta):
    theta_ref = ray.put(theta)
    loss_refs = [loss(theta_ref, xs_ref, ys_ref) for (xs_ref, ys_ref) in batch_refs]
    return sum([ray.get(loss_ref) for loss_ref in loss_refs])

  def full_grad(theta):
    theta_ref = ray.put(theta)
    grad_refs = [grad(theta_ref, xs_ref, ys_ref) for (xs_ref, ys_ref) in batch_refs]
    return sum([ray.get(grad_ref) for grad_ref in grad_refs]).astype("float64") # This conversion is necessary for use with fmin_l_bfgs_b.

  theta_init = np.zeros(dim)
  result = scipy.optimize.fmin_l_bfgs_b(full_loss, theta_init, maxiter=10, fprime=full_grad, disp=True)

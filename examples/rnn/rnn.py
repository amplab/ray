import ray
import numpy as np
import tensorflow as tf

scale = 200
num_steps = 10

batch_size = scale - 1
xdim = scale * 10
h1dim = (scale + 1) * 10
h2dim = (scale + 2) * 10
h3dim = (scale + 3) * 10
h4dim = (scale + 4) * 10
h5dim = (scale + 5) * 10
ydim = (2 * scale + 6) * 10


def net_initialization():
  x_in = tf.placeholder(tf.float32, [batch_size, xdim])
  h1_in = tf.placeholder(tf.float32, [batch_size, h1dim])
  h2_in = tf.placeholder(tf.float32, [batch_size, h2dim])
  h3_in = tf.placeholder(tf.float32, [batch_size, h3dim])
  h4_in = tf.placeholder(tf.float32, [batch_size, h4dim])
  h5_in = tf.placeholder(tf.float32, [batch_size, h5dim])

  W1_p = tf.Variable(tf.truncated_normal([xdim, h1dim]))
  W1_h = tf.Variable(tf.truncated_normal([h1dim, h1dim]))

  W2_p = tf.Variable(tf.truncated_normal([h1dim, h2dim]))
  W2_h = tf.Variable(tf.truncated_normal([h2dim, h2dim]))

  W3_p = tf.Variable(tf.truncated_normal([h2dim, h3dim]))
  W3_h = tf.Variable(tf.truncated_normal([h3dim, h3dim]))

  W4_p = tf.Variable(tf.truncated_normal([h3dim, h4dim]))
  W4_h = tf.Variable(tf.truncated_normal([h4dim, h4dim]))

  W5_p = tf.Variable(tf.truncated_normal([h4dim, h5dim]))
  W5_h = tf.Variable(tf.truncated_normal([h5dim, h5dim]))

  W6_p = tf.Variable(tf.truncated_normal([h5dim, ydim]))

  h1 = tf.matmul(x_in, W1_p) + tf.matmul(h1_in, W1_h)
  h2 = tf.matmul(h1_in, W2_p) + tf.matmul(h2_in, W2_h)
  h3 = tf.matmul(h2_in, W3_p) + tf.matmul(h3_in, W3_h)
  h4 = tf.matmul(h3_in, W4_p) + tf.matmul(h4_in, W4_h)
  h5 = tf.matmul(h4_in, W5_p) + tf.matmul(h5_in, W5_h)
  y = tf.matmul(h5_in, W6_p)


  init = tf.initialize_all_variables()
  sess = tf.Session()
  sess.run(init)

  return sess, h1, h2, h3, h4, h5, y, x_in, h1_in, h2_in, h3_in, h4_in, h5_in

def net_reinitialization(net_vars):
  return net_vars

mono_net_vars = []

@ray.remote
def first_layer(x_val, h1_val):
  sess, h1, _, _, _, _, _, x_in, h1_in, _, _, _, _ = ray.reusables.net_vars
  return sess.run(h1, feed_dict={x_in: x_val, h1_in: h1_val})

@ray.remote
def second_layer(h1_val, h2_val):
  sess, _, h2, _, _, _, _, _, h1_in, h2_in, _, _, _ = ray.reusables.net_vars
  return sess.run(h2, feed_dict={h1_in: h1_val, h2_in: h2_val})

@ray.remote
def third_layer(h2_val, h3_val):
  sess, _, _, h3, _, _, _, _, _, h2_in, h3_in, _, _ = ray.reusables.net_vars
  return sess.run(h3, feed_dict={h2_in: h2_val, h3_in: h3_val})

@ray.remote
def fourth_layer(h3_val, h4_val):
  sess, _, _, _, h4, _, _, _, _, _, h3_in, h4_in, _ = ray.reusables.net_vars
  return sess.run(h4, feed_dict={h3_in: h3_val, h4_in: h4_val})

@ray.remote
def fifth_layer(h4_val, h5_val):
  sess, _, _, _, _, h5, _, _, _, _, _, h4_in, h5_in = ray.reusables.net_vars
  return sess.run(h5, feed_dict={h4_in: h4_val, h5_in: h5_val})

@ray.remote
def sixth_layer(h5_val):
  sess, _, _, _, _, _, y, _, _, _, _, _, h5_in = ray.reusables.net_vars
  return sess.run(y, feed_dict={h5_in: h5_val})

def first_layer_mono(x_val, h1_val):
  sess, h1, _, _, _, _, _, x_in, h1_in, _, _, _, _ = mono_net_vars
  return sess.run(h1, feed_dict={x_in: x_val, h1_in: h1_val})

def second_layer_mono(h1_val, h2_val):
  sess, _, h2, _, _, _, _, _, h1_in, h2_in, _, _, _ = mono_net_vars
  return sess.run(h2, feed_dict={h1_in: h1_val, h2_in: h2_val})

def third_layer_mono(h2_val, h3_val):
  sess, _, _, h3, _, _, _, _, _, h2_in, h3_in, _, _ = mono_net_vars
  return sess.run(h3, feed_dict={h2_in: h2_val, h3_in: h3_val})

def fourth_layer_mono(h3_val, h4_val):
  sess, _, _, _, h4, _, _, _, _, _, h3_in, h4_in, _ = mono_net_vars
  return sess.run(h4, feed_dict={h3_in: h3_val, h4_in: h4_val})

def fifth_layer_mono(h4_val, h5_val):
  sess, _, _, _, _, h5, _, _, _, _, _, h4_in, h5_in = mono_net_vars
  return sess.run(h5, feed_dict={h4_in: h4_val, h5_in: h5_val})

def sixth_layer_mono(h5_val):
  sess, _, _, _, _, _, y, _, _, _, _, _, h5_in = mono_net_vars
  return sess.run(y, feed_dict={h5_in: h5_val})

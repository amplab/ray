import orchpy
import numpy as np
import tensorflow as tf

batch_size = 499
xdim = 500 * 10
h1dim = 501 * 10
h2dim = 502 * 10
h3dim = 503 * 10
h4dim = 504 * 10
h5dim = 505 * 10
ydim = 1006 * 10
num_steps = 10
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

# monolithic tensorflow code
h1_mono = tf.Variable(tf.zeros([batch_size, h1dim]))
h2_mono = tf.Variable(tf.zeros([batch_size, h2dim]))
h3_mono = tf.Variable(tf.zeros([batch_size, h3dim]))
h4_mono = tf.Variable(tf.zeros([batch_size, h4dim]))
h5_mono = tf.Variable(tf.zeros([batch_size, h5dim]))
inputs_monolithic = [tf.placeholder(tf.float32, [batch_size, xdim]) for _ in range(num_steps)]
y_monolithic = []
for t in range(num_steps):
  h1_mono = tf.matmul(inputs_monolithic[t], W1_p) + tf.matmul(h1_mono, W1_h)
  h2_mono = tf.matmul(h1_mono, W2_p) + tf.matmul(h2_mono, W2_h)
  h3_mono = tf.matmul(h2_mono, W3_p) + tf.matmul(h3_mono, W3_h)
  h4_mono = tf.matmul(h3_mono, W4_p) + tf.matmul(h4_mono, W4_h)
  h5_mono = tf.matmul(h4_mono, W5_p) + tf.matmul(h5_mono, W5_h)
  y_monolithic.append(tf.matmul(h5_mono, W6_p))
# end monolithic tensorflow code

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

@orchpy.distributed([np.ndarray, np.ndarray], [np.ndarray])
def first_layer(x_val, h1_val):
  return sess.run(h1, feed_dict={x_in: x_val, h1_in: h1_val})

@orchpy.distributed([np.ndarray, np.ndarray], [np.ndarray])
def second_layer(h1_val, h2_val):
  return sess.run(h2, feed_dict={h1_in: h1_val, h2_in: h2_val})

@orchpy.distributed([np.ndarray, np.ndarray], [np.ndarray])
def third_layer(h2_val, h3_val):
  return sess.run(h3, feed_dict={h2_in: h2_val, h3_in: h3_val})

@orchpy.distributed([np.ndarray, np.ndarray], [np.ndarray])
def fourth_layer(h3_val, h4_val):
  return sess.run(h4, feed_dict={h3_in: h3_val, h4_in: h4_val})

@orchpy.distributed([np.ndarray, np.ndarray], [np.ndarray])
def fifth_layer(h4_val, h5_val):
  return sess.run(h5, feed_dict={h4_in: h4_val, h5_in: h5_val})

@orchpy.distributed([np.ndarray], [np.ndarray])
def sixth_layer(h5_val):
  return sess.run(y, feed_dict={h5_in: h5_val})

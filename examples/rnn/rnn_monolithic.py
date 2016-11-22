import rnn
import tensorflow as tf
import numpy as np
import time

# Run monolithic RNN

W1_p = tf.Variable(tf.truncated_normal([rnn.xdim, rnn.h1dim]))
W1_h = tf.Variable(tf.truncated_normal([rnn.h1dim, rnn.h1dim]))

W2_p = tf.Variable(tf.truncated_normal([rnn.h1dim, rnn.h2dim]))
W2_h = tf.Variable(tf.truncated_normal([rnn.h2dim, rnn.h2dim]))

W3_p = tf.Variable(tf.truncated_normal([rnn.h2dim, rnn.h3dim]))
W3_h = tf.Variable(tf.truncated_normal([rnn.h3dim, rnn.h3dim]))

W4_p = tf.Variable(tf.truncated_normal([rnn.h3dim, rnn.h4dim]))
W4_h = tf.Variable(tf.truncated_normal([rnn.h4dim, rnn.h4dim]))

W5_p = tf.Variable(tf.truncated_normal([rnn.h4dim, rnn.h5dim]))
W5_h = tf.Variable(tf.truncated_normal([rnn.h5dim, rnn.h5dim]))

W6_p = tf.Variable(tf.truncated_normal([rnn.h5dim, rnn.ydim]))

h1_mono = tf.Variable(tf.zeros([rnn.batch_size, rnn.h1dim]))
h2_mono = tf.Variable(tf.zeros([rnn.batch_size, rnn.h2dim]))
h3_mono = tf.Variable(tf.zeros([rnn.batch_size, rnn.h3dim]))
h4_mono = tf.Variable(tf.zeros([rnn.batch_size, rnn.h4dim]))
h5_mono = tf.Variable(tf.zeros([rnn.batch_size, rnn.h5dim]))
inputs_monolithic = [tf.placeholder(tf.float32, [rnn.batch_size, rnn.xdim]) for _ in range(rnn.num_steps)]
y_monolithic = []
for t in range(rnn.num_steps):
  h1_mono = tf.matmul(inputs_monolithic[t], W1_p) + tf.matmul(h1_mono, W1_h)
  h2_mono = tf.matmul(h1_mono, W2_p) + tf.matmul(h2_mono, W2_h)
  h3_mono = tf.matmul(h2_mono, W3_p) + tf.matmul(h3_mono, W3_h)
  h4_mono = tf.matmul(h3_mono, W4_p) + tf.matmul(h4_mono, W4_h)
  h5_mono = tf.matmul(h4_mono, W5_p) + tf.matmul(h5_mono, W5_h)
  y_monolithic.append(tf.matmul(h5_mono, W6_p))

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

inputs = [np.random.normal(size=[rnn.batch_size, rnn.xdim]) for _ in range(rnn.num_steps)]
feed_dict = dict(zip(inputs_monolithic, inputs))

start_time = time.time()
outputs = sess.run(h1_mono, feed_dict=feed_dict)
end_time = time.time()
print "Monolithic RNN, 1 layer, elapsed_time = {} seconds.".format(end_time - start_time)

start_time = time.time()
outputs = sess.run(h2_mono, feed_dict=feed_dict)
end_time = time.time()
print "Monolithic RNN, 2 layer, elapsed_time = {} seconds.".format(end_time - start_time)

start_time = time.time()
outputs = sess.run(h3_mono, feed_dict=feed_dict)
end_time = time.time()
print "Monolithic RNN, 3 layer, elapsed_time = {} seconds.".format(end_time - start_time)

start_time = time.time()
outputs = sess.run(h4_mono, feed_dict=feed_dict)
end_time = time.time()
print "Monolithic RNN, 4 layer, elapsed_time = {} seconds.".format(end_time - start_time)

start_time = time.time()
outputs = sess.run(h5_mono, feed_dict=feed_dict)
end_time = time.time()
print "Monolithic RNN, 5 layer, elapsed_time = {} seconds.".format(end_time - start_time)

start_time = time.time()
outputs = sess.run(y_monolithic, feed_dict=feed_dict)
end_time = time.time()
print "Monolithic RNN, 6 layer, elapsed_time = {} seconds.".format(end_time - start_time)

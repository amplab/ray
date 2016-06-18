"""
./scheduler 52.50.92.141:10001

import orchpy.services as services
test_path = "/home/ubuntu/orch/examples/tensorflow/benchmark.py"
services.start_node("52.50.92.141:10001", "52.50.92.141", 1, worker_path=test_path)

import orchpy.services as services
test_path = "/home/ubuntu/halo/examples/tensorflow/benchmark.py"
services.start_node("52.50.92.141:10001", "52.49.170.133", 1, worker_path=test_path)

import orchpy.services as services
test_path = "/home/ubuntu/halo/examples/tensorflow/benchmark.py"
services.start_node("52.50.92.141:10001", "52.51.101.205", 1, worker_path=test_path)
"""

import orchpy as op
import orchpy.services as services
import rnn
import os
import time
import numpy as np

import arrays.single as single

# test_dir = os.path.dirname(os.path.abspath(__file__))
# test_path = os.path.join(test_dir, "benchmark.py")
# services.start_singlenode_cluster(return_drivers=False, num_workers_per_objstore=6, worker_path=test_path)
# time.sleep(10) # Wait to let each worker import rnn (which builds a tensorflow graph)
op.connect("52.50.92.141:10001", "52.50.92.141:20001", "52.50.92.141:90899")

h1 = single.zeros([rnn.batch_size, rnn.h1dim], "float")
h2 = single.zeros([rnn.batch_size, rnn.h2dim], "float")
h3 = single.zeros([rnn.batch_size, rnn.h3dim], "float")
h4 = single.zeros([rnn.batch_size, rnn.h4dim], "float")
h5 = single.zeros([rnn.batch_size, rnn.h5dim], "float")

inputs = [single.random.normal([rnn.batch_size, rnn.xdim]) for _ in range(rnn.num_steps)]
outputs = []

# Run distributed RNN
start_time = time.time()
for t in range(rnn.num_steps):
  h1 = rnn.first_layer(inputs[t], h1)
op.pull(h1)
end_time = time.time()
print "Distributed RNN, 1 layer, elapsed_time = {} seconds.".format(end_time - start_time)

start_time = time.time()
for t in range(rnn.num_steps):
  h1 = rnn.first_layer(inputs[t], h1)
  h2 = rnn.second_layer(h1, h2)
op.pull(h2)
end_time = time.time()
print "Distributed RNN, 2 layer, elapsed_time = {} seconds.".format(end_time - start_time)

start_time = time.time()
for t in range(rnn.num_steps):
  h1 = rnn.first_layer(inputs[t], h1)
  h2 = rnn.second_layer(h1, h2)
  h3 = rnn.third_layer(h2, h3)
op.pull(h3)
end_time = time.time()
print "Distributed RNN, 3 layer, elapsed_time = {} seconds.".format(end_time - start_time)

start_time = time.time()
for t in range(rnn.num_steps):
  h1 = rnn.first_layer(inputs[t], h1)
  h2 = rnn.second_layer(h1, h2)
  h3 = rnn.third_layer(h2, h3)
  h4 = rnn.fourth_layer(h3, h4)
op.pull(h4)
end_time = time.time()
print "Distributed RNN, 4 layer, elapsed_time = {} seconds.".format(end_time - start_time)

start_time = time.time()
for t in range(rnn.num_steps):
  h1 = rnn.first_layer(inputs[t], h1)
  h2 = rnn.second_layer(h1, h2)
  h3 = rnn.third_layer(h2, h3)
  h4 = rnn.fourth_layer(h3, h4)
  h5 = rnn.fifth_layer(h4, h5)
op.pull(h5)
end_time = time.time()
print "Distributed RNN, 5 layer, elapsed_time = {} seconds.".format(end_time - start_time)

start_time = time.time()
for t in range(rnn.num_steps):
  h1 = rnn.first_layer(inputs[t], h1)
  h2 = rnn.second_layer(h1, h2)
  h3 = rnn.third_layer(h2, h3)
  h4 = rnn.fourth_layer(h3, h4)
  h5 = rnn.fifth_layer(h4, h5)
  outputs.append(rnn.sixth_layer(h5))
for t in range(rnn.num_steps):
  op.pull(outputs[t])
end_time = time.time()
print "Distributed RNN, 6 layer, elapsed_time = {} seconds.".format(end_time - start_time)

# Run monolithic RNN
inputs = [np.random.normal(size=[rnn.batch_size, rnn.xdim]) for _ in range(rnn.num_steps)]
feed_dict = dict(zip(rnn.inputs_monolithic, inputs))

start_time = time.time()
outputs = rnn.sess.run(rnn.h1_mono, feed_dict=feed_dict)
end_time = time.time()
print "Monolithic RNN, 1 layer, elapsed_time = {} seconds.".format(end_time - start_time)

start_time = time.time()
outputs = rnn.sess.run(rnn.h2_mono, feed_dict=feed_dict)
end_time = time.time()
print "Monolithic RNN, 2 layer, elapsed_time = {} seconds.".format(end_time - start_time)

start_time = time.time()
outputs = rnn.sess.run(rnn.h3_mono, feed_dict=feed_dict)
end_time = time.time()
print "Monolithic RNN, 3 layer, elapsed_time = {} seconds.".format(end_time - start_time)

start_time = time.time()
outputs = rnn.sess.run(rnn.h4_mono, feed_dict=feed_dict)
end_time = time.time()
print "Monolithic RNN, 4 layer, elapsed_time = {} seconds.".format(end_time - start_time)

start_time = time.time()
outputs = rnn.sess.run(rnn.h5_mono, feed_dict=feed_dict)
end_time = time.time()
print "Monolithic RNN, 5 layer, elapsed_time = {} seconds.".format(end_time - start_time)

start_time = time.time()
outputs = rnn.sess.run(rnn.y_monolithic, feed_dict=feed_dict)
end_time = time.time()
print "Monolithic RNN, 6 layer, elapsed_time = {} seconds.".format(end_time - start_time)

services.cleanup()

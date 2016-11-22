import rnn
import numpy as np
import time

rnn.mono_net_vars = rnn.net_initialization()

# Run monolithic task RNN
h1 = np.zeros([rnn.batch_size, rnn.h1dim])
h2 = np.zeros([rnn.batch_size, rnn.h2dim])
h3 = np.zeros([rnn.batch_size, rnn.h3dim])
h4 = np.zeros([rnn.batch_size, rnn.h4dim])
h5 = np.zeros([rnn.batch_size, rnn.h5dim])

inputs = [np.random.normal(size=[rnn.batch_size, rnn.xdim]) for _ in range(rnn.num_steps)]

# Run monolithic task RNN
start_time = time.time()
for t in range(rnn.num_steps):
  h1 = rnn.first_layer_mono(inputs[t], h1)
end_time = time.time()
print "Monolithic Task RNN, 1 layer, elapsed_time = {} seconds.".format(end_time - start_time)

start_time = time.time()
for t in range(rnn.num_steps):
  h1 = rnn.first_layer_mono(inputs[t], h1)
  h2 = rnn.second_layer_mono(h1, h2)
end_time = time.time()
print "Monolithic Task RNN, 2 layer, elapsed_time = {} seconds.".format(end_time - start_time)

start_time = time.time()
for t in range(rnn.num_steps):
  h1 = rnn.first_layer_mono(inputs[t], h1)
  h2 = rnn.second_layer_mono(h1, h2)
  h3 = rnn.third_layer_mono(h2, h3)
end_time = time.time()
print "Monolithic Task RNN, 3 layer, elapsed_time = {} seconds.".format(end_time - start_time)

start_time = time.time()
for t in range(rnn.num_steps):
  h1 = rnn.first_layer_mono(inputs[t], h1)
  h2 = rnn.second_layer_mono(h1, h2)
  h3 = rnn.third_layer_mono(h2, h3)
  h4 = rnn.fourth_layer_mono(h3, h4)
end_time = time.time()
print "Monolithic Task RNN, 4 layer, elapsed_time = {} seconds.".format(end_time - start_time)

start_time = time.time()
for t in range(rnn.num_steps):
  h1 = rnn.first_layer_mono(inputs[t], h1)
  h2 = rnn.second_layer_mono(h1, h2)
  h3 = rnn.third_layer_mono(h2, h3)
  h4 = rnn.fourth_layer_mono(h3, h4)
  h5 = rnn.fifth_layer_mono(h4, h5)
end_time = time.time()
print "Monolithic Task RNN, 5 layer, elapsed_time = {} seconds.".format(end_time - start_time)

start_time = time.time()
outputs = []
for t in range(rnn.num_steps):
  h1 = rnn.first_layer_mono(inputs[t], h1)
  h2 = rnn.second_layer_mono(h1, h2)
  h3 = rnn.third_layer_mono(h2, h3)
  h4 = rnn.fourth_layer_mono(h3, h4)
  h5 = rnn.fifth_layer_mono(h4, h5)
  outputs.append(rnn.sixth_layer_mono(h5))
end_time = time.time()
print "Monolithic Task RNN, 6 layer, elapsed_time = {} seconds.".format(end_time - start_time)

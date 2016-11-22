import ray
import rnn
import time
import ray.array.remote as ra

ray.init(start_ray_local=True, num_workers=10)
ray.reusables.net_vars = ray.Reusable(rnn.net_initialization, rnn.net_reinitialization)

h1 = ra.zeros.remote([rnn.batch_size, rnn.h1dim])
h2 = ra.zeros.remote([rnn.batch_size, rnn.h2dim])
h3 = ra.zeros.remote([rnn.batch_size, rnn.h3dim])
h4 = ra.zeros.remote([rnn.batch_size, rnn.h4dim])
h5 = ra.zeros.remote([rnn.batch_size, rnn.h5dim])

inputs = [ra.random.normal.remote([rnn.batch_size, rnn.xdim]) for _ in range(rnn.num_steps)]

# Run distributed RNN
start_time = time.time()
for t in range(rnn.num_steps):
  h1 = rnn.first_layer.remote(inputs[t], h1)
ray.get(h1)
end_time = time.time()
print "Distributed RNN, 1 layer, elapsed_time = {} seconds.".format(end_time - start_time)

start_time = time.time()
for t in range(rnn.num_steps):
  h1 = rnn.first_layer.remote(inputs[t], h1)
  h2 = rnn.second_layer.remote(h1, h2)
ray.get(h2)
end_time = time.time()
print "Distributed RNN, 2 layer, elapsed_time = {} seconds.".format(end_time - start_time)

start_time = time.time()
for t in range(rnn.num_steps):
  h1 = rnn.first_layer.remote(inputs[t], h1)
  h2 = rnn.second_layer.remote(h1, h2)
  h3 = rnn.third_layer.remote(h2, h3)
ray.get(h3)
end_time = time.time()
print "Distributed RNN, 3 layer, elapsed_time = {} seconds.".format(end_time - start_time)

start_time = time.time()
for t in range(rnn.num_steps):
  h1 = rnn.first_layer.remote(inputs[t], h1)
  h2 = rnn.second_layer.remote(h1, h2)
  h3 = rnn.third_layer.remote(h2, h3)
  h4 = rnn.fourth_layer.remote(h3, h4)
ray.get(h4)
end_time = time.time()
print "Distributed RNN, 4 layer, elapsed_time = {} seconds.".format(end_time - start_time)

start_time = time.time()
for t in range(rnn.num_steps):
  h1 = rnn.first_layer.remote(inputs[t], h1)
  h2 = rnn.second_layer.remote(h1, h2)
  h3 = rnn.third_layer.remote(h2, h3)
  h4 = rnn.fourth_layer.remote(h3, h4)
  h5 = rnn.fifth_layer.remote(h4, h5)
ray.get(h5)
end_time = time.time()
print "Distributed RNN, 5 layer, elapsed_time = {} seconds.".format(end_time - start_time)

start_time = time.time()
outputs = []
for t in range(rnn.num_steps):
  h1 = rnn.first_layer.remote(inputs[t], h1)
  h2 = rnn.second_layer.remote(h1, h2)
  h3 = rnn.third_layer.remote(h2, h3)
  h4 = rnn.fourth_layer.remote(h3, h4)
  h5 = rnn.fifth_layer.remote(h4, h5)
  outputs.append(rnn.sixth_layer.remote(h5))
for t in range(rnn.num_steps):
  ray.get(outputs[t])
end_time = time.time()
print "Distributed RNN, 6 layer, elapsed_time = {} seconds.".format(end_time - start_time)

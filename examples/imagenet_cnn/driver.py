import numpy as np
import ray
import ray.services as services
import os
import tensorflow as tf

import functions

def oneandzeros(x):
  zero = np.zeros([1000])
  zero[x] = 1.0
  return(zero)

num_workers = 3

worker_dir = os.path.dirname(os.path.abspath(__file__))
worker_path = os.path.join(worker_dir, "worker.py")
services.start_ray_local(num_workers=num_workers, worker_path=worker_path)

batchnum = 0
Y = np.asarray(map(oneandzeros,np.random.randint(0, 1000, size=[200])))
X = np.random.uniform(size=[200,224,224,3])
xref = ray.put(X)
yref = ray.put(Y)
weights = []
for placeholder in functions.placeholders:
  weights.append(np.random.normal(scale = 1e-1, size=placeholder.get_shape()))
print "weights inited"
functions.sess.run(functions.assignment, feed_dict=dict(zip(functions.placeholders,weights)))
print "Weights passed"
while True:
  results = []
  print "Start of loop"
  weights = functions.sess.run(functions.parameters)
  print "Weights recieved"
  weightrefs = map(ray.put, weights)
  print "Weightrefs created"
  for i in range(num_workers):
    functions.updateweights(*weightrefs)
  print "Workers updated"
  for i in range(num_workers):
    print "Sending stuff"
    results.append(functions.computegrad(xref,yref))
  print "Grads recieved"
  actualresult = map(lambda(x):map(ray.get,x), results)
  print map(len, actualresult)
  grads = [np.asarray([gradset[i] for gradset in actualresult]) for i in range(16)]
  print grads[0].shape
  gradientvalues = map(lambda(x):tf.convert_to_tensor(np.mean(x,axis=0)), grads)
  functions.sess.run(functions.opt.apply_gradients(zip(gradientvalues,functions.parameters)))
  if (batchnum == 100):
    print functions.accuracy.eval(feed_dict={images:X, y_true:Y, dropout:0.0})

services.cleanup()

import numpy as np
import ray
import os

import functions

num_workers = 3
samples = 50
epochs = 100

worker_dir = os.path.dirname(os.path.abspath(__file__))
worker_path = os.path.join(worker_dir, "worker.py")
ray.services.start_ray_local(num_workers=num_workers, worker_path=worker_path)

best_params = None
best_accuracy = 0

results = []

for i in range(samples):
  learning_rate = 10 ** np.random.uniform(-6, 1)
  batch_size = np.random.randint(30, 100)
  dropout = np.random.uniform(0, 1)
  stddev = 10 ** np.random.uniform(-3, 1)
  randparams = {"learning_rate": learning_rate, "batch_size": batch_size, "dropout": dropout, "stddev": stddev}
  results.append((randparams, functions.train_cnn(randparams, epochs)))

for i in range(samples):
  params, ref = results[i]
  accuracy = ray.get(ref)
  print "With hyperparameters {}, we achieve an accuracy of {:.4}%.".format(params, 100 * accuracy)
  if accuracy > best_accuracy:
    best_params = params
    best_accuracy = accuracy
    print "Best parameters are now {}.".format(params)

print "Best parameters over {} samples was {}, with an accuracy of {:.4}%.".format(samples, best_params, 100 * best_accuracy)

import ray
import numpy as np

# Start a scheduler, an object store, and some workers.
# ray.init(node_ip_address="", scheduler_address="")

# Define a remote function for estimating pi.
@ray.remote([int], [float])
def estimate_pi(n):
      x = np.random.uniform(size=n)
      y = np.random.uniform(size=n)
      return 4 * np.mean(x ** 2 + y ** 2 < 1)

# Launch 10 tasks, each of which estimates pi.
result_ids = []
for _ in range(10):
        result_ids.append(estimate_pi.remote(100))

# Fetch the results of the tasks and print their average.
estimate = np.mean([ray.get(result_id) for result_id in result_ids])
print "Pi is approximately {}.".format(estimate)

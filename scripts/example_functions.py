import ray
import numpy as np
from typing import List

@ray.remote([int], [float])
def estimate_pi(n):
  x = np.random.uniform(size=n)
  y = np.random.uniform(size=n)
  return 4 * np.mean(x ** 2 + y ** 2 < 1)

@ray.remote([int], [int])
def increment(x):
  return x + 1

@ray.remote([int, int], [int])
def add(a, b):
  return a + b

@ray.remote([List], [np.ndarray])
def zeros(shape):
  return np.zeros(shape)

@ray.remote([np.ndarray, np.ndarray], [np.ndarray])
def dot(a, b):
  return np.dot(a, b)

@ray.remote([], [])
def throw_exception():
  raise Exception("This function intentionally failed.")

@ray.remote([], [])
def no_op():
  pass

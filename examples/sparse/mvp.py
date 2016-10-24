# the worker code for this script is available at
# https://github.com/cathywu/distSLA/blob/cholesky/examples/matmul/sparse_linear_algebra_worker.py
# currently, this code needs to be run on the 'sparse' branch of halo

import cPickle
import numpy as np

import orchpy as op
import orchpy.services as services
import sparse.single

worker_path = "/home/ubuntu/distSLA/examples/matmul/sparse_linear_algebra_worker.py"
services.start_node("54.210.61.0:10001", "54.210.61.0", 4, worker_path=worker_path)

M = cPickle.load(open("/home/ubuntu/distSLA/M_orkut.pickle", "r"))
num_partitions = 32
v = np.random.random(M.shape[1])

indices = np.array_split(np.arange(M.shape[0]), num_partitions)
M_partitioned = [M[indices[i],:] for i in range(0, num_partitions)]
M_partitions = [op.push(partition) for partition in M_partitioned] # push matrix to object store

def mvp(M_partitions, v):
  v_remote = op.push(v) # v starts off locally
  r_remote = [sparse.single.dot(partition, v_remote) for partition in M_partitions]
  return np.concatenate([op.pull(r) for r in r_remote]) # and is returned locally

# benchmark code:
total_time = 0.0
for i in range(30):
  a = time.time(); mvp(M_partitions, v); b = time.time() - a
  total_time += b

print "execution took", total_time/30

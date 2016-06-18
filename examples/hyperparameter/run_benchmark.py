"""
./scheduler 52.50.92.141:10001

import orchpy.services as services
test_path = "/home/ubuntu/orch/examples/hyperparameter/benchmark.py"
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
import hyperparameter
import os
import time
import numpy as np


test_path = "/home/ubuntu/photon/examples/hyperparameter/benchmark.py"
services.start_singlenode_cluster(return_drivers=False, num_objstores=1, num_workers_per_objstore=3, worker_path=test_path)

start_time = time.time()
results = []
n = 10000
d = 400
x = np.random.random((n,d))
y = np.random.random((n,1))
logx = np.append(np.ones((n,1)), np.random.random((n,2)), axis = 1)
logy = np.where(logx.dot(np.array([0,3,1])) >= 1.5, 1, 0)
results.append(hyperparameter.linreg_gd((x, y)))
results.append(hyperparameter.linreg_sgd((x, y)))
results.append(hyperparameter.logreg_gd((logx,logy)))
results.append(hyperparameter.logreg_sgd((logx,logy.T)))

for i in range(2):
    w = op.pull(results[i])
    #print 2 * (x.T.dot(x.dot(w)) - x.T.dot(y))
    print np.linalg.norm(np.linalg.solve(x.T.dot(x), x.T).dot(y) - w)
for i in range(2):
    w = op.pull(results[i+2])
    print w
    print np.sum(logy == np.squeeze(hyperparameter.pred_values(w, logx)))

end_time = time.time()
print "Hyperparameter optimization, elapsed_time = {} seconds.".format(end_time - start_time)

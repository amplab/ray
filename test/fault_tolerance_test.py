import unittest
import orchpy
import orchpy.serialization as serialization
import orchpy.services as services
import orchpy.worker as worker
import numpy as np
import time
import subprocess32 as subprocess
import os

from google.protobuf.text_format import *

import orchestra_pb2
import types_pb2

import test_functions
import arrays.single as single
import arrays.dist as dist

class FaultToleranceTest(unittest.TestCase):

  def testMethods1(self):
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(test_dir, "testrecv.py")
    services.start_singlenode_cluster(return_drivers=False, num_objstores=4, num_workers_per_objstore=2, worker_path=test_path)

    x = dist.zeros([10 * dist.BLOCK_SIZE, 10 * dist.BLOCK_SIZE], "float")
    y = dist.zeros([10 * dist.BLOCK_SIZE, 10 * dist.BLOCK_SIZE], "float")
    z = dist.dot(x, y)
    orchpy.kill_objstore(1)

    w = dist.dot(x, y)
    orchpy.kill_objstore(2)

    z_full = dist.assemble(z)
    w_full = dist.assemble(w)
    z_val = orchpy.pull(z_full)
    w_val = orchpy.pull(w_full)
    self.assertTrue(np.alltrue(z_val == w_val))

    services.cleanup()

  def testMethods2(self):
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(test_dir, "testrecv.py")
    services.start_singlenode_cluster(return_drivers=False, num_objstores=4, num_workers_per_objstore=2, worker_path=test_path)

    x = dist.zeros([10 * dist.BLOCK_SIZE, 10 * dist.BLOCK_SIZE], "float")
    y = dist.zeros([10 * dist.BLOCK_SIZE, 10 * dist.BLOCK_SIZE], "float")
    orchpy.kill_objstore(1)
    orchpy.kill_objstore(2)
    z = dist.dot(x, y)
    w = dist.dot(x, y)

    z_full = dist.assemble(z)
    w_full = dist.assemble(w)
    z_val = orchpy.pull(z_full)
    w_val = orchpy.pull(w_full)
    self.assertTrue(np.alltrue(z_val == w_val))

    services.cleanup()

  def testMethods3(self):
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(test_dir, "testrecv.py")
    services.start_singlenode_cluster(return_drivers=False, num_objstores=3, num_workers_per_objstore=10, worker_path=test_path)

    x = dist.zeros([10 * dist.BLOCK_SIZE, 10 * dist.BLOCK_SIZE], "float")
    y = dist.zeros([10 * dist.BLOCK_SIZE, 10 * dist.BLOCK_SIZE], "float")
    z = dist.dot(x, y)
    w = dist.dot(x, y)
    time.sleep(2)
    orchpy.kill_objstore(1)

    z_full = dist.assemble(z)
    w_full = dist.assemble(w)
    z_val = orchpy.pull(z_full)
    w_val = orchpy.pull(w_full)
    self.assertTrue(np.alltrue(z_val == w_val))

    services.cleanup()

if __name__ == '__main__':
    unittest.main()

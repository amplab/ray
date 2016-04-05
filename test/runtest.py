import unittest
import orchpy
import orchpy.serialization as serialization
import orchpy.services as services
import orchpy.worker as worker
import numpy as np
import time
import subprocess32 as subprocess
import os

import arrays.single as single

from google.protobuf.text_format import *

from grpc.beta import implementations
import orchestra_pb2
import types_pb2

def connect_to_scheduler(host, port):
  channel = implementations.insecure_channel(host, port)
  return orchestra_pb2.beta_create_Scheduler_stub(channel)

def connect_to_objstore(host, port):
  channel = implementations.insecure_channel(host, port)
  return orchestra_pb2.beta_create_ObjStore_stub(channel)


class SerializationTest(unittest.TestCase):

  def roundTripTest(self, data):
    serialized = serialization.serialize(data)
    result = serialization.deserialize(serialized)
    self.assertEqual(data, result)

  def numpyTypeTest(self, typ):
    a = np.random.randint(0, 10, size=(100, 100)).astype(typ)
    b = serialization.serialize(a)
    c = serialization.deserialize(b)
    self.assertTrue((a == c).all())

  def testSerialize(self):
    self.roundTripTest([1, "hello", 3.0])
    self.roundTripTest(42)
    self.roundTripTest("hello world")
    self.roundTripTest(42.0)
    self.roundTripTest((1.0, "hi"))

    self.roundTripTest({"hello" : "world", 1: 42, 1.0: 45})
    self.roundTripTest({})

    a = np.zeros((100, 100))
    res = serialization.serialize(a)
    b = serialization.deserialize(res)
    self.assertTrue((a == b).all())

    self.numpyTypeTest('int8')
    self.numpyTypeTest('uint8')
    # self.numpyTypeTest('int16') # TODO(pcm): implement this
    # self.numpyTypeTest('int32') # TODO(pcm): implement this
    self.numpyTypeTest('float32')
    self.numpyTypeTest('float64')

    a = np.array([[orchpy.lib.ObjRef(0), orchpy.lib.ObjRef(1)], [orchpy.lib.ObjRef(41), orchpy.lib.ObjRef(42)]])
    capsule = serialization.serialize(a)
    result = serialization.deserialize(capsule)
    self.assertTrue((a == result).all())

class OrchPyLibTest(unittest.TestCase):

    def testOrchPyLib(self):
      w = services.start_cluster()

      w.put_object(orchpy.lib.ObjRef(0), 'hello world')
      result = w.get_object(orchpy.lib.ObjRef(0))

      self.assertEqual(result, 'hello world')

      services.cleanup()

class ObjStoreTest(unittest.TestCase):

  # Test setting up object stores, transfering data between them and retrieving data to a client
  def testObjStore(self):
    w = services.start_cluster()

    # pushing and pulling an object shouldn't change it
    for data in ["h", "h" * 10000, 0, 0.0]:
      objref = orchpy.push(data, w)
      result = orchpy.pull(objref, w)
      self.assertEqual(result, data)

    # pushing an object, shipping it to another worker, and pulling it shouldn't change it
    # for data in ["h", "h" * 10000, 0, 0.0]:
    #   objref = worker.push(data, worker1)
    #   response = objstore1_stub.DeliverObj(orchestra_pb2.DeliverObjRequest(objref=objref.val, objstore_address=address(IP_ADDRESS, objstore2_port)), TIMEOUT_SECONDS)
    #   result = worker.pull(objref, worker2)
    #   self.assertEqual(result, data)

    services.cleanup()

class SchedulerTest(unittest.TestCase):

  def testCall(self):
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(test_dir, "testrecv.py")
    w = services.start_cluster(1, test_path)

    value_before = "test_string"
    objref = w.remote_call("__main__.print_string", [value_before])

    time.sleep(0.2)

    value_after = orchpy.pull(objref[0], w)
    self.assertEqual(value_before, value_after)

    time.sleep(0.1)

    services.cleanup()

class WorkerTest(unittest.TestCase):

  def testPushPull(self):
    w = services.start_cluster()

    for i in range(100):
      value_before = i * 10 ** 6
      objref = orchpy.push(value_before, w)
      value_after = orchpy.pull(objref, w)
      self.assertEqual(value_before, value_after)

    for i in range(100):
      value_before = i * 10 ** 6 * 1.0
      objref = orchpy.push(value_before, w)
      value_after = orchpy.pull(objref, w)
      self.assertEqual(value_before, value_after)

    for i in range(100):
      value_before = "h" * i
      objref = orchpy.push(value_before, w)
      value_after = orchpy.pull(objref, w)
      self.assertEqual(value_before, value_after)

    for i in range(100):
      value_before = [1] * i
      objref = orchpy.push(value_before, w)
      value_after = orchpy.pull(objref, w)
      self.assertEqual(value_before, value_after)

    services.cleanup()

"""
class APITest(unittest.TestCase):

  def testObjRefAliasing(self):
    services.start_scheduler(address(IP_ADDRESS, new_scheduler_port()))
    time.sleep(0.1)
    services.start_objstore(address(IP_ADDRESS, scheduler_port), address(IP_ADDRESS, new_objstore_port()))
    time.sleep(0.2)
    worker1 = worker.Worker()
    orchpy.connect(address(IP_ADDRESS, scheduler_port), address(IP_ADDRESS, objstore_port), address(IP_ADDRESS, new_worker_port()), worker1)
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(test_dir, "testrecv.py")
    services.start_worker(test_path, address(IP_ADDRESS, scheduler_port), address(IP_ADDRESS, objstore_port), address(IP_ADDRESS, new_worker_port()))
"""


if __name__ == '__main__':
    unittest.main()

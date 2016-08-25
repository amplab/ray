import importlib
import numpy as np

import libraylib as raylib

# The following definitions are required because Python doesn't allow custom
# attributes for primitive types. We need custom attributes for (a) implementing
# destructors that close the shared memory segment that the object resides in
# and (b) fixing https://github.com/amplab/ray/issues/72.

class Int(int):
  pass

class Long(long):
  pass

class Float(float):
  pass

class List(list):
  pass

class Dict(dict):
  pass

class Tuple(tuple):
  pass

class Str(str):
  pass
  
class Unicode(unicode):
  pass

class NDArray(np.ndarray):
  pass

def to_primitive(value):
  if not hasattr(value, "__dict__"):
    return value
  result = {"py/module": type(value).__module__, "py/type": type(value).__name__}
  if hasattr(value, "__getnewargs__"):
    result.update({"py/newargs": value.__getnewargs__()})
  # result.update(dict((k, to_primitive(v)) for (k, v) in value.__dict__.iteritems()))
  return result

def from_primitive(dictionary):
  if isinstance(dictionary, dict) and dictionary.has_key("py/module"):
    module = importlib.import_module(dictionary["py/module"])
    type_name = dictionary["py/type"]
    newargs = dictionary["py/newargs"] if dictionary.has_key("py/newargs") else []
    cls = module.__dict__[type_name]
    obj = cls.__new__(cls, *map(from_primitive, newargs))
    return obj
  else:
    return dictionary

def serialize(worker_capsule, obj):
  primitive_obj = to_primitive(obj)
  obj_capsule, contained_objectids = raylib.serialize_object(worker_capsule, primitive_obj) # contained_objectids is a list of the objectids contained in obj
  return obj_capsule, contained_objectids

def deserialize(worker_capsule, capsule):
  primitive_obj = raylib.deserialize_object(worker_capsule, capsule)
  return from_primitive(primitive_obj)

def serialize_task(worker_capsule, func_name, args):
  primitive_args = [(arg if isinstance(arg, raylib.ObjectID) else to_primitive(arg)) for arg in args]
  return raylib.serialize_task(worker_capsule, func_name, primitive_args)

def deserialize_task(worker_capsule, task):
  func_name, primitive_args, return_objectids = task
  args = [(arg if isinstance(arg, raylib.ObjectID) else from_primitive(arg)) for arg in primitive_args]
  return func_name, args, return_objectids

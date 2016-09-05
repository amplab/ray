import numpy as np
import libraylib as raylib
import libnumbuf

# This field keeps track of a whitelisted set of classes that Ray will
# serialize.
whitelisted_classes = {}
classes_to_pickle = set()
custom_serializers = {}
custom_deserializers = {}

def class_identifier(typ):
  return "{}.{}".format(typ.__module__, typ.__name__)

def is_named_tuple(cls):
  b = cls.__bases__
  if len(b) != 1 or b[0] != tuple:
    return False
  f = getattr(cls, "_fields", None)
  if not isinstance(f, tuple):
    return False
  return all(type(n) == str for n in f)

def add_class_to_whitelist(cls, pickle=False, custom_serializer=None, custom_deserializer=None):
  class_id = class_identifier(cls)
  whitelisted_classes[class_id] = cls
  if pickle:
    classes_to_pickle.add(class_id)
  if custom_serializer is not None:
    custom_serializers[class_id] = custom_serializer
    custom_deserializers[class_id] = custom_deserializer

# Here we define a custom serializer and deserializer for handling numpy
# arrays that contain objects.
def array_custom_serializer(obj):
  return obj.tolist(), obj.dtype.str
def array_custom_deserializer(serialized_obj):
  return np.array(serialized_obj[0], dtype=np.dtype(serialized_obj[1]))
add_class_to_whitelist(np.ndarray, pickle=False, custom_serializer=array_custom_serializer, custom_deserializer=array_custom_deserializer)

def serialize(obj):
  # Later, the class identifier should uniquely identify the class.
  class_id = class_identifier(type(obj))
  if class_id not in whitelisted_classes:
    raise Exception("Ray does not know how to serialize the object {}. To fix this, call 'ray.register_class' on the class of the object.".format(obj))
  if class_id in classes_to_pickle:
    serialized_obj = {"data": pickling.dumps(obj)}
  elif class_id in custom_serializers.keys():
    serialized_obj = {"data": custom_serializers[class_id](obj)}
  else:
    if not hasattr(obj, "__dict__"):
      raise Exception("We do not know how to serialize the object '{}'".format(obj))
    serialized_obj = obj.__dict__
    if is_named_tuple(type(obj)):
      # Handle the namedtuple case.
      serialized_obj["_ray_getnewargs_"] = obj.__getnewargs__()
    elif hasattr(obj, "__slots__"):
      print "This object has a __slots__ attribute, so a custom serializer must be used."
      raise Exception("This object has a __slots__ attribute, so a custom serializer must be used.")
  result = dict(serialized_obj, **{"_pytype_": class_id})
  return result

def deserialize(serialized_obj):
  class_id = serialized_obj["_pytype_"]
  cls = whitelisted_classes[class_id]
  if class_id in classes_to_pickle:
    obj = pickling.loads(serialized_obj["data"])
  elif class_id in custom_deserializers.keys():
    obj = custom_deserializers[class_id](serialized_obj["data"])
  else:
    # In this case, serialized_obj should just be the __dict__ field.
    if "_ray_getnewargs_" in serialized_obj:
      obj = cls.__new__(cls, *serialized_obj["_ray_getnewargs_"])
      serialized_obj.pop("_ray_getnewargs_")
    else:
      obj = cls.__new__(cls)
    serialized_obj.pop("_pytype_")
    obj.__dict__.update(serialized_obj)
  return obj

libnumbuf.register_callbacks(serialize, deserialize)

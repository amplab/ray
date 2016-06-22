// TODO: - Implement other datatypes for ndarray

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <structmember.h>
#define PY_ARRAY_UNIQUE_SYMBOL NUMBUF_ARRAY_API
#include <numpy/arrayobject.h>
#include <arrow/api.h>
#include <iostream>

#include "types.pb.h"
#include "worker.h"
#include "utils.h"

RayConfig global_ray_config;

extern "C" {

int PyObjectToWorker(PyObject* object, Worker **worker);

// Object references

typedef struct {
    PyObject_HEAD
    ObjRef val;
    Worker* worker;
} PyObjRef;

static void PyObjRef_dealloc(PyObjRef *self) {
  std::vector<ObjRef> objrefs;
  objrefs.push_back(self->val);
  self->worker->decrement_reference_count(objrefs);
  self->ob_type->tp_free((PyObject*) self);
}

static PyObject* PyObjRef_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  PyObjRef* self = (PyObjRef*) type->tp_alloc(type, 0);
  if (self != NULL) {
    self->val = 0;
  }
  return (PyObject*) self;
}

static int PyObjRef_init(PyObjRef *self, PyObject *args, PyObject *kwds) {
  if (!PyArg_ParseTuple(args, "iO&", &self->val, &PyObjectToWorker, &self->worker)) {
    return -1;
  }
  std::vector<ObjRef> objrefs;
  objrefs.push_back(self->val);
  RAY_LOG(RAY_REFCOUNT, "In PyObjRef_init, calling increment_reference_count for objref " << objrefs[0]);
  self->worker->increment_reference_count(objrefs);
  return 0;
};

static int PyObjRef_compare(PyObject* a, PyObject* b) {
  PyObjRef* A = (PyObjRef*) a;
  PyObjRef* B = (PyObjRef*) b;
  if (A->val < B->val) {
    return -1;
  }
  if (A->val > B->val) {
    return 1;
  }
  return 0;
}

static PyMemberDef PyObjRef_members[] = {
  {"val", T_INT, offsetof(PyObjRef, val), 0, "object reference"},
  {NULL}
};

static PyTypeObject PyObjRefType = {
  PyObject_HEAD_INIT(NULL)
  0,                         /* ob_size */
  "ray.ObjRef",           /* tp_name */
  sizeof(PyObjRef),          /* tp_basicsize */
  0,                         /* tp_itemsize */
  (destructor)PyObjRef_dealloc,          /* tp_dealloc */
  0,                         /* tp_print */
  0,                         /* tp_getattr */
  0,                         /* tp_setattr */
  PyObjRef_compare,          /* tp_compare */
  0,                         /* tp_repr */
  0,                         /* tp_as_number */
  0,                         /* tp_as_sequence */
  0,                         /* tp_as_mapping */
  0,                         /* tp_hash */
  0,                         /* tp_call */
  0,                         /* tp_str */
  0,                         /* tp_getattro */
  0,                         /* tp_setattro */
  0,                         /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,        /* tp_flags */
  "Ray objects",            /* tp_doc */
  0,                         /* tp_traverse */
  0,                         /* tp_clear */
  0,                         /* tp_richcompare */
  0,                         /* tp_weaklistoffset */
  0,                         /* tp_iter */
  0,                         /* tp_iternext */
  0,                         /* tp_methods */
  PyObjRef_members,          /* tp_members */
  0,                         /* tp_getset */
  0,                         /* tp_base */
  0,                         /* tp_dict */
  0,                         /* tp_descr_get */
  0,                         /* tp_descr_set */
  0,                         /* tp_dictoffset */
  (initproc)PyObjRef_init,   /* tp_init */
  0,                         /* tp_alloc */
  PyObjRef_new,              /* tp_new */
};

// create PyObjRef from C++ (could be made more efficient if neccessary)
PyObject* make_pyobjref(PyObject* worker_capsule, ObjRef objref) {
  PyObject* arglist = Py_BuildValue("(iO)", objref, worker_capsule);
  PyObject* result = PyObject_CallObject((PyObject*) &PyObjRefType, arglist);
  Py_DECREF(arglist);
  return result;
}

// Error handling

static PyObject *RayError;

// Pass arguments from Python to C++

int PyObjectToTask(PyObject* object, Task **task) {
  if (PyCapsule_IsValid(object, "task")) {
    *task = static_cast<Task*>(PyCapsule_GetPointer(object, "task"));
    return 1;
  } else {
    PyErr_SetString(PyExc_TypeError, "must be a 'task' capsule");
    return 0;
  }
}

int PyObjectToObj(PyObject* object, Obj **obj) {
  if (PyCapsule_IsValid(object, "obj")) {
    *obj = static_cast<Obj*>(PyCapsule_GetPointer(object, "obj"));
    return 1;
  } else {
    PyErr_SetString(PyExc_TypeError, "must be a 'obj' capsule");
    return 0;
  }
}

int PyObjectToWorker(PyObject* object, Worker **worker) {
  if (PyCapsule_IsValid(object, "worker")) {
    *worker = static_cast<Worker*>(PyCapsule_GetPointer(object, "worker"));
    return 1;
  } else {
    PyErr_SetString(PyExc_TypeError, "must be a 'worker' capsule");
    return 0;
  }
}

int PyObjectToObjRef(PyObject* object, ObjRef *objref) {
  if (PyObject_IsInstance(object, (PyObject*)&PyObjRefType)) {
    *objref = ((PyObjRef*) object)->val;
    return 1;
  } else {
    PyErr_SetString(PyExc_TypeError, "must be an object reference");
    return 0;
  }
}

// Destructors

void ObjCapsule_Destructor(PyObject* capsule) {
  Obj* obj = static_cast<Obj*>(PyCapsule_GetPointer(capsule, "obj"));
  delete obj;
}

void WorkerCapsule_Destructor(PyObject* capsule) {
  Worker* obj = static_cast<Worker*>(PyCapsule_GetPointer(capsule, "worker"));
  delete obj;
}

void TaskCapsule_Destructor(PyObject* capsule) {
  Task* obj = static_cast<Task*>(PyCapsule_GetPointer(capsule, "task"));
  delete obj;
}

// Helper methods

// Pass ownership of both the key and the value to the PyDict.
// This is only required for PyDicts, not for PyLists or PyTuples, compare
// https://docs.python.org/2/c-api/dict.html
// https://docs.python.org/2/c-api/list.html
// https://docs.python.org/2/c-api/tuple.html

void set_dict_item_and_transfer_ownership(PyObject* dict, PyObject* key, PyObject* val) {
  PyDict_SetItem(dict, key, val);
  Py_XDECREF(key);
  Py_XDECREF(val);
}

// Serialization

// serialize will serialize the python object val into the protocol buffer
// object obj, returns 0 if successful and something else if not
// FIXME(pcm): This currently only works for contiguous arrays
// This method will push all of the object references contained in `obj` to the `objrefs` vector.
int serialize(PyObject* worker_capsule, PyObject* val, Obj* obj, std::vector<ObjRef> &objrefs) {
  if (PyBool_Check(val)) {
    // The bool case must precede the int case because PyInt_Check passes for bools
    Bool* data = obj->mutable_bool_data();
    if (val == Py_False) {
      data->set_data(false);
    } else {
      data->set_data(true);
    }
  } else if (PyInt_Check(val)) {
    Int* data = obj->mutable_int_data();
    long d = PyInt_AsLong(val);
    data->set_data(d);
  } else if (PyFloat_Check(val)) {
    Double* data = obj->mutable_double_data();
    double d = PyFloat_AsDouble(val);
    data->set_data(d);
  } else if (PyTuple_Check(val)) {
    Tuple* data = obj->mutable_tuple_data();
    for (size_t i = 0, size = PyTuple_Size(val); i < size; ++i) {
      Obj* elem = data->add_elem();
      if (serialize(worker_capsule, PyTuple_GetItem(val, i), elem, objrefs) != 0) {
        return -1;
      }
    }
  } else if (PyList_Check(val)) {
    List* data = obj->mutable_list_data();
    for (size_t i = 0, size = PyList_Size(val); i < size; ++i) {
      Obj* elem = data->add_elem();
      if (serialize(worker_capsule, PyList_GetItem(val, i), elem, objrefs) != 0) {
        return -1;
      }
    }
  } else if (PyDict_Check(val)) {
    PyObject *pykey, *pyvalue;
    Py_ssize_t pos = 0;
    Dict* data = obj->mutable_dict_data();
    while (PyDict_Next(val, &pos, &pykey, &pyvalue)) {
      DictEntry* elem = data->add_elem();
      Obj* key = elem->mutable_key();
      if (serialize(worker_capsule, pykey, key, objrefs) != 0) {
        return -1;
      }
      Obj* value = elem->mutable_value();
      if (serialize(worker_capsule, pyvalue, value, objrefs) != 0) {
        return -1;
      }
    }
  } else if (PyString_Check(val)) {
    char* buffer;
    Py_ssize_t length;
    PyString_AsStringAndSize(val, &buffer, &length); // creates pointer to internal buffer
    obj->mutable_string_data()->set_data(buffer, length);
  } else if (val == Py_None) {
    obj->mutable_empty_data(); // allocate an Empty object, this is a None
  } else if (PyObject_IsInstance(val, (PyObject*) &PyObjRefType)) {
    ObjRef objref = ((PyObjRef*) val)->val;
    Ref* data = obj->mutable_objref_data();
    data->set_data(objref);
    objrefs.push_back(objref);
  } else if (PyArray_Check(val)) {
    PyArrayObject* array = PyArray_GETCONTIGUOUS((PyArrayObject*) val);
    Array* data = obj->mutable_array_data();
    npy_intp size = PyArray_SIZE(array);
    for (int i = 0; i < PyArray_NDIM(array); ++i) {
      data->add_shape(PyArray_DIM(array, i));
    }
    int typ = PyArray_TYPE(array);
    data->set_dtype(typ);
    switch (typ) {
      case NPY_FLOAT: {
          npy_float* buffer = (npy_float*) PyArray_DATA(array);
          for (npy_intp i = 0; i < size; ++i) {
            data->add_float_data(buffer[i]);
          }
        }
        break;
      case NPY_DOUBLE: {
          npy_double* buffer = (npy_double*) PyArray_DATA(array);
          for (npy_intp i = 0; i < size; ++i) {
            data->add_double_data(buffer[i]);
          }
        }
        break;
      case NPY_INT8: {
          npy_int8* buffer = (npy_int8*) PyArray_DATA(array);
          for (npy_intp i = 0; i < size; ++i) {
            data->add_int_data(buffer[i]);
          }
        }
        break;
      case NPY_INT64: {
          npy_int64* buffer = (npy_int64*) PyArray_DATA(array);
          for (npy_intp i = 0; i < size; ++i) {
            data->add_int_data(buffer[i]);
          }
        }
        break;
      case NPY_UINT8: {
          npy_uint8* buffer = (npy_uint8*) PyArray_DATA(array);
          for (npy_intp i = 0; i < size; ++i) {
            data->add_uint_data(buffer[i]);
          }
        }
        break;
      case NPY_UINT64: {
          npy_uint64* buffer = (npy_uint64*) PyArray_DATA(array);
          for (npy_intp i = 0; i < size; ++i) {
            data->add_uint_data(buffer[i]);
          }
        }
        break;
      case NPY_OBJECT: { // FIXME(pcm): Support arbitrary python objects, not only objrefs
          PyArrayIterObject* iter = (PyArrayIterObject*) PyArray_IterNew((PyObject*)array);
          while (PyArray_ITER_NOTDONE(iter)) {
            PyObject** item = (PyObject**) PyArray_ITER_DATA(iter);
            ObjRef objref;
            if (PyObject_IsInstance(*item, (PyObject*) &PyObjRefType)) {
              objref = ((PyObjRef*) (*item))->val;
            } else {
              PyErr_SetString(PyExc_TypeError, "must be an object reference"); // TODO: improve error message
              return -1;
            }
            data->add_objref_data(objref);
            objrefs.push_back(objref);
            PyArray_ITER_NEXT(iter);
          }
          Py_XDECREF(iter);
        }
        break;
      default:
        PyErr_SetString(RayError, "serialization: numpy datatype not know");
        return -1;
    }
    Py_DECREF(array); // TODO(rkn): is this right?
  } else {
    PyErr_SetString(RayError, "serialization: type not know");
    return -1;
  }
  return 0;
}

// This method will push all of the object references contained in `obj` to the `objrefs` vector.
PyObject* deserialize(PyObject* worker_capsule, const Obj& obj, std::vector<ObjRef> &objrefs) {
  if (obj.has_int_data()) {
    return PyInt_FromLong(obj.int_data().data());
  } else if (obj.has_double_data()) {
    return PyFloat_FromDouble(obj.double_data().data());
  } else if (obj.has_bool_data()) {
    if (obj.bool_data().data()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else if (obj.has_tuple_data()) {
    const Tuple& data = obj.tuple_data();
    size_t size = data.elem_size();
    PyObject* tuple = PyTuple_New(size);
    for (size_t i = 0; i < size; ++i) {
      PyTuple_SetItem(tuple, i, deserialize(worker_capsule, data.elem(i), objrefs));
    }
    return tuple;
  } else if (obj.has_list_data()) {
    const List& data = obj.list_data();
    size_t size = data.elem_size();
    PyObject* list = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
      PyList_SetItem(list, i, deserialize(worker_capsule, data.elem(i), objrefs));
    }
    return list;
  } else if (obj.has_dict_data()) {
    const Dict& data = obj.dict_data();
    PyObject* dict = PyDict_New();
    size_t size = data.elem_size();
    for (size_t i = 0; i < size; ++i) {
      PyObject* pykey = deserialize(worker_capsule, data.elem(i).key(), objrefs);
      PyObject* pyval = deserialize(worker_capsule, data.elem(i).value(), objrefs);
      set_dict_item_and_transfer_ownership(dict, pykey, pyval);
    }
    return dict;
  } else if (obj.has_string_data()) {
    const char* buffer = obj.string_data().data().data();
    Py_ssize_t length = obj.string_data().data().size();
    return PyString_FromStringAndSize(buffer, length);
  } else if (obj.has_empty_data()) {
    Py_RETURN_NONE;
  } else if (obj.has_objref_data()) {
    objrefs.push_back(obj.objref_data().data());
    return make_pyobjref(worker_capsule, obj.objref_data().data());
  } else if (obj.has_array_data()) {
    const Array& array = obj.array_data();
    std::vector<npy_intp> dims;
    for (int i = 0; i < array.shape_size(); ++i) {
      dims.push_back(array.shape(i));
    }
    PyArrayObject* pyarray = (PyArrayObject*) PyArray_SimpleNew(array.shape_size(), &dims[0], array.dtype());
    if (array.double_data_size() > 0) { // TODO: handle empty array
      npy_intp size = array.double_data_size();
      npy_double* buffer = (npy_double*) PyArray_DATA(pyarray);
      for (npy_intp i = 0; i < size; ++i) {
        buffer[i] = array.double_data(i);
      }
    } else if (array.float_data_size() > 0) {
      npy_intp size = array.float_data_size();
      npy_float* buffer = (npy_float*) PyArray_DATA(pyarray);
      for (npy_intp i = 0; i < size; ++i) {
        buffer[i] = array.float_data(i);
      }
    } else if (array.int_data_size() > 0) {
      npy_intp size = array.int_data_size();
      switch (array.dtype()) {
        case NPY_INT8: {
            npy_int8* buffer = (npy_int8*) PyArray_DATA(pyarray);
            for (npy_intp i = 0; i < size; ++i) {
              buffer[i] = array.int_data(i);
            }
          }
          break;
        case NPY_INT64: {
            npy_int64* buffer = (npy_int64*) PyArray_DATA(pyarray);
            for (npy_intp i = 0; i < size; ++i) {
              buffer[i] = array.int_data(i);
            }
          }
          break;
        default:
          PyErr_SetString(RayError, "deserialization: internal error (array type not implemented)");
          return NULL;
      }
    } else if (array.uint_data_size() > 0) {
      npy_intp size = array.uint_data_size();
      switch (array.dtype()) {
        case NPY_UINT8: {
            npy_uint8* buffer = (npy_uint8*) PyArray_DATA(pyarray);
            for (npy_intp i = 0; i < size; ++i) {
              buffer[i] = array.uint_data(i);
            }
          }
          break;
        case NPY_UINT64: {
            npy_uint64* buffer = (npy_uint64*) PyArray_DATA(pyarray);
            for (npy_intp i = 0; i < size; ++i) {
              buffer[i] = array.uint_data(i);
            }
          }
          break;
        default:
          PyErr_SetString(RayError, "deserialization: internal error (array type not implemented)");
          return NULL;
      }
    } else if (array.objref_data_size() > 0) {
      npy_intp size = array.objref_data_size();
      PyObject** buffer = (PyObject**) PyArray_DATA(pyarray);
      for (npy_intp i = 0; i < size; ++i) {
        buffer[i] = make_pyobjref(worker_capsule, array.objref_data(i));
        objrefs.push_back(array.objref_data(i));
      }
    } else {
      PyErr_SetString(RayError, "deserialization: internal error (array type not implemented)");
      return NULL;
    }
    return (PyObject*) pyarray;
  } else {
    PyErr_SetString(RayError, "deserialization: internal error (type not implemented)");
    return NULL;
  }
}

// This returns the serialized object and a list of the object references contained in that object.
PyObject* serialize_object(PyObject* self, PyObject* args) {
  Obj* obj = new Obj(); // TODO: to be freed in capsul destructor
  PyObject* worker_capsule;
  PyObject* pyval;
  if (!PyArg_ParseTuple(args, "OO", &worker_capsule, &pyval)) {
    return NULL;
  }
  std::vector<ObjRef> objrefs;
  if (serialize(worker_capsule, pyval, obj, objrefs) != 0) {
    return NULL;
  }
  Worker* worker;
  PyObjectToWorker(worker_capsule, &worker);
  PyObject* contained_objrefs = PyList_New(objrefs.size());
  for (int i = 0; i < objrefs.size(); ++i) {
    PyList_SetItem(contained_objrefs, i, make_pyobjref(worker_capsule, objrefs[i]));
  }
  PyObject* t = PyTuple_New(2); // We set the items of the tuple using PyTuple_SetItem, because that transfers ownership to the tuple.
  PyTuple_SetItem(t, 0, PyCapsule_New(static_cast<void*>(obj), "obj", &ObjCapsule_Destructor));
  PyTuple_SetItem(t, 1, contained_objrefs);
  return t;
}

PyObject* put_arrow(PyObject* self, PyObject* args) {
  Worker* worker;
  ObjRef objref;
  PyObject* value;
  if (!PyArg_ParseTuple(args, "O&O&O", &PyObjectToWorker, &worker, &PyObjectToObjRef, &objref, &value)) {
    return NULL;
  }
  worker->put_arrow(objref, value);
  Py_RETURN_NONE;
}

PyObject* get_arrow(PyObject* self, PyObject* args) {
  Worker* worker;
  ObjRef objref;
  if (!PyArg_ParseTuple(args, "O&O&", &PyObjectToWorker, &worker, &PyObjectToObjRef, &objref)) {
    return NULL;
  }
  SegmentId segmentid;
  PyObject* value = worker->get_arrow(objref, segmentid);
  PyObject* val_and_segmentid = PyList_New(2);
  PyList_SetItem(val_and_segmentid, 0, value);
  PyList_SetItem(val_and_segmentid, 1, PyInt_FromLong(segmentid));
  return val_and_segmentid;
}

PyObject* is_arrow(PyObject* self, PyObject* args) {
  Worker* worker;
  ObjRef objref;
  if (!PyArg_ParseTuple(args, "O&O&", &PyObjectToWorker, &worker, &PyObjectToObjRef, &objref)) {
    return NULL;
  }
  if (worker->is_arrow(objref))
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

PyObject* deserialize_object(PyObject* self, PyObject* args) {
  PyObject* worker_capsule;
  Obj* obj;
  if (!PyArg_ParseTuple(args, "OO&", &worker_capsule, &PyObjectToObj, &obj)) {
    return NULL;
  }
  std::vector<ObjRef> objrefs; // This is a vector of all the objrefs that are serialized in this task, including objrefs that are contained in Python objects that are passed by value.
  return deserialize(worker_capsule, *obj, objrefs);
  // TODO(rkn): Should we do anything with objrefs?
}

PyObject* serialize_task(PyObject* self, PyObject* args) {
  PyObject* worker_capsule;
  Task* task = new Task(); // TODO: to be freed in capsule destructor
  char* name;
  int len;
  PyObject* arguments;
  if (!PyArg_ParseTuple(args, "Os#O", &worker_capsule, &name, &len, &arguments)) {
    return NULL;
  }
  task->set_name(name, len);
  std::vector<ObjRef> objrefs; // This is a vector of all the objrefs that are serialized in this task, including objrefs that are contained in Python objects that are passed by value.
  if (PyList_Check(arguments)) {
    for (size_t i = 0, size = PyList_Size(arguments); i < size; ++i) {
      PyObject* element = PyList_GetItem(arguments, i);
      if (PyObject_IsInstance(element, (PyObject*)&PyObjRefType)) {
        ObjRef objref = ((PyObjRef*) element)->val;
        task->add_arg()->set_ref(objref);
        objrefs.push_back(objref);
      } else {
        Obj* arg = task->add_arg()->mutable_obj();
        serialize(worker_capsule, PyList_GetItem(arguments, i), arg, objrefs);
      }
    }
  } else {
    PyErr_SetString(RayError, "serialize_task: second argument needs to be a list");
    return NULL;
  }
  Worker* worker;
  PyObjectToWorker(worker_capsule, &worker);
  if (objrefs.size() > 0) {
    RAY_LOG(RAY_REFCOUNT, "In serialize_task, calling increment_reference_count for contained objrefs");
    worker->increment_reference_count(objrefs);
  }
  return PyCapsule_New(static_cast<void*>(task), "task", &TaskCapsule_Destructor);
}

PyObject* deserialize_task(PyObject* self, PyObject* args) {
  PyObject* worker_capsule;
  Task* task;
  if (!PyArg_ParseTuple(args, "OO&", &worker_capsule, &PyObjectToTask, &task)) {
    return NULL;
  }
  std::vector<ObjRef> objrefs; // This is a vector of all the objrefs that were serialized in this task, including objrefs that are contained in Python objects that are passed by value.
  PyObject* string = PyString_FromStringAndSize(task->name().c_str(), task->name().size());
  int argsize = task->arg_size();
  PyObject* arglist = PyList_New(argsize);
  for (int i = 0; i < argsize; ++i) {
    const Value& val = task->arg(i);
    if (!val.has_obj()) {
      PyList_SetItem(arglist, i, make_pyobjref(worker_capsule, val.ref()));
      objrefs.push_back(val.ref());
    } else {
      PyList_SetItem(arglist, i, deserialize(worker_capsule, val.obj(), objrefs));
    }
  }
  Worker* worker;
  PyObjectToWorker(worker_capsule, &worker);
  worker->decrement_reference_count(objrefs);
  int resultsize = task->result_size();
  std::vector<ObjRef> result_objrefs;
  PyObject* resultlist = PyList_New(resultsize);
  for (int i = 0; i < resultsize; ++i) {
    PyList_SetItem(resultlist, i, make_pyobjref(worker_capsule, task->result(i)));
    result_objrefs.push_back(task->result(i));
  }
  worker->decrement_reference_count(result_objrefs); // The corresponding increment is done in SubmitTask in the scheduler.
  PyObject* t = PyTuple_New(3); // We set the items of the tuple using PyTuple_SetItem, because that transfers ownership to the tuple.
  PyTuple_SetItem(t, 0, string);
  PyTuple_SetItem(t, 1, arglist);
  PyTuple_SetItem(t, 2, resultlist);
  return t;
}

// Ray Python API

PyObject* create_worker(PyObject* self, PyObject* args) {
  const char* scheduler_addr;
  const char* objstore_addr;
  const char* worker_addr;
  if (!PyArg_ParseTuple(args, "sss", &scheduler_addr, &objstore_addr, &worker_addr)) {
    return NULL;
  }
  auto scheduler_channel = grpc::CreateChannel(scheduler_addr, grpc::InsecureChannelCredentials());
  auto objstore_channel = grpc::CreateChannel(objstore_addr, grpc::InsecureChannelCredentials());
  Worker* worker = new Worker(std::string(worker_addr), scheduler_channel, objstore_channel);
  worker->register_worker(std::string(worker_addr), std::string(objstore_addr));
  return PyCapsule_New(static_cast<void*>(worker), "worker", &WorkerCapsule_Destructor);
}

PyObject* disconnect(PyObject* self, PyObject* args) {
  Worker* worker;
  if (!PyArg_ParseTuple(args, "O&", &PyObjectToWorker, &worker)) {
    return NULL;
  }
  worker->disconnect();
  Py_RETURN_NONE;
}

PyObject* connected(PyObject* self, PyObject* args) {
  Worker* worker;
  if (!PyArg_ParseTuple(args, "O&", &PyObjectToWorker, &worker)) {
    return NULL;
  }
  if (worker->connected()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

PyObject* wait_for_next_task(PyObject* self, PyObject* args) {
  Worker* worker;
  if (!PyArg_ParseTuple(args, "O&", &PyObjectToWorker, &worker)) {
    return NULL;
  }
  Task* task = worker->receive_next_task();
  return PyCapsule_New(static_cast<void*>(task), "task", NULL); // This task is owned by the C++ worker class, so we do not deallocate it.
}

PyObject* submit_task(PyObject* self, PyObject* args) {
  PyObject* worker_capsule;
  Task* task;
  if (!PyArg_ParseTuple(args, "OO&", &worker_capsule, &PyObjectToTask, &task)) {
    return NULL;
  }
  Worker* worker;
  PyObjectToWorker(worker_capsule, &worker);
  SubmitTaskRequest request;
  request.set_allocated_task(task);
  SubmitTaskReply reply = worker->submit_task(&request);
  if (!reply.function_registered()) {
    PyErr_SetString(RayError, "task: function not registered");
    return NULL;
  }
  request.release_task(); // TODO: Make sure that task is not moved, otherwise capsule pointer needs to be updated
  int size = reply.result_size();
  PyObject* list = PyList_New(size);
  std::vector<ObjRef> result_objrefs;
  for (int i = 0; i < size; ++i) {
    PyList_SetItem(list, i, make_pyobjref(worker_capsule, reply.result(i)));
    result_objrefs.push_back(reply.result(i));
  }
  worker->decrement_reference_count(result_objrefs); // The corresponding increment is done in SubmitTask in the scheduler.
  return list;
}

PyObject* notify_task_completed(PyObject* self, PyObject* args) {
  Worker* worker;
  PyObject* task_succeeded_obj;
  const char* error_message_ptr;
  if (!PyArg_ParseTuple(args, "O&Os", &PyObjectToWorker, &worker, &task_succeeded_obj, &error_message_ptr)) {
    return NULL;
  }
  std::string error_message(error_message_ptr);
  bool task_succeeded = PyObject_IsTrue(task_succeeded_obj);
  worker->notify_task_completed(task_succeeded, error_message);
  Py_RETURN_NONE;
}

PyObject* register_function(PyObject* self, PyObject* args) {
  Worker* worker;
  const char* function_name;
  int num_return_vals;
  if (!PyArg_ParseTuple(args, "O&si", &PyObjectToWorker, &worker, &function_name, &num_return_vals)) {
    return NULL;
  }
  worker->register_function(std::string(function_name), num_return_vals);
  Py_RETURN_NONE;
}

PyObject* get_objref(PyObject* self, PyObject* args) {
  PyObject* worker_capsule;
  if (!PyArg_ParseTuple(args, "O", &worker_capsule)) {
    return NULL;
  }
  Worker* worker;
  PyObjectToWorker(worker_capsule, &worker);
  ObjRef objref = worker->get_objref();
  return make_pyobjref(worker_capsule, objref);
}

PyObject* put_object(PyObject* self, PyObject* args) {
  Worker* worker;
  ObjRef objref;
  Obj* obj;
  PyObject* contained_objrefs;
  if (!PyArg_ParseTuple(args, "O&O&O&O", &PyObjectToWorker, &worker, &PyObjectToObjRef, &objref, &PyObjectToObj, &obj, &contained_objrefs)) {
    return NULL;
  }
  RAY_CHECK(PyList_Check(contained_objrefs), "The contained_objrefs argument must be a list.")
  std::vector<ObjRef> vec_contained_objrefs;
  size_t size = PyList_Size(contained_objrefs);
  for (size_t i = 0; i < size; ++i) {
    ObjRef contained_objref;
    PyObjectToObjRef(PyList_GetItem(contained_objrefs, i), &contained_objref);
    vec_contained_objrefs.push_back(contained_objref);
  }
  worker->put_object(objref, obj, vec_contained_objrefs);
  Py_RETURN_NONE;
}

PyObject* get_object(PyObject* self, PyObject* args) {
  // get_object assumes that objref is a canonical objref
  Worker* worker;
  ObjRef objref;
  if (!PyArg_ParseTuple(args, "O&O&", &PyObjectToWorker, &worker, &PyObjectToObjRef, &objref)) {
    return NULL;
  }
  slice s = worker->get_object(objref);
  Obj* obj = new Obj(); // TODO: Make sure this will get deleted
  obj->ParseFromString(std::string(reinterpret_cast<char*>(s.data), s.len));
  PyObject* result = PyList_New(2);
  PyList_SetItem(result, 0, PyCapsule_New(static_cast<void*>(obj), "obj", &ObjCapsule_Destructor));
  PyList_SetItem(result, 1, PyInt_FromLong(s.segmentid));
  return result;
}

PyObject* request_object(PyObject* self, PyObject* args) {
  Worker* worker;
  ObjRef objref;
  if (!PyArg_ParseTuple(args, "O&O&", &PyObjectToWorker, &worker, &PyObjectToObjRef, &objref)) {
    return NULL;
  }
  worker->request_object(objref);
  Py_RETURN_NONE;
}

PyObject* alias_objrefs(PyObject* self, PyObject* args) {
  Worker* worker;
  ObjRef alias_objref;
  ObjRef target_objref;
  if (!PyArg_ParseTuple(args, "O&O&O&", &PyObjectToWorker, &worker, &PyObjectToObjRef, &alias_objref, &PyObjectToObjRef, &target_objref)) {
    return NULL;
  }
  worker->alias_objrefs(alias_objref, target_objref);
  Py_RETURN_NONE;
}

PyObject* start_worker_service(PyObject* self, PyObject* args) {
  Worker* worker;
  if (!PyArg_ParseTuple(args, "O&", &PyObjectToWorker, &worker)) {
    return NULL;
  }
  worker->start_worker_service();
  Py_RETURN_NONE;
}

PyObject* scheduler_info(PyObject* self, PyObject* args) {
  Worker* worker;
  if (!PyArg_ParseTuple(args, "O&", &PyObjectToWorker, &worker)) {
    return NULL;
  }
  ClientContext context;
  SchedulerInfoRequest request;
  SchedulerInfoReply reply;
  worker->scheduler_info(context, request, reply);

  PyObject* target_objref_list = PyList_New(reply.target_objref_size());
  for (size_t i = 0; i < reply.target_objref_size(); ++i) {
    PyList_SetItem(target_objref_list, i, PyInt_FromLong(reply.target_objref(i)));
  }
  PyObject* reference_count_list = PyList_New(reply.reference_count_size());
  for (size_t i = 0; i < reply.reference_count_size(); ++i) {
    PyList_SetItem(reference_count_list, i, PyInt_FromLong(reply.reference_count(i)));
  }

  PyObject* dict = PyDict_New();
  set_dict_item_and_transfer_ownership(dict, PyString_FromString("target_objrefs"), target_objref_list);
  set_dict_item_and_transfer_ownership(dict, PyString_FromString("reference_counts"), reference_count_list);
  return dict;
}

PyObject* task_info(PyObject* self, PyObject* args) {
  Worker* worker;
  if (!PyArg_ParseTuple(args, "O&", &PyObjectToWorker, &worker)) {
    return NULL;
  }
  ClientContext context;
  TaskInfoRequest request;
  TaskInfoReply reply;
  worker->task_info(context, request, reply);

  PyObject* failed_tasks_list = PyList_New(reply.failed_task_size());
  for (size_t i = 0; i < reply.failed_task_size(); ++i) {
    const TaskStatus& info = reply.failed_task(i);
    PyObject* info_dict = PyDict_New();
    set_dict_item_and_transfer_ownership(info_dict, PyString_FromString("worker_address"), PyString_FromStringAndSize(info.worker_address().data(), info.worker_address().size()));
    set_dict_item_and_transfer_ownership(info_dict, PyString_FromString("function_name"), PyString_FromStringAndSize(info.function_name().data(), info.function_name().size()));
    set_dict_item_and_transfer_ownership(info_dict, PyString_FromString("operationid"), PyInt_FromLong(info.operationid()));
    set_dict_item_and_transfer_ownership(info_dict, PyString_FromString("error_message"), PyString_FromStringAndSize(info.error_message().data(), info.error_message().size()));
    PyList_SetItem(failed_tasks_list, i, info_dict);
  }

  PyObject* running_tasks_list = PyList_New(reply.running_task_size());
  for (size_t i = 0; i < reply.running_task_size(); ++i) {
    const TaskStatus& info = reply.running_task(i);
    PyObject* info_dict = PyDict_New();
    set_dict_item_and_transfer_ownership(info_dict, PyString_FromString("worker_address"), PyString_FromStringAndSize(info.worker_address().data(), info.worker_address().size()));
    set_dict_item_and_transfer_ownership(info_dict, PyString_FromString("function_name"), PyString_FromStringAndSize(info.function_name().data(), info.function_name().size()));
    set_dict_item_and_transfer_ownership(info_dict, PyString_FromString("operationid"), PyInt_FromLong(info.operationid()));
    PyList_SetItem(running_tasks_list, i, info_dict);
  }

  PyObject* dict = PyDict_New();
  set_dict_item_and_transfer_ownership(dict, PyString_FromString("failed_tasks"), failed_tasks_list);
  set_dict_item_and_transfer_ownership(dict, PyString_FromString("running_tasks"), running_tasks_list);
  set_dict_item_and_transfer_ownership(dict, PyString_FromString("num_succeeded"), PyInt_FromLong(reply.num_succeeded()));
  return dict;
}

PyObject* set_log_config(PyObject* self, PyObject* args) {
  const char* log_file_name;
  if (!PyArg_ParseTuple(args, "s", &log_file_name)) {
    return NULL;
  }
  create_log_dir_or_die(log_file_name);
  global_ray_config.log_to_file = true;
  global_ray_config.logfile.open(log_file_name);
  Py_RETURN_NONE;
}

static PyMethodDef RayLibMethods[] = {
 { "serialize_object", serialize_object, METH_VARARGS, "serialize an object to protocol buffers" },
 { "deserialize_object", deserialize_object, METH_VARARGS, "deserialize an object from protocol buffers" },
 { "put_arrow", put_arrow, METH_VARARGS, "put an arrow array on the local object store"},
 { "get_arrow", get_arrow, METH_VARARGS, "get an arrow array from the local object store"},
 { "is_arrow", is_arrow, METH_VARARGS, "is the object in the local object store an arrow object?"},
 { "serialize_task", serialize_task, METH_VARARGS, "serialize a task to protocol buffers" },
 { "deserialize_task", deserialize_task, METH_VARARGS, "deserialize a task from protocol buffers" },
 { "create_worker", create_worker, METH_VARARGS, "connect to the scheduler and the object store" },
 { "disconnect", disconnect, METH_VARARGS, "disconnect the worker from the scheduler and the object store" },
 { "connected", connected, METH_VARARGS, "check if the worker is connected to the scheduler and the object store" },
 { "register_function", register_function, METH_VARARGS, "register a function with the scheduler" },
 { "put_object", put_object, METH_VARARGS, "put a protocol buffer object (given as a capsule) on the local object store" },
 { "get_object", get_object, METH_VARARGS, "get protocol buffer object from the local object store" },
 { "get_objref", get_objref, METH_VARARGS, "register a new object reference with the scheduler" },
 { "request_object" , request_object, METH_VARARGS, "request an object to be delivered to the local object store" },
 { "alias_objrefs", alias_objrefs, METH_VARARGS, "make two objrefs refer to the same object" },
 { "wait_for_next_task", wait_for_next_task, METH_VARARGS, "get next task from scheduler (blocking)" },
 { "submit_task", submit_task, METH_VARARGS, "call a remote function" },
 { "notify_task_completed", notify_task_completed, METH_VARARGS, "notify the scheduler that a task has been completed" },
 { "start_worker_service", start_worker_service, METH_VARARGS, "start the worker service" },
 { "scheduler_info", scheduler_info, METH_VARARGS, "get info about scheduler state" },
 { "task_info", task_info, METH_VARARGS, "get task statuses" },
 { "set_log_config", set_log_config, METH_VARARGS, "set filename for raylib logging" },
 { NULL, NULL, 0, NULL }
};

PyMODINIT_FUNC initlibraylib(void) {
  PyObject* m;
  PyObjRefType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyObjRefType) < 0) {
    return;
  }
  m = Py_InitModule3("libraylib", RayLibMethods, "Python C Extension for Ray");
  Py_INCREF(&PyObjRefType);
  PyModule_AddObject(m, "ObjRef", (PyObject *)&PyObjRefType);
  RayError = PyErr_NewException("ray.error", NULL, NULL);
  Py_INCREF(RayError);
  PyModule_AddObject(m, "error", RayError);
  import_array();
}

}

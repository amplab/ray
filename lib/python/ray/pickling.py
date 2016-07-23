# Code adapted from dill library and Python standard library.
# Dill Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Dill Copyright (c) 2008-2015 California Institute of Technology.
# Dill License: 3-clause BSD.  The full license text is available at:
# - http://trac.mystic.cacr.caltech.edu/project/pathos/browser/dill/LICENSE

import dill
import typing
import worker

def dumps(obj, protocol=dill.DEFAULT_PROTOCOL):
  stringio = dill.dill.StringIO()
  pickler = BetterPickler(stringio, protocol)
  pickler.dump(obj)
  return stringio.getvalue()

def loads(string):
  return dill.loads(string)

def _copy_closure_cells(dest, source):
  # WARNING: Do NOT close over any variables here!
  python_dll = dill.dill.ctypes.cdll.python27
  for i, v in enumerate(source):
    python_dll.PyCell_Set(dill.dill.ctypes.c_void_p(id(dest[i])), dill.dill.ctypes.c_void_p(id(v.cell_contents)))

def _create_type(type_repr):
  return eval(type_repr.replace("~", ""), locals=(lambda d: d.setdefault("typing", typing) and None or d)(dict(typing.__dict__)))


class BetterPickler(dill.Pickler):
  def __init__(self, *args, **kwargs):
    kwargs = kwargs.copy()
    kwargs.setdefault("byref", True)
    kwargs.setdefault("recurse", True)
    dill.Pickler.__init__(self, *args, **kwargs)
    self.globalvars_seen = {}
  def globalvars(self, func, recurse, builtin, depth=0):
    """get objects defined in global scope that are referred to by func
  
    return a dict of {name:object}"""
    detect = dill.detect
    result = self.globalvars_seen.get(id(func))
    if result is None:
      result = {}
      self.globalvars_seen[id(func)] = result
      if not detect.isfunction(func) or (func.__module__ in ('__main__',) or func.__name__ in ('__call__',)):
        if detect.PY3:
          im_func = '__func__'
          func_code = '__code__'
          func_globals = '__globals__'
          func_closure = '__closure__'
        else:
          im_func = 'im_func'
          func_code = 'func_code'
          func_globals = 'func_globals'
          func_closure = 'func_closure'
        globs = None
        if detect.ismethod(func): func = getattr(func, im_func)
        if detect.isfunction(func):
          globs = vars(detect.getmodule(sum)) if builtin else {}
          # get references from within closure
          orig_func, func = func, set()
          for obj in getattr(orig_func, func_closure) or {}:
            _vars = self.globalvars(obj.cell_contents, recurse, builtin, depth + 1) or {}
            func.update(_vars) #XXX: (above) be wary of infinte recursion?
            globs.update(_vars)
          # get globals
          globs.update(getattr(orig_func, func_globals) or {})
          # get names of references
          if not recurse:
            func.update(getattr(orig_func, func_code).co_names)
          else:
            func.update(detect.nestedglobals(getattr(orig_func, func_code)))
            # find globals for all entries of func
            for key in func.copy(): #XXX: unnecessary...?
              nested_func = globs.get(key)
              if nested_func == orig_func:
                 #func.remove(key) if key in func else None
                continue  #XXX: self.globalvars(func, False)?
              func.update(self.globalvars(nested_func, True, builtin, depth + 1))
        elif detect.iscode(func):
          globs = vars(detect.getmodule(sum)) if builtin else {}
           #globs.update(globals())
          if not recurse:
            func = func.co_names # get names
          else:
            orig_func = func.co_name # to stop infinite recursion
            func = set(detect.nestedglobals(func))
            # find globals for all entries of func
            for key in func.copy(): #XXX: unnecessary...?
              if key == orig_func:
                 #func.remove(key) if key in func else None
                continue  #XXX: self.globalvars(func, False)?
              nested_func = globs.get(key)
              func.update(self.globalvars(nested_func, True, builtin, depth + 1))
        if globs is not None:
           result.update((name,globs[name]) for name in func if name in globs)
    return result
  def save_function(self, obj):
    dill_impl = dill.dill
    if not dill_impl._locate_function(obj):
      if getattr(self, '_recurse', False):
        globs = self.globalvars(obj, recurse=True, builtin=True)
        if obj in globs.values():
          globs = obj.__globals__ if dill_impl.PY3 else obj.func_globals
      else:
        globs = obj.__globals__ if dill_impl.PY3 else obj.func_globals
      if dill_impl.PY3:
        closure = obj.__closure__
        empty_closure = tuple(map(lambda _: dill_impl._create_cell(None), closure)) if closure else closure
        self.save_reduce(dill_impl._create_function, (obj.__code__, globs, obj.__name__, obj.__defaults__, empty_closure, obj.__dict__), obj=obj)
      else:
        closure = obj.func_closure
        empty_closure = tuple(map(lambda _: dill_impl._create_cell(None), closure)) if closure else closure
        self.save_reduce(dill_impl._create_function, (obj.func_code, globs, obj.func_name, obj.func_defaults, empty_closure, obj.__dict__), obj=obj)
      if closure:
        self.save_reduce(_copy_closure_cells, (empty_closure, closure), obj=obj)
    else:
      self.save_global(obj)
  def memoize(self, obj):
    # HACK: is this the right way to do things?
    if id(obj) not in self.memo:
      dill.Pickler.memoize(self, obj)
  def save_type(self, obj):
    try: return self.save_global(obj)
    except dill.PicklingError:
      self.save_reduce(_create_type, (repr(obj),), obj=obj)
  dispatch = dill.Pickler.dispatch.copy()
  dispatch[dill.dill.FunctionType] = save_function
  dispatch[typing.GenericMeta] = save_type

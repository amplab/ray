import halo

import numpy as np

# Test simple functionality

@halo.remote([str], [str])
def print_string(string):
  print "called print_string with", string
  f = open("asdfasdf.txt", "w")
  f.write("successfully called print_string with argument {}.".format(string))
  return string

@halo.remote([int, int], [int, int])
def handle_int(a, b):
  return a + 1, b + 1

# Test aliasing

@halo.remote([], [np.ndarray])
def test_alias_f():
  return np.ones([3, 4, 5])

@halo.remote([], [np.ndarray])
def test_alias_g():
  return test_alias_f()

@halo.remote([], [np.ndarray])
def test_alias_h():
  return test_alias_g()

# Test timing

@halo.remote([], [])
def empty_function():
  return ()

@halo.remote([], [int])
def trivial_function():
  return 1

# Test keyword arguments

@halo.remote([int, str], [str])
def keyword_fct1(a, b="hello"):
  return "{} {}".format(a, b)

@halo.remote([str, str], [str])
def keyword_fct2(a="hello", b="world"):
  return "{} {}".format(a, b)

@halo.remote([int, int, str, str], [str])
def keyword_fct3(a, b, c="hello", d="world"):
  return "{} {} {} {}".format(a, b, c, d)

@halo.remote([int], [str])
def varargs_fct(*a):
  return " ".join(map(str, a))

@halo.remote([int, int], [str])
def varargs_fct2(a, *b):
  return " ".join(map(str, b))

try:
  @halo.remote([int, int, int], [])
  def varargs_throw_exception(a, *b, **c):
    return ()

  varargs_exception_thrown = False
except:
  varargs_exception_thrown = True

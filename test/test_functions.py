import quartz

import numpy as np

# Test simple functionality

@quartz.remote([str], [str])
def print_string(string):
  print "called print_string with", string
  f = open("asdfasdf.txt", "w")
  f.write("successfully called print_string with argument {}.".format(string))
  return string

@quartz.remote([int, int], [int, int])
def handle_int(a, b):
  return a + 1, b + 1

# Test aliasing

@quartz.remote([], [np.ndarray])
def test_alias_f():
  return np.ones([3, 4, 5])

@quartz.remote([], [np.ndarray])
def test_alias_g():
  return test_alias_f()

@quartz.remote([], [np.ndarray])
def test_alias_h():
  return test_alias_g()

# Test timing

@quartz.remote([], [])
def empty_function():
  return ()

@quartz.remote([], [int])
def trivial_function():
  return 1

# Test keyword arguments

@quartz.remote([int, str], [str])
def keyword_fct1(a, b="hello"):
  return "{} {}".format(a, b)

@quartz.remote([str, str], [str])
def keyword_fct2(a="hello", b="world"):
  return "{} {}".format(a, b)

@quartz.remote([int, int, str, str], [str])
def keyword_fct3(a, b, c="hello", d="world"):
  return "{} {} {} {}".format(a, b, c, d)

# Test variable numbers of arguments

@quartz.remote([int], [str])
def varargs_fct1(*a):
  return " ".join(map(str, a))

@quartz.remote([int, int], [str])
def varargs_fct2(a, *b):
  return " ".join(map(str, b))

try:
  @quartz.remote([int], [])
  def kwargs_throw_exception(**c):
    return ()
  kwargs_exception_thrown = False
except:
  kwargs_exception_thrown = True

try:
  @quartz.remote([int, str, int], [str])
  def varargs_and_kwargs_throw_exception(a, b="hi", *c):
    return "{} {} {}".format(a, b, c)
  varargs_and_kwargs_exception_thrown = False
except:
  varargs_and_kwargs_exception_thrown = True

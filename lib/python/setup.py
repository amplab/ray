import sys

from setuptools import setup, Extension, find_packages
import setuptools

# because of relative paths, this must be run from inside quartz/lib/python/

MACOSX = (sys.platform in ["darwin"])

setup(
  name = "quartz",
  version = "0.1.dev0",
  use_2to3=True,
  packages=find_packages(),
  package_data = {
    "quartz": ["libquartzlib.dylib" if MACOSX else "libquartzlib.so",
               "scheduler",
               "objstore"]
  },
  zip_safe=False
)

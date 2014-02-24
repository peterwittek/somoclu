#!/usr/bin/env python2

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension

import numpy

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

somoclu_module = Extension('_somoclu',
                           sources=['somoclu_wrap.cxx', 'somocluWrap.cpp'],
                           extra_objects=['somoclu.o', 'denseCpuKernels.o',
                                          'io.o', 'sparseCpuKernels.o',
                                          'training.o',
                                          'mapDistanceFunctions.o'],
                           libraries=['gomp'],
                           include_dirs=[numpy_include]
                           )

setup(name='somoclu',
      version='1.2',
      author="peterwittek",
      description="""a cluster-oriented implementation \
      of self-organizing maps""",
      ext_modules=[somoclu_module],
      py_modules=["somoclu"],
      )

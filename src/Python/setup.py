#!/usr/bin/env python2

"""
setup.py file for SWIG example
"""

from setuptools import setup, Extension
from setuptools.command.install import install
from subprocess import call
import numpy
import os
import sys

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()
        
sources_files=['somoclu_wrap.cxx',
				'src/src/somocluWrap.cpp',
				'src/src/denseCpuKernels.cpp',
				'src/src/io.cpp',
				'src/src/sparseCpuKernels.cpp',
				'src/src/training.cpp',
				'src/src/mapDistanceFunctions.cpp',
				'src/src/trainOneEpoch.cpp',
				'src/src/uMatrix.cpp']
if sys.platform.startswith('win'):
    extra_compile_args = ['-openmp']
else:
    extra_compile_args = ['-fopenmp']
    extra_link_args = [
        '-lgomp'
    ]        

somoclu_module = Extension('_somoclu',
                           sources=sources_files,						   
                           include_dirs=[numpy_include]
                           )


setup(name='somoclu',
      version='1.4',
      license='GPL3',
      author="peterwittek",
      author_email="",
      maintainer="shichaogao",
      maintainer_email="xgdgsc@gmail.com",
      url="http://peterwittek.github.io/somoclu/",
      platforms="unix",
      description="a cluster-oriented implementation of self-organizing maps",
      ext_modules=[somoclu_module],
      py_modules=["somoclu"],
      install_requires=['numpy'],
      )


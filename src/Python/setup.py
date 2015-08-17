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
        
sources_files=[	'somoclu/src/somocluWrap.cpp',
                'somoclu/src/denseCpuKernels.cpp',
                'somoclu/src/io.cpp',
                'somoclu/src/sparseCpuKernels.cpp',
                'somoclu/src/mapDistanceFunctions.cpp',
                'somoclu/src/trainOneEpoch.cpp',
                'somoclu/src/uMatrix.cpp',
                'somoclu/somoclu_wrap.cxx']
if sys.platform.startswith('win'):
    extra_compile_args = ['-openmp']
    extra_link_args = []
else:
    extra_compile_args = ['-fopenmp']
    extra_link_args = [
        '-lgomp'
    ]        

somoclu_module = Extension('_somoclu_wrap',
                           sources=sources_files,						   
                           include_dirs=[numpy_include, 'src'],
                           extra_compile_args=extra_compile_args,
                           extra_link_args=extra_link_args
                           )


setup(name='somoclu',
      version='1.4.1.1',
      license='GPL3',
      author="peterwittek",
      author_email="",
      maintainer="shichaogao",
      maintainer_email="xgdgsc@gmail.com",
      url="http://peterwittek.github.io/somoclu/",
      platforms=["unix", "windows"],
      description="a massively parallel implementation of "
      "self-organizing maps",
      ext_modules=[somoclu_module],
      py_modules=["somoclu"],
      packages=["somoclu"],
      install_requires=['numpy']
      )


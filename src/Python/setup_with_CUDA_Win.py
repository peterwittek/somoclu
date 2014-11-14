#!/usr/bin/env python2

"""
setup.py file for CUDA
use

sudo python2 setup_with_CUDA.py install

to install
"""

from setuptools import setup, Extension
from setuptools.command.install import install
from subprocess import call
import numpy
import os

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


somoclu_module = Extension('_somoclu',
                           sources=['somoclu_wrap.cxx',
                                    'src/src/somocluWrap.cpp'],
       #                    extra_objects=['src/src/somoclu.obj',
							extra_objects=[
                                          'src/src/denseCpuKernels.obj',
                                          'src/src/io.obj',
                                          'src/src/sparseCpuKernels.obj',
                                          'src/src/training.obj',
                                          'src/src/mapDistanceFunctions.obj',
                                          'src/src/trainOneEpoch.obj',
                                          'src/src/uMatrix.obj',
                                          'src/src/denseGpuKernels.cu.obj'],
                           define_macros=[('CUDA', None)],
                           #PATH to CUDA library here, 64 for 64 bit
                           library_dirs=["C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v6.5/lib/x64"],
                           libraries=['cudart','cublas'],
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
      install_requires=['numpy']
      )


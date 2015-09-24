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


somoclu_module = Extension('_somoclu_wrap',
                           sources=['somoclu/somoclu_wrap.cxx'],
       #                    extra_objects=['somoclu/src/somoclu.obj',
							extra_objects=[
                                          'somoclu/src/denseCpuKernels.obj',
                                          'somoclu/src/io.obj',
                                          'somoclu/src/sparseCpuKernels.obj',
                                          'somoclu/src/training.obj',
                                          'somoclu/src/mapDistanceFunctions.obj',
                                          'somoclu/src/trainOneEpoch.obj',
                                          'somoclu/src/uMatrix.obj',
                                          'somoclu/src/denseGpuKernels.cu.obj'],
                           define_macros=[('CUDA', None)],
                           #PATH to CUDA library here, 64 for 64 bit
                           library_dirs=["C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v6.5/lib/x64"],
                           libraries=['cudart','cublas'],
                           include_dirs=[numpy_include]
                           )


setup(name='somoclu',
      version='1.5',
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
      packages=["somoclu"],
      install_requires=['numpy']
      )


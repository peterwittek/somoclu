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

        
class MyInstall(install):

    def run(self):
        #call(["src/autogen.sh"])
        call(["pwd"])
        os.chdir('somoclu')
        call(["bash", "autogen.sh"])
        ## CUDA PATH here:
        call(["./configure", "--without-mpi","--with-cuda=/opt/cuda/"])
        call(["make"])
        os.chdir('../')
        install.run(self)
        

somoclu_module = Extension('_somoclu_wrap',
                           sources=['somoclu/somoclu_wrap.cxx'],
                           extra_objects=['somoclu/src/somoclu.o',
                                          'somoclu/src/denseCpuKernels.o',
                                          'somoclu/src/io.o',
                                          'somoclu/src/sparseCpuKernels.o',
                                          'somoclu/src/training.o',
                                          'somoclu/src/mapDistanceFunctions.o',
                                          'somoclu/src/trainOneEpoch.o',
                                          'somoclu/src/uMatrix.o',
                                          'somoclu/src/denseGpuKernels.cu.co'],
                           define_macros=[('CUDA', None)],
                           #PATH to CUDA library here, 64 for 64 bit
                           library_dirs=['/opt/cuda/lib64'],
                           libraries=['gomp','cudart','cublas'],
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
      description="a massively parallel implementation of "
      "self-organizing maps",
      ext_modules=[somoclu_module],
      py_modules=["somoclu"],
      packages=["somoclu"],      
      install_requires=['numpy'],
      # test_suite="tests",
      cmdclass={'install': MyInstall}  # , 'build_ext': MyBuildExt
      )


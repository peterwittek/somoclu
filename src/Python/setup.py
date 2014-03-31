#!/usr/bin/env python2

"""
setup.py file for SWIG example
"""

from setuptools import setup, Extension
from setuptools.command.install import install
from subprocess import call
import numpy
#from setuptools.command.build_ext import build_ext
import os

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

    
# class MyBuildExt(build_ext):

#     def run(self):
#         call(["pwd"])
#         call(["cd", "src"])
#         call(["pwd"])
#         call(["bash", "autogen.sh"])
#         build_ext.run(self)

        
class MyInstall(install):

    def run(self):
        #call(["src/autogen.sh"])
        call(["pwd"])
        os.chdir('src')
        call(["bash", "autogen.sh"])
        call(["./configure", "--without-mpi", "--without-cuda"])
        call(["make"])
        os.chdir('../')
        install.run(self)
        

somoclu_module = Extension('_somoclu',
                           sources=['somoclu_wrap.cxx',
                                    'src/src/somocluWrap.cpp'],
                           extra_objects=['src/src/somoclu.o',
                                          'src/src/denseCpuKernels.o',
                                          'src/src/io.o',
                                          'src/src/sparseCpuKernels.o',
                                          'src/src/training.o',
                                          'src/src/mapDistanceFunctions.o'],
                           libraries=['gomp'],
                           include_dirs=[numpy_include]
                           )


setup(name='somoclu',
      version='1.3',
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
      # test_suite="tests",
      cmdclass={'install': MyInstall}  # , 'build_ext': MyBuildExt
      )


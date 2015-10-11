#!/usr/bin/env python
from setuptools import setup, Extension
from setuptools.command.install import install
from subprocess import call
import numpy
import os
import sys
import platform

# Path to CUDA on Linux and OS X
cuda_dir = "/usr/local/cuda"
# Path to CUDA library on Windows, 64 for 64 bit
win_cuda_dir = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.5"

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

arch = int(platform.architecture()[0][0:2])
cmdclass = {}
if sys.platform.startswith('win') and os.path.exists(win_cuda_dir):
    somoclu_module = Extension('_somoclu_wrap',
                               sources=['somoclu/somoclu_wrap.cxx'],
                               extra_objects=[
                                        'somoclu/src/denseCpuKernels.obj',
                                        'somoclu/src/sparseCpuKernels.obj',
                                        'somoclu/src/training.obj',
                                        'somoclu/src/mapDistanceFunctions.obj',
                                        'somoclu/src/uMatrix.obj',
                                        'somoclu/src/denseGpuKernels.cu.obj'],
                               define_macros=[('CUDA', None)],
                               library_dirs=[win_cuda_dir+"/lib/x"+str(arch)],
                               libraries=['cudart', 'cublas'],
                               include_dirs=[numpy_include])
elif os.path.exists(cuda_dir):
    class MyInstall(install):

        def run(self):
            os.chdir('somoclu')
            call(["./configure", "--without-mpi", "--with-cuda=" + cuda_dir])
            call(["make", "lib"])
            os.chdir('../')
            install.run(self)

    if arch == 32:
        cuda_lib_dir = cuda_dir + "/lib"
    else:
        cuda_lib_dir = cuda_dir + "/lib64"
    somoclu_module = Extension('_somoclu_wrap',
                               sources=['somoclu/somoclu_wrap.cxx'],
                               extra_objects=[
                                          'somoclu/src/denseCpuKernels.o',
                                          'somoclu/src/sparseCpuKernels.o',
                                          'somoclu/src/training.o',
                                          'somoclu/src/mapDistanceFunctions.o',
                                          'somoclu/src/uMatrix.o',
                                          'somoclu/src/denseGpuKernels.cu.co'],
                               define_macros=[('CUDA', None)],
                               library_dirs=[cuda_lib_dir],
                               libraries=['gomp', 'cudart', 'cublas'],
                               include_dirs=[numpy_include])
    cmdclass = {'install': MyInstall}

else:
    if sys.platform.startswith('win'):
        extra_compile_args = ['-openmp']
        extra_link_args = []
    else:
        extra_compile_args = ['-fopenmp']
        extra_link_args = [
            '-lgomp'
        ]
    sources_files = ['somoclu/src/denseCpuKernels.cpp',
                     'somoclu/src/io.cpp',
                     'somoclu/src/sparseCpuKernels.cpp',
                     'somoclu/src/mapDistanceFunctions.cpp',
                     'somoclu/src/training.cpp',
                     'somoclu/src/uMatrix.cpp',
                     'somoclu/somoclu_wrap.cxx']
    somoclu_module = Extension('_somoclu_wrap',
                               sources=sources_files,
                               include_dirs=[numpy_include, 'src'],
                               extra_compile_args=extra_compile_args,
                               extra_link_args=extra_link_args
                               )

try:
    setup(name='somoclu',
          version='1.5.0.1',
          license='GPL3',
          author="peterwittek",
          author_email="xgdgsc@gmail.com",
          maintainer="shichaogao",
          maintainer_email="xgdgsc@gmail.com",
          url="http://peterwittek.github.io/somoclu/",
          platforms=["unix", "windows"],
          description="Massively parallel implementation of self-organizing maps",
          ext_modules=[somoclu_module],
          py_modules=["somoclu"],
          packages=["somoclu"],
          install_requires=['numpy', 'matplotlib'],
          cmdclass=cmdclass
          )
except:
    setup(name='somoclu',
          version='1.5.0.1',
          license='GPL3',
          author="peterwittek",
          author_email="xgdgsc@gmail.com",
          maintainer="shichaogao",
          maintainer_email="xgdgsc@gmail.com",
          url="http://peterwittek.github.io/somoclu/",
          platforms=["unix", "windows"],
          description="Massively parallel implementation of self-organizing maps",
          py_modules=["somoclu"],
          packages=["somoclu"],
          install_requires=['numpy', 'matplotlib'],
          cmdclass=cmdclass
          )  

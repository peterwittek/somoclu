#!/usr/bin/env python
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy
import os
import sys
import platform
import traceback
win_cuda_dir = None


def find_cuda():
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = os.path.join(home, 'bin', 'nvcc')
    else:
        nvcc = None
        for dir in os.environ['PATH'].split(os.pathsep):
            binpath = os.path.join(dir, 'nvcc')
            if os.path.exists(binpath):
                nvcc = os.path.abspath(binpath)
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be located in '
                                   'your $PATH. Either add it to your path, or'
                                   'set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))
    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': os.path.join(home, 'include')}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in '
                                   '%s' % (k, v))
    libdir = os.path.join(home, 'lib')
    arch = int(platform.architecture()[0][0:2])
    if sys.platform.startswith('win'):
        os.path.join(libdir, "x"+str(arch))
    if os.path.exists(os.path.join(home, libdir + "64")):
        cudaconfig['lib'] = libdir + "64"
    elif os.path.exists(os.path.join(home, libdir)):
        cudaconfig['lib'] = libdir
    else:
        raise EnvironmentError('The CUDA libraries could not be located')
    return cudaconfig

try:
    CUDA = find_cuda()
except EnvironmentError:
    CUDA = None
    print("Proceeding without CUDA")

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

def customize_compiler_for_nvcc(self):
    '''This is a verbatim copy of the NVCC compiler extension from
    https://github.com/rmcgibbo/npcuda-example
    '''
    self.src_extensions.append('.cu')
    default_compiler_so = self.compiler_so
    super = self._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            self.set_executable('compiler_so', CUDA['nvcc'])
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['cc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        self.compiler_so = default_compiler_so
    self._compile = _compile


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

cmdclass = {}
if sys.platform.startswith('win') and win_cuda_dir is not None:
    if win_cuda_dir == "":
        if 'CUDA_PATH' in os.environ:
            win_cuda_dir = os.environ['CUDA_PATH']
    elif os.path.exists(win_cuda_dir):
        pass
    else:
        win_cuda_dir = None
    if win_cuda_dir:
        arch = int(platform.architecture()[0][0:2])
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
else:
    sources_files = ['somoclu/src/denseCpuKernels.cpp',
                     'somoclu/src/sparseCpuKernels.cpp',
                     'somoclu/src/mapDistanceFunctions.cpp',
                     'somoclu/src/training.cpp',
                     'somoclu/src/uMatrix.cpp',
                     'somoclu/somoclu_wrap.cxx']
    if sys.platform.startswith('win'):
        extra_compile_args = ['-openmp']
        cmdclass = {}
        somoclu_module = Extension('_somoclu_wrap',
                               sources=sources_files,
                               include_dirs=[numpy_include, 'src'],
                               extra_compile_args=extra_compile_args,
                               )
    else:
        extra_compile_args = ['-fopenmp']
        if 'CC' in os.environ and 'clang-omp' in os.environ['CC']:
            openmp = 'iomp5'
        else:
            openmp = 'gomp'
        cmdclass = {'build_ext': custom_build_ext}
        somoclu_module = Extension('_somoclu_wrap',
                                   sources=sources_files,
                                   include_dirs=[numpy_include, 'src'],
                                   extra_compile_args={'cc': extra_compile_args},
                                   libraries=[openmp],
                                   )
    if CUDA is not None:
        somoclu_module.sources.append('somoclu/src/denseGpuKernels.cu')
        somoclu_module.define_macros = [('CUDA', None)]
        somoclu_module.include_dirs.append(CUDA['include'])
        somoclu_module.library_dirs = [CUDA['lib']]
        somoclu_module.libraries += ['cudart', 'cublas']
        somoclu_module.runtime_library_dirs = [CUDA['lib']]
        somoclu_module.extra_compile_args['nvcc']=['-use_fast_math',
                                                   '--ptxas-options=-v', '-c',
                                                   '--compiler-options','-fPIC ' +
                                                   extra_compile_args[0]]



try:
    setup(name='somoclu',
          version='1.6.1',
          license='GPL3',
          author="Peter Wittek, Shi Chao Gao",
          author_email="",
          maintainer="shichaogao",
          maintainer_email="xgdgsc@gmail.com",
          url="https://somoclu.readthedocs.io/",
          platforms=["unix", "windows"],
          description="Massively parallel implementation of self-organizing maps",
          ext_modules=[somoclu_module],
          packages=["somoclu"],
          install_requires=['numpy', 'matplotlib'],
          classifiers=[
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Operating System :: OS Independent',
            'Development Status :: 5 - Production/Stable',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Scientific/Engineering :: Visualization',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python',
            'Programming Language :: C++'
          ],
          cmdclass=cmdclass,
          )
except:
    traceback.print_exc()
    setup(name='somoclu',
          version='1.6.1',
          license='GPL3',
          author="Peter Wittek, Shi Chao Gao",
          author_email="",
          maintainer="shichaogao",
          maintainer_email="xgdgsc@gmail.com",
          url="https://somoclu.readthedocs.io/",
          platforms=["unix", "windows"],
          description="Massively parallel implementation of self-organizing maps",
          packages=["somoclu"],
          classifiers=[
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Operating System :: OS Independent',
            'Development Status :: 5 - Production/Stable',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Scientific/Engineering :: Visualization',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python',
            'Programming Language :: C++'
          ],
          install_requires=['numpy', 'matplotlib']
          )

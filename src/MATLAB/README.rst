Version: 1.6.2

Maintainer: ShichaoGao<xgdgsc at gmail.com>

URL: http://peterwittek.github.io/somoclu/

BugReports: https://github.com/peterwittek/somoclu/issues

License: GPL-3

OS_type: unix, windows

Somoclu MATLAB Extension Build Guide (Linux/Mac):
=================================================

   **(OS X users see the Note below first)**

1. Configure the library to compile without MPI support. Optionally, specify the root of your MATLAB installation. E.g.::

    $ ./configure --without-mpi --with-matlab=/usr/local/MATLAB/R2014a

If you want CUDA support, specify the CUDA directory as well.

2. Build MATLAB Extension by running:
   ::
      make matlab

3. Then ``MexSomoclu.mexa64`` or ``MexSomoclu.mexa32`` is generated for use, you can test by installing som-toolbox from https://github.com/ilarinieminen/SOM-Toolbox, and running the ``mex_interface_test.m`` or ``mex_interface_test_gpu.m``.

If you encounter errors like:
::
  /usr/local/MATLAB/R2013a/sys/os/glnx86/libstdc++.so.6:
  version `GLIBCXX_3.4.20' not found
  
You can rename ``libstdc++.so.6*`` under ``MATLAB_ROOT/sys/os/glnxa64`` to solve this issue.

Note for Mac OS X users:
================================
Using GCC
---------------
As of OS X 10.10, gcc is just symlink to clang. To build somoclu and this extension using GCC, it is recommended to install gcc using something like:
::
   $ brew install gcc --without-multilib

and set environment using:
::
    export CC=/usr/local/bin/gcc-5
    export CXX=/usr/local/bin/g++-5
    export CPP=/usr/local/bin/cpp-5
    export LD=/usr/local/bin/gcc-5
    alias c++=/usr/local/bin/c++-5
    alias g++=/usr/local/bin/g++-5
    alias gcc=/usr/local/bin/gcc-5
    alias cpp=/usr/local/bin/cpp-5
    alias ld=/usr/local/bin/gcc-5
    alias cc=/usr/local/bin/gcc-5
    
before running ``./configure`` .

Then follow the instructions at https://github.com/peterwittek/somoclu to build somoclu itself.


Using clang-omp
---------------
To install clang-omp, follow instructions at http://clang-omp.github.io/. And set environment using:
::
    export CC=/usr/local/bin/clang-omp
    export CXX=/usr/local/bin/clang-omp++
    export CPP=/usr/local/bin/clang-omp++
    export LD=/usr/local/bin/clang-omp
    alias c++=/usr/local/bin/clang-omp++
    alias g++=/usr/local/bin/clang-omp++
    alias gcc=/usr/local/bin/clang-omp
    alias cpp=/usr/local/bin/clang-omp++
    alias ld=/usr/local/bin/clang-omp
    alias cc=/usr/local/bin/clang-omp
    export PATH=/usr/local/bin/:$PATH
    export C_INCLUDE_PATH=/usr/local/include/:$C_INCLUDE_PATH
    export CPLUS_INCLUDE_PATH=/usr/local/include/:$CPLUS_INCLUDE_PATH
    export LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH


before running ``./configure`` .


Building Mex Extension on OS X:
===============================
Using GCC
---------------
To build the extension on OS X, we need to make mex use gcc instead of the default clang compiler which doesn' t support openmp (As of OSX 10.10.5). We need to copy ``MATLAB_ROOT/bin/mexopts.sh`` to ``~/.matlab/VERSION/mexopts.sh`` , replace ``MATLAB_ROOT`` with your installation path of MATLAB and replace ``VERSION`` with your MATLAB version in that folder. Example:
::
   cp /Applications/MATLAB_R2013a.app/bin/mexopts.sh ~/.matlab/R2013a/mexopts.sh

Then modify ``~/.matlab/VERSION/mexopts.sh`` to use gcc as follows:

1. change ``CC='gcc'`` and comment out all ``CC=`` statements after that.
2. change ``CXX='g++'`` and comment out all ``CXX=`` statements after that.
3. change ``MACOSX_DEPLOYMENT_TARGET='10.9'`` where ``10.9`` is your OS X version number.

an example is given at https://gist.github.com/xgdgsc/9832340, then you can follow the instruction step 2 at the top to build the extension and test.
  

Using clang-omp
---------------
Similar to above GCC approach, we need to make mex use clang-omp by modifying ``~/.matlab/VERSION/mexopts.sh``, an example is given at https://gist.github.com/xgdgsc/6cfeda967ee44fef4603 . Note ``CXXFLAGS = -std=c++11``, ``LDFLAGS="$LDFLAGS -fopenmp"``

Then you can follow the instruction step 2 at the top to build the extension and test. If you encounter errors including ``libiomp5.dylib`` when running the test after build, renaming the file packed with MATLAB under ``/Applications/MATLAB_R2013a.app/sys/os/maci64/libiomp5.dylib`` would fix it.

Building Mex Extension on Windows:
===================================

First, you should install some supported version of Visual Studio that includes the Visual C++ compiler by your MATLAB version like on `this <http://www.mathworks.com/support/compilers/R2013a/index.html?sec=win64/>`_ page. With MATLAB and Visual Studio installed properly, running ``mex -setup`` in CMD will prompt fpr available compilers and you can choose the appropriate version.

Then run the script in this folder makeMex.bat in CMD and the ``MexSomoclu.mexa64`` or ``MexSomoclu.mexa32`` is generated for use, you can test by running the ``mex_interface_test.m``.

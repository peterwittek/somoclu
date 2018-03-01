Version: 1.7.5

Maintainer: ShichaoGao<xgdgsc at gmail.com>

URL: http://peterwittek.github.io/somoclu/

BugReports: https://github.com/peterwittek/somoclu/issues

License: GPL-3

OS_type: unix, windows

Somoclu MATLAB Extension Build Guide (Linux/Mac):
=================================================

   **(macOS users see the Note below first)**

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

Note for macOS users:
================================
Using GCC
---------------
As of macOS 10.10, gcc is just symlink to clang. To build somoclu and this extension using GCC, it is recommended to install gcc using something like:
::
   $ brew install gcc --without-multilib

and set environment using:
::
    export CC=/usr/local/bin/gcc-7
    export CXX=/usr/local/bin/g++-7
    export CPP=/usr/local/bin/cpp-7
    export LD=/usr/local/bin/gcc-7
    alias c++=/usr/local/bin/c++-7
    alias g++=/usr/local/bin/g++-7
    alias gcc=/usr/local/bin/gcc-7
    alias cpp=/usr/local/bin/cpp-7
    alias ld=/usr/local/bin/gcc-7
    alias cc=/usr/local/bin/gcc-7

before running ``./configure`` .

Then follow the instructions at https://github.com/peterwittek/somoclu to build somoclu itself.


Building Mex Extension on macOS:
===============================
Using GCC
---------------
To build the extension on macOS, we need to make mex use gcc instead of the default clang compiler which doesn' t support openmp (As of OSX 10.10.5). We need to copy ``MATLAB_ROOT/bin/mexopts.sh`` to ``~/.matlab/VERSION/mexopts.sh`` , replace ``MATLAB_ROOT`` with your installation path of MATLAB and replace ``VERSION`` with your MATLAB version in that folder. Example:
::
   cp /Applications/MATLAB_R2013a.app/bin/mexopts.sh ~/.matlab/R2013a/mexopts.sh

Then modify ``~/.matlab/VERSION/mexopts.sh`` to use gcc as follows:

1. change ``CC='gcc'`` and comment out all ``CC=`` statements after that.
2. change ``CXX='g++'`` and comment out all ``CXX=`` statements after that.
3. change ``MACOSX_DEPLOYMENT_TARGET='10.9'`` where ``10.9`` is your macOS version number.

an example is given at https://gist.github.com/xgdgsc/9832340, then you can follow the instruction step 2 at the top to build the extension and test.


Building Mex Extension on Windows:
===================================

First, you should install some supported version of Visual Studio that includes the Visual C++ compiler by your MATLAB version like on `this <http://www.mathworks.com/support/compilers/R2013a/index.html?sec=win64/>`_ page. With MATLAB and Visual Studio installed properly, running ``mex -setup`` in CMD will prompt fpr available compilers and you can choose the appropriate version.

Then run the script in this folder makeMex.bat in CMD and the ``MexSomoclu.mexa64`` or ``MexSomoclu.mexa32`` is generated for use, you can test by running the ``mex_interface_test.m``.

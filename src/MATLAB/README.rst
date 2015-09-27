Version: 1.5

Maintainer: ShichaoGao<xgdgsc at gmail.com>

URL: http://peterwittek.github.io/somoclu/

BugReports: https://github.com/peterwittek/somoclu/issues

License: GPL-3

OS_type: unix, windows

Somoclu MATLAB Extension Build Guide (Linux/Mac):
================================

1. Follow the instructions to build Somoclu itself at: https://github.com/peterwittek/somoclu

   **(OS X users see the Note below first)**

2. Build MATLAB Extension by running:
   ::
      MEX_BIN="/usr/local/MATLAB/R2013a/bin/mex" ./makeMex.sh
    
where ``MEX_BIN`` is the path to the MATLAB installation mex binary.

3. Then ``MexSomoclu.mexa64`` or ``MexSomoclu.mexa32`` is generated for use, you can test by running the ``mex_interface_test.m``.

Note for Mac OS X users:
================================

As of OS X 10.9, gcc is just symlink to clang. To build somoclu and this extension correctly, it is recommended to install gcc using something like:
::
   $ brew install gcc48

and set environment using:
::
    export CC=/usr/local/bin/gcc
    export CXX=/usr/local/bin/g++
    export CPP=/usr/local/bin/cpp
    export LD=/usr/local/bin/gcc
    alias c++=/usr/local/bin/c++
    alias g++=/usr/local/bin/g++	
    alias gcc=/usr/local/bin/gcc
    alias cpp=/usr/local/bin/cpp
    alias ld=/usr/local/bin/gcc
    alias cc=/usr/local/bin/gcc

Then follow the instructions at https://github.com/peterwittek/somoclu to build somoclu itself.

Building Mex Extension on OS X:
===============================

To build the extension on OS X, we need to make mex use gcc instead of the default clang compiler. So we need to copy ``MATLAB_ROOT/bin/mexopts.sh`` to ``~/.matlab/VERSION/mexopts.sh`` , replace ``MATLAB_ROOT`` with your installation path of MATLAB and replace ``VERSION`` with your MATLAB version in that folder. Example:
::
   cp /Applications/MATLAB_R2013a.app/bin/mexopts.sh ~/.matlab/R2013a/mexopts.sh

Then modify ``~/.matlab/VERSION/mexopts.sh`` to use gcc as follows:

1. change ``CC='gcc'`` and comment out all ``CC=`` statements after that.
2. change ``CXX='g++'`` and comment out all ``CXX=`` statements after that.
3. change ``MACOSX_DEPLOYMENT_TARGET='10.9'`` where ``10.9`` is your OS X version number.

an example is given at https://gist.github.com/xgdgsc/9832340, then you can follow the instruction step 2 at the top to build the extension and test.

Building Mex Extension on Windows:
===================================

First, you should install some supported version of Visual Studio that includes the Visual C++ compiler by your MATLAB version like on `this <http://www.mathworks.com/support/compilers/R2013a/index.html?sec=win64/>`_ page. With MATLAB and Visual Studio installed properly, running ``mex -setup`` in CMD will prompt fpr available compilers and you can choose the appropriate version. 

Then run the script in this folder makeMex.bat in CMD and the ``MexSomoclu.mexa64`` or ``MexSomoclu.mexa32`` is generated for use, you can test by running the ``mex_interface_test.m``.

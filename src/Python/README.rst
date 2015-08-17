Somoclu - Python Interface
================================

Somoclu is a massively parallel implementation of self-organizing maps. It relies on OpenMP for multicore execution, MPI for distributing the workload, and it can be accelerated by CUDA on a GPU cluster. A sparse kernel is also included, which is useful for training maps on vector spaces generated in text mining processes. The topology of the grid is rectangular. Currently a subset of the command line version is supported with this package.

Key features of the Python interface:

* Fast execution by parallelization: OpenMP and CUDA are supported.
* Multi-platform: Linux, OS X, and Windows are supported.
* Planar and toroid maps.
* Both dense and sparse input data are supported.
* Large maps of several hundred thousand neurons are feasible.


Usage
------
The following file uses rgbs.txt, which can be found at https://github.com/peterwittek/somoclu/tree/master/data

.. code-block:: python
		
    #!/usr/bin/env python2
    # -*- coding: utf-8 -*-
    import somoclu
    import numpy as np

    data = np.loadtxt('rgbs.txt', dtype=np.float32)
    nSomX, nSomY = 50, 50

    som = somoclu.Somoclu(data, nSomX, nSomY)
    som.train()
    print som.codebook
    print som.globalBmus
    print som.uMatrix




Installation
------------
The code is available on PyPI, hence it can be installed by

::

    $ sudo pip install somoclu

If you want the latest git version, follow the standard procedure for installing Python modules:

::

    $ sudo python setup.py install

Build on Mac OS X
--------------------
Before installing using pip, gcc should be installed first. As of OS X 10.9, gcc is just symlink to clang. To build somoclu and this extension correctly, it is recommended to install gcc using something like:
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

Then you can issue
::
   
    $ sudo pip install somoclu

    
Build with CUDA support on Linux and OS X:
------------------------------------------
You need to clone the GitHub repo or download the latest release tarball, and use the following script:

::
   
    $ cd src/Python
    $ bash makepy.sh

Then if your CUDA installation is located at /opt/cuda and 64bit, you can do the following to install:

::
   
    $ sudo python2 setup_with_CUDA.py install

Otherwise, you should modify the setup_with_CUDA.py ,
change the path to CUDA installation accordingly:

::
   
   call(["./configure", "--without-mpi","--with-cuda=/opt/cuda/"])

and

::
   
   library_dirs=['/opt/cuda/lib64']

Then run the install command

::
   
    $ sudo python2 setup_with_CUDA.py install

Then you can use the python interface like before, with CUDA support.

Build with CUDA support on Windows:
--------------------------------------
You should first follow the instructions to build the windows binary at https://github.com/peterwittek/somoclu with MPI disabled with the same version Visual Studio as your Python is built.(Since currently Python is built by VS2008 by default and CUDA v6.5 removed VS2008 support, you may use CUDA 6.0 with VS2008 or find a Python prebuilt with VS2010. And remember to install VS2010 or Windows SDK7.1 to get the option in Platform Toolset if you use VS2013.) Then you should copy the .obj files generated in the release build path to the Python/src/src folder. 

Then modify the library_dirs in setup_with_CUDA_Win.py  to your CUDA path.

Then run the install command

::
   
    $ sudo python2 setup_with_CUDA_Win.py install
	
Then it should be able to build and install the extension.

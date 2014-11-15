Somoclu - python interface
================================

Somoclu is a massively parallel implementation of self-organizing maps. It relies on OpenMP for multicore execution, MPI for distributing the workload, and it can be accelerated by CUDA on a GPU cluster. A sparse kernel is also included, which is useful for training maps on vector spaces generated in text mining processes. The topology of the grid is rectangular.

Currently a subset of the C++ version is supported with this package.

Homepage: `https://github.com/peterwittek/somoclu <https://github.com/peterwittek/somoclu/>`_

Example, in which the data file rgbs.txt can be found at https://github.com/peterwittek/somoclu/tree/master/data

.. code-block:: python
		
    #!/usr/bin/env python2
    # -*- coding: utf-8 -*-
    import somoclu
    import numpy as np

    data = np.loadtxt('rgbs.txt')
    print(data)
    data = np.float32(data)
    nSomX = 50
    nSomY = 50
    nVectors = data.shape[0]
    nDimensions = data.shape[1]
    data1D = np.ndarray.flatten(data)
    nEpoch = 10
    radius0 = 0
    radiusN = 0
    radiusCooling = "linear"
    scale0 = 0
    scaleN = 0.01
    scaleCooling = "linear"
    kernelType = 0
    mapType = "planar"
    snapshots = 0
    initialCodebookFilename = ''
    codebook_size = nSomY * nSomX * nDimensions
    codebook = np.zeros(codebook_size, dtype=np.float32)
    globalBmus_size = int(nVectors * int(np.ceil(nVectors/nVectors))*2)
    globalBmus = np.zeros(globalBmus_size, dtype=np.intc)
    uMatrix_size = nSomX * nSomY
    uMatrix = np.zeros(uMatrix_size, dtype=np.float32)
    somoclu.trainWrapper(data1D, nEpoch, nSomX, nSomY,
                         nDimensions, nVectors,
                         radius0, radiusN,
                         radiusCooling, scale0, scaleN,
                         scaleCooling, snapshots,
                         kernelType, mapType,
                         initialCodebookFilename,
                         codebook, globalBmus, uMatrix)
    print codebook
    print globalBmus
    print uMatrix



Get it now
----------

::
   
    $ sudo pip install somoclu

Build on Mac OS X:
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

Then you can
::
   
    $ sudo pip install somoclu

    
Build with CUDA support on Linux:
--------------------------------------
You need to clone this repo or download the latest release tarball, and

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

Then you can use the python interface like before, with CUDA support. Building on OS X should be quite similar to linux.

Build with CUDA support on Windows:
--------------------------------------
You should first follow the instructions to build the windows binary at https://github.com/peterwittek/somoclu with MPI disabled with the same version Visual Studio as your Python is built.(Since currently Python is built by VS2008 by default and CUDA v6.5 removed VS2008 support, you may use CUDA 6.0 with VS2008 or find a Python prebuilt with VS2010. And remember to install VS2010 or Windows SDK7.1 to get the option in Platform Toolset if you use VS2013.) Then you should copy the .obj files generated in the release build path to the Python/src/src folder. 

Then modify the library_dirs in setup_with_CUDA_Win.py  to your CUDA path.

Then run the install command

::
   
    $ sudo python2 setup_with_CUDA_Win.py install
	
Then it should be able to build and install the extension.
Somoclu - Python Interface
================================

Somoclu is a massively parallel implementation of self-organizing maps. It relies on OpenMP for multicore execution, MPI for distributing the workload, and it can be accelerated by CUDA. A sparse kernel is also included, which is useful for training maps on vector spaces generated in text mining processes. The topology of map is either planar or toroid, the grid is square or hexagonal. Currently a subset of the command line version is supported with this Python module.

Key features of the Python interface:

* Fast execution by parallelization: OpenMP and CUDA are supported.
* Multi-platform: Linux, OS X, and Windows are supported.
* Planar and toroid maps.
* Square and hexagonal grids.
* Visualization of maps, including those that were trained outside of Python.

The documentation is available online. Further details are found in the following paper:

Peter Wittek, Shi Chao Gao, Ik Soo Lim, Li Zhao (2015). Somoclu: An Efficient Parallel Library for Self-Organizing Maps. `arXiv:1305.1422 <http://arxiv.org/abs/1305.1422>`_.


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
If your CUDA is installed elsewhere than /usr/local/cuda, you cannot directly install the module from PyPI. Please download the `source distribution <https://pypi.python.org/packages/source/n/somoclu/somoclu-1.5.tar.gz>`_ from PyPI. Open the setup.py file in an editor and modify the path to your CUDA installation directory:

::
   
   cuda_dir = /path/to/cuda

Then run the install command

::
   
    $ sudo python setup.py install

Build with CUDA support on Windows:
--------------------------------------
You should first follow the instructions to `build the Windows binary <https://github.com/peterwittek/somoclu>`_ with MPI disabled with the same version Visual Studio as your Python is built with.(Since currently Python is built by VS2008 by default and CUDA v6.5 removed VS2008 support, you may use CUDA 6.0 with VS2008 or find a Python prebuilt with VS2010. And remember to install VS2010 or Windows SDK7.1 to get the option in Platform Toolset if you use VS2013.) Then you should copy the .obj files generated in the release build path to the Python/src folder. 

Then modify the win_cuda_dir in setup.py to your CUDA path and run the install command

::
   
    $ sudo python setup.py install
	
Then it should be able to build and install the module.

*************************
Download and Installation
*************************
The package is available in the `Python Package Index <https://pypi.python.org/pypi/somoclu/>`_, containing the source, documentation, and examples. The latest development version is available on `GitHub <https://github.com/peterwittek/somoclu>`_.

Dependencies
============
The module requires `Numpy <http://www.numpy.org/>`_ and `matplotlib <http://www.matplotlib.org/>`_. The code is compatible with both Python 2 and 3. 

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
If the ``CUDAHOME`` variable is set, the usual install command will build and install the library:

::
   
    $ sudo python setup.py install

Build with CUDA support on Windows:
--------------------------------------
You should first follow the instructions to `build the Windows binary <https://github.com/peterwittek/somoclu>`_ with MPI disabled with the same version Visual Studio as your Python is built with.(Since currently Python is built by VS2008 by default and CUDA v6.5 removed VS2008 support, you may use CUDA 6.0 with VS2008 or find a Python prebuilt with VS2010. And remember to install VS2010 or Windows SDK7.1 to get the option in Platform Toolset if you use VS2013.) Then you should copy the .obj files generated in the release build path to the Python/src folder. 

Then modify the win_cuda_dir in setup.py to your CUDA path and run the install command

::
   
    $ sudo python setup.py install
	
Then it should be able to build and install the module.

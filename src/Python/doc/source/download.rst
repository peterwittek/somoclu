*************************
Download and Installation
*************************
The entire package for is available as a `gzipped tar <https://pypi.python.org/packages/source/n/somoclu/somoclu-1.5.tar.gz>`_ file from the `Python Package Index <https://pypi.python.org/pypi/somoclu/>`_, containing the source, documentation, and examples.

The latest development version is available on `GitHub <https://github.com/peterwittek/somoclu>`_.

Dependencies
============
The module requires `Numpy <http://www.numpy.org/>`_ and `matplotlib <http://www.matplotlib.org/>`_. The code is compatible with both Python 2 and 3. 

Installation
============
Follow the standard procedure for installing Python modules:

::

    $ pip install somoclu

If you use the development version, install it from the source code:

::

    $ git clone https://github.com/peterwittek/somoclu.git
    $ cd somoclu
    $ python setup.py install

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

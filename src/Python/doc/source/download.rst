*************************
Download and Installation
*************************
The package is available in the `Python Package Index <https://pypi.python.org/pypi/somoclu/>`_, containing the source, documentation, and examples. The latest development version is available on `GitHub <https://github.com/peterwittek/somoclu>`_.

Dependencies
============
The module requires `Numpy <http://www.numpy.org/>`_ and `matplotlib <http://www.matplotlib.org/>`_. The code is compatible with both Python 2 and 3.

On Linux and macOS, you need a standard C++ compile chain, for instance, GCC, ICC and clang are known to work.

On Windows, having ``MSVCP90.DLL`` and ``VCOMP90.DLL`` is usually sufficient. See `this issue <https://github.com/peterwittek/somoclu/issues/28#issuecomment-238419778>`_ if you have problems.

Installation
------------
The code is available on PyPI, hence it can be installed by

::

    $ pip install somoclu

Alternatively, it is also available on [conda-forge](https://github.com/conda-forge/somoclu-feedstock):

::

    $ conda install somoclu

If you want the latest git version, clone the repository, make the Python target, and follow the standard procedure for installing Python modules:

::

    $ git clone https://github.com/peterwittek/somoclu.git
    $ cd somoclu
    $ ./autogen.sh
    $ ./configure
    $ make python
    $ cd src/Python
    $ sudo python setup.py install


Build with CUDA support on Linux and macOS:
-------------------------------------------
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

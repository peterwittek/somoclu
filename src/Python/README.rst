Somoclu - Python Interface
================================

Somoclu is a massively parallel implementation of self-organizing maps. It relies on OpenMP for multicore execution and it can be accelerated by CUDA. The topology of map is either planar or toroid, the grid is rectangular or hexagonal. Currently a subset of the command line version is supported with this Python module.

Key features of the Python interface:

* Fast execution by parallelization: OpenMP and CUDA are supported.
* Multi-platform: Linux, OS X, and Windows are supported.
* Planar and toroid maps.
* Rectangular and hexagonal grids.
* Gaussian or bubble neighborhood functions.
* Visualization of maps, including those that were trained outside of Python.
* PCA initialization of codebook.

The documentation is available on `Read the Docs <https://somoclu.readthedocs.io/>`_. Further details are found in the manuscript describing the library [1].

Usage
-----
A simple example is below. For more example, please refer to the `documentation <https://somoclu.readthedocs.io/>`_ and a more thorough ipython notebook example at `Somoclu in Python.ipynb <http://nbviewer.ipython.org/github/peterwittek/ipython-notebooks/blob/master/Somoclu%20in%20Python.ipynb>`_.

::

    import somoclu
    import numpy as np
    import matplotlib.pyplot as plt

    c1 = np.random.rand(50, 2)/5
    c2 = (0.2, 0.5) + np.random.rand(50, 2)/5
    c3 = (0.4, 0.1) + np.random.rand(50, 2)/5
    data = np.float32(np.concatenate((c1, c2, c3)))
    colors = ["red"] * 50
    colors.extend(["green"] * 50)
    colors.extend(["blue"] * 50)

    labels = list(range(150))
    n_rows, n_columns = 30, 50
    som = somoclu.Somoclu(n_columns, n_rows, maptype="planar",
                          gridtype="rectangular")
    som.train(data, epochs=10)
    som.view_umatrix(bestmatches=True, bestmatchcolors=colors, labels=labels)

Installation
------------
The code is available on PyPI, hence it can be installed by

::

    $ pip install somoclu

Alternatively, it is also available on [conda-forge](https://github.com/conda-forge/somoclu-feedstock):

::

    $ conda install somoclu

Some pre-built binaries in the wheel format or windows installer are provided at `PyPI Dowloads <https://pypi.python.org/pypi/somoclu#downloads>`_, they are tested with `Anaconda <https://www.continuum.io/downloads>`_ distributions. If you encounter errors like `ImportError: DLL load failed: The specified module could not be found` when `import somoclu`, you may need to use `Dependency Walker <http://www.dependencywalker.com/>`_ as shown `here <http://stackoverflow.com/a/24704384/1136027>`_ on ``_somoclu_wrap.pyd`` to find out missing DLLs and place them at the write place. Usually right version (32/64bit) of ``vcomp90.dll, msvcp90.dll, msvcr90.dll`` should be put to ``C:\Windows\System32`` or ``C:\Windows\SysWOW64``.

If you want the latest git version, first git clone the repo, install `swig <http://www.swig.org/>`_ and run:

::

    $ ./autogen.sh
    $ ./configure [options]
    $ make
    $ make python

to generate Python interface files.

Then follow the standard procedure for installing Python modules:

::

    $ python setup.py install

Build with CUDA support on Linux and OS X:
------------------------------------------
If the ``CUDAHOME`` variable is set, the usual install command will build and install the library:

::

    $ sudo python setup.py install

Build with CUDA support on Windows:
--------------------------------------
You should first follow the instructions to `build the Windows binary <https://github.com/peterwittek/somoclu>`_ with ``HAVE_MPI`` and ``CLI`` disabled with the same version Visual Studio as your Python is built with.(Since currently Python is built by VS2008 by default and CUDA v6.5 removed VS2008 support, you may use CUDA 6.0 with VS2008 or find a Python prebuilt with VS2010. And remember to install VS2010 or Windows SDK7.1 to get the option in Platform Toolset if you use VS2013.) The recommended configuration is VS2010  Platform Toolset with Python 3.4. Then you should copy the .obj files generated in the release build path to the ``Python\somoclu\src`` folder.

Then modify the environment variable ``CUDA_PATH`` or ``win_cuda_dir`` in ``setup.py`` to your CUDA path and run the install command

::

    $ sudo python setup.py install

Then it should be able to build and install the module.

Citation
--------

1. Peter Wittek, Shi Chao Gao, Ik Soo Lim, Li Zhao (2017). Somoclu: An Efficient Parallel Library for Self-Organizing Maps.  Journal of Statistical Software, 78(9), pp.1--21. DOI:`10.18637/jss.v078.i09 <https://doi.org/10.18637/jss.v078.i09>`_. arXiv:`1305.1422 <https://arxiv.org/abs/1305.1422>`_.

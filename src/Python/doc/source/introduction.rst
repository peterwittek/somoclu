============
Introduction
============
Somoclu is a massively parallel implementation of self-organizing maps. It relies on OpenMP for multicore execution and it can be accelerated by CUDA. The topology of map is either planar or toroid, the grid is rectangular or hexagonal. Currently a subset of the command line version is supported with this Python module.

Key features of the Python interface:

* Fast execution by parallelization: OpenMP and CUDA are supported.
* Multi-platform: Linux, macOS, and Windows are supported.
* Planar and toroid maps.
* Rectangular and hexagonal grids.
* Gaussian or bubble neighborhood functions.
* Visualization of maps, including those that were trained outside of Python.
* PCA initialization of codebook.

The documentation is available online. Further details are found in the following paper:

Peter Wittek, Shi Chao Gao, Ik Soo Lim, Li Zhao (2017). Somoclu: An Efficient Parallel Library for Self-Organizing Maps.  Journal of Statistical Software, 78(9), pp.1--21. DOI:`10.18637/jss.v078.i09 <https://doi.org/10.18637/jss.v078.i09>`_. arXiv:`1305.1422 <https://arxiv.org/abs/1305.1422>`_.

Copyright and License
---------------------
Somoclu is free software; you can redistribute it and/or modify it under the terms of the `MIT License <https://opensource.org/license/mit/>`_ as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

Somoclu is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the `MIT License <https://opensource.org/license/mit/>`_ for more details. 


Acknowledgment
--------------
This work is supported by the European Commission Seventh Framework Programme under Grant Agreement Number FP7-601138 `PERICLES <http://pericles-project.eu/>`_ and by the AWS in Education Machine Learning Grant award.

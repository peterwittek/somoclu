============
Introduction
============
Somoclu is a massively parallel implementation of self-organizing maps. It relies on OpenMP for multicore execution, MPI for distributing the workload, and it can be accelerated by CUDA. A sparse kernel is also included, which is useful for training maps on vector spaces generated in text mining processes. The topology of map is either planar or toroid, the grid is rectangular or hexagonal. Currently a subset of the command line version is supported with this Python module.

Key features of the Python interface:

* Fast execution by parallelization: OpenMP and CUDA are supported.
* Multi-platform: Linux, OS X, and Windows are supported.
* Planar and toroid maps.
* Rectangular and hexagonal grids.
* Gaussian or bubble neighborhood function.
* Visualization of maps, including those that were trained outside of Python.

The documentation is available online. Further details are found in the following paper:

Peter Wittek, Shi Chao Gao, Ik Soo Lim, Li Zhao (2015). Somoclu: An Efficient Parallel Library for Self-Organizing Maps. `arXiv:1305.1422 <http://arxiv.org/abs/1305.1422>`_.

Copyright and License
---------------------
Somoclu is free software; you can redistribute it and/or modify it under the terms of the `GNU General Public License <http://www.gnu.org/licenses/gpl-3.0.html>`_ as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

Somoclu is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the `GNU General Public License <http://www.gnu.org/licenses/gpl-3.0.html>`_ for more details. 


Acknowledgment
--------------
This work is supported by the European Commission Seventh Framework Programme under Grant Agreement Number FP7-601138 `PERICLES <http://pericles-project.eu/>`_ and by the AWS in Education Machine Learning Grant award.

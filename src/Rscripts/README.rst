Somoclu - R interface
================================

Somoclu is a massively parallel implementation of self-organizing maps. It relies on OpenMP for multicore execution, MPI for distributing the workload, and it can be accelerated by CUDA on a GPU cluster. A sparse kernel is also included, which is useful for training maps on vector spaces generated in text mining processes.

Currently a subset of the C++ version is supported with this package.

Homepage: `https://github.com/peterwittek/somoclu <https://github.com/peterwittek/somoclu/>`_

Note for Building on Mac OS X:
----------
Before installing using R commands, gcc should be installed first. As of OS X 10.9, gcc is just symlink to clang. To build somoclu and this extension correctly, it is recommended to install gcc using something like:
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

Then you can install within R:
::
    install.packages("Rsomoclu")

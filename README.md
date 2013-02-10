Somoclu
==
Somoclu is a cluster-oriented implementation of self-organizing maps. It relies on MPI for distributing the workload, and it can be accelerated by CUDA on a GPU cluster. A sparse kernel is also included, which is useful for training maps on vector spaces generated in text mining processes.

Usage
==
Somoclu takes a plain text input file. The dense format should include an instance in one row. The sparse format follows the libsvm sparse matrix format. Example files are included.

    $ [mpirun -np NPROC] somoclu [OPTIONs] INPUT_FILE OUTPUT_PREFIX

Arguments:

    -e NUMBER     Maximum number of epochs
    -k NUMBER     Kernel type
                     0: Dense CPU
                     1: Dense GPU
                     2: Sparse CPU
    -s            Enable snapshots of U-matrix
    -x NUMBER     Dimension of SOM in direction x
    -y NUMBER     Dimension of SOM in direction y

Examples:

    $ somoclu data/rgbs.txt data/rgbs
    $ mpirun -np 4 somoclu -k 0 -x 20 -y 20 data/rgbs.txt data/rgbs

Dependencies
==
Note that MPI is required. CUDA support is optional.

Compilation & Installation
==
From GIT repository first run

    $ ./autogen.sh

Then follow the standard POSIX procedure:

    $ ./configure [options]
    $ make
    $ make install


Options for configure
--

    --prefix=PATH           Set directory prefix for installation


By default Somoclu is installed into /usr/local. If you prefer a
different location, use this option to select an installation
directory.

    --with-cuda=/path/to/cuda           Set path for CUDA

Somoclu looks for CUDA in /usr/local/cuda. If your installation is not there, then specify the path with this parameter. If you do not want CUDA enabled, set the parameter to ```--without-cuda```.

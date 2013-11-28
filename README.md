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
    -m NUMBER     Map type
                     0: Planar
                     1: Toroid
    -r NUMBER     Initial radius (default: half of the map in direction x)                     
    -s            Enable snapshots of U-matrix
    -x NUMBER     Dimension of SOM in direction x
    -y NUMBER     Dimension of SOM in direction y

Examples:

    $ somoclu data/rgbs.txt data/rgbs
    $ mpirun -np 4 somoclu -k 0 -x 20 -y 20 data/rgbs.txt data/rgbs

Visualisation
==
The primary purpose of generating a map is visualisation. Somoclu does not come with its own functions for visualisation, since there are numerous generic tools that are capable of plotting high-quality figures. A simplot gnuplot script is provided with the source code as plot_som.gp. This script takes a U-matrix (umat.umx), and outputs a plot (som.png).

The output formats of the U-matrix and the codebook are compatible with [Databionic ESOM Tools](http://databionic-esom.sourceforge.net/) for more advanced visualisation.

Dependencies
==
You need a working MPI installation on your system to compile Somoclu. A single-core or a single-GPU variant will run without the MPI runtime. The package was tested with OpenMPI, but it should work with other MPI flavours. 

CUDA support is optional. CUDA 4.1 and 5.0 versions are known to work.

Compilation & Installation
==
From GIT repository first run

    $ ./autogen.sh

Then follow the standard POSIX procedure:

    $ ./configure [options]
    $ make
    $ make install


Options for configure

    --prefix=PATH           Set directory prefix for installation


By default Somoclu is installed into /usr/local. If you prefer a
different location, use this option to select an installation
directory.

    --with-mpi-compilers=DIR or --with-mpi-compilers=yes
                              use MPI compiler (mpicxx) found in directory DIR, or
                              in your PATH if =yes
    --with-mpi=MPIROOT      use MPI root directory.
    --with-mpi-libs="LIBS"  MPI libraries [default "-lmpi"]
    --with-mpi-incdir=DIR   MPI include directory [default MPIROOT/include]
    --with-mpi-libdir=DIR   MPI library directory [default MPIROOT/lib]

The above flags allow the identification of the correct MPI library the user wishes to use. The flags are especially useful if MPI is installed in a non-standard location, or when multiple MPI libraries are available.

    --with-cuda=/path/to/cuda           Set path for CUDA

Somoclu looks for CUDA in /usr/local/cuda. If your installation is not there, then specify the path with this parameter. If you do not want CUDA enabled, set the parameter to ```--without-cuda```.

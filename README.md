Somoclu
==
Somoclu is a cluster-oriented implementation of self-organizing maps. It relies on MPI for distributing the workload, and it can be accelerated by CUDA on a GPU cluster. A sparse kernel is also included, which is useful for training maps on vector spaces generated in text mining processes.

Key features:

* Fast execution by parallelization: MPI and CUDA are supported.
* Planar and toroid maps.
* Both dense and sparse input data are supported.
* Large maps of several hundred thousand neurons are feasible.
* Integration with Databionic ESOM Tools.

For more information, refer to the following paper:

Peter Wittek (2013). Somoclu: An Efficient Distributed Library for Self-Organizing Maps. [arXiv:1305.1422](http://arxiv.org/abs/1305.1422).


Usage
==
Somoclu takes a plain text input file -- either dense or sparse data. Example files are included.

    $ [mpirun -np NPROC] somoclu [OPTIONs] INPUT_FILE OUTPUT_PREFIX

Arguments:

    -c FILENAME              Specify an initial codebook for the map.
    -e NUMBER                Maximum number of epochs
    -k NUMBER                Kernel type
                                0: Dense CPU
                                1: Dense GPU
                                2: Sparse CPU
    -m NUMBER                Map type
                                0: Planar
                                1: Toroid
    -r NUMBER                Initial radius (default: half the number of columns)
    -s NUMBER             Save interim files (default: 0):\n" \
                                0: Do not save interim files\n" \
                                1: Save U-matrix only\n" \
                                2: Also save codebook and best matching     -x, --columns NUMBER     Number of columns in map (size of SOM in direction x)
    -y, --rows    NUMBER     Number of rows in map (size of SOM in direction y)

Examples:

    $ somoclu data/rgbs.txt data/rgbs
    $ mpirun -np 4 somoclu -k 0 --rows 20 --columns 20 data/rgbs.txt data/rgbs

Input File Formats
==
One sparse and two dense data formats are supported. All of them are plain text files. The entries can be separated by any white-space character. One row represents one data instance across all formats. Comment lines starting with a hash mark are ignored.

The sparse format follows the [libsvm](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) guidelines. The first feature is zero-indexed. For instance, the vector [ 1.2 0 0 3.4] is represented as the following line in the file:
0:1.2 3:3.4. The file is parsed twice: once to get the number of instances and features, and the second time to read the data in the individual threads.

The basic dense format includes the coordinates of the data vectors, separated by a white-space. Just like the sparse format, this file is parsed twice to get the basic dimensions right. 

The .lrn file of [Databionic ESOM Tools](http://databionic-esom.sourceforge.net/) is also accepted and it is parsed only once. The format is described as follows:

% n

% m

% s1		s2			..		sm

% var_name1	var_name2		..		var_namem	

x11		x12			..		x1m

x21		x22			..		x2m

.		.			.		.

.		.			.		.

xn1		xn2			..		xnm

Here n is the number of rows in the file, that is, the number of data instances. Parameter m defines the number of columns in the file. The next row defines the column mask: the value 1 for a column means the column should be used in the training. Note that the first column in this format is always a unique key, so this should have the value 9 in the column mask. The row with the variable names is ignore by Somoclu. The elements of the matrix follow -- from here, the file is identical to the basic dense format, with the addition of the first column as the unique key.

If the input file is sparse, but a dense kernel is invoked, Somoclu will execute and results will be incorrect. Invoking a sparse kernel on a dense input file is likely to lead to a segmentation fault.

Visualisation
==
The primary purpose of generating a map is visualisation. Somoclu does not come with its own functions for visualisation, since there are numerous generic tools that are capable of plotting high-quality figures. A simplot gnuplot script is provided with the source code as plot_som.gp. This script takes a U-matrix (umat.umx), and outputs a plot (som.png).

The output formats of the U-matrix and the codebook are compatible with [Databionic ESOM Tools](http://databionic-esom.sourceforge.net/) for more advanced visualisation.

Dependencies
==
The only dependency is GCC, albeit other compiler chains might also work.

Distributed systems and single-machine multicore execution is supported through MPI. The package was tested with OpenMPI, versions 1.3.2 and 1.6.5 were tested. It should also work with other MPI flavours. 

CUDA support is optional. CUDA versions 4.1, 5.0 and 5.5 are known to work.

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

    --without-mpi           Disregard any MPI installation found.
    --with-mpi=MPIROOT      Use MPI root directory.
    --with-mpi-compilers=DIR or --with-mpi-compilers=yes
                              use MPI compiler (mpicxx) found in directory DIR, or
                              in your PATH if =yes
    --with-mpi-libs="LIBS"  MPI libraries [default "-lmpi"]
    --with-mpi-incdir=DIR   MPI include directory [default MPIROOT/include]
    --with-mpi-libdir=DIR   MPI library directory [default MPIROOT/lib]

The above flags allow the identification of the correct MPI library the user wishes to use. The flags are especially useful if MPI is installed in a non-standard location, or when multiple MPI libraries are available.

    --with-cuda=/path/to/cuda           Set path for CUDA

Somoclu looks for CUDA in /usr/local/cuda. If your installation is not there, then specify the path with this parameter. If you do not want CUDA enabled, set the parameter to ```--without-cuda```.

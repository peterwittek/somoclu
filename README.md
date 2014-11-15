Somoclu
===
Somoclu is a massively parallel implementation of self-organizing maps. It exploits multicore CPUs, it is able to rely on MPI for distributing the workload in a cluster, and it can be accelerated by CUDA. A sparse kernel is also included, which is useful for training maps on vector spaces generated in text mining processes. The topology of the grid of neurons is rectangular.

Key features:

* Fast execution by parallelization: OpenMP, MPI, and CUDA are supported.
* Planar and toroid maps.
* Both dense and sparse input data are supported.
* Large maps of several hundred thousand neurons are feasible.
* Integration with Databionic ESOM Tools.
* Python, R, and MATLAB interfaces for the dense CPU and GPU kernels.

For more information, refer to the following paper:

Peter Wittek (2013). Somoclu: An Efficient Distributed Library for Self-Organizing Maps. [arXiv:1305.1422](http://arxiv.org/abs/1305.1422).

Usage
===
Basic Use
---
Somoclu takes a plain text input file -- either dense or sparse data. Example files are included.

    $ [mpirun -np NPROC] somoclu [OPTIONs] INPUT_FILE OUTPUT_PREFIX

Arguments:

    -c FILENAME              Specify an initial codebook for the map.
    -e NUMBER                Maximum number of epochs
    -k NUMBER                Kernel type
                                0: Dense CPU
                                1: Dense GPU
                                2: Sparse CPU
    -m TYPE                  Map type: planar or toroid (default: planar)
    -t STRATEGY              Radius cooling strategy: linear or exponential (default: linear)
    -r NUMBER                Start radius (default: half of the map in direction min(x,y))
    -R NUMBER                End radius (default: 1)
    -T STRATEGY              Learning rate cooling strategy: linear or exponential (default: linear)
    -l NUMBER                Starting learning rate (default: 0.1)
    -L NUMBER                Finishing learning rate (default: 0.01)
    -s NUMBER                Save interim files (default: 0):
                                0: Do not save interim files
                                1: Save U-matrix only
                                2: Also save codebook and best matching
    -x, --columns NUMBER     Number of columns in map (size of SOM in direction x)
    -y, --rows    NUMBER     Number of rows in map (size of SOM in direction y)

Examples:

    $ somoclu data/rgbs.txt data/rgbs
    $ mpirun -np 4 somoclu -k 0 --rows 20 --columns 20 data/rgbs.txt data/rgbs

Efficient Parallel Execution
---
The CPU kernels use OpenMP to load multicore processors. On a single node, this is more efficient than launching tasks with MPI to match the number of cores. The MPI tasks replicated the codebook, which is especially inefficient for large maps. 

For instance, given a single node with eight cores, the following execution will use 1/8th of the memory, and will run 10-20% faster:

    $ somoclu -x 200 -y 200 data/rgbs.txt data/rgbs

Or, equivalently:

    $ OMP_NUM_THREADS=8 somoclu -x 200 -y 200 data/rgbs.txt data/rgbs

Avoid the following on a single node:

    $ OMP_NUM_THREADS=1 mpirun -np 8 somoclu -x 200 -y 200 data/rgbs.txt data/rgbs

The same caveats apply for the sparse CPU kernel.

Visualisation
---
The primary purpose of generating a map is visualisation. Somoclu does not come with its own functions for visualisation, since there are numerous generic tools that are capable of plotting high-quality figures. 

The output formats of the U-matrix and the codebook are compatible with [Databionic ESOM Tools](http://databionic-esom.sourceforge.net/) for more advanced visualisation.


Input File Formats
===
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

Interfaces
===
Python, R, and MATLAB interfaces are available for the dense CPU kernel. MPI, CUDA, and the sparse kernel are not support through the interfaces. The connection to the C++ library is seamless, data structures are not duplicated. For respective examples, see the folders in src. All versions require GCC to compile the code.

The Python version is also available in Pypi. You can install it with

    $ sudo pip install somoclu
    
The R version is available on CRAN. You can install it with
    
    install.packages("Rsomoclu")

For using the MATLAB toolbox, define the location of the mex compiler in MEX_BIN. Then invoke makeMex.sh in the src/MATLAB folder.

For more information on the respective interfaces, refer to the subfolders in src.

Compilation & Installation
===
The only dependency is GCC, although other compiler chains might also work.

Distributed systems and single-machine multicore execution is supported through MPI. The package was tested with OpenMPI. It should also work with other MPI flavours. 

CUDA support is optional.

Linux or Mac OS X
---
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

Windows
---
Use the `somoclu.sln` under src/Windows/somoclu as an example visual studio 2013 solution. Modify the CUDA version or VC compiler version according to your needs. 

The default solution enables all of OpenMP, MPI, and CUDA. The default MPI installation path is `C:\Program Files\Microsoft MPI`, modify the settings if yours is in a different path. The configuration default CUDA version is 6.5.  Disable MPI by removing `HAVE_MPI` macro in the project properties (`Properties -> Configuration Properties -> C/C++ -> Preprocessor`). Disable CUDA by removing `CUDA` macro in the solution properties.

The usage is identical to the linux version through command line (see the relevant section). 

When using the somoclu Python interface on Windows, if you encounter errors like: 

    ImportError: DLL load failed: The specified module could not be found
    
You may need to find the right version (32/64bit) of `vcomp90.dll, msvcp90.dll, msvcr90.dll` and put to `C:\Windows\System32` or `C:\Windows\SysWOW64`.

Instructions on building python extension with CUDA support on windows is at [here](https://github.com/peterwittek/somoclu/tree/master/src/Python)

Known Issues
===
The MATLAB CUDA interface crashes with unknown reasons.

The maps generated by the GPU and the CPU kernels are likely to be different. For computational efficiency, Somoclu uses single-precision floats. This occasionally results in identical distances between a data instance and the neurons. The CPU version will pick the best matching unit with the lowest coordinate values. Such sequentiality cannot be guaranteed in the reduction kernel of the GPU variant. This is not a bug, but it is better to be aware of it.

Acknowledgment
===
This work was supported by the European Commission Seventh Framework Programme under Grant Agreement Number FP7-601138 PERICLES and by the AWS in Education Machine Learning Grant award.

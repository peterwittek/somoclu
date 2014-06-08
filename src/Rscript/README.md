Instructions for building the R package with CUDA support on Linux
==
First after unzip the tarball, you can cd to the directory that contains the script to help you build the package with CUDA support:

	$ cd src/Rscript

Then export the variable `CUDA_HOME` which indicates your installation:

	$ export CUDA_HOME=/opt/cuda

or like:

	$ export CUDA_HOME=/usr/local/cuda

then run the script to build:

	$ ./makeR-CUDA.sh
	
If there isn' t error. You will then find the package `Rsomoclu_VERION.tar.gz` at the upper src folder. Install it with:

	$ R CMD INSTALL Rsomoclu_VERION.tar.gz

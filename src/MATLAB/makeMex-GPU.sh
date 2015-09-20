#!/bin/sh
if [ -z "$MATLAB_ROOT" ]
    then MATLAB_ROOT="/usr/local/MATLAB/R2013a/"
fi
MEX_BIN="$MATLAB_ROOT/bin/mex"
if [ -z "$CUDA_LIB" ]
    then CUDA_LIB="/opt/cuda/lib64"
fi
cp -f ../denseGpuKernels.cu.co ../denseGpuKernels.cu.o
$MEX_BIN -I"$MATLAB_ROOT/toolbox/distcomp/gpu/extern/include/" -I../ MexSomoclu.cpp -DCUDA ../denseCpuKernels.o ../io.o ../sparseCpuKernels.o ../training.o ../mapDistanceFunctions.o ../uMatrix.o ../denseGpuKernels.cu.o -lgomp -L$CUDA_LIB -lcudart -lcublas -lnvblas -lmwgpu

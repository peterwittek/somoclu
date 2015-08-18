#!/bin/sh
if [ -z "$MEX_BIN" ]
    then MEX_BIN="/usr/local/MATLAB/R2013a/bin/mex"
fi
CUDA_LIB="/opt/cuda/lib64"
$MEX_BIN -I../ MexSomoclu.cpp -DCUDA ../denseCpuKernels.o ../io.o ../sparseCpuKernels.o ../training.o ../mapDistanceFunctions.o ../uMatrix.o ../denseGpuKernels.cu.co -lgomp -L$CUDA_LIB -lcudart -lcublas -lnvblas

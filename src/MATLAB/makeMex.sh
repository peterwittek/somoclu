#!/bin/sh
if [ -z "$MATLAB_ROOT" ]
     then MATLAB_ROOT="/usr/local/MATLAB/R2013a/"
fi
MEX_BIN="$MATLAB_ROOT/bin/mex"
$MEX_BIN -I../ MexSomoclu.cpp ../denseCpuKernels.o ../io.o ../sparseCpuKernels.o ../training.o ../mapDistanceFunctions.o ../uMatrix.o -lgomp

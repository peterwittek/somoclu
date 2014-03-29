#!/bin/sh
if [ -z "$MEX_BIN" ]
    then MEX_BIN="/usr/local/MATLAB/R2013a/bin/mex"
fi
$MEX_BIN -I../ MexSomoclu.cpp ../somocluWrap.cpp ../somoclu.o ../denseCpuKernels.o ../io.o ../sparseCpuKernels.o ../training.o ../mapDistanceFunctions.o -lgomp

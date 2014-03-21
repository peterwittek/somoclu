#!/bin/sh
MEX_BIN="/home/gsc/data/MATLAB/R2013a/bin/mex"
$MEX_BIN MexSomoclu.cpp somocluWrap.cpp somoclu.o denseCpuKernels.o io.o sparseCpuKernels.o training.o mapDistanceFunctions.o -lgomp

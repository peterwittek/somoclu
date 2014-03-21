#!/bin/sh
# swig -c++ -r -module somocluR somoclu_R.i
rm Rsomoclu.so
export PKG_CXXFLAGS=`Rscript -e "Rcpp:::CxxFlags()"`
R CMD SHLIB -lgomp -o Rsomoclu.so Rsomoclu.cpp somocluWrap.cpp somoclu.o denseCpuKernels.o io.o sparseCpuKernels.o training.o mapDistanceFunctions.o

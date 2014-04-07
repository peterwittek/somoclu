#!/bin/sh
# rm Rsomoclu.so
# export PKG_CXXFLAGS=`Rscript -e "Rcpp:::CxxFlags()"`
# R CMD SHLIB -lgomp -o Rsomoclu.so Rsomoclu.cpp somocluWrap.cpp somoclu.o denseCpuKernels.o io.o sparseCpuKernels.o training.o mapDistanceFunctions.o
#rm -rf ./build
cd ../R
mkdir data
cp ../../src/*.h ./src/
cp ../../src/mapDistanceFunctions.cpp ./src/
cp ../../src/trainOneEpoch.cpp ./src/
cp ../../src/uMatrix.cpp ./src/
cp ../../data/rgbs.txt ./data/
gzip ./data/rgbs.txt

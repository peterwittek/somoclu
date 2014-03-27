#!/bin/sh
# rm Rsomoclu.so
# export PKG_CXXFLAGS=`Rscript -e "Rcpp:::CxxFlags()"`
# R CMD SHLIB -lgomp -o Rsomoclu.so Rsomoclu.cpp somocluWrap.cpp somoclu.o denseCpuKernels.o io.o sparseCpuKernels.o training.o mapDistanceFunctions.o
#rm -rf ./build
cp ../../autogen.sh ./src
cp -r ../../m4 ./src
cp ../../configure.ac ./src
cp ../../Makefile.in ./src
cp ../../src/*.h ./src/src
cp ../../src/*.cpp ./src/src
cp ../../src/*.cu ./src/src
cp ../../src/Makefile.in ./src/src
cp ../../data/rgbs.txt tests/
cp ../../data/rgbs.txt data/

#!/bin/sh
rm -rf ../R-CUDA
cp -r ../R ../R-CUDA
cp configure ../R-CUDA
cp tests/R_interface_test_CUDA.R ../R-CUDA/tests
cd ../R-CUDA
mkdir data
cp -r ../../m4/ ./src/
cp ../../autogen.sh ./src/
cp ../../configure.ac ./src/
cp ../../Makefile.in ./src/
mkdir ./src/src
cp ../../src/*.cpp ./src/src
cp ../../src/*.h ./src/src
cp ../../src/*.cu ./src/src
cp ../../src/Makefile.in ./src/src
cp ../../data/rgbs.txt ./data/
gzip -f ./data/rgbs.txt
cd src
mv Rsomoclu.cpp src
rm *.cpp
rm *.h
rm Makevars*
#cd ../
#R CMD build R-CUDA

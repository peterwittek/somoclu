#!/bin/sh
cd ../R
mkdir data
cp ../../src/*.h ./src/
cp ../../src/mapDistanceFunctions.cpp ./src/
cp ../../src/trainOneEpoch.cpp ./src/
cp ../../src/uMatrix.cpp ./src/
cp ../../src/denseCpuKernels.cpp ./src/
cp ../../src/sparseCpuKernels.cpp ./src/
cp ../../src/somocluWrap.cpp ./src/
cp ../../data/rgbs.txt ./data/
gzip ./data/rgbs.txt 
cd ../
R CMD build R

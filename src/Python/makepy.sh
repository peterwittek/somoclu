#!/bin/sh
# rm _somoclu.so
# rm somoclu.pyc
# swig -c++ -python -module somoclu somoclu.i   
# ./setup.py clean
# ./setup.py build_ext --inplace
rm -rf ./src
rm -rf ./dist
rm -rf ./build
mkdir src
mkdir src/src
mkdir src/src/Windows
cp ../../autogen.sh ./src
cp -r ../../m4 ./src
cp ../../configure.ac ./src
cp ../../Makefile.in ./src
cp ../../src/*.h ./src/src
cp ../../src/*.cpp ./src/src
cp ../../src/*.cu ./src/src
cp ../../src/Windows/*.h ./src/src/Windows
cp ../../src/Windows/*.c ./src/src/Windows
cp ../../src/Makefile.in ./src/src
cp ../../data/rgbs.txt tests/
python2 setup.py sdist
#sudo python2 setup.py install

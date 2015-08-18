#!/bin/sh
rm -rf ./somoclu/src
rm -rf ./somoclu/*so
rm -rf ./somoclu/*c
rm -rf ./somoclu/*cxx
rm -rf ./dist
rm -rf ./build
mkdir somoclu/src
mkdir somoclu/src/Windows
cp ../../src/*.h ./somoclu/src/
cp ../../src/*.cpp ./somoclu/src/
cp ../../src/*.cu ./somoclu/src/
cp ../../data/rgbs.txt tests/
swig -c++ -python somoclu/somoclu.i
python2 setup.py build
cp  build/lib.linux-x86_64-2.7/_somoclu_wrap.so somoclu/
#sudo python2 setup.py install

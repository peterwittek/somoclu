#!/bin/sh
rm _somoclu.so
rm somoclu.pyc
swig -c++ -python -module somoclu somoclu.i
#mpic++ -O2 -fPIC -c somoclu_wrap.cxx -I/usr/include/python2.7 
#mpic++ -lgomp -shared *.o -o _somoclu.so      
./setup.py build_ext --inplace

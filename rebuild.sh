#!/bin/bash
./autogen.sh
./configure --with-mpi=/usr/lib/openmpi
make clean
make
cd src
rm _somoclu.so
./makepy.sh

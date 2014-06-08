#!/bin/bash
cd ../R/src/
shopt -s extglob
rm !(Makevars|Makevars.win|Rsomoclu.cpp)

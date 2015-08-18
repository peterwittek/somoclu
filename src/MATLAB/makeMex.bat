mex -D_OPENMP COMPFLAGS="/openmp $COMPFLAGS" -I../ MexSomoclu.cpp ../denseCpuKernels.cpp ../io.cpp ../sparseCpuKernels.cpp ../training.cpp ../mapDistanceFunctions.cpp ../uMatrix.cpp

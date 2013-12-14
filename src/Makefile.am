somoclu_SOURCES=sparseCpuKernels.cpp somoclu.cpp io.cpp denseCpuKernels.cpp mapDistanceFunctions.cpp training.cpp

if HAVE_CUDA
somoclu_SOURCES += denseGpuKernels.cu
EXTRA_DIST = denseGpuKernels.cu
somoclu_LDADD = $(CUDA_LIBS)
somoclu_LDFLAGS = $(CUDA_LDFLAGS)
endif

.cu.o:
	$(NVCC) $(DEFS) $(CUDA_CFLAGS) ${MPI_INC} -I$(srcdir) -I.. -o $@ -c $(srcdir)/$<

bin_PROGRAMS=somoclu

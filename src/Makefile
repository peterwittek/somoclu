# src/Makefile.  Generated from Makefile.in by configure.

# Package-related substitution variables
package	= Somoclu
version	= 1.0
tarname	= somoclu
distdir	= $(tarname)-$(version)

# Prefix-related substitution variables
prefix	 = /usr/local
exec_prefix    = ${prefix}
bindir	 = ${exec_prefix}/bin

# Tool-related substitution variables
CXX		         = mpic++
CXXFLAGS       = -O3 -fopenmp 
LIBS	         = 
DEFS           = -DHAVE_CONFIG_H
INSTALL	       = /usr/bin/install -c
INSTALL_DATA   = ${INSTALL} -m 644
INSTALL_PROGRAM= ${INSTALL}
INSTALL_SCRIPT = ${INSTALL}
CUDA_CFLAGS    = -I/usr/local/cuda/include -use_fast_math -gencode arch=compute_10,code=sm_10 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -Xcompiler "-O3 -fopenmp"
CUDA_LIBS      = -lcudart -lcublas
CUDA_LDFLAGS   = -L/usr/local/cuda/lib64
NVCC       	   = /usr/local/cuda/bin/nvcc
MPI_INC        = 
MPI_LIBDIR     = 
MPI_LIBS       = 

# VPATH-related substitution variables
srcdir	 = .


OBJS=sparseCpuKernels.o somoclu.o io.o denseCpuKernels.o training.o

ifdef CUDA_LIBS
	OBJS+=denseGpuKernels.cu.co
endif

all: somoclu

somoclu: $(OBJS)
	$(CXX) $(DEFS) $(CXXFLAGS) $(CUDA_LDFLAGS) ${MPI_LIBDIR} -o $@ $^ $(LIBS) $(CUDA_LIBS) ${MPI_LIBS}

%.o: %.cpp
		$(CXX) $(DEFS) $(CXXFLAGS) ${MPI_INC} -I$(srcdir) -I.. -o $@ -c $(srcdir)/$<

%.cu.co: %.cu
		$(NVCC) $(DEFS) $(CUDA_CFLAGS) ${MPI_INC} -I$(srcdir) -I.. -o $@ -c $(srcdir)/$<

clean:
	-rm somoclu $(OBJS) 1>/dev/null

install:
	$(INSTALL) -d $(DESTDIR)$(bindir)/somoclu
	$(INSTALL_PROGRAM) -m 0755 somoclu \
	 $(DESTDIR)$(bindir)/somoclu

uninstall:
	-rm $(DESTDIR)$(bindir)/somoclu &>/dev/null

Makefile: Makefile.in ../config.status
	cd .. && ./config.status $@

../config.status: ../configure
	cd .. && ./config.status --recheck

.PHONY: all clean install uninstall

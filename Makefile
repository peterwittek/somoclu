CUDA_DIR=/usr/local/cuda
MPI_DIR=/usr

CXX=$(MPI_DIR)/bin/mpicxx
CXXFLAGS = -g -Wall

NVCC=$(CUDA_DIR)/bin/nvcc
NVCCFLAGS=-arch sm_20 -g --compiler-options -rdynamic

INCLUDES=-I$(MPI_DIR)/include -I$(CUDA_DIR)/include
LIBS=-L $(CUDA_DIR)/lib64 -lcublas -lcudart

OBJDIR=obj
OBJS := $(addprefix $(OBJDIR)/,sparseCpuKernels.o somoclu.o io.o denseCpuKernels.o denseGpuKernels.cu.co training.o)

TARGETS=somoclu

all: $(TARGETS)

somoclu: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<

$(OBJDIR)/%.cu.co: %.cu
		$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c $<
	     
$(OBJS): | $(OBJDIR)
     
$(OBJDIR):
		mkdir $(OBJDIR)

	
clean:
		rm -f $(OBJS) $(TARGETS)

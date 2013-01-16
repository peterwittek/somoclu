CUDA_DIR=/usr/local/cuda
MPI_DIR=/usr

CXX=$(MPI_DIR)/bin/mpicxx
CXXFLAGS = -g -Wall

NVCC=$(CUDA_DIR)/bin/nvcc
NVCCFLAGS=-arch sm_11 -g --compiler-options -rdynamic

INCLUDES=-I$(MPI_DIR)/include -I$(CUDA_DIR)/include
LIBS=-L $(CUDA_DIR)/lib64 -lcublas

OBJDIR=obj
OBJS := $(addprefix $(OBJDIR)/,io.o denseTraining.o denseCpuKernels.o denseGpuKernels.cu.co somoclu.o)

TARGETS=somoclu

all: $(TARGETS)

somoclu: $(OBJS)
	$(CXX) $(CXXFLAGS) $(LIBS) -o $@ $^

$(OBJDIR)/%.o: %.cpp
		$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<

$(OBJDIR)/%.cu.co: %.cu
		$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c $<
	     
$(OBJS): | $(OBJDIR)
     
$(OBJDIR):
		mkdir $(OBJDIR)

	
clean:
		rm -f $(OBJS) $(TARGETS)

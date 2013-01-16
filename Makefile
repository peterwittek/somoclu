CUDA_DIR=/usr/local/cuda
MPI_DIR=/usr

CXX=$(MPI_DIR)/bin/mpicxx
CXXFLAGS = -g -Wall

NVCC=$(CUDA_DIR)/bin/nvcc
NVCCFLAGS=-arch sm_11 -g --compiler-options -rdynamic

INCLUDE=-I$(MPI_DIR)/include -I$(CUDA_DIR)/include
LIBS=-L $(CUDA_DIR)/lib64 -lcublas

all: mrsom

mrsom: obj/mrsom.o obj/mrsom.cu.o obj/io.o obj/train.o
	$(CXX) obj/mrsom.o obj/mrsom.cu.o obj/io.o obj/train.o $(LIBS) -o mrsom

obj/mrsom.o:
	mkdir -p obj
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c mrsom.cpp -o obj/mrsom.o

obj/io.o:
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c io.cpp -o obj/io.o

obj/train.o:
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c train.cpp -o obj/train.o

obj/mrsom.cu.o: 
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) mrsom.cu -c -o obj/mrsom.cu.o
	
clean:
	rm -rf obj mrsom

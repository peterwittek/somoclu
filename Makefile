CUDA_DIR=/usr/local/cuda
MPI_DIR=/usr
MRMPI_DIR=../mrmpi

CXX=$(MPI_DIR)/bin/mpicxx
CXXFLAGS = -g -Wall

NVCC=$(CUDA_DIR)/bin/nvcc
NVCCFLAGS=-arch sm_11 -I $(CUDA_DIR)/include -g --compiler-options -rdynamic

INCLUDE=-I $(MRMPI_DIR)
LIBS=$(MRMPI_DIR)/libmrmpi.a -L $(CUDA_DIR)/lib64 -lcublas

all: mrsom

mrsom: obj/mrsom.o obj/mrsom.cu.o obj/io.o
	$(CXX) obj/mrsom.o obj/mrsom.cu.o obj/io.o $(LIBS) -o mrsom

obj/mrsom.o:
	mkdir -p obj
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c mrsom.cpp -o obj/mrsom.o

obj/io.o:
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c io.cpp -o obj/io.o

obj/mrsom.cu.o: 
	$(NVCC) $(NVCCFLAGS) mrsom.cu -c -o obj/mrsom.cu.o
	
clean:
	rm -rf obj mrsom

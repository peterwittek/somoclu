/**
 * Self-Organizing Maps on a cluster
 *  Copyright (C) 2013 Peter Wittek
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#include <cmath> 
#include <cstdlib>
#include <mpi.h>
#include <unistd.h>

#include "somoclu.h"
 
using namespace std;

/// For syncronized timing
#ifndef MPI_WTIME_IS_GLOBAL
#define MPI_WTIME_IS_GLOBAL 1
#endif

// Default parameters
#define N_EPOCH 10
#define N_SOM_X 50 
#define N_SOM_Y 50
#define KERNEL_TYPE 0
#define MAP_TYPE 0
#define ENABLE_SNAPSHOTS false

void processCommandLine(int argc, char** argv, char* inFileName, 
                        char* outPrefix, unsigned int *nEpoch, 
                        unsigned int *nSomX, unsigned int *nSomY, 
                        unsigned int *kernelType, unsigned int *mapType,
                        bool *enableSnapshots);

/* -------------------------------------------------------------------------- */
int main(int argc, char** argv)
/* -------------------------------------------------------------------------- */
{    
  ///
  /// MPI init
  ///
  MPI_Init(&argc, &argv);
  int rank, nProcs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
  MPI_Barrier(MPI_COMM_WORLD); 
  
  unsigned int nEpoch = 0;
  unsigned int nSomX = 0;
  unsigned int nSomY = 0;
  unsigned int kernelType = 0;
  unsigned int mapType = 0;
  bool enableSnapshots = false;
  char *inFileName = new char[255];
  char *outPrefix = new char[255];

  if (rank==0) {
      processCommandLine(argc, argv, inFileName, outPrefix, 
                         &nEpoch, &nSomX, &nSomY, 
                         &kernelType, &mapType, &enableSnapshots);
#ifndef CUDA
      if (kernelType == DENSE_GPU){
          cerr << "Somoclu was compile without GPU support!\n";
          MPI_Abort(MPI_COMM_WORLD, 1);          
      }
#endif
  }
  MPI_Bcast(&nEpoch, 1, MPI_INT, 0, MPI_COMM_WORLD);  
  MPI_Bcast(&nSomX, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nSomY, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&kernelType, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&mapType, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(inFileName, 255, MPI_CHAR, 0, MPI_COMM_WORLD);
  
  double profile_time = MPI_Wtime();

  float * dataRoot = NULL;
  unsigned int nDimensions = 0;    
  unsigned int nVectors = 0;
  if(rank == 0 ){
      if (kernelType == DENSE_CPU || kernelType == DENSE_GPU) {
        dataRoot = readMatrix(inFileName, nVectors, nDimensions);
      } else {
        readSparseMatrixDimensions(inFileName, nVectors, nDimensions);
      }
  }
  MPI_Barrier(MPI_COMM_WORLD); 

  MPI_Bcast(&nVectors, 1, MPI_INT, 0, MPI_COMM_WORLD);
  unsigned int nVectorsPerRank = ceil(nVectors / (1.0*nProcs));    
  MPI_Bcast(&nDimensions, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Allocate a buffer on each node
  float* data;
  svm_node **sparseData;
  if (kernelType == DENSE_CPU || kernelType == DENSE_GPU) {
    data = new float[nVectorsPerRank*nDimensions];
    // Dispatch a portion of the input data to each node
    MPI_Scatter(dataRoot, nVectorsPerRank*nDimensions, MPI_FLOAT,
              data, nVectorsPerRank*nDimensions, MPI_FLOAT,
              0, MPI_COMM_WORLD);
  } else {
    int currentRankProcessed = 0;
    while (currentRankProcessed < nProcs) {
      if (rank == currentRankProcessed) {
        sparseData=readSparseMatrixChunk(inFileName, nVectors, nVectorsPerRank, 
                              rank*nVectorsPerRank);
      }
      currentRankProcessed++;
      MPI_Barrier(MPI_COMM_WORLD); 
    }
  }
  
  if(rank == 0){
      // No need for root data any more
      if (kernelType == DENSE_CPU || kernelType == DENSE_GPU) {
        delete [] dataRoot;
      }
      cout << "nVectors: " << nVectors << " ";
      cout << "nVectorsPerRank: " << nVectorsPerRank << " ";
      cout << "nDimensions: " << nDimensions << " ";
      cout << endl;
  }

#ifdef CUDA  
  if (kernelType == DENSE_GPU){
    setDevice(rank, nProcs);
    initializeGpu(data, nVectorsPerRank, nDimensions, nSomX, nSomY);
  }
#endif
  
  MPI_Barrier(MPI_COMM_WORLD); 
  
  // TRAINING
  train(rank, data, sparseData, nSomX, nSomY, 
        nDimensions, nVectors, nVectorsPerRank,
        nEpoch, outPrefix, enableSnapshots, kernelType, mapType);
  
  if (kernelType == DENSE_CPU || kernelType == DENSE_GPU) {
    delete [] data;
  } else {
    delete [] sparseData;
  }
  
  profile_time = MPI_Wtime() - profile_time;
  if (rank == 0) {
    cerr << "Total Execution Time: " << profile_time << endl;
  }

#ifdef CUDA  
  if (kernelType == DENSE_GPU){
    freeGpu();
  }
#endif  
  MPI_Finalize();
  return 0;
}

void printUsage() {
    cout << "Usage:\n" \
              "     [mpirun -np NPROC] somoclu [OPTIONs] INPUT_FILE OUTPUT_PREFIX\n" \
              "Arguments:\n" \
              "     -e NUMBER     Maximum number of epochs (default: " << N_EPOCH << ")\n" \
              "     -k NUMBER     Kernel type (default: " << KERNEL_TYPE << "): \n" \
              "                      0: Dense CPU\n" \
              "                      1: Dense GPU\n" \
              "                      2: Sparse CPU\n" \
              "     -m NUMBER     Map type (default: " << MAP_TYPE << "): \n" \
              "                      0: Planar\n" \
              "                      1: Toroid\n" \
              "     -s            Enable snapshots of U-matrix (default: false)\n" \
              "     -x NUMBER     Dimension of SOM in direction x (default: " << N_SOM_X << ")\n" \
              "     -y NUMBER     Dimension of SOM in direction y (default: " << N_SOM_Y << ")\n" \
              "Examples:\n" \
              "     somoclu data/rgbs.txt data/rgbs\n"
              "     mpirun -np 4 somoclu -k 0 -x 20 -y 20 data/rgbs.txt data/rgbs\n";
}

void processCommandLine(int argc, char** argv, char* inFileName, 
                        char* outPrefix, unsigned int *nEpoch, 
                        unsigned int *nSomX, unsigned int *nSomY, 
                        unsigned int *kernelType, unsigned int *mapType, bool *enableSnapshots) {
  
    // Setting default values
    *nEpoch = N_EPOCH;
    *nSomX = N_SOM_X;
    *nSomY = N_SOM_Y;
    *kernelType = KERNEL_TYPE;
    *enableSnapshots = ENABLE_SNAPSHOTS;
    *mapType = MAP_TYPE;
    
    int c;
    extern int optind, optopt;
    while ((c = getopt (argc, argv, "hsx:y:e:k:m:")) != -1) {
        switch (c) {
        case 'e':
            *nEpoch = atoi(optarg);
            if (*nEpoch<=0) {
                fprintf (stderr, "The argument of option -e should be a positive integer.\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            break;
        case 'h':
            printUsage();
            MPI_Abort(MPI_COMM_WORLD, 0);
            break;
        case 'k':
            *kernelType = atoi(optarg);
            if (*kernelType<DENSE_CPU||*kernelType>SPARSE_CPU) {
                fprintf (stderr, "The argument of option -k should be a valid kernel.\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            break;
        case 'm':
            *mapType = atoi(optarg);
            if (*mapType<PLANAR||*mapType>TOROID) {
                fprintf (stderr, "The argument of option -m should be a valid map type.\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            break;            
        case 's':
              *enableSnapshots = true;
              break;
        case 'x':
            *nSomX = atoi(optarg);
            if (*nSomX<=0) {
                fprintf (stderr, "The argument of option -x should be a positive integer.\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            break;
        case 'y':
            *nSomY = atoi(optarg);
            if (*nSomY<=0) {
                fprintf (stderr, "The argument of option -y should be a positive integer.\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            break;
        case '?':
            if (optopt == 'e' || optopt == 'k' || optopt == 's' || 
                optopt == 'x'    || optopt == 'y') {
                fprintf (stderr, "Option -%c requires an argument.\n", optopt);
                printUsage();
                MPI_Abort(MPI_COMM_WORLD, 1);
            } else if (isprint (optopt)) {
                fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                printUsage();
                MPI_Abort(MPI_COMM_WORLD, 1);
            } else {
                fprintf (stderr, "Unknown option character `\\x%x'.\n",  optopt);
                printUsage();
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        default:
            abort ();
        }
    }
    if (argc-optind!=2) {
                fprintf(stderr, "Incorrect number of mandatory parameters");
                printUsage();
                MPI_Abort(MPI_COMM_WORLD, 1);
    }
    strcpy(inFileName, argv[optind++]);
    strcpy(outPrefix, argv[optind++]);
}

/** Shut down MPI cleanly if something goes wrong
 * @param err - error code to print
 */
void my_abort(int err)
{
  cout << "Test FAILED\n";
  MPI_Abort(MPI_COMM_WORLD, err);
}

/// EOF

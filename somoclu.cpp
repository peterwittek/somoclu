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

#include "somoclu.h"
 
using namespace std;

/// For syncronized timing
#ifndef MPI_WTIME_IS_GLOBAL
#define MPI_WTIME_IS_GLOBAL 1
#endif

/* -------------------------------------------------------------------------- */
int main(int argc, char** argv)
/* -------------------------------------------------------------------------- */
{    
  unsigned int nEpoch = 10;
  unsigned int nSomX = 10;
  unsigned int nSomY = 10;
  
  if (argc!=3){
    cout << "Usage: mrsom input_filename output_prefix\n";
    exit(1);
  }
          
  const char *inFileName = argv[1];
  const char *outPrefix = argv[2];

  ///
  /// MPI init
  ///
  MPI_Init(&argc, &argv);
  int MPI_myId, MPI_nProcs;
  MPI_Comm_rank(MPI_COMM_WORLD, &MPI_myId);
  MPI_Comm_size(MPI_COMM_WORLD, &MPI_nProcs);
  MPI_Barrier(MPI_COMM_WORLD); 
  
  double profile_time = MPI_Wtime();

  float * dataRoot = NULL;    
  unsigned int nDimensions = 0;    
  unsigned int nVectors = 0;           /// Total num of feature vectors 
  if(MPI_myId == 0){
      dataRoot = readMatrix(inFileName, nVectors, nDimensions);
  }
  MPI_Barrier(MPI_COMM_WORLD); 

  MPI_Bcast(&nVectors, 1, MPI_INT, 0, MPI_COMM_WORLD);
  unsigned int nVectorsPerRank = ceil(nVectors / (1.0*MPI_nProcs));    
  MPI_Bcast(&nDimensions, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Allocate a buffer on each node
  float* data = new float[nVectorsPerRank*nDimensions];
  
  // Dispatch a portion of the input data to each node
  MPI_Scatter(dataRoot, nVectorsPerRank*nDimensions, MPI_FLOAT,
              data, nVectorsPerRank*nDimensions, MPI_FLOAT,
              0, MPI_COMM_WORLD);
  
  if(MPI_myId == 0){
      // No need for root data any more
      delete [] dataRoot;
      cout << "nVectors: " << nVectors << " ";
      cout << "nVectorsPerRank: " << nVectorsPerRank << " ";
      cout << "nDimensions: " << nDimensions << " ";
      cout << endl;
  }
  
  setDevice(MPI_myId, MPI_nProcs);
  initializeGpu(data, nVectorsPerRank, nDimensions);

  // TRAINING
  train(MPI_myId, data, nSomX, nSomY, nDimensions, nVectors, nVectorsPerRank,
        nEpoch, outPrefix, true, 1);

  delete [] data;
  
  profile_time = MPI_Wtime() - profile_time;
  if (MPI_myId == 0) {
    cerr << "Total Execution Time: " << profile_time << endl;
  }
  
  shutdownGpu();
  MPI_Finalize();
  return 0;
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

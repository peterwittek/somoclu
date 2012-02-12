/**
 * GPU-Accelerated MapReduce-Based Self-Organizing Maps
 *  Copyright (C) 2012 Peter Wittek
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

/// MapReduce-MPI
#include "mapreduce.h"
#include "keyvalue.h"

#include "mrsom.h"
 
using namespace MAPREDUCE_NS;
using namespace std;

/// For syncronized timing
#ifndef MPI_WTIME_IS_GLOBAL
#define MPI_WTIME_IS_GLOBAL 1
#endif

/// For CODEBOOK
float *CODEBOOK;
float *NUMER2;
float *DENOM2;

void initializeCodebook(unsigned int seed, float *codebook, unsigned int SOM_X, 
                        unsigned int SOM_Y, unsigned int NDIMEN);
void mpireduce_train_batch(int itask, KeyValue* kv, void* ptr);

/// GLOBALS
unsigned int SOM_X=10;                 /// Width of SOM MAP 
unsigned int SOM_Y=10;                 /// Height of SOM MAP
unsigned int SOM_D=2;                 /// Dimension of SOM MAP, 2=2D
unsigned int NDIMEN;            /// Num of dimensionality
unsigned int NVECS;           /// Total num of feature vectors 
unsigned int NVECSPERRANK;    /// Num of feature vectors per task
float* DATA = NULL;        /// Feature data
float R;                

/* -------------------------------------------------------------------------- */
int main(int argc, char** argv)
/* -------------------------------------------------------------------------- */
{    
  unsigned int nEpoch = 10;
  
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
  if(MPI_myId == 0){
      dataRoot = readMatrix(inFileName, NVECS, NDIMEN);
  }
  MPI_Barrier(MPI_COMM_WORLD); 

  MPI_Bcast(&NVECS, 1, MPI_INT, 0, MPI_COMM_WORLD);
  NVECSPERRANK = ceil(NVECS / (1.0*MPI_nProcs));    
  MPI_Bcast(&NDIMEN, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Allocate a buffer on each node
  DATA = new float[NVECSPERRANK*NDIMEN];
  
  // Dispatch a portion of the input data to each node
  MPI_Scatter(dataRoot, NVECSPERRANK*NDIMEN, MPI_FLOAT,
              DATA, NVECSPERRANK*NDIMEN, MPI_FLOAT,
              0, MPI_COMM_WORLD);
  
  if(MPI_myId == 0){
      // No need for root data any more
      delete [] dataRoot;
      cout << "NVECS: " << NVECS << " ";
      cout << "NVECSPERRANK: " << NVECSPERRANK << " ";
      cout << "NDIMEN: " << NDIMEN << " ";
      cout << endl;
  }
  
  setDevice(MPI_myId, MPI_nProcs);
  initializeGpu(DATA, NVECSPERRANK, NDIMEN);
  
  /// 
  /// Codebook
  ///
  CODEBOOK= new float[SOM_Y*SOM_X*NDIMEN];
  if (MPI_myId == 0) {
    NUMER2 = new float[SOM_Y*SOM_X*NDIMEN];
    DENOM2 = new float[SOM_Y*SOM_X];
    initializeCodebook(0, CODEBOOK, SOM_X, SOM_Y, NDIMEN);
  }

  /// 
  /// MR-MPI
  ///
  MapReduce* mr = new MapReduce(MPI_COMM_WORLD);
  mr->verbosity = 0;
  mr->timer = 0;
  mr->mapstyle = 0;       /// chunk. NOTE: MPI_reduce() does not work with 
                          /// master/slave mode
  mr->memsize = 64;   /// page size
  mr->keyalign = 8;       /// default: key type = uint_64t = 8 bytes
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  ///
  /// Parameters for SOM
  ///
  float N = (float)nEpoch;       /// iterations
  float R0;
  R0 = SOM_X / 2.0f;              /// init radius for updating neighbors
  R = R0;
  unsigned int x = 0;             /// 0...N-1
      
  ///
  /// Training
  ///
  while (nEpoch && R > 1.0) {
    double epoch_time = MPI_Wtime();
    if (MPI_myId == 0) {
        R = R0 * exp(-10.0f * (x * x) / (N * N));
        x++;
        printf("BATCH-  epoch: %d   R: %.2f \n", (nEpoch - 1), R);
    }
      
    MPI_Bcast(&R, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(CODEBOOK, SOM_Y*SOM_X*NDIMEN, MPI_FLOAT, 
              0, MPI_COMM_WORLD);

    /// 1. Each task fills NUMER1 and DENOM1
    /// 2. MPI_reduce sums up each tasks NUMER1 and DENOM1 to the root's 
    ///    NUMER2 and DENOM2.
    /// 3. Update CODEBOOK using NUMER2 and DENOM2
    if (MPI_myId == 0) {
      for (unsigned int som_y = 0; som_y < SOM_Y; som_y++) { 
        for (unsigned int som_x = 0; som_x < SOM_X; som_x++) {
          DENOM2[som_y*SOM_X + som_x] = 0.0;
          for (unsigned int d = 0; d < NDIMEN; d++) {
            NUMER2[som_y*SOM_X*NDIMEN + som_x*NDIMEN + d] = 0.0;
          }
        }
      }
    }

    mr->map(MPI_nProcs, &mpireduce_train_batch, NULL);
      
    if (MPI_myId == 0) {
      for (unsigned int som_y = 0; som_y < SOM_Y; som_y++) { 
        for (unsigned int som_x = 0; som_x < SOM_X; som_x++) {
          float denom = DENOM2[som_y*SOM_X + som_x];
          for (unsigned int d = 0; d < NDIMEN; d++) {
              float newWeight = NUMER2[som_y*SOM_X*NDIMEN 
                                  + som_x*NDIMEN + d] / denom;
              if (newWeight > 0.0) 
                  CODEBOOK[som_y*SOM_X*NDIMEN+som_x*NDIMEN+d] = newWeight;
          }
        }
      }
    }
      ///
#if 0
    if (MPI_myId == 0) {
    printf("INFO: Saving interim U-Matrix...\n");
    char umatInterimFileName[50];
    sprintf(umatInterimFileName, "%s-umat-%03d.txt", outPrefix,  x);
    int ret = saveUMat(umatInterimFileName, CODEBOOK, SOM_X, SOM_Y, NDIMEN);
    }
#endif
    nEpoch--;
    epoch_time = MPI_Wtime() - epoch_time;
    if (MPI_myId == 0) {
      cerr << "Epoch Time: " << epoch_time << endl;
    }
  }  
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  ///
  /// Save SOM map and u-mat
  ///
  if (MPI_myId == 0) {
    ///
    /// Save U-mat
    ///
    char umatFileName[50];
    sprintf(umatFileName, "%s-umat.txt", outPrefix);
    int ret = saveUMat(umatFileName, CODEBOOK, SOM_X, SOM_Y, NDIMEN);
    if (ret < 0) 
        printf("    Failed to save u-matrix. !\n");
    else {
        printf("    Done (1) !\n");
    }        
     
    ///
    /// Save codebook
    ///
    char codebookInterimFileName[50];
    sprintf(codebookInterimFileName, "%s-codebook.txt", outPrefix);
    saveCodebook(codebookInterimFileName, CODEBOOK, SOM_X, SOM_Y, NDIMEN);
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  delete mr;
  delete [] DATA;
  delete [] CODEBOOK;
  
  profile_time = MPI_Wtime() - profile_time;
  if (MPI_myId == 0) {
    cerr << "Total Execution Time: " << profile_time << endl;
    delete [] NUMER2;
    delete [] DENOM2;
  }
  
  shutdownGpu();
  MPI_Finalize();
  return 0;
}

/** MR-MPI user-defined map function - batch training with MPI_reduce()
 * @param itask - number of work items
 * @param kv
 * @param ptr
 */
      
void mpireduce_train_batch(int itask, KeyValue* kv, void* ptr)
{           
  int p1[SOM_D];
  int p2[SOM_D];
  float *NUMER1;
  float *DENOM1;
  NUMER1 = new float[SOM_Y*SOM_X*NDIMEN];
  DENOM1 = new float[SOM_Y*SOM_X];
    
  for (unsigned int som_y = 0; som_y < SOM_Y; som_y++) {
      for (unsigned int som_x = 0; som_x < SOM_X; som_x++) {
          DENOM1[som_y*SOM_X + som_x] = 0.0;
          for (unsigned int d = 0; d < NDIMEN; d++) 
              NUMER1[som_y*SOM_X*NDIMEN + som_x*NDIMEN + d] = 0.0;
      }
  }
  int bmus[NVECSPERRANK*2];
  getBmusOnGpu(bmus, CODEBOOK, SOM_X, SOM_Y, NDIMEN, NVECSPERRANK);
  for (unsigned int n = 0; n < NVECSPERRANK; n++) {
    if (itask*NVECSPERRANK+n<NVECS){    
      /// get the best matching unit
      p1[0]=bmus[n*2];
      p1[1]=bmus[n*2+1];
          
      /// Accumulate denoms and numers
      for (unsigned int som_y = 0; som_y < SOM_Y; som_y++) { 
        for (unsigned int som_x = 0; som_x < SOM_X; som_x++) {
          p2[0] = som_x;
          p2[1] = som_y;
          float dist = 0.0f;
          for (unsigned int p = 0; p < SOM_D; p++)
            dist += (p1[p] - p2[p]) * (p1[p] - p2[p]);
          dist = sqrt(dist);
          
          float neighbor_fuct = 0.0f;
          neighbor_fuct = exp(-(1.0f * dist * dist) / (R * R));
          
          for (unsigned int d = 0; d < NDIMEN; d++) {
            NUMER1[som_y*SOM_X*NDIMEN + som_x*NDIMEN + d] += 
              1.0f * neighbor_fuct 
              * (*((DATA) + n*NDIMEN + d));
          }
          DENOM1[som_y*SOM_X + som_x] += neighbor_fuct;
        }
      }
    }
  }     
      
  MPI_Reduce(NUMER1, NUMER2, 
          SOM_Y*SOM_X*NDIMEN, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(DENOM1, DENOM2, 
          SOM_Y*SOM_X, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);  
  delete [] NUMER1;
  delete [] DENOM1;
}

/** Initialize SOM codebook with random values
 * @param seed - random seed
 * @param codebook - the codebook to fill in
 * @param nSomX - dimensions of SOM map in the x direction
 * @param nSomY - dimensions of SOM map in the y direction
 * @param nDimensions - dimensions of a data instance
 */

void initializeCodebook(unsigned int seed, float *codebook, unsigned int nSomX, 
                        unsigned int nSomY, unsigned int nDimensions)
{
  ///
  /// Fill initial random weights
  ///
  srand(seed);
  for (unsigned int som_y = 0; som_y < nSomY; som_y++) {        
    for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
      for (unsigned int d = 0; d < nDimensions; d++) {
        int w = 0xFFF & rand();
        w -= 0x800;
        codebook[som_y*nSomX*nDimensions+som_x*nDimensions+d] = (float)w / 4096.0f;
      }
    }
  }
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

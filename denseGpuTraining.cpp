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
#include <mpi.h>
#include <cstdlib>

#include "somoclu.h"

using namespace std;

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

/** MR-MPI user-defined map function - batch training with MPI_reduce()
 * @param itask - number of work items
 * @param kv
 * @param ptr
 */
      
void trainOneEpoch(int itask, float *data, float *numerator, 
                           float *denominator, float *codebook, 
                           unsigned int nSomX, unsigned int nSomY, 
                           unsigned int nDimensions, unsigned int nVectors,
                           unsigned int nVectorsPerRank, float radius)
{           
  int p1[2];
  int p2[2];
  float *localNumerator;
  float *localDenominator;
  localNumerator = new float[nSomY*nSomX*nDimensions];
  localDenominator = new float[nSomY*nSomX];
    
  for (unsigned int som_y = 0; som_y < nSomY; som_y++) {
      for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
          localDenominator[som_y*nSomX + som_x] = 0.0;
          for (unsigned int d = 0; d < nDimensions; d++) 
              localNumerator[som_y*nSomX*nDimensions + som_x*nDimensions + d] = 0.0;
      }
  }
  int bmus[nVectorsPerRank*2];
  getBmusOnGpu(bmus, codebook, nSomX, nSomY, nDimensions, nVectorsPerRank);
  for (unsigned int n = 0; n < nVectorsPerRank; n++) {
    if (itask*nVectorsPerRank+n<nVectors){    
      /// get the best matching unit
      p1[0]=bmus[n*2];
      p1[1]=bmus[n*2+1];
          
      /// Accumulate denoms and numers
      for (unsigned int som_y = 0; som_y < nSomY; som_y++) { 
        for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
          p2[0] = som_x;
          p2[1] = som_y;
          float dist = 0.0f;
          for (unsigned int p = 0; p < 2; p++)
            dist += (p1[p] - p2[p]) * (p1[p] - p2[p]);
          dist = sqrt(dist);
          
          float neighbor_fuct = 0.0f;
          neighbor_fuct = exp(-(1.0f * dist * dist) / (radius * radius));
          
          for (unsigned int d = 0; d < nDimensions; d++) {
            localNumerator[som_y*nSomX*nDimensions + som_x*nDimensions + d] += 
              1.0f * neighbor_fuct 
              * (*((data) + n*nDimensions + d));
          }
          localDenominator[som_y*nSomX + som_x] += neighbor_fuct;
        }
      }
    }
  }     
      
  MPI_Reduce(localNumerator, numerator, 
          nSomY*nSomX*nDimensions, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(localDenominator, denominator, 
          nSomY*nSomX, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);  
  delete [] localNumerator;
  delete [] localDenominator;
}

void train(int itask, float *data, unsigned int nSomX, unsigned int nSomY, 
           unsigned int nDimensions, unsigned int nVectors, 
           unsigned int nVectorsPerRank, unsigned int nEpoch, 
           const char *outPrefix, bool shouldSaveInterim)
{
  /// 
  /// Codebook
  ///
  float *codebook= new float[nSomY*nSomX*nDimensions];
  float *numerator;
  float *denominator;
  if (itask == 0) {
    numerator = new float[nSomY*nSomX*nDimensions];
    denominator = new float[nSomY*nSomX];
    initializeCodebook(0, codebook, nSomX, nSomY, nDimensions);
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  ///
  /// Parameters for SOM
  ///
  float N = (float)nEpoch;       /// iterations
  float radius0;
  radius0 = nSomX / 2.0f;              /// init radius for updating neighbors
  float radius = radius0;
  unsigned int x = 0;             /// 0...N-1
      
  ///
  /// Training
  ///
  while (nEpoch && radius > 1.0) {
    double epoch_time = MPI_Wtime();
    if (itask == 0) {
        radius = radius0 * exp(-10.0f * (x * x) / (N * N));
        x++;
        cout << "Epoch: " << (nEpoch-1) << " Radius: " << radius << endl;
    }
      
    MPI_Bcast(&radius, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Bcast(codebook, nSomY*nSomX*nDimensions, MPI_FLOAT, 
              0, MPI_COMM_WORLD);

    /// 1. Each task fills localNumerator and localDenominator
    /// 2. MPI_reduce sums up each tasks localNumerator and localDenominator to the root's 
    ///    numerator and denominator.
    /// 3. Update codebook using numerator and denominator
    if (itask == 0) {
      for (unsigned int som_y = 0; som_y < nSomY; som_y++) { 
        for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
          denominator[som_y*nSomX + som_x] = 0.0;
          for (unsigned int d = 0; d < nDimensions; d++) {
            numerator[som_y*nSomX*nDimensions + som_x*nDimensions + d] = 0.0;
          }
        }
      }
    }

    trainOneEpoch(itask, data, numerator, denominator, codebook,
                          nSomX, nSomY, nDimensions, nVectors, nVectorsPerRank,
                          radius);
    MPI_Barrier(MPI_COMM_WORLD);  
    if (itask == 0) {
      for (unsigned int som_y = 0; som_y < nSomY; som_y++) { 
        for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
          float denom = denominator[som_y*nSomX + som_x];
          for (unsigned int d = 0; d < nDimensions; d++) {
              float newWeight = numerator[som_y*nSomX*nDimensions 
                                  + som_x*nDimensions + d] / denom;
              if (newWeight > 0.0) 
                  codebook[som_y*nSomX*nDimensions+som_x*nDimensions+d] = newWeight;
          }
        }
      }
    }
      ///
    if (shouldSaveInterim && itask == 0) {
      cout << "Saving interim U-Matrix..." << endl;      
      char umatInterimFileName[50];
      sprintf(umatInterimFileName, "%s-umat-%03d.txt", outPrefix,  x);
      saveUMat(umatInterimFileName, codebook, nSomX, nSomY, nDimensions);
    }
    nEpoch--;
    epoch_time = MPI_Wtime() - epoch_time;
    if (itask == 0) {
      cerr << "Epoch Time: " << epoch_time << endl;
    }
  }  
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  ///
  /// Save SOM map and u-mat
  ///
  if (itask == 0) {
    ///
    /// Save U-mat
    ///
    char umatFileName[50];
    sprintf(umatFileName, "%s-umat.txt", outPrefix);
    int ret = saveUMat(umatFileName, codebook, nSomX, nSomY, nDimensions);
    if (ret < 0) 
        cout << "    Failed to save u-matrix. !" << endl; 
    else {
        cout << "    Done!" << endl;
    }        
     
    ///
    /// Save codebook
    ///
    char codebookInterimFileName[50];
    sprintf(codebookInterimFileName, "%s-codebook.txt", outPrefix);
    saveCodebook(codebookInterimFileName, codebook, nSomX, nSomY, nDimensions);
  }
  
  MPI_Barrier(MPI_COMM_WORLD);

  delete [] codebook;
  if (itask == 0) {  
    delete [] numerator;
    delete [] denominator;
  }
  
}

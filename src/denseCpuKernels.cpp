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

/** Distance b/w a feature vector and a weight vector
 * = Euclidean
 * @param som_y
 * @param som_x
 * @param r - row number in the input feature file
  */

float get_distance(float* codebook, float* data, 
                   unsigned int som_y, unsigned int som_x, unsigned int nSomX,
                   unsigned int nDimensions, unsigned int r)
{
    float distance = 0.0f;
    for (unsigned int d = 0; d < nDimensions; d++)
        distance += (codebook[som_y*nSomX*nDimensions+som_x*nDimensions+d] - 
                    *(data + r*nDimensions + d))
                    *
                    (codebook[som_y*nSomX*nDimensions+som_x*nDimensions+d] - 
                    *(data + r*nDimensions + d));
    return distance;
}

/** Get node coords for the best matching unit (BMU)
 * @param coords - BMU coords
 * @param n - row num in the input feature file
 */
void get_bmu_coord(float* codebook, float* data, 
                   unsigned int nSomY, unsigned int nSomX, 
                   unsigned int nDimensions, int* coords, unsigned int n)
{ 
    float mindist = 9999.99;
    float dist = 0.0f;
    
    /// Check nSomX * nSomY nodes one by one and compute the distance 
    /// D(W_K, Fvec) and get the mindist and get the coords for the BMU.
    ///
    for (unsigned int som_y = 0; som_y < nSomY; som_y++) { 
        for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
            dist = get_distance(codebook, data, som_y, som_x, nSomX,
                                nDimensions, n);
            if (dist < mindist) { 
                mindist = dist;
                coords[0] = som_x;
                coords[1] = som_y;
            }
        }
    }
}

void trainOneEpochDenseCPU(int itask, float *data, float *numerator, 
                           float *denominator, float *codebook, 
                           unsigned int nSomX, unsigned int nSomY, 
                           unsigned int nDimensions, unsigned int nVectors,
                           unsigned int nVectorsPerRank, float radius)
{           
  int p1[2];
  int p2[2];
  float *localNumerator = new float[nSomY*nSomX*nDimensions];
  float *localDenominator = new float[nSomY*nSomX];

  /// v2
  for (unsigned int som_y = 0; som_y < nSomY; som_y++) {
      for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
          localDenominator[som_y*nSomX + som_x] = 0.0;
          for (unsigned int d = 0; d < nDimensions; d++) 
              localNumerator[som_y*nSomX*nDimensions + som_x*nDimensions + d] = 0.0;
      }
  }
      
  for (unsigned int n = 0; n < nVectorsPerRank; n++) {
    if (itask*nVectorsPerRank+n<nVectors){    
      /// get the best matching unit
      get_bmu_coord(codebook, data, nSomY, nSomX,
                    nDimensions, p1, n);

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
              * (*(data + n*nDimensions + d));
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
}

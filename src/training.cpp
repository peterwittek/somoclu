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
#include <iostream>
#include <sstream>

#include "somoclu.h"

using namespace std;

/** Initialize SOM codebook with random values
 * @param seed - random seed
 * @param codebook - the codebook to fill in
 * @param nSomX - dimensions of SOM map in the currentEpoch direction
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

float linearCooling(float start, float end, float nEpoch, float epoch) {
  float diff = (start - end) / (nEpoch-1);
  return start - (epoch * diff);
}

float exponentialCooling(float start, float end, float nEpoch, float epoch) {
  float diff = 0;
  if (end == 0.0)
  {
      diff = -log(0.1) / nEpoch;
  }
  else
  {
      diff = -log(end / start) / nEpoch;
  }
  return start * exp(-epoch * diff);
}

core_data trainOneEpoch(int itask, float *data, svm_node **sparseData,
           core_data coreData, unsigned int nEpoch, unsigned int currentEpoch,
           bool enableCalculatingUMatrix,
           unsigned int nSomX, unsigned int nSomY,
           unsigned int nDimensions, unsigned int nVectors,
           unsigned int nVectorsPerRank,
           unsigned int radius0, unsigned int radiusN,
           string radiusCooling,
           float scale0, float scaleN,
           string scaleCooling,
           unsigned int kernelType, string mapType){

    float N = (float)nEpoch;
    float *numerator;
    float *denominator;
    float scale = scale0;
    float radius = radius0;             
    if (itask == 0) {
        numerator = new float[nSomY*nSomX*nDimensions];
        denominator = new float[nSomY*nSomX];
        for (unsigned int som_y = 0; som_y < nSomY; som_y++) {
            for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
                denominator[som_y*nSomX + som_x] = 0.0;
                for (unsigned int d = 0; d < nDimensions; d++) {
                    numerator[som_y*nSomX*nDimensions + som_x*nDimensions + d] = 0.0;
                }
            }
        }

        if (radiusCooling == "linear") {
          radius = linearCooling(float(radius0), radiusN, N, currentEpoch);
        } else {
          radius = exponentialCooling(radius0, radiusN, N, currentEpoch);
        }
        if (scaleCooling == "linear") {
          scale = linearCooling(scale0, scaleN, N, currentEpoch);
        } else {
          scale = exponentialCooling(scale0, scaleN, N, currentEpoch);
        }
        cout << "Epoch: " << currentEpoch << " Radius: " << radius << endl;
    }
#ifdef HAVE_MPI
    MPI_Bcast(&radius, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&scale, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(coreData.codebook, nSomY*nSomX*nDimensions, MPI_FLOAT,
              0, MPI_COMM_WORLD);
#endif

    /// 1. Each task fills localNumerator and localDenominator
    /// 2. MPI_reduce sums up each tasks localNumerator and localDenominator to the root's
    ///    numerator and denominator.
    switch (kernelType) {
    default:
    case DENSE_CPU:
        trainOneEpochDenseCPU(itask, data, numerator, denominator,
                              coreData.codebook, nSomX, nSomY, nDimensions,
                              nVectors, nVectorsPerRank, radius, scale, 
                              mapType, coreData.globalBmus);
        break;
#ifdef CUDA
    case DENSE_GPU:
        trainOneEpochDenseGPU(itask, data, numerator, denominator,
                              coreData.codebook, nSomX, nSomY, nDimensions,
                              nVectors, nVectorsPerRank, radius, scale, 
                              mapType, coreData.globalBmus);
        break;
#endif
    case SPARSE_CPU:
        trainOneEpochSparseCPU(itask, sparseData, numerator, denominator,
                               coreData.codebook, nSomX, nSomY, nDimensions,
                               nVectors, nVectorsPerRank, radius, scale, 
                               mapType, coreData.globalBmus);
        break;
    }

    /// 3. Update codebook using numerator and denominator
#ifdef HAVE_MPI        
    MPI_Barrier(MPI_COMM_WORLD);
#endif        
    if (itask == 0) {
        #pragma omp parallel for
        for (unsigned int som_y = 0; som_y < nSomY; som_y++) {
            for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
                float denom = denominator[som_y*nSomX + som_x];
                for (unsigned int d = 0; d < nDimensions; d++) {
                    float newWeight = numerator[som_y*nSomX*nDimensions
                                                + som_x*nDimensions + d] / denom;
                    if (newWeight > 0.0) {
                        coreData.codebook[som_y*nSomX*nDimensions+som_x*nDimensions+d] = newWeight;
                    }
                }
            }
        }
    }
    if (enableCalculatingUMatrix) {
        coreData.uMatrix = calculateUMatrix(coreData.codebook, nSomX, nSomY, nDimensions, mapType);
    }
    if (itask == 0) {
        delete [] numerator;
        delete [] denominator;
    }
    return coreData;
}


/** Main training loop
 * @param itask - number of work items
 * @param kv
 * @param ptr
 */

void train(int itask, float *data, svm_node **sparseData,
           unsigned int nSomX, unsigned int nSomY,
           unsigned int nDimensions, unsigned int nVectors,
           unsigned int nVectorsPerRank, unsigned int nEpoch,
           unsigned int radius0, unsigned int radiusN, 
           string radiusCooling,
           float scale0, float scaleN,
           string scaleCooling,
           string outPrefix, unsigned int snapshots,
           unsigned int kernelType, string mapType,
           string initialCodebookFilename)
{
    ///
    /// Codebook
    ///
    core_data coreData;
    coreData.codebook = new float[nSomY*nSomX*nDimensions];
    coreData.globalBmus = NULL;
    coreData.uMatrix = NULL;    
    if (itask == 0) {
        coreData.globalBmus = new int[nVectorsPerRank*int(ceil(nVectors/(double)nVectorsPerRank))*2];
        
        if (initialCodebookFilename.empty()){
            initializeCodebook(0, coreData.codebook, nSomX, nSomY, nDimensions);
        } else {
            unsigned int nSomXY = 0;
            unsigned int tmpNDimensions = 0;
            delete [] coreData.codebook;
            coreData.codebook = readMatrix(initialCodebookFilename, nSomXY, tmpNDimensions);
            if (tmpNDimensions != nDimensions) {
                cerr << "Dimension of initial codebook does not match data!\n";
                my_abort(5);
            } else if (nSomXY / nSomY != nSomX) {
                cerr << "Dimension of initial codebook does not match specified SOM grid!\n";
                my_abort(6);
            }
            cout << "Read initial codebook: " << initialCodebookFilename << "\n";
        }
    }
    ///
    /// Parameters for SOM
    ///
    if (radius0 == 0) {
        unsigned int minDim = min(nSomX, nSomY);
        radius0 = minDim / 2.0f;              /// init radius for updating neighbors
    }
    if (radiusN == 0) {
        radiusN = 1;
    }
    if (scale0 == 0) {
      scale0 = 1.0;
    }
        
    unsigned int currentEpoch = 0;             /// 0...nEpoch-1
    
    ///
    /// Training
    ///
#ifdef HAVE_MPI    
    double training_time = MPI_Wtime();
#endif    

    while ( currentEpoch < nEpoch ) {

#ifdef HAVE_MPI      
        double epoch_time = MPI_Wtime();
#endif        

        coreData = trainOneEpoch(itask, data, sparseData,
                                 coreData, nEpoch, currentEpoch,
                                 snapshots > 0,
                                 nSomX, nSomY,
                                 nDimensions, nVectors,
                                 nVectorsPerRank,
                                 radius0, radiusN,
                                 radiusCooling,
                                 scale0, scaleN,
                                 scaleCooling,
                                 kernelType, mapType);

        if (snapshots > 0 && itask == 0) {
            cout << "Saving interim U-Matrix..." << endl;
            stringstream sstm;
            sstm << outPrefix << "." << currentEpoch + 1;
            saveUMatrix(sstm.str() + string(".umx"), coreData.uMatrix, nSomX, nSomY);
            if (snapshots == 2){
                saveBmus(sstm.str() + string(".bm"), coreData.globalBmus, nSomX, nSomY, nVectors); 
                saveCodebook(sstm.str() + string(".wts"), coreData.codebook, nSomX, nSomY, nDimensions);                
            }
        }
        currentEpoch++;
#ifdef HAVE_MPI        
        if (itask == 0) {
            epoch_time = MPI_Wtime() - epoch_time;
            cerr << "Epoch Time: " << epoch_time << endl;
        }
#endif        
    }
#ifdef HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    training_time = MPI_Wtime() - training_time;
    if (itask == 0) {
        cerr << "Total training Time: " << training_time << endl;
    }
#endif
    ///
    /// Save SOM map and u-mat
    ///
    if (itask == 0) {
        ///
        /// Save U-mat
        ///
        coreData.uMatrix = calculateUMatrix(coreData.codebook, nSomX, nSomY, nDimensions, mapType);
        int ret =  saveUMatrix(outPrefix + string(".umx"), coreData.uMatrix, nSomX, nSomY);        
        if (ret < 0)
            cout << "    Failed to save u-matrix. !" << endl;
        else {
            cout << "    Done!" << endl;
        }
        saveBmus(outPrefix + string(".bm"), coreData.globalBmus, nSomX, nSomY, nVectors); 
        ///
        /// Save codebook
        ///
        saveCodebook(outPrefix + string(".wts"), coreData.codebook, nSomX, nSomY, nDimensions);
    }
#ifdef HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    delete [] coreData.codebook;
    delete [] coreData.globalBmus;
    delete [] coreData.uMatrix;
}

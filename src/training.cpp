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
#ifdef CLI
#include <iostream>
#include <iomanip>
#include <sstream>
#endif
#ifdef HAVE_R
#include <R.h>
#endif
#include "somoclu.h"

using namespace std;

void train(float *data, int data_length, unsigned int nEpoch,
           unsigned int nSomX, unsigned int nSomY,
           unsigned int nDimensions, unsigned int nVectors,
           unsigned int radius0, unsigned int radiusN, string radiusCooling,
           float scale0, float scaleN, string scaleCooling,
           unsigned int kernelType, string mapType,
           string gridType, bool compact_support,
           float *codebook, int codebook_size,
           int *globalBmus, int globalBmus_size,
           float *uMatrix, int uMatrix_size) {
    train(0, data, NULL, codebook, globalBmus, uMatrix, nSomX, nSomY,
          nDimensions, nVectors, nVectors,
          nEpoch, radius0, radiusN, radiusCooling,
          scale0, scaleN, scaleCooling,
          kernelType, mapType,
          gridType, compact_support
#ifdef CLI
          , "", 0);
#else
         );
#endif
    calculateUMatrix(uMatrix, codebook, nSomX, nSomY, nDimensions, mapType,
                     gridType);
}

void train(int itask, float *data, svm_node **sparseData,
           float *codebook, int *globalBmus, float *uMatrix,
           unsigned int nSomX, unsigned int nSomY,
           unsigned int nDimensions, unsigned int nVectors,
           unsigned int nVectorsPerRank, unsigned int nEpoch,
           unsigned int radius0, unsigned int radiusN, string radiusCooling,
           float scale0, float scaleN, string scaleCooling,
           unsigned int kernelType, string mapType,
           string gridType, bool compact_support
#ifdef CLI
           , string outPrefix, unsigned int snapshots)
#else
          )
#endif
{
    int nProcs = 1;
#ifdef HAVE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
#endif
#ifdef CUDA
    if (kernelType == DENSE_GPU) {
        setDevice(itask, nProcs);
        initializeGpu(data, nVectorsPerRank, nDimensions, nSomX, nSomY);
    }
#endif
    // (Re-)Initialize codebook with random values only if requested through
    // the passed codebook -- meaning that the user did not have an initial
    // codebook

    if (codebook[0] == 1000 && codebook[1] == 2000) {
        initializeCodebook(0, codebook, nSomX, nSomY, nDimensions);
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
        scale0 = 0.1;
    }

    ///
    /// Training
    ///
    unsigned int currentEpoch = 0;             /// 0...nEpoch-1
    while ( currentEpoch < nEpoch ) {

#ifdef HAVE_MPI
        double epoch_time = MPI_Wtime();
#endif

        trainOneEpoch(itask, data, sparseData, codebook, globalBmus,
                      nEpoch, currentEpoch,
                      nSomX, nSomY, nDimensions, nVectors, nVectorsPerRank,
                      radius0, radiusN, radiusCooling,
                      scale0, scaleN, scaleCooling, kernelType, mapType,
                      gridType, compact_support);
#ifdef CLI
        if (snapshots > 0 && itask == 0) {
            calculateUMatrix(uMatrix, codebook, nSomX, nSomY, nDimensions,
                             mapType, gridType);
            stringstream sstm;
            sstm << outPrefix << "." << currentEpoch + 1;
            saveUMatrix(sstm.str() + string(".umx"), uMatrix, nSomX, nSomY);
            if (snapshots == 2) {
                saveBmus(sstm.str() + string(".bm"), globalBmus, nSomX, nSomY, nVectors);
                saveCodebook(sstm.str() + string(".wts"), codebook, nSomX, nSomY, nDimensions);
            }
        }
#endif
        ++currentEpoch;

#ifdef CLI
#ifdef HAVE_MPI
        if (itask == 0) {
            epoch_time = MPI_Wtime() - epoch_time;
            cerr << "Epoch Time: " << epoch_time << endl;
            if ( (currentEpoch != nEpoch) && (currentEpoch % (nEpoch / 100 + 1) != 0) ) {}
            else {
                float ratio  =  currentEpoch / (float)nEpoch;
                int   c      =  ratio * 50 + 1;
                cout << std::setw(7) << (int)(ratio * 100) << "% [";
                for (int x = 0; x < c; x++) cout << "=";
                for (int x = c; x < 50; x++) cout << " ";
                cout << "]\n" << flush;
            }
        }
#endif
#endif
    }
#ifdef CUDA
    if (kernelType == DENSE_GPU) {
        freeGpu();
    }
#endif
}

float linearCooling(float start, float end, float nEpoch, float epoch) {
    float diff = (start - end) / (nEpoch - 1);
    return start - (epoch * diff);
}

float exponentialCooling(float start, float end, float nEpoch, float epoch) {
    float diff = 0;
    if (end == 0.0) {
        diff = -log(0.1) / nEpoch;
    }
    else {
        diff = -log(end / start) / nEpoch;
    }
    return start * exp(-epoch * diff);
}



/** Initialize SOM codebook with random values
 * @param seed - random seed
 * @param codebook - the codebook to fill in
 * @param nSomX - dimensions of SOM map in the currentEpoch direction
 * @param nSomY - dimensions of SOM map in the y direction
 * @param nDimensions - dimensions of a data instance
 */

void initializeCodebook(unsigned int seed, float *codebook, unsigned int nSomX,
                        unsigned int nSomY, unsigned int nDimensions) {
    ///
    /// Fill initial random weights
    ///
#ifdef HAVE_R
    GetRNGstate();
#else
    srand(seed);
#endif
    for (unsigned int som_y = 0; som_y < nSomY; som_y++) {
        for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
            for (unsigned int d = 0; d < nDimensions; d++) {
#ifdef HAVE_R
                int w = 0xFFF & (int) (RAND_MAX * unif_rand());
#else
                int w = 0xFFF & rand();
#endif
                w -= 0x800;
                codebook[som_y * nSomX * nDimensions + som_x * nDimensions + d] = (float)w / 4096.0f;
            }
        }
    }
#ifdef HAVE_R
    PutRNGstate();
#endif
}

void trainOneEpoch(int itask, float *data, svm_node **sparseData,
                   float *codebook, int *globalBmus,
                   unsigned int nEpoch, unsigned int currentEpoch,
                   unsigned int nSomX, unsigned int nSomY,
                   unsigned int nDimensions, unsigned int nVectors,
                   unsigned int nVectorsPerRank,
                   unsigned int radius0, unsigned int radiusN,
                   string radiusCooling,
                   float scale0, float scaleN,
                   string scaleCooling,
                   unsigned int kernelType, string mapType,
                   string gridType, bool compact_support) {

    float N = (float)nEpoch;
    float *numerator;
    float *denominator;
    float scale = scale0;
    float radius = radius0;
    if (itask == 0) {
        numerator = new float[nSomY * nSomX * nDimensions];
        denominator = new float[nSomY * nSomX];
        for (unsigned int som_y = 0; som_y < nSomY; som_y++) {
            for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
                denominator[som_y * nSomX + som_x] = 0.0;
                for (unsigned int d = 0; d < nDimensions; d++) {
                    numerator[som_y * nSomX * nDimensions + som_x * nDimensions + d] = 0.0;
                }
            }
        }

        if (radiusCooling == "linear") {
            radius = linearCooling(float(radius0), radiusN, N, currentEpoch);
        }
        else {
            radius = exponentialCooling(radius0, radiusN, N, currentEpoch);
        }
        if (scaleCooling == "linear") {
            scale = linearCooling(scale0, scaleN, N, currentEpoch);
        }
        else {
            scale = exponentialCooling(scale0, scaleN, N, currentEpoch);
        }
//        cout << "Epoch: " << currentEpoch << " Radius: " << radius << endl;
    }
#ifdef HAVE_MPI
    MPI_Bcast(&radius, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&scale, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(codebook, nSomY * nSomX * nDimensions, MPI_FLOAT,
              0, MPI_COMM_WORLD);
#endif

    /// 1. Each task fills localNumerator and localDenominator
    /// 2. MPI_reduce sums up each tasks localNumerator and localDenominator to the root's
    ///    numerator and denominator.
    switch (kernelType) {
    default:
    case DENSE_CPU:
        trainOneEpochDenseCPU(itask, data, numerator, denominator,
                              codebook, nSomX, nSomY, nDimensions,
                              nVectors, nVectorsPerRank, radius, scale,
                              mapType, gridType, compact_support, globalBmus);
        break;
#ifdef CUDA
    case DENSE_GPU:
        trainOneEpochDenseGPU(itask, data, numerator, denominator,
                              codebook, nSomX, nSomY, nDimensions,
                              nVectors, nVectorsPerRank, radius, scale,
                              mapType, gridType, compact_support, globalBmus);
        break;
#endif
    case SPARSE_CPU:
        trainOneEpochSparseCPU(itask, sparseData, numerator, denominator,
                               codebook, nSomX, nSomY, nDimensions,
                               nVectors, nVectorsPerRank, radius, scale,
                               mapType, gridType, compact_support, globalBmus);
        break;
    }

    /// 3. Update codebook using numerator and denominator
#ifdef HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    if (itask == 0) {
        #pragma omp parallel for
#ifdef _WIN32
        for (int som_y = 0; som_y < nSomY; som_y++) {
#else
        for (unsigned int som_y = 0; som_y < nSomY; som_y++) {
#endif
            for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
                float denom = denominator[som_y * nSomX + som_x];
                for (unsigned int d = 0; d < nDimensions; d++) {
                    float newWeight = numerator[som_y * nSomX * nDimensions
                                                + som_x * nDimensions + d] / denom;
                    if (newWeight > 0.0) {
                        codebook[som_y * nSomX * nDimensions + som_x * nDimensions + d] = newWeight;
                    }
                }
            }
        }
    }
    if (itask == 0) {
        delete [] numerator;
        delete [] denominator;
    }
}

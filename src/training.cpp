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
#ifndef CLI
#include <stdexcept>
#else
#include <sstream>
#endif  // CLI

#include "somoclu.h"

using namespace std;

// From https://stackoverflow.com/questions/17432502/how-can-i-measure-cpu-time-and-wall-clock-time-on-both-linux-windows
//  Windows
#ifdef _WIN32
#include <Windows.h>
double get_wall_time(){
    LARGE_INTEGER time,freq;
    if (!QueryPerformanceFrequency(&freq)){
        //  Handle error
        return 0;
    }
    if (!QueryPerformanceCounter(&time)){
        //  Handle error
        return 0;
    }
    return (double)time.QuadPart / freq.QuadPart;
}

//  Posix/Linux
#else
#include <time.h>
#include <sys/time.h>
double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
#endif

#ifdef HAVE_R
#include <Rcpp.h>
#else
#include <iostream>
#include <iomanip>
#endif  // HAVE_R

void my_abort(string err) {
#ifdef CLI
#ifdef HAVE_MPI
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        cerr << "Error: " << err << endl;
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
#else
    cerr << "Error: " << err << endl;
    exit(1);
#endif  // HAVE_MPI
#else
    throw std::runtime_error(err);
#endif  // CLI
}

void train(float *data, int data_length, unsigned int nEpoch,
           unsigned int nSomX, unsigned int nSomY,
           unsigned int nDimensions, unsigned int nVectors,
           float radius0, float radiusN, string radiusCooling,
           float scale0, float scaleN, string scaleCooling,
           unsigned int kernelType, string mapType,
           string gridType, bool compact_support, bool gaussian,
           float std_coeff, unsigned int verbose,
           float *codebook, int codebook_size,
           int *globalBmus, int globalBmus_size,
           float *uMatrix, int uMatrix_size) {
#ifdef HAVE_R
#ifndef CUDA
    if(kernelType == DENSE_GPU){
        Rprintf("Error: CUDA kernel not compiled \n");
        return;
    }
#endif // CUDA
#endif // HAVE_R
    train(0, data, NULL, codebook, globalBmus, uMatrix, nSomX, nSomY,
          nDimensions, nVectors, nVectors,
          nEpoch, radius0, radiusN, radiusCooling,
          scale0, scaleN, scaleCooling,
          kernelType, mapType,
          gridType, compact_support, gaussian, std_coeff, verbose
#ifdef CLI
          , "", 0);
#else
         );
#endif
    calculateUMatrix(uMatrix, codebook, nSomX, nSomY, nDimensions, mapType,
                     gridType);
}

void julia_train(float *data, int data_length, unsigned int nEpoch,
           unsigned int nSomX, unsigned int nSomY,
           unsigned int nDimensions, unsigned int nVectors,
           float radius0, float radiusN, unsigned int _radiusCooling,
           float scale0, float scaleN, unsigned int _scaleCooling,
           unsigned int kernelType, unsigned int _mapType,
           unsigned int _gridType, bool compact_support, bool gaussian,
           float std_coeff, unsigned int verbose,
           float *codebook, int codebook_size,
           int *globalBmus, int globalBmus_size,
           float *uMatrix, int uMatrix_size) {
    string radiusCooling;
    string scaleCooling;
    string mapType;
    string gridType;
    if (_radiusCooling == 0) {
        radiusCooling = "linear";
    } else {
        radiusCooling = "exponential";
    }
    if (_scaleCooling == 0) {
        scaleCooling = "linear";
    } else {
        scaleCooling = "exponential";
    }
    if (_mapType == 0) {
        mapType = "planar";
    } else {
        mapType = "toroid";
    }
    if (_gridType == 0) {
        gridType = "square";
    } else {
        gridType = "hexagonal";
    }
    train(0, data, NULL, codebook, globalBmus, uMatrix, nSomX, nSomY,
          nDimensions, nVectors, nVectors,
          nEpoch, radius0, radiusN, radiusCooling,
          scale0, scaleN, scaleCooling,
          kernelType, mapType,
          gridType, compact_support, gaussian, std_coeff, verbose
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
           float radius0, float radiusN, string radiusCooling,
           float scale0, float scaleN, string scaleCooling,
           unsigned int kernelType, string mapType,
           string gridType, bool compact_support, bool gaussian,
           float std_coeff, unsigned int verbose
#ifdef CLI
           , string outPrefix, unsigned int snapshots)
#else
          )
#endif
{
    int nProcs = 1;
    float * X2 = NULL;
#ifdef HAVE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
#endif
#ifdef CUDA
    if (kernelType == DENSE_GPU) {
        setDevice(itask, nProcs);
        initializeGpu(data, nVectorsPerRank, nDimensions, nSomX, nSomY);
    }
#endif

    if (kernelType == SPARSE_CPU) {
        // Pre-compute the squared norm of all the vectors

        X2 = new float[nVectorsPerRank];

#ifdef _OPENMP
    #pragma omp parallel for
#endif
#ifdef _WIN32
        for (int i=0; i<nVectorsPerRank; ++i) {
#else
        for (unsigned int i=0; i<nVectorsPerRank; ++i) {
#endif
            if (itask * nVectorsPerRank + i < nVectors) {
                float acc=0.f;
                for (unsigned int j=0; sparseData[i][j].index!=-1; ++j) {
                    acc += sparseData[i][j].value * sparseData[i][j].value;
                }
                X2[i] = acc;
            }
        }
    }

    // (Re-)Initialize codebook with random values only if requested through
    // the passed codebook -- meaning that the user did not have an initial
    // codebook

    if (codebook[0] == 1000 && codebook[1] == 2000) {
        initializeCodebook(get_wall_time(), codebook, nSomX, nSomY, nDimensions);
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

        double epoch_time = get_wall_time();
        trainOneEpoch(itask, data, sparseData, X2, codebook, globalBmus,
                      nEpoch, currentEpoch,
                      nSomX, nSomY, nDimensions, nVectors, nVectorsPerRank,
                      radius0, radiusN, radiusCooling,
                      scale0, scaleN, scaleCooling, kernelType, mapType,
                      gridType, compact_support, gaussian, std_coeff);
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
#ifndef HAVE_R
        if (itask == 0 && verbose > 0) {
            epoch_time = get_wall_time() - epoch_time;
            cerr << "Time for epoch " << currentEpoch << ": " << std::setw(4) << std::setprecision(4) << epoch_time << " ";
            if ( (currentEpoch != nEpoch) && (currentEpoch % (nEpoch / 100 + 1) != 0) ) {} else {
                float ratio  =  currentEpoch / (float)nEpoch;
                int   c      =  ratio * 50 + 1;
                cout << std::setw(7) << (int)(ratio * 100) << "% [";
                for (int x = 0; x < c; x++) cout << "=";
                for (int x = c; x < 50; x++) cout << " ";
                if (verbose == 1) {
                    cout << "]\r" << flush;
                } else {
                    cout << "]\n" << flush;
                }
            }
        }
#endif
    }
#ifndef HAVE_R
    if (itask == 0 && verbose > 0) {
        cout << endl;
    }
#endif
    trainOneEpoch(itask, data, sparseData, X2, codebook, globalBmus,
                  nEpoch, currentEpoch,
                  nSomX, nSomY, nDimensions, nVectors, nVectorsPerRank,
                  radius0, radiusN, radiusCooling,
                  scale0, scaleN, scaleCooling, kernelType, mapType,
                  gridType, compact_support, gaussian, std_coeff, true);
#ifdef CUDA
    if (kernelType == DENSE_GPU) {
        freeGpu();
    }
#endif

    if (kernelType == SPARSE_CPU) {
        delete [] X2;
    }
}

float linearCooling(float start, float end, float nEpoch, float epoch) {
    float diff = (start - end) / (nEpoch-1);
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
    #pragma omp parallel for
#ifdef _WIN32
    for (int som_y = 0; som_y < nSomY; som_y++) {
#else
    for (unsigned int som_y = 0; som_y < nSomY; som_y++) {
#endif
        for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
            for (unsigned int d = 0; d < nDimensions; d++) {
#ifdef HAVE_R
                int w = 0xFFF & (int) (RAND_MAX * R::runif(0,1));
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

void trainOneEpoch(int itask, float *data, svm_node **sparseData, float *X2,
                   float *codebook, int *globalBmus,
                   unsigned int nEpoch, unsigned int currentEpoch,
                   unsigned int nSomX, unsigned int nSomY,
                   unsigned int nDimensions, unsigned int nVectors,
                   unsigned int nVectorsPerRank,
                   float radius0, float radiusN,
                   string radiusCooling,
                   float scale0, float scaleN,
                   string scaleCooling,
                   unsigned int kernelType, string mapType,
                   string gridType, bool compact_support, bool gaussian,
                   float std_coeff, bool only_bmus) {

    float N = (float)nEpoch;
    float *numerator;
    float *denominator;
    float scale = scale0;
    float radius = radius0;
    if (itask == 0 && !only_bmus) {
#ifdef HAVE_MPI
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
#endif
        if (radiusCooling == "linear") {
            radius = linearCooling(radius0, radiusN, N, currentEpoch);
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
        //  << "Epoch: " << currentEpoch << " Radius: " << radius << endl;
    }
#ifdef HAVE_MPI
    if (!only_bmus) {
        MPI_Bcast(&radius, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&scale, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(codebook, nSomY * nSomX * nDimensions, MPI_FLOAT,
                  0, MPI_COMM_WORLD);
    }
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
                              mapType, gridType, compact_support, gaussian,
                              globalBmus, only_bmus, std_coeff);
        break;
    case DENSE_GPU:
#ifdef CUDA
        trainOneEpochDenseGPU(itask, data, numerator, denominator,
                              codebook, nSomX, nSomY, nDimensions,
                              nVectors, nVectorsPerRank, radius, scale,
                              mapType, gridType, compact_support, gaussian, globalBmus, only_bmus, std_coeff);
#else
        my_abort("Compiled without CUDA!");
#endif
        break;
    case SPARSE_CPU:
        trainOneEpochSparseCPU(itask, sparseData, X2, numerator, denominator,
                               codebook, nSomX, nSomY, nDimensions,
                               nVectors, nVectorsPerRank, radius, scale,
                               mapType, gridType, compact_support, gaussian,
                               globalBmus, only_bmus, std_coeff);
        break;
    }

    /// 3. Update codebook using numerator and denominator
#ifdef HAVE_MPI
    if (!only_bmus) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (itask == 0 && !only_bmus) {
          #pragma omp parallel for
#ifdef _WIN32
          for (int som_y = 0; som_y < nSomY; som_y++) {
#else
          for (unsigned int som_y = 0; som_y < nSomY; som_y++) {
#endif
              for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
                  float denom = denominator[som_y * nSomX + som_x];
                  if (denom != 0) {
                    for (unsigned int d = 0; d < nDimensions; d++) {
                        float newWeight = numerator[som_y * nSomX * nDimensions
                                                    + som_x * nDimensions + d] / denom;
                        codebook[som_y * nSomX * nDimensions + som_x * nDimensions + d] = newWeight;
                    }
                  }
              }
          }
          delete [] numerator;
          delete [] denominator;
      }
    }
#endif // HAVE_MPI
}

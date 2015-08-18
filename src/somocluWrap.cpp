#include <iostream>
#include <string.h>

#include "somocluWrap.h"

using namespace std;

void trainWrapper(float *data, int data_length,
                  unsigned int nEpoch,
                  unsigned int nSomX, unsigned int nSomY,
                  unsigned int nDimensions, unsigned int nVectors,
                  unsigned int radius0, unsigned int radiusN,
                  string radiusCooling,
                  float scale0, float scaleN,
                  string scaleCooling, unsigned int kernelType, string mapType,
                  float *codebook, int codebook_size,
                  int *globalBmus, int globalBmus_size,
                  float *uMatrix, int uMatrix_size) {   
    
    // Initialize codebook with random values only if requested through
    // the passed codebook -- meaning that the user did not have an initial
    // codebook
    if (codebook[0] == 1000 && codebook[1] == 2000) {
        initializeCodebook(0, codebook, nSomX, nSomY, nDimensions);
    }

#ifdef CUDA
    if(kernelType==DENSE_GPU){
        int rank = 0;
        int nProcs = 1;
        setDevice(rank, nProcs);
        initializeGpu(data, nVectorsPerRank, nDimensions, nSomX, nSomY);
    }
#endif

    if (radius0 == 0) {
        radius0 = nSomX / 2.0f;
    }
    if (radiusN == 0) {
        radiusN = 1;
    }
    if (scale0 == 0) {
        scale0 = 0.1;
    }

    for (unsigned int currentEpoch = 0; currentEpoch < nEpoch; ++currentEpoch) {
        trainOneEpoch(0, data, NULL, codebook, globalBmus, nEpoch, currentEpoch,
                      nSomX, nSomY, nDimensions, nVectors, nVectors,
                      radius0, radiusN, radiusCooling,
                      scale0, scaleN, scaleCooling, kernelType, mapType);
    }

    calculateUMatrix(uMatrix, codebook, nSomX, nSomY, nDimensions, mapType);
#ifdef CUDA
    if (kernelType == DENSE_GPU) {
        freeGpu();
    }
#endif
}

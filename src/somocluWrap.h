#ifndef SOMOCLUWRAP_H
#define SOMOCLUWRAP_H
#include"somoclu.h"
using namespace std;

core_data trainWrapper(float *data, int data_length,
//                       float *codebook, int codebook_length,
//                       int *globalBmus, int globalBmus_length,
//                       float *uMatrix, int uMatrix_length,
                       unsigned int nEpoch,
                       unsigned int nSomX, unsigned int nSomY,
                       unsigned int nDimensions, unsigned int nVectors,
                       unsigned int radius0, unsigned int radiusN,
                       string radiusCooling,
                       float scale0, float scaleN,
                       string scaleCooling, unsigned int snapshots,
                       unsigned int kernelType, string mapType,
                       string initialCodebookFilename);
#endif // SOMOCLUWRAP_H

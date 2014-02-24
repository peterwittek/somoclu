
#include"somocluWrap.h"

core_data trainWrapper(int itask,
                       float *data, int data_length,
                       svm_node **sparseData,
                       float *codebook, int codebook_length,
                       int *globalBmus, int globalBmus_length,
                       float *uMatrix, int uMatrix_length,
                       unsigned int nEpoch, unsigned int currentEpoch,
                       bool enableCalculatingUMatrix,
                       unsigned int nSomX, unsigned int nSomY,
                       unsigned int nDimensions, unsigned int nVectors,
                       unsigned int nVectorsPerRank,
                       unsigned int radius0, unsigned int radiusN,
                       string radiusCooling,
                       float scale0, float scaleN,
                       string scaleCooling,
                       unsigned int kernelType, string mapType)
{
  core_data coreData;
  coreData.codebook = codebook;
  coreData.globalBmus = globalBmus;
  coreData.uMatrix = uMatrix;
  return trainOneEpoch(itask, data, sparseData, coreData, nEpoch, currentEpoch,
                       enableCalculatingUMatrix, nSomX, nSomY, nDimensions,
                       nVectors, nVectorsPerRank, radius0, radiusN, radiusCooling,
                       scale0, scaleN, scaleCooling, kernelType, mapType);
}


#include"somocluWrap.h"
#include <cmath>
#include <iostream>
#include <string.h>

using namespace std;



void trainWrapper(float *data, int data_length,
                  unsigned int nEpoch,
                  unsigned int nSomX, unsigned int nSomY,
                  unsigned int nDimensions, unsigned int nVectors,
                  unsigned int radius0, unsigned int radiusN,
                  string radiusCooling,
                  float scale0, float scaleN,
                  string scaleCooling, unsigned int snapshots,
                  unsigned int kernelType, string mapType,
                  string initialCodebookFilename,
                  float *codebook, int codebook_size,
                  int *globalBmus, int globalBmus_size,
                  float *uMatrix, int uMatrix_size)
{
  ///
  /// Codebook
  ///

  int itask = 0;
  svm_node ** sparseData = NULL;
  core_data coreData;
  coreData.codebook_size = nSomY*nSomX*nDimensions;
  coreData.codebook = new float[coreData.codebook_size];
  coreData.globalBmus = NULL;
  coreData.uMatrix = NULL;
  unsigned int nVectorsPerRank = nVectors;
  coreData.globalBmus_size = nVectorsPerRank*int(ceil(nVectors/(double)nVectorsPerRank))*2;
  if (itask == 0) {
      coreData.globalBmus = new int[coreData.globalBmus_size];

      if (initialCodebookFilename.empty()){
          initializeCodebook(0, coreData.codebook, nSomX, nSomY, nDimensions);
      } else {
          unsigned int nSomXY = 0;
          unsigned int tmpNDimensions = 0;
          delete [] coreData.codebook;
          coreData.codebook = readMatrix(initialCodebookFilename, nSomXY, tmpNDimensions);
          if (tmpNDimensions != nDimensions) {
              cerr << "Dimension of initial codebook does not match data!\n";
//              my_abort(5);
          } else if (nSomXY / nSomY != nSomX) {
              cerr << "Dimension of initial codebook does not match specified SOM grid!\n";
//              my_abort(6);
          }
          cout << "Read initial codebook: " << initialCodebookFilename << "\n";
      }
  }
  ///
  /// Parameters for SOM
  ///
  if (radius0 == 0) {
      radius0 = nSomX / 2.0f;              /// init radius for updating neighbors
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

  while ( currentEpoch < nEpoch ) {

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

      currentEpoch++;
    }

  if (itask == 0) {
      ///
      /// Save U-mat
      ///
      coreData.uMatrix = calculateUMatrix(coreData.codebook, nSomX, nSomY, nDimensions, mapType);
      coreData.uMatrix_size = nSomX * nSomY;
  }
  if(coreData.codebook != NULL){
      memcpy(codebook, coreData.codebook, sizeof(float) *  codebook_size);
      delete [] coreData.codebook;
    }
  if(coreData.globalBmus != NULL){
      memcpy(globalBmus, coreData.globalBmus, sizeof(int) *  globalBmus_size);
      delete [] coreData.globalBmus;
    }
  if(coreData.uMatrix != NULL){
      memcpy(uMatrix, coreData.uMatrix, sizeof(float) *  uMatrix_size);
      delete [] coreData.uMatrix;
    }
}

void trainWrapperR(float *data, int data_length,
                  unsigned int nEpoch,
                  unsigned int nSomX, unsigned int nSomY,
                  unsigned int nDimensions, unsigned int nVectors,
                  unsigned int radius0, unsigned int radiusN,
                  string radiusCooling,
                  float scale0, float scaleN,
                  string scaleCooling, unsigned int snapshots,
                  unsigned int kernelType, string mapType,
//                  string initialCodebookFilename,
                  float *codebook, int codebook_size,
                  int *globalBmus, int globalBmus_size,
                  float *uMatrix, int uMatrix_size)
{
  ///
  /// Codebook
  ///

  int itask = 0;
  svm_node ** sparseData = NULL;
  core_data coreData;
  coreData.codebook_size = nSomY*nSomX*nDimensions;
  coreData.codebook = new float[coreData.codebook_size];
  coreData.globalBmus = NULL;
  coreData.uMatrix = NULL;
  unsigned int nVectorsPerRank = nVectors;
  coreData.globalBmus_size = nVectorsPerRank*int(ceil(nVectors/(double)nVectorsPerRank))*2;
  if (itask == 0) {
      coreData.globalBmus = new int[coreData.globalBmus_size];

//      if (initialCodebookFilename.empty()){
          initializeCodebook(0, coreData.codebook, nSomX, nSomY, nDimensions);
//      } else {
//          unsigned int nSomXY = 0;
//          unsigned int tmpNDimensions = 0;
//          delete [] coreData.codebook;
//          coreData.codebook = readMatrix(initialCodebookFilename, nSomXY, tmpNDimensions);
//          if (tmpNDimensions != nDimensions) {
//              cerr << "Dimension of initial codebook does not match data!\n";
//              my_abort(5);
//          } else if (nSomXY / nSomY != nSomX) {
//              cerr << "Dimension of initial codebook does not match specified SOM grid!\n";
//              my_abort(6);
//          }
//          cout << "Read initial codebook: " << initialCodebookFilename << "\n";
//      }
  }
  ///
  /// Parameters for SOM
  ///
  if (radius0 == 0) {
      radius0 = nSomX / 2.0f;              /// init radius for updating neighbors
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

  while ( currentEpoch < nEpoch ) {

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

      currentEpoch++;
    }

  if (itask == 0) {
      ///
      /// Save U-mat
      ///
      coreData.uMatrix = calculateUMatrix(coreData.codebook, nSomX, nSomY, nDimensions, mapType);
      coreData.uMatrix_size = nSomX * nSomY;
  }
  if(coreData.codebook != NULL){
      memcpy(codebook, coreData.codebook, sizeof(float) *  codebook_size);
      delete [] coreData.codebook;
    }
  if(coreData.globalBmus != NULL){
      memcpy(globalBmus, coreData.globalBmus, sizeof(int) *  globalBmus_size);
      delete [] coreData.globalBmus;
    }
  if(coreData.uMatrix != NULL){
      memcpy(uMatrix, coreData.uMatrix, sizeof(float) *  uMatrix_size);
      delete [] coreData.uMatrix;
    }
}

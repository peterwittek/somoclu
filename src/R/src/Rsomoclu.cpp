
#include<Rcpp.h>
//#include<R/R.h>
using namespace std;
using namespace Rcpp;

#include"somocluWrap.h"


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
      initializeCodebook(0, coreData.codebook, nSomX, nSomY, nDimensions);
  }
#ifdef CUDA
    if(kernelType==DENSE_GPU){
        int rank = 0;
        int nProcs = 1;
        setDevice(rank, nProcs);
        initializeGpu(data, nVectorsPerRank, nDimensions, nSomX, nSomY);
    }
#endif
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
#ifdef CUDA
  if (kernelType == DENSE_GPU) {
      freeGpu();
  }
#endif
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


RcppExport SEXP Rtrain(SEXP data_p,
                       SEXP nEpoch_p,
                       SEXP nSomX_p, SEXP nSomY_p,
                       SEXP radius0_p, SEXP radiusN_p,
                       SEXP radiusCooling_p,
                       SEXP scale0_p, SEXP scaleN_p,
                       SEXP scaleCooling_p, SEXP snapshots_p,
                       SEXP kernelType_p, SEXP mapType_p)
{
  Rcpp::NumericMatrix dataMatrix(data_p);
  int nVectors = dataMatrix.rows();
  int nDimensions = dataMatrix.cols();
  int nEpoch = as<int>(nEpoch_p);
  unsigned int nSomX = (unsigned int) as<int> (nSomX_p);
  unsigned int nSomY = (unsigned int) as<int> (nSomY_p);
  unsigned int radius0 = (unsigned int) as<int> (radius0_p);
  unsigned int radiusN = (unsigned int) as<int> (radiusN_p);
  string radiusCooling = as<string>(radiusCooling_p);
  unsigned int scale0 = (unsigned int) as<int> (scale0_p);
  unsigned int scaleN = (unsigned int) as<int> (scaleN_p);
  string scaleCooling = as<string> (scaleCooling_p);
  unsigned int snapshots = (unsigned int) as<int>(snapshots_p);
  unsigned int kernelType = (unsigned int) as<int>(kernelType_p);
  string mapType = as<string>(mapType_p);
  int data_length = nVectors * nDimensions;
  float* data = new float[data_length];
  // convert matrix to data c float array
  for(int i = 0; i < nVectors; i++){
      for(int j = 0; j < nDimensions; j++){
          data[i * nDimensions + j] = (float) dataMatrix(i,j);
        }
    }

  int codebook_size =  nSomY * nSomX * nDimensions;
  int globalBmus_size = nVectors * 2;
  int uMatrix_size = nSomX * nSomY;
  float* codebook = new float[codebook_size];
  int* globalBmus = new int[globalBmus_size];
  float* uMatrix = new float[uMatrix_size];
    trainWrapperR(data, data_length, nEpoch, nSomX, nSomY,
                 nDimensions, nVectors, radius0, radiusN,
                 radiusCooling, scale0, scaleN, scaleCooling,
                 snapshots, kernelType, mapType,
                 codebook, codebook_size, globalBmus, globalBmus_size,
                 uMatrix, uMatrix_size);
  Rcpp::NumericVector codebook_vec(codebook_size);
  Rcpp::NumericVector globalBmus_vec(globalBmus_size);
  Rcpp::NumericVector uMatrix_vec(uMatrix_size);
  if(codebook != NULL){
      for(int i = 0; i < codebook_size; i++){
          codebook_vec(i) = codebook[i];
        }
    }
  if(globalBmus != NULL){
      for(int i = 0; i < globalBmus_size; i++){
          globalBmus_vec(i) = globalBmus[i];
        }
    }
  if(uMatrix != NULL){
      for(int i = 0; i < uMatrix_size; i++){
          uMatrix_vec(i) = uMatrix[i];
        }
    }
  delete[] codebook;
  delete[] globalBmus;
  delete[] uMatrix;
  return Rcpp::List::create(Rcpp::Named("codebook") = codebook_vec,
                            Rcpp::Named("globalBmus") = globalBmus_vec,
                            Rcpp::Named("uMatrix") = uMatrix_vec);;
}

RCPP_MODULE(Rsomoclu){
  Rcpp::function("Rtrain", &Rtrain);
}

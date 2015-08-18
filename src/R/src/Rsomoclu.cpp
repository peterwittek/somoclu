#include <Rcpp.h>

using namespace std;
using namespace Rcpp;

#include"somocluWrap.h"

RcppExport SEXP Rtrain(SEXP data_p,
                       SEXP nEpoch_p,
                       SEXP nSomX_p, SEXP nSomY_p,
                       SEXP radius0_p, SEXP radiusN_p,
                       SEXP radiusCooling_p,
                       SEXP scale0_p, SEXP scaleN_p,
                       SEXP scaleCooling_p,
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
  trainWrapper(data, data_length, nEpoch, nSomX, nSomY,
               nDimensions, nVectors, radius0, radiusN,
               radiusCooling, scale0, scaleN, scaleCooling,
               kernelType, mapType,
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

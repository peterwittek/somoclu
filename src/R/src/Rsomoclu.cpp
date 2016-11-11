#include <Rcpp.h>
#include <iostream>

using namespace std;
using namespace Rcpp;

#include"somoclu.h"

RcppExport SEXP Rtrain(SEXP data_p,
                       SEXP nEpoch_p,
                       SEXP nSomX_p, SEXP nSomY_p,
                       SEXP radius0_p, SEXP radiusN_p,
                       SEXP radiusCooling_p,
                       SEXP scale0_p, SEXP scaleN_p,
                       SEXP scaleCooling_p,
                       SEXP kernelType_p, SEXP mapType_p,
                       SEXP gridType_p, SEXP compactSupport_p,
                       SEXP neighborhood_p,
                       SEXP stdCoeff_p,
                       SEXP codebook_p) {
    Rcpp::NumericMatrix dataMatrix(data_p);
    Rcpp::NumericMatrix codebookMatrix(codebook_p);
    int nVectors = dataMatrix.rows();
    int nDimensions = dataMatrix.cols();
    int nEpoch = as<int>(nEpoch_p);
    unsigned int nSomX = (unsigned int) as<int> (nSomX_p);
    unsigned int nSomY = (unsigned int) as<int> (nSomY_p);
    float radius0 = as<float> (radius0_p);
    float radiusN = as<float> (radiusN_p);
    string radiusCooling = as<string>(radiusCooling_p);
    float scale0 = as<float> (scale0_p);
    float scaleN = as<float> (scaleN_p);
    float std_coeff = as<float> (stdCoeff_p);
    string scaleCooling = as<string> (scaleCooling_p);
    unsigned int kernelType = (unsigned int) as<int>(kernelType_p);
    bool compactSupport = as<bool>(compactSupport_p);
    string mapType = as<string>(mapType_p);
    string gridType = as<string>(gridType_p);
    string neighborhood = as<string>(neighborhood_p);
    int data_length = nVectors * nDimensions;
    float* data = new float[data_length];
    int uMatrix_size = nSomX * nSomY;
    // convert matrix to data c float array
    for(int i = 0; i < nVectors; i++) {
        for(int j = 0; j < nDimensions; j++) {
            data[i * nDimensions + j] = (float) dataMatrix(i, j);
        }
    }
    int codebook_size =  nSomY * nSomX * nDimensions;
    float* codebook = new float[codebook_size];
    for(int i = 0; i < uMatrix_size; i++) {
      for(int j = 0; j < nDimensions; j++) {
        codebook[i * nDimensions + j] = (float) codebookMatrix(i ,j);
      }
    }
    int globalBmus_size = nVectors * 2;
    int* globalBmus = new int[globalBmus_size];
    float* uMatrix = new float[uMatrix_size];
    train(data, data_length, nEpoch, nSomX, nSomY,
          nDimensions, nVectors, radius0, radiusN,
          radiusCooling, scale0, scaleN, scaleCooling,
          kernelType, mapType,
          gridType, compactSupport, neighborhood == "gaussian",
          std_coeff,
          codebook, codebook_size, globalBmus, globalBmus_size,
          uMatrix, uMatrix_size);
    Rcpp::NumericMatrix globalBmusMatrix(nVectors, 2);
    Rcpp::NumericMatrix uMatrixMatrix(nSomX, nSomY);
    if(codebook != NULL) {
    	for(int i = 0; i < uMatrix_size; i++) {
    			for(int j = 0; j < nDimensions; j++) {
    				codebookMatrix(i ,j) = (float) codebook[i * nDimensions + j];
    			}
    		}
    }
    if(globalBmus != NULL) {
        for(int i = 0; i < nVectors; i++) {
        	for (int j = 0; j < 2; j++) {
        		globalBmusMatrix(i, j) = globalBmus[i * 2 + j];
        	}
        }
    }
    if(uMatrix != NULL) {
        for(int i = 0; i < nSomX; i++) {
        	for (int j = 0; j < nSomY; j++) {
        		uMatrixMatrix(i, j) = uMatrix[i * nSomY + j];
        	}
        }
    }
    delete[] codebook;
    delete[] globalBmus;
    delete[] uMatrix;
    return Rcpp::List::create(Rcpp::Named("codebook") = codebookMatrix,
                              Rcpp::Named("globalBmus") = globalBmusMatrix,
                              Rcpp::Named("uMatrix") = uMatrixMatrix);;
}

RCPP_MODULE(Rsomoclu) {
    Rcpp::function("Rtrain", &Rtrain);
}

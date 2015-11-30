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
                       SEXP codebook_p) {
    Rcpp::NumericMatrix dataMatrix(data_p);
    Rcpp::NumericVector codebook_vec(codebook_p);
    int nVectors = dataMatrix.rows();
    int nDimensions = dataMatrix.cols();
//    Rcpp::Rcout<<"data"<<endl;
//    Rcpp::Rcout<<nVectors<<' '<<nDimensions<<endl;
//    Rcpp::Rcout<<dataMatrix(0, 0)<<' '<<dataMatrix(0, 1)<<' '<<dataMatrix(0, 2)<<endl;
//    Rcpp::Rcout<<dataMatrix(1, 0)<<' '<<dataMatrix(1, 1)<<' '<<dataMatrix(1, 2)<<endl;
//    Rcpp::Rcout<<"codebook"<<endl;
//    Rcpp::Rcout<<codebook_vec.size()<<endl;
//    Rcpp::Rcout<<codebook_vec[0]<<' '<<codebook_vec[1]<<' '<<codebook_vec[2]<<endl;
//    Rcpp::Rcout<<codebook_vec[3]<<' '<<codebook_vec[4]<<' '<<codebook_vec[5]<<endl;
//    Rcpp::Rcout<<codebook_vec(0, 0)<<' '<<codebook_vec(0, 1)<<' '<<codebook_vec(0, 2)<<endl;
//    Rcpp::Rcout<<codebook_vec(1, 0)<<' '<<codebook_vec(1, 1)<<' '<<codebook_vec(1, 2)<<endl;
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
//    for(int som_y = 0; som_y < nSomY; ++som_y) {
//        for(int som_x = 0; som_x < nSomX; ++som_x) {
//            for(int d = 0; d < nDimensions; ++d) {
//                codebook[som_y * nSomX * nDimensions + som_x * nDimensions + d] = (float) codebook_vec(som_y * nSomX * nDimensions + som_x * nDimensions + d);
//            }
//        }
//    }
	for(int i = 0; i < uMatrix_size; i++) {
		for(int j = 0; j < nDimensions; j++) {
			codebook[i * nDimensions + j] = (float) codebook_vec(i ,j);
			//mexPrintf("%f\n",data[i * nDimensions + j] );
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
          codebook, codebook_size, globalBmus, globalBmus_size,
          uMatrix, uMatrix_size);
    Rcpp::NumericVector globalBmus_vec(globalBmus_size);
    Rcpp::NumericVector uMatrix_vec(uMatrix_size);
    if(codebook != NULL) {
//        for(int som_y = 0; som_y < nSomY; ++som_y) {
//            for(int som_x = 0; som_x < nSomX; ++som_x) {
//                for(int d = 0; d < nDimensions; ++d) {
//                    codebook_vec(som_y * nSomX * nDimensions + som_x * nDimensions + d) = codebook[som_y * nSomX * nDimensions + som_x * nDimensions + d];
//                }
//            }
//        }
    	for(int i = 0; i < uMatrix_size; i++) {
    			for(int j = 0; j < nDimensions; j++) {
    				codebook_vec(i ,j) = (float) codebook[i * nDimensions + j]  ;
    				//mexPrintf("%f\n",data[i * nDimensions + j] );
    			}
    		}
    }
    if(globalBmus != NULL) {
        for(int i = 0; i < globalBmus_size; i++) {
            globalBmus_vec(i) = globalBmus[i];
        }
    }
    if(uMatrix != NULL) {
        for(int i = 0; i < uMatrix_size; i++) {
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

RCPP_MODULE(Rsomoclu) {
    Rcpp::function("Rtrain", &Rtrain);
}

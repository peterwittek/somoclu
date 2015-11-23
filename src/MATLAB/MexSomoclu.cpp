#include "mex.h"
#include "somoclu.h"

#ifdef CUDA
#include "gpu/mxGPUArray.h"
#endif

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
#ifdef CUDA
	mxInitGPU();
#endif
    int        i;

    /* Examine input (right-hand-side) arguments. */
    //mexPrintf("\nThere are %d right-hand-side argument(s).", nrhs);
//   for (i=0; i<nrhs; i++)  {
//       mexPrintf("\n\tInput Arg %i is of type:\t%s ",i,mxGetClassName(prhs[i]));
//     }

    //Get input values
    int nVectors = mxGetM(prhs[0]);
    int nDimensions = mxGetN(prhs[0]);
    int data_length = nVectors * nDimensions;
    //    mexPrintf("\n%d\n", nVectors);
    //    mexPrintf("\n%d\n", nDimensions);
    float* data = new float[data_length];
    double * pData = mxGetPr(prhs[0]);

    for(int i = 0; i < nVectors; i++) {
        for(int j = 0; j < nDimensions; j++) {
            data[i * nDimensions + j] = (float) pData[j * nVectors + i];
            //mexPrintf("%f\n",data[i * nDimensions + j] );
        }
    }

    int nEpoch = (int) mxGetPr(prhs[1])[0];
    unsigned int nSomX = (unsigned int) mxGetPr(prhs[2])[0];
    unsigned int nSomY = (unsigned int) mxGetPr(prhs[3])[0];


    int codebook_size =  nSomY * nSomX * nDimensions;
    int globalBmus_size = nVectors * 2;
    int uMatrix_size = nSomX * nSomY;
    float* codebook = new float[codebook_size];
    int* globalBmus = new int[globalBmus_size];
    float* uMatrix = new float[uMatrix_size];

    double * pCodebook = mxGetPr(prhs[14]);
    if(pCodebook != NULL) {
    	for(int i = 0; i < uMatrix_size; i++) {
    		for(int j = 0; j < nDimensions; j++) {
    			codebook[i * nDimensions + j] = (float) pCodebook[j * uMatrix_size + i];
    			//mexPrintf("%f\n",data[i * nDimensions + j] );
    		}
    	}
    }

    unsigned int radius0 = (unsigned int) mxGetPr(prhs[4])[0];
    unsigned int radiusN = (unsigned int) mxGetPr(prhs[5])[0];
    char* radiusCooling_c = mxArrayToString(prhs[6]);
    string radiusCooling;
    if(radiusCooling_c != NULL) {
        radiusCooling = radiusCooling_c;
    }
    else {
        radiusCooling = "";
    }
    mxFree(radiusCooling_c);
    unsigned int scale0 = (unsigned int) mxGetPr(prhs[7])[0];
    unsigned int scaleN = (unsigned int) mxGetPr(prhs[8])[0];
    char* scaleCooling_c = mxArrayToString(prhs[9]);
    string scaleCooling;
    if(scaleCooling_c != NULL) {
        scaleCooling = scaleCooling_c;
    }
    else {
        scaleCooling = "";
    }
    mxFree(scaleCooling_c);
    unsigned int kernelType = (unsigned int) mxGetPr(prhs[10])[0];
    string mapType;
    char* mapType_c = mxArrayToString(prhs[11]);
    if(mapType_c != NULL) {
        mapType = mapType_c;
    }
    else {
        mapType = "planar";
    }
    mxFree(mapType_c);

    string gridType;
    char* gridType_c = mxArrayToString(prhs[12]);
    if(gridType_c != NULL) {
        gridType = gridType_c;
    }
    else {
        gridType = "rectangular";
    }
    mxFree(gridType_c);
    bool compactSupport = (bool) mxGetPr(prhs[13])[0];
    bool gaussian = (bool) mxGetPr(prhs[14])[0];

    //Call train routine
    train(data, data_length, nEpoch, nSomX, nSomY,
          nDimensions, nVectors, radius0, radiusN,
          radiusCooling, scale0, scaleN, scaleCooling,
          kernelType, mapType, gridType, compactSupport, gaussian,
          codebook, codebook_size, globalBmus, globalBmus_size,
          uMatrix, uMatrix_size);

    /* Examine output (left-hand-side) arguments. */
    //mexPrintf("\n\nThere are %d left-hand-side argument(s).\n", nlhs);
    if (nlhs > nrhs)
        mexErrMsgIdAndTxt( "MATLAB:mexfunction:inputOutputMismatch",
                           "Cannot specify more outputs than inputs.\n");

    plhs[0] = mxCreateDoubleMatrix(nSomY * nSomX, nDimensions, mxREAL);
    double* codebook_p = mxGetPr(plhs[0]);
    for(int i=0; i < uMatrix_size; i++) {
    	for(int j=0; j < nDimensions; j++) {
    		codebook_p[j * uMatrix_size + i] = codebook[i * nDimensions + j];
    	}
    }

    if(globalBmus != NULL) {
    	plhs[1] = mxCreateDoubleMatrix(nVectors, 2, mxREAL);
    	double* globalBmus_p = mxGetPr(plhs[1]);
    	for(int i=0; i < nVectors; i++) {
    		for(int j=0; j < 2; j++) {
    			globalBmus_p[j * nVectors + i] = (double) globalBmus[i * 2 + j];
    		}
    	}
    }

    else {
        plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
        *mxGetPr(plhs[1]) = 1;
    }

    plhs[2] = mxCreateDoubleMatrix(nSomX, nSomY, mxREAL);
    double* uMatrix_p = mxGetPr(plhs[2]);
    for(int i=0; i < nSomX; i++) {
    	for(int j=0; j < nSomY; j++) {
    		uMatrix_p[j * nSomX + i] = uMatrix[i * nSomY + j];
    	}
    }
    delete[] codebook;
    delete[] globalBmus;
    delete[] uMatrix;
}


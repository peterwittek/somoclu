#ifndef SOMOCLUWRAP_H
#define SOMOCLUWRAP_H
#include"somoclu.h"
#include<string>


using namespace std;

//struct core_data
//{
//	float *codebook;
//	int *globalBmus;
//	float *uMatrix;
//	int codebook_size;
//	int globalBmus_size;
//	int uMatrix_size;
//};

//struct svm_node
//{
//	int index;
//	float value;
//};

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
                  float* codebook, int codebook_size,
                  int* globalBmus, int globalBmus_size,
                  float* uMatrix, int uMatrix_size);

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
                  float *uMatrix, int uMatrix_size);

#endif // SOMOCLUWRAP_H

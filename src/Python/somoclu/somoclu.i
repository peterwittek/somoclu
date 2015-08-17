%module somoclu_wrap
%include "std_string.i"
%{
#define SWIG_FILE_WITH_INIT
#include "src/somocluWrap.h"
%}
%include "numpy.i"
%init %{
import_array();
%}
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* data, int data_length)}
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* codebook, int codebook_size)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* globalBmus, int globalBmus_size)}
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* uMatrix, int uMatrix_size)}

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
                  float* codebook, int codebook_size,
                  int* globalBmus, int globalBmus_size,
                  float* uMatrix, int uMatrix_size);

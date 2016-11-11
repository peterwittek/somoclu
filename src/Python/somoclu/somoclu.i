%module somoclu_wrap
%include "std_string.i"
%{
#define SWIG_FILE_WITH_INIT
#include "src/somoclu.h"
%}
%include "numpy.i"
%init %{
import_array();
%}
%apply (float* IN_ARRAY1, int DIM1) {(float* data, int data_length)}
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* codebook, int codebook_size)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* globalBmus, int globalBmus_size)}
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* uMatrix, int uMatrix_size)}

using namespace std;

%exception train {
   try {
      $action
   } catch (runtime_error &e) {
      PyErr_SetString(PyExc_RuntimeError, const_cast<char*>(e.what()));
      return NULL;
   }
}

void train(float *data, int data_length,
           unsigned int nEpoch,
           unsigned int nSomX, unsigned int nSomY,
           unsigned int nDimensions, unsigned int nVectors,
           float radius0, float radiusN,
           string radiusCooling,
           float scale0, float scaleN,
           string scaleCooling,
           unsigned int kernelType, string mapType,
           string gridType, bool compact_support, bool gaussian,
           float std_coeff,
           float* codebook, int codebook_size,
           int* globalBmus, int globalBmus_size,
           float* uMatrix, int uMatrix_size);

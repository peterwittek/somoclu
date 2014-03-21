%module somoclu
%include "std_string.i"
%{
#define SWIG_FILE_WITH_INIT
#include "src/src/somocluWrap.h"
%}
%include "numpy.i"
%init %{
import_array();
%}
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* data, int data_length)}
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* codebook, int codebook_size)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* globalBmus, int globalBmus_size)}
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* uMatrix, int uMatrix_size)}
/* %typemap (in,numinputs=0) core_data * (core_data temp) { */
/*   $1 = &temp; */
/*  } */

/* %typemap (argout) core_data * { */
/*   /\* codebook *\/ */
/*   { */
/*     npy_intp dims[1] = { $1->codebook_size }; */
/*     PyObject * array = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, (void*)($1->codebook)); */
/*     if (!array) SWIG_fail; */
/*     $result = SWIG_Python_AppendOutput($result,array); */
/*   } */
/*   /\* globalBmus *\/ */
/*   { */
/*     npy_intp dims[1] = { $1->globalBmus_size }; */
/*     PyObject * array = PyArray_SimpleNewFromData(1, dims, NPY_INT32, (void*)($1->globalBmus)); */
/*     if (!array) SWIG_fail; */
/*     $result = SWIG_Python_AppendOutput($result,array); */
/*   } */
/*   /\* uMatrix *\/ */
/*   { */
/*     npy_intp dims[1] = { $1->uMatrix_size }; */
/*     PyObject * array = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, (void*)($1->uMatrix)); */
/*     if (!array) SWIG_fail; */
/*     $result = SWIG_Python_AppendOutput($result,array); */
/*   } */
/*  } */
/* %ignore core_data; */
/* struct core_data */
/* { */
/* 	float *codebook; */
/* 	int *globalBmus; */
/*         float *uMatrix; */
/*   	int codebook_size; */
/* 	int globalBmus_size; */
/* 	int uMatrix_size; */
/* }; */

%include "src/src/somocluWrap.h"
















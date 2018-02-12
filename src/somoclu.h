/**
 * Self-Organizing Maps on a cluster
 *  Copyright (C) 2013 Peter Wittek
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#include <string>
using namespace std;

#ifndef SOMOCLU_H
#define SOMOCLU_H

#if HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#ifdef _WIN32
#include <algorithm>
#endif

#define DENSE_CPU 0
#define DENSE_GPU 1
#define SPARSE_CPU 2

/// The neighbor_fuct value below which we consider
/// the impact zero for a given node in the map
#define NEIGHBOR_THRESHOLD 0.05

/// Sparse structures and routines
struct svm_node {
    int index;
    float value;
};

#ifdef _WIN32
typedef int omp_iter_t;
#else
typedef unsigned int omp_iter_t;
#endif


class Distance{
private:    
    unsigned int dim;
public:
    Distance(unsigned int d):dim(d){}
    virtual ~Distance(){}
    virtual float operator()(float* v1, float* v2) const = 0;
    virtual void precompute(){}
    unsigned int Dim() const {return dim;}
};

class EuclideanDistance: public Distance{
public:
    EuclideanDistance(unsigned int d):Distance(d){}
    virtual ~EuclideanDistance(){}
    virtual float operator()(float* vec1, float* vec2) const;
};

/// Som parameters
struct som {
    unsigned int nSomX;
    unsigned int nSomY;
    unsigned int nDimensions;
    unsigned int nVectors;
    string mapType;
    string gridType;
    const Distance& get_distance;
    float* uMatrix;
    float *codebook;
    int *bmus;
};

class Snapshot {
private:
    unsigned int snapshots;
    string outPrefix;
public:
    Snapshot(unsigned int snapshots, string outPrefix):snapshots(snapshots),
    outPrefix(outPrefix){}
    virtual ~Snapshot(){}
    virtual void write(unsigned int currentEpoch, som map);
};

float euclideanDistanceOnToroidMap(const unsigned int som_x, const unsigned int som_y, const unsigned int x, const unsigned int y, const unsigned int nSomX, const unsigned int nSomY);
float euclideanDistanceOnPlanarMap(const unsigned int som_x, const unsigned int som_y, const unsigned int x, const unsigned int y);
float euclideanDistanceOnHexagonalToroidMap(const unsigned int som_x, const unsigned int som_y, const unsigned int x, const unsigned int y, const unsigned int nSomX, const unsigned int nSomY);
float euclideanDistanceOnHexagonalPlanarMap(const unsigned int som_x, const unsigned int som_y, const unsigned int x, const unsigned int y);
double get_wall_time();
float getWeight(float distance, float radius, float scaling, bool compact_support, bool gaussian, float std_coeff);
int saveCodebook(string cbFileName, som map);
void calculateUMatrix(som map);
int saveUMatrix(string fname, som map);
int saveBmus(string filename, som map);
float *readMatrix(const string inFilename,
                  unsigned int &nRows, unsigned int &nCols);
void readSparseMatrixDimensions(const string filename, unsigned int &nRows,
                                unsigned int &nColumns, bool& zerobased);
svm_node** readSparseMatrixChunk(const string filename, unsigned int nRows,
                                 unsigned int nRowsToRead,
                                 unsigned int rowOffset,
                                 unsigned int colOffset=0);
void trainOneEpoch(int itask, float *data, svm_node **sparseData, float *X2,
                   som map,
                   unsigned int nEpoch, unsigned int currentEpoch,
                   unsigned int nVectorsPerRank,
                   float radius0, float radiusN,
                   string radiusCooling,
                   float scale0, float scaleN,
                   string scaleCooling,
                   unsigned int kernelType, bool compact_support, bool gaussian,
                   float std_coeff=0.5, bool only_bmus=false);
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
           float std_coeff, unsigned int verbose,
           float* codebook, int codebook_size,
           int* globalBmus, int globalBmus_size,
           float* uMatrix, int uMatrix_size);
void train(int itask, float *data, svm_node **sparseData,
           som map, unsigned int nVectorsPerRank, unsigned int nEpoch,
           float radius0, float radiusN,
           string radiusCooling,
           float scale0, float scaleN,
           string scaleCooling,
           unsigned int kernelType, bool compact_support, bool gaussian,
           float std_coeff, unsigned int verbose, Snapshot *snapshot=NULL);

void trainOneEpochDenseCPU(int itask, float *data, float *numerator,
                           float *denominator, som map,
                           unsigned int nVectorsPerRank, float radius,
                           float scale, bool compact_support, bool gaussian,
                           bool only_bmus, float std_coeff);
void trainOneEpochSparseCPU(int itask, svm_node **sparseData, float *X2,
                            float *numerator, float *denominator, som map,
                            unsigned int nVectorsPerRank, float radius,
                            float scale, bool compact_support, bool gaussian,
                            bool only_bmus, float std_coeff);
void initializeCodebook(unsigned int seed, som map);


extern "C" {
#ifdef CUDA
    void setDevice(int commRank, int commSize);
    void freeGpu();
    void initializeGpu(float *hostData, int nVectorsPerRank, som map);
    void trainOneEpochDenseGPU(int itask, float *data, float *numerator,
                               float *denominator, som map,
                               unsigned int nVectorsPerRank, float radius,
                               float scale, bool compact_support, bool gaussian,
                               bool only_bmus, float std_coeff);

#endif
#ifdef _WIN32
    __declspec(dllexport) void my_abort(string err);
#else
    void my_abort(string err);
#endif
}
#endif  // SOMOCLU_H

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

#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <iostream>
#include <sstream>
#include <map>
#include <vector>
#include <stdio.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#include "somoclu.h"

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif

// Error handling macro
#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        cudaError_t err = cudaGetLastError(); \
        stringstream sstm; \
        sstm << "CUDA error calling \""#call"\", code is " << err; \
        my_abort(sstm.str()); }

//Globals
cublasHandle_t handle;
thrust::device_vector<float> deviceData;
thrust::device_vector<float> deviceDataNorms;
thrust::device_vector<float> deviceCodebook;
thrust::device_vector<float> deviceCodebookNorms;

// convert a linear index to a row index
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T, T> {
    T C; // number of columns

    __host__ __device__
    linear_index_to_row_index(T C) : C(C) {}

    __host__ __device__
    T operator()(T i) {
        return i / C;
    }
};

// note: functor inherits from unary_function
template <typename T>
struct square : public thrust::unary_function<T, T> {
    __host__ __device__
    T operator()(T x) const {
        return x * x;
    }
};

typedef thrust::tuple<int, float> argMinType;

struct argMin : public thrust::binary_function<argMinType, argMinType, argMinType> {
    __host__ __device__
    argMinType operator()(const argMinType& a, const argMinType& b) const {
        if (thrust::get<1>(a) < thrust::get<1>(b)) {
            return a;
        }
        else {
            return b;
        }
    }
};

template <typename T>
thrust::device_vector<T> normsOfRowSpace(thrust::device_vector<T> A, int nRows, int nColumns) {
    // allocate storage for row sums and indices
    thrust::device_vector<T> row_sums(nRows);
    thrust::device_vector<int> row_indices(nRows);

    // compute row sums by summing values with equal row indices
    thrust::reduce_by_key
    (thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(nColumns)),
     thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(nColumns)) + (nRows * nColumns),
     thrust::make_transform_iterator(A.begin(), square<T>()),
     row_indices.begin(),
     row_sums.begin(),
     thrust::equal_to<int>(),
     thrust::plus<T>());

    return row_sums;
}

thrust::device_vector<argMinType> minsOfRowSpace(thrust::device_vector<float> A, int nRows, int nColumns) {
    // allocate storage for row sums and indices
    thrust::device_vector<argMinType> row_sums(nRows);
    thrust::device_vector<int> row_indices(nRows);

    // compute row sums by summing values with equal row indices
    thrust::reduce_by_key
    (thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(nColumns)),
     thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(nColumns)) + (nRows * nColumns),
     thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(0), A.begin())),
     row_indices.begin(),
     row_sums.begin(),
     thrust::equal_to<int>(),
     argMin());
    return row_sums;
}

template <int BLOCK_DIM>
__global__ void euclidean(float *anorm2, float *bnorm2, float *M, int height, int width) {
    unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
    unsigned int yStartIndex = blockIdx.y * BLOCK_DIM;
    if (xIndex < width) {
        float bNormForX = bnorm2[xIndex];
        unsigned int yEndIndex = (yStartIndex + BLOCK_DIM < height ? yStartIndex + BLOCK_DIM : height);
        for (unsigned int yIndex = yStartIndex; yIndex < yEndIndex; yIndex++) {
            unsigned int index = yIndex * width + xIndex;
            M[index] = anorm2[yIndex] - 2 * M[index] + bNormForX;
        }
    }
}

template <typename T>
void printMatrix(thrust::device_vector<T> A, int nRows, int nColumns) {
    for (size_t i = 0; i < nRows; i++) {
        for (size_t j = 0; j < nColumns; j++) {
            std::cout << A[i * nColumns + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

/** Clear the device memory and shut down CUBLAS
 *
 */
void freeGpu() {
    deviceData.clear();
    deviceDataNorms.clear();
    deviceCodebook.clear();
    deviceCodebookNorms.clear();
    thrust::device_vector<float>().swap(deviceData);
    thrust::device_vector<float>().swap(deviceDataNorms);
    thrust::device_vector<float>().swap(deviceCodebook);
    thrust::device_vector<float>().swap(deviceCodebookNorms);
    cublasStatus_t status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        my_abort("CuBLAS shutdown error");
    }
}

/** Find the best matching units -- called from the map function
 * @param bmus - array of best matching units
 * @param codebook - the codebook to save
 * @param nSomX - dimensions of SOM map in the x direction
 * @param nSomY - dimensions of SOM map in the y direction
 * @param nDimensions - dimensions of a data instance
 * @param nVectorsPerRank - the number of data points assigned to this GPU
 */

void getBmusOnGpu(int *bmus, float *codebook, int nSomX, int nSomY, int nDimensions, int nVectorsPerRank) {
    deviceCodebook = thrust::device_vector<float>(codebook, codebook + nSomX * nSomY * nDimensions);
    deviceCodebookNorms = normsOfRowSpace<float>(deviceCodebook, nSomX * nSomY, nDimensions);
    thrust::device_vector<float> deviceGramMatrix(nSomX * nSomY * nVectorsPerRank, 0);
    //Calculate the inner products of the data vectors and the weight vectors

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                        nSomX * nSomY, nVectorsPerRank, nDimensions,
                                        &alpha, thrust::raw_pointer_cast(&deviceCodebook[0]), nDimensions,
                                        thrust::raw_pointer_cast(&deviceData[0]), nDimensions,
                                        &beta,  thrust::raw_pointer_cast(&deviceGramMatrix[0]), nSomX * nSomY);

    if (status != CUBLAS_STATUS_SUCCESS) {
        my_abort("Kernel execution error.");
    }

    //All components of the vectorized Euclidean distance are available
    // 32 is a magic number, this is the block size that works best on Tesla C2050
    int BLOCK_DIM = 32;
    dim3 grid((nSomX * nSomY + BLOCK_DIM - 1) / BLOCK_DIM, (nVectorsPerRank + BLOCK_DIM - 1) / BLOCK_DIM, 1);
    dim3 threads(BLOCK_DIM, 1, 1);
    if (BLOCK_DIM == 32) {
        euclidean<32> <<< grid, threads>>>(thrust::raw_pointer_cast(&deviceDataNorms[0]),
                                           thrust::raw_pointer_cast(&deviceCodebookNorms[0]),
                                           thrust::raw_pointer_cast(&deviceGramMatrix[0]),
                                           nVectorsPerRank, nSomX * nSomY);
    }
    //Finding minimums
    thrust::host_vector<argMinType> minsOfA = minsOfRowSpace(deviceGramMatrix, nVectorsPerRank, nSomX * nSomY);
    CUDA_CHECK(cudaDeviceSynchronize());

    //Getting back SOM coordinates from minimums
    for (int i = 0; i < nVectorsPerRank; i++) {
        argMinType tmp = minsOfA[i];
        int somCoordinate = thrust::get<0>(tmp) % (nSomX * nSomY);
        bmus[i * 2] = somCoordinate % nSomX;
        bmus[i * 2 + 1] = somCoordinate / nSomX;
    }
}

/** Initialize CUBLAS and device data
 * @param hostData - the data in the main memory
 * @param height - number of data points assigned to this GPU
 * @param width - dimensions of a data instance
 */

void initializeGpu(float *hostData, int nVectorsPerRank, int nDimensions, int nSomX, int nSomY) {
    /* Initialize CUBLAS */
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        my_abort("CuBLAS initialization error");
    }
    deviceData = thrust::device_vector<float>(hostData, hostData + nVectorsPerRank * nDimensions);
    deviceDataNorms = normsOfRowSpace<float>(deviceData, nVectorsPerRank, nDimensions);
    deviceCodebook = thrust::device_vector<float>(nSomX * nSomY * nDimensions, 0);
    deviceCodebookNorms = thrust::device_vector<float>(nSomX * nSomY, 0);
}

/** Check and initialize a device attached to a node
 * @param commRank - the MPI rank of this process
 * @param commSize - the size of MPI comm world
 */

/// Note that this function was lifted from http://code.google.com/p/gpmr/
void setDevice(int commRank, int commSize) {
    int devCount;
    int deviceNum = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
#ifdef HAVE_MPI
#ifdef _WIN32
    FILE * fp = popen("hostname.exe", "r");
#else
    FILE * fp = popen("/bin/hostname", "r");
#endif
    char buf[1024];
    if (fgets(buf, 1023, fp) == NULL) strcpy(buf, "localhost");
    pclose(fp);
    string host = buf;
    host = host.substr(0, host.size() - 1);
    strcpy(buf, host.c_str());
    if (commRank == 0) {
        map<string, vector<int> > hosts;
        map<string, int> devCounts;
        hosts[buf].push_back(0);
        devCounts[buf] = devCount;

        MPI_Status stat;
        MPI_Request req;
        for (int i = 1; i < commSize; ++i) {
            MPI_Recv(buf, 1024, MPI_CHAR, i, 0, MPI_COMM_WORLD, &stat);
            MPI_Recv(&devCount, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &stat);

            // check to make sure each process on each node reports the same number of devices.
            hosts[buf].push_back(i);
            if (devCounts.find(buf) != devCounts.end()) {
                if (devCounts[buf] != devCount) {
                    printf("Error, device count mismatch %d != %d on %s\n", devCounts[buf], devCount, buf);
                    fflush(stdout);
                }
            }
            else devCounts[buf] = devCount;
        }
        // check to make sure that we don't have more jobs on a node than we have GPUs.
        for (map<string, vector<int> >::iterator it = hosts.begin(); it != hosts.end(); ++it) {
            if (it->second.size() > static_cast<unsigned int>(devCounts[it->first])) {
                stringstream sstm;
                sstm << "Error, more jobs running on " << it->first.c_str() << " than devices - " << static_cast<int>(it->second.size()) << " jobs > " << devCounts[it->first] << " devices.";
                my_abort(sstm.str());
            }
        }

        // send out the device number for each process to use.
        MPI_Irecv(&deviceNum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &req);
        for (map<string, vector<int> >::iterator it = hosts.begin(); it != hosts.end(); ++it) {
            for (unsigned int i = 0; i < it->second.size(); ++i) {
                int devID = i;
                MPI_Send(&devID, 1, MPI_INT, it->second[i], 0, MPI_COMM_WORLD);
            }
        }
        MPI_Wait(&req, &stat);
    }
    else {
        // send out the hostname and device count for your local node, then get back the device number you should use.
        MPI_Status stat;
        MPI_Send(buf, strlen(buf) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&devCount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Recv(&deviceNum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &stat);
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    CUDA_CHECK(cudaSetDevice(deviceNum));
}

/** One epoch on the GPU, dense variant
 */
void trainOneEpochDenseGPU(int itask, float *data, float *numerator,
                           float *denominator, float *codebook,
                           unsigned int nSomX, unsigned int nSomY,
                           unsigned int nDimensions, unsigned int nVectors,
                           unsigned int nVectorsPerRank, float radius,
                           float scale, string mapType, string gridType,
                           bool compact_support, bool gaussian,
                           int *globalBmus, bool only_bmus, float std_coeff) {
    int *bmus;
#ifdef HAVE_MPI
    bmus = new int[nVectorsPerRank * 2];
#else
    bmus = globalBmus;
#endif
    getBmusOnGpu(bmus, codebook, nSomX, nSomY, nDimensions, nVectorsPerRank);
    if (only_bmus) {
#ifdef HAVE_MPI
        MPI_Gather(bmus, nVectorsPerRank * 2, MPI_INT, globalBmus, nVectorsPerRank * 2, MPI_INT, 0, MPI_COMM_WORLD);
        delete [] bmus;
#endif
        return;
    }
#ifdef HAVE_MPI
    float *localNumerator = new float[nSomY * nSomX * nDimensions];
    float *localDenominator = new float[nSomY * nSomX];
    #pragma omp for
#ifdef _WIN32
    for (int som_y = 0; som_y < nSomY; som_y++) {
#else
    for (unsigned int som_y = 0; som_y < nSomY; som_y++) {
#endif // _WIN32
        for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
            localDenominator[som_y * nSomX + som_x] = 0.0;
            for (unsigned int d = 0; d < nDimensions; d++)
                localNumerator[som_y * nSomX * nDimensions + som_x * nDimensions + d] = 0.0;
        }
    }
    #pragma omp parallel default(shared)
#else  // not HAVE_MPI
    float *localNumerator;
    float localDenominator = 0;
    #pragma omp parallel default(shared) private(localDenominator) private(localNumerator)
#endif
    {
#ifndef HAVE_MPI
        localNumerator = new float[nDimensions];
#endif // HAVE_MPI
        #pragma omp for
#ifdef _WIN32
        for (int som_y = 0; som_y < nSomY; som_y++) {
#else
        for (unsigned int som_y = 0; som_y < nSomY; som_y++) {
#endif
            for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
                for (unsigned int n = 0; n < nVectorsPerRank; n++) {
                    if (itask * nVectorsPerRank + n < nVectors) {
                        float dist = 0.0f;
                        if (gridType == "rectangular") {
                            if (mapType == "planar") {
                                dist = euclideanDistanceOnPlanarMap(som_x, som_y, bmus[2 * n], bmus[2 * n + 1]);
                            }
                            else if (mapType == "toroid") {
                                dist = euclideanDistanceOnToroidMap(som_x, som_y, bmus[2 * n], bmus[2 * n + 1], nSomX, nSomY);
                            }
                        }
                        else {
                            if (mapType == "planar") {
                                dist = euclideanDistanceOnHexagonalPlanarMap(som_x, som_y, bmus[2 * n], bmus[2 * n + 1]);
                            }
                            else if (mapType == "toroid") {
                                dist = euclideanDistanceOnHexagonalToroidMap(som_x, som_y, bmus[2 * n], bmus[2 * n + 1], nSomX, nSomY);
                            }
                        }
                        float neighbor_fuct = getWeight(dist, radius, scale, compact_support, gaussian, std_coeff);
#ifdef HAVE_MPI
                        for (unsigned int d = 0; d < nDimensions; d++) {
                            localNumerator[som_y * nSomX * nDimensions + som_x * nDimensions + d] +=
                                1.0f * neighbor_fuct
                                * (*(data + n * nDimensions + d));
                        }
                        localDenominator[som_y * nSomX + som_x] += neighbor_fuct;
#else // In this case, we can update in place
                        if (n == 0) {
                            localDenominator = neighbor_fuct;
                            for (unsigned int d = 0; d < nDimensions; d++) {
                                localNumerator[d] = 1.0f * neighbor_fuct
                                    * (*(data + n * nDimensions + d));
                            }
                         } else {
                            localDenominator += neighbor_fuct;
                            for (unsigned int d = 0; d < nDimensions; d++) {
                                localNumerator[d] += 1.0f * neighbor_fuct
                                    * (*(data + n * nDimensions + d));
                            }
                         }
#endif // HAVE_MPI                        
                    }
                } // Looping over data instances
#ifndef HAVE_MPI // We update in-place
                for (unsigned int d = 0; d < nDimensions; d++) {
                    float newWeight = localNumerator[d] / localDenominator;
                    codebook[som_y * nSomX * nDimensions + som_x * nDimensions + d] = newWeight;
                }
#endif
            } // Looping over som_x
        } // Looping over som_y
#ifndef HAVE_MPI
    delete [] localNumerator;
#endif
    } // OPENMP
#ifdef HAVE_MPI
    MPI_Reduce(localNumerator, numerator,
               nSomY * nSomX * nDimensions, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(localDenominator, denominator,
               nSomY * nSomX, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Gather(bmus, nVectorsPerRank * 2, MPI_INT, globalBmus, nVectorsPerRank * 2, MPI_INT, 0, MPI_COMM_WORLD);
    delete [] bmus;
    delete [] localNumerator;
    delete [] localDenominator;
#endif
}

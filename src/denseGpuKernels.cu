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
        cerr << "CUDA error calling \""#call"\", code is " << err << endl; \
        my_abort(err); }

//Globals
cublasHandle_t handle;

#define OPT_CUDA

#ifdef OPT_CUDA
//configure for reduction kernel
const int maxThreads = 256;  // number of threads per block
const int whichKernel = 6;
const int maxBlocks = 64;
int numBlocks = 0;
int numThreads = 0;

float *p_device_data;
float *p_device_data_norms;
float *p_device_codebook;
float *p_device_codebook_norms;
#else
thrust::device_vector<float> deviceData;
thrust::device_vector<float> deviceDataNorms;
thrust::device_vector<float> deviceCodebook;
thrust::device_vector<float> deviceCodebookNorms;
#endif
// convert a linear index to a row index
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T>
{
    T C; // number of columns

    __host__ __device__
    linear_index_to_row_index(T C) : C(C) {}

    __host__ __device__
    T operator()(T i)
    {
        return i / C;
    }
};

// note: functor inherits from unary_function
template <typename T>
struct square : public thrust::unary_function<T,T>
{
    __host__ __device__
    T operator()(T x) const
    {
        return x*x;
    }
};

typedef thrust::tuple<int,float> argMinType;

struct argMin : public thrust::binary_function<argMinType,argMinType,argMinType>
{
    __host__ __device__
    argMinType operator()(const argMinType& a, const argMinType& b) const
    {
        if (thrust::get<1>(a) < thrust::get<1>(b)) {
            return a;
        } else {
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
     thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(nColumns)) + (nRows*nColumns),
     thrust::make_transform_iterator(A.begin(), square<T>()),
     row_indices.begin(),
     row_sums.begin(),
     thrust::equal_to<int>(),
     thrust::plus<T>());
    return row_sums;
}
#ifdef OPT_CUDA
unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}
#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif
////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// For the kernels >= 3, we set threads / block to the minimum of maxThreads and
// n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel
// 6, we observe the maximum specified number of blocks, because each thread in
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{

    //get device capability, to avoid block/grid size excceed the upbound
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    if (whichKernel < 3)
    {
        threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
        blocks = (n + threads - 1) / threads;
    }
    else
    {
        threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
    }

    if ((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
    {
        printf("n is too large, please choose a smaller number!\n");
    }

    if (blocks > prop.maxGridSize[0])
    {
        printf("Grid size <%d> excceeds the device capability <%d>, set block size as %d (original %d)\n",
               blocks, prop.maxGridSize[0], threads*2, threads);

        blocks /= 2;
        threads *= 2;
    }

    if (whichKernel == 6)
    {
        blocks = MIN(maxBlocks, blocks);
    }
}

template<typename T>
__global__ void squareArray(T *d_array, int length) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < length) d_array[idx] = d_array[idx] * d_array[idx];
}

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

extern "C"
bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}
/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += g_idata[i+blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    __syncthreads();

    if ((blockSize >= 256) &&(tid < 128))
    {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

     __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = mySum = mySum + sdata[tid +  64];
    }

    __syncthreads();

#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
            mySum += __shfl_down(mySum, offset);
        }
    }
#else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
    }

    __syncthreads();

    if ((blockSize >=  32) && (tid < 16))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 16];
    }

    __syncthreads();

    if ((blockSize >=  16) && (tid <  8))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  8];
    }

    __syncthreads();

    if ((blockSize >=   8) && (tid <  4))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  4];
    }

    __syncthreads();

    if ((blockSize >=   4) && (tid <  2))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  2];
    }

    __syncthreads();

    if ((blockSize >=   2) && ( tid <  1))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  1];
    }

    __syncthreads();
#endif

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void
reduce(int size, int threads, int blocks,
       int whichKernel, T *d_idata, T *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    // choose which of the optimized versions of reduction to launch
    switch (whichKernel)
    {
        case 6:
        default:
            if (isPow2(size))
            {
                switch (threads)
                {
                    case 512:
                        reduce6<T, 512, true><<< dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                        break;

                    case 256:
                        reduce6<T, 256, true><<< dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                        break;

                    case 128:
                        reduce6<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 64:
                        reduce6<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 32:
                        reduce6<T,  32, true><<< dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                        break;

                    case 16:
                        reduce6<T,  16, true><<< dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                        break;

                    case  8:
                        reduce6<T,   8, true><<< dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                        break;

                    case  4:
                        reduce6<T,   4, true><<< dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                        break;

                    case  2:
                        reduce6<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case  1:
                        reduce6<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;
                }
            }
            else
            {
                switch (threads)
                {
                    case 512:
                        reduce6<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 256:
                        reduce6<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 128:
                        reduce6<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 64:
                        reduce6<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 32:
                        reduce6<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case 16:
                        reduce6<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case  8:
                        reduce6<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case  4:
                        reduce6<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case  2:
                        reduce6<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;

                    case  1:
                        reduce6<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                        break;
                }
            }

            break;
    }
}

// Instantiate the reduction function for 3 types
template void
reduce<int>(int size, int threads, int blocks,
            int whichKernel, int *d_idata, int *d_odata);

template void
reduce<float>(int size, int threads, int blocks,
              int whichKernel, float *d_idata, float *d_odata);

template void
reduce<double>(int size, int threads, int blocks,
               int whichKernel, double *d_idata, double *d_odata);

template <typename T>
void normsOfRowSpaceKernel(T *d_indata, T *d_odata, int nRows, int nColumns) {
	//calc square
	int length = nRows * nColumns;
    T* d_indata_copy;
    cudaMalloc((void **)& d_indata_copy, sizeof(float) * length);
    cudaMemcpy(d_indata_copy, d_indata, sizeof(float) * length, cudaMemcpyDeviceToDevice);
	getNumBlocksAndThreads(whichKernel, nRows * nColumns, maxBlocks, maxThreads, numBlocks, numThreads);
	int n_blocks = length/numThreads + (length%numThreads == 0 ? 0:1);
	squareArray <<< n_blocks, numThreads >>> (d_indata_copy, length);
	//reduce squared rows
	getNumBlocksAndThreads(whichKernel, nColumns, maxBlocks, maxThreads, numBlocks, numThreads);

	for (int i = 0; i < nRows; i++) {
		reduce<T>(nColumns, numThreads, numBlocks,
		              whichKernel, &d_indata_copy[i*nColumns], &d_odata[i]);
	}
	cudaFree(d_indata_copy);
}

template void
normsOfRowSpaceKernel<float>(float *d_indata,float *d_odata, int nRows, int nColumns);
#endif

thrust::device_vector<argMinType> minsOfRowSpace(thrust::device_vector<float> A, int nRows, int nColumns) {
    // allocate storage for row sums and indices
    thrust::device_vector<argMinType> row_sums(nRows);
    thrust::device_vector<int> row_indices(nRows);

    // compute row sums by summing values with equal row indices
    thrust::reduce_by_key
    (thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(nColumns)),
     thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(nColumns)) + (nRows*nColumns),
     thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(0),A.begin())),
     row_indices.begin(),
     row_sums.begin(),
     thrust::equal_to<int>(),
     argMin());
    return row_sums;
}

template <int BLOCK_DIM>
__global__ void euclidean(float *anorm2, float *bnorm2, float *M, int height, int width)
{
    unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
    unsigned int yStartIndex = blockIdx.y * BLOCK_DIM;
    if (xIndex < width) {
        float bNormForX = bnorm2[xIndex];
        unsigned int yEndIndex=(yStartIndex+BLOCK_DIM < height ? yStartIndex+BLOCK_DIM : height);
        for (unsigned int yIndex=yStartIndex; yIndex<yEndIndex; yIndex++) {
            unsigned int index=yIndex*width+xIndex;
            M[index] = anorm2[yIndex]-2*M[index]+bNormForX;
        }
    }
}

template <typename T>
void printMatrix(thrust::device_vector<T> A, int nRows, int nColumns) {
    for (size_t i = 0; i < nRows; i++) {
        for (size_t j = 0; j < nColumns; j++) {
            std::cout << A[i*nColumns+j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

/** Clear the device memory and shut down CUBLAS
 *
 */
void freeGpu()
{
#ifndef OPT_CUDA
    deviceData.clear();
    deviceDataNorms.clear();
    deviceCodebook.clear();
    deviceCodebookNorms.clear();
    thrust::device_vector<float>().swap(deviceData);
    thrust::device_vector<float>().swap(deviceDataNorms);
    thrust::device_vector<float>().swap(deviceCodebook);
    thrust::device_vector<float>().swap(deviceCodebookNorms);
#else
    cudaFree(p_device_data);
    cudaFree(p_device_data_norms);
    cudaFree(p_device_codebook);
    cudaFree(p_device_codebook_norms);
#endif
    cublasStatus_t status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cerr << "!!!! shutdown error (A)\n";
        my_abort(-1);
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

void getBmusOnGpu(int *bmus, float *codebook, int nSomX, int nSomY, int nDimensions, int nVectorsPerRank)
{
#ifndef OPT_CUDA
    deviceCodebook = thrust::device_vector<float>(codebook, codebook+nSomX*nSomY*nDimensions);
    deviceCodebookNorms = normsOfRowSpace<float>(deviceCodebook, nSomX*nSomY, nDimensions);
#else
    cudaMalloc((void**)& p_device_codebook, sizeof(float) * nSomX*nSomY*nDimensions);
    cudaMalloc((void**)& p_device_codebook_norms, sizeof(float) * nSomX*nSomY);
    cudaMemcpy(p_device_codebook, codebook, sizeof(float) * nSomX*nSomY*nDimensions, cudaMemcpyHostToDevice);
    normsOfRowSpaceKernel(p_device_codebook, p_device_codebook_norms, nSomX*nSomY, nDimensions);
#endif
    thrust::device_vector<float> deviceGramMatrix(nSomX*nSomY*nVectorsPerRank, 0);

    //Calculate the inner products of the data vectors and the weight vectors

    float alpha = 1.0f;
    float beta = 0.0f;
#ifndef OPT_CUDA
    cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                        nSomX*nSomY, nVectorsPerRank, nDimensions,
                                        &alpha, thrust::raw_pointer_cast(&deviceCodebook[0]), nDimensions,
                                        thrust::raw_pointer_cast(&deviceData[0]), nDimensions,
                                        &beta,  thrust::raw_pointer_cast(&deviceGramMatrix[0]), nSomX*nSomY);
#else

    cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                            nSomX*nSomY, nVectorsPerRank, nDimensions,
                                            &alpha, p_device_codebook, nDimensions,
                                            p_device_data, nDimensions,
                                            &beta,  thrust::raw_pointer_cast(&deviceGramMatrix[0]), nSomX*nSomY);

#endif
    if (status != CUBLAS_STATUS_SUCCESS) {
        cerr << "!!!! kernel execution error.\n";
        my_abort(-1);
    }

    //All components of the vectorized Euclidean distance are available
    // 32 is a magic number, this is the block size that works best on Tesla C2050
    int BLOCK_DIM=32;
    dim3 grid((nSomX*nSomY+BLOCK_DIM-1)/BLOCK_DIM, (nVectorsPerRank+BLOCK_DIM-1)/BLOCK_DIM,1);
    dim3 threads(BLOCK_DIM,1,1);
    if (BLOCK_DIM==32) {
#ifndef OPT_CUDA
        euclidean<32><<<grid, threads>>>(thrust::raw_pointer_cast(&deviceDataNorms[0]),
                                         thrust::raw_pointer_cast(&deviceCodebookNorms[0]),
                                         thrust::raw_pointer_cast(&deviceGramMatrix[0]),
                                         nVectorsPerRank, nSomX*nSomY);
#else
        euclidean<32><<<grid, threads>>>(p_device_data_norms,
                                                 p_device_codebook_norms,
                                                 thrust::raw_pointer_cast(&deviceGramMatrix[0]),
                                                 nVectorsPerRank, nSomX*nSomY);
#endif
    }
    //Finding minimums
    thrust::host_vector<argMinType> minsOfA=minsOfRowSpace(deviceGramMatrix, nVectorsPerRank, nSomX*nSomY);
    CUDA_CHECK(cudaDeviceSynchronize());

    //Getting back SOM coordinates from minimums
    for (int i=0; i<nVectorsPerRank; i++) {
        argMinType tmp=minsOfA[i];
        int somCoordinate=thrust::get<0>(tmp) % (nSomX*nSomY);
        bmus[i*2] = somCoordinate % nSomX;
        bmus[i*2+1] = somCoordinate / nSomX;
    }
}

/** Initialize CUBLAS and device data
 * @param hostData - the data in the main memory
 * @param height - number of data points assigned to this GPU
 * @param width - dimensions of a data instance
 */

void initializeGpu(float *hostData, int nVectorsPerRank, int nDimensions, int nSomX, int nSomY)
{
    /* Initialize CUBLAS */
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cerr << "!!!! CUBLAS initialization error\n";
        my_abort(-1);
    }
#ifndef OPT_CUDA
    deviceData = thrust::device_vector<float>(hostData, hostData+nVectorsPerRank*nDimensions);
    deviceDataNorms = normsOfRowSpace<float>(deviceData, nVectorsPerRank, nDimensions);

//    cout<<"norms of row space:\n";
//    for(int i=0; i< deviceDataNorms.size() ; i++)
//    {
//    	cout<<deviceDataNorms[i]<<endl;
//    }
//    cout<<endl;
    deviceCodebook = thrust::device_vector<float>(nSomX*nSomY*nDimensions,0);
    deviceCodebookNorms = thrust::device_vector<float>(nSomX*nSomY,0);
#else
    cudaMalloc((void **)& p_device_data, sizeof(float) * nVectorsPerRank * nDimensions);
    cudaMemcpy(p_device_data, hostData, sizeof(float) * nVectorsPerRank * nDimensions, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&p_device_data_norms, sizeof(float) * nVectorsPerRank);
    normsOfRowSpaceKernel<float>(p_device_data, p_device_data_norms, nVectorsPerRank, nDimensions);

#endif
}

/** Check and initialize a device attached to a node
 * @param commRank - the MPI rank of this process
 * @param commSize - the size of MPI comm world
 */

/// Note that this function was lifted from http://code.google.com/p/gpmr/
void setDevice(int commRank, int commSize)
{
    int devCount;
    int deviceNum=0;
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
    if (commRank == 0)
    {
        map<string, vector<int> > hosts;
        map<string, int> devCounts;
        hosts[buf].push_back(0);
        devCounts[buf] = devCount;

        MPI_Status stat;
        MPI_Request req;
        for (int i = 1; i < commSize; ++i)
        {
            MPI_Recv(buf, 1024, MPI_CHAR, i, 0, MPI_COMM_WORLD, &stat);
            MPI_Recv(&devCount, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &stat);

            // check to make sure each process on each node reports the same number of devices.
            hosts[buf].push_back(i);
            if (devCounts.find(buf) != devCounts.end())
            {
                if (devCounts[buf] != devCount)
                {
                    printf("Error, device count mismatch %d != %d on %s\n", devCounts[buf], devCount, buf);
                    fflush(stdout);
                }
            }
            else devCounts[buf] = devCount;
        }
        // check to make sure that we don't have more jobs on a node than we have GPUs.
        for (map<string, vector<int> >::iterator it = hosts.begin(); it != hosts.end(); ++it)
        {
            if (it->second.size() > static_cast<unsigned int>(devCounts[it->first]))
            {
                printf("Error, more jobs running on '%s' than devices - %d jobs > %d devices.\n",
                       it->first.c_str(), static_cast<int>(it->second.size()), devCounts[it->first]);
                fflush(stdout);
                my_abort(1);
            }
        }

        // send out the device number for each process to use.
        MPI_Irecv(&deviceNum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &req);
        for (map<string, vector<int> >::iterator it = hosts.begin(); it != hosts.end(); ++it)
        {
            for (unsigned int i = 0; i < it->second.size(); ++i)
            {
                int devID = i;
                MPI_Send(&devID, 1, MPI_INT, it->second[i], 0, MPI_COMM_WORLD);
            }
        }
        MPI_Wait(&req, &stat);
    }
    else
    {
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
                           float scale, string mapType, int *globalBmus)
{
#ifdef _WIN32
	int* bmus = (int *)alloca(sizeof(int) * nVectorsPerRank * 2);
#else
    int bmus[nVectorsPerRank*2];
#endif
    getBmusOnGpu(bmus, codebook, nSomX, nSomY, nDimensions, nVectorsPerRank);

    float *localNumerator = new float[nSomY*nSomX*nDimensions];
    float *localDenominator = new float[nSomY*nSomX];

    #pragma omp parallel default(shared)
    {
        #pragma omp for
        for (unsigned int som_y = 0; som_y < nSomY; som_y++) {
            for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
                localDenominator[som_y*nSomX + som_x] = 0.0;
                for (unsigned int d = 0; d < nDimensions; d++)
                    localNumerator[som_y*nSomX*nDimensions + som_x*nDimensions + d] = 0.0;
            }
        }
        /// Accumulate denoms and numers
        #pragma omp for
        for (unsigned int som_y = 0; som_y < nSomY; som_y++) {
            for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
                for (unsigned int n = 0; n < nVectorsPerRank; n++) {
                    if (itask*nVectorsPerRank+n<nVectors) {
                        float dist = 0.0f;
                        if (mapType == "planar") {
                            dist = euclideanDistanceOnPlanarMap(som_x, som_y, bmus[2*n], bmus[2*n+1]);
                        } else if (mapType == "toroid") {
                            dist = euclideanDistanceOnToroidMap(som_x, som_y, bmus[2*n], bmus[2*n+1], nSomX, nSomY);
                        }
                        float neighbor_fuct = getWeight(dist, radius, scale);
                        for (unsigned int d = 0; d < nDimensions; d++) {
                            localNumerator[som_y*nSomX*nDimensions + som_x*nDimensions + d] +=
                                1.0f * neighbor_fuct
                                * (*(data + n*nDimensions + d));
                        }
                        localDenominator[som_y*nSomX + som_x] += neighbor_fuct;
                    }
                }
            }
        }
    }
#ifdef HAVE_MPI         
    MPI_Reduce(localNumerator, numerator,
               nSomY*nSomX*nDimensions, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(localDenominator, denominator,
               nSomY*nSomX, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Gather(bmus, nVectorsPerRank*2, MPI_INT, globalBmus, nVectorsPerRank*2, MPI_INT, 0, MPI_COMM_WORLD);
#else
    for (unsigned int i=0; i < nSomY*nSomX*nDimensions; ++i) {
        numerator[i] = localNumerator[i];
    }
    for (unsigned int i=0; i < nSomY*nSomX; ++i) {
        denominator[i] = localDenominator[i];
    }
    for (unsigned int i=0; i < 2*nVectorsPerRank; ++i) {
      globalBmus[i]=bmus[i];
    }
#endif
    delete [] localNumerator;
    delete [] localDenominator;
}

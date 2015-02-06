#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <stdio.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <string>
#include <ctime>
using namespace std;

int maxThreads = 256;  // number of threads per block
//const int whichKernel = 6;
int maxBlocks = 64;
int maxBlocks_y = 64;
int maxBlocks_z = 64;
int numBlocks = 0;
int numThreads = 0;
cudaDeviceProp prop;
int device;

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
extern "C"
bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}
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

template<>
struct SharedMemory<float>
{
    __device__ inline operator       float *()
    {
        extern __shared__ int __smem[];
        return (float *)__smem;
    }

    __device__ inline operator const float *() const
    {
        extern __shared__ int __smem[];
        return (float *)__smem;
    }
};
void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{

    //get device capability, to avoid block/grid size excceed the upbound
    threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);

    if ((float)threads*blocks > (float) maxBlocks * maxThreads)
    {
        printf("n is too large, please choose a smaller number!\n");
    }

    if (blocks > maxBlocks)
    {
        printf("Grid size <%d> excceeds the device capability <%d>, set block size as %d (original %d)\n",
               blocks, maxBlocks, threads*2, threads);

        blocks /= 2;
        threads *= 2;
    }

    blocks = MIN(maxBlocks, blocks);

}
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6(T *g_idata, T *g_odata, int nRows, int nCols)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    unsigned int row = blockIdx.z * gridDim.y + blockIdx.y;
    if(row < nRows) {
    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < nCols)
    {
    	mySum += g_idata[nCols*row + i] * g_idata[nCols*row + i];

    	// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
    	if (nIsPow2 || i + blockSize < nCols)
    		mySum += g_idata[nCols*row + i + blockSize] * g_idata[nCols*row + i + blockSize];

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
    if (tid == 0) g_odata[row] = mySum;
    }
}
template <class T>
void
reduce(int nRows,int nCols, int threads, int blocks,
		T *d_idata, T *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    int nDimz = ceil((float)nRows/maxBlocks_y);
    int nDimy = nRows > maxBlocks_y? maxBlocks_y: nRows;
    dim3 dimGrid(blocks, nDimy, nDimz);
    cout<<"rows:"<<nRows<<endl;
    cout<<"cols:"<<nCols<<endl;
    cout<<"blocks:"<<blocks<<endl;
    cout<<"blocks_y:"<<nDimy<<endl;
    cout<<"blocks_z:"<<nDimz<<endl;
    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    // choose which of the optimized versions of reduction to launch

    if (isPow2(nCols))
    {
    	switch (threads)
    	{
    	case 1024:
    		reduce6<T, 1024, true><<< dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, nRows, nCols);
    		break;

    	case 512:
    		reduce6<T, 512, true><<< dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, nRows, nCols);
    		break;

    	case 256:
    		reduce6<T, 256, true><<< dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, nRows, nCols);
    		break;

    	case 128:
    		reduce6<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nRows, nCols);
    		break;

    	case 64:
    		reduce6<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nRows, nCols);
    		break;

    	case 32:
    		reduce6<T,  32, true><<< dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, nRows, nCols);
    		break;

    	case 16:
    		reduce6<T,  16, true><<< dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, nRows, nCols);
    		break;

    	case  8:
    		reduce6<T,   8, true><<< dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, nRows, nCols);
    		break;

    	case  4:
    		reduce6<T,   4, true><<< dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, nRows, nCols);
    		break;

    	case  2:
    		reduce6<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nRows, nCols);
    		break;

    	case  1:
    		reduce6<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nRows, nCols);
    		break;
    	}
    }
    else
    {
    	switch (threads)
    	{
    	case 1024:
    		reduce6<T, 1024, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nRows, nCols);
    		break;

    	case 512:
    		reduce6<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nRows, nCols);
    		break;

    	case 256:
    		reduce6<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nRows, nCols);
    		break;

    	case 128:
    		reduce6<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nRows, nCols);
    		break;

    	case 64:
    		reduce6<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nRows, nCols);
    		break;

    	case 32:
    		reduce6<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nRows, nCols);
    		break;

    	case 16:
    		reduce6<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nRows, nCols);
    		break;

    	case  8:
    		reduce6<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nRows, nCols);
    		break;

    	case  4:
    		reduce6<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nRows, nCols);
    		break;

    	case  2:
    		reduce6<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nRows, nCols);
    		break;

    	case  1:
    		reduce6<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nRows, nCols);
    		break;
    	}
    }
}

// Instantiate the reduction function for 3 types
template void
reduce<int>(int nRows,int nCols, int threads, int blocks,
		 	 int *d_idata, int *d_odata);

template void
reduce<float>(int nRows,int nCols, int threads, int blocks,
               float *d_idata, float *d_odata);

template void
reduce<double>(int nRows,int nCols, int threads, int blocks,
				double *d_idata, double *d_odata);

template <typename T>
void normsOfRowSpaceKernel(T *d_indata, T *d_odata, int nRows, int nColumns) {
	getNumBlocksAndThreads(nColumns, maxBlocks, maxThreads, numBlocks, numThreads);
	cout<<"numBlocks:"<<numBlocks<<endl;
	cout<<"numThreads:"<<numThreads<<endl;
	reduce<T>(nRows, nColumns, numThreads, numBlocks, d_indata, d_odata);
}

template void
normsOfRowSpaceKernel<float>(float *d_indata,float *d_odata, int nRows, int nColumns);

float *readMatrix(string inFilename, unsigned int &nRows, unsigned int &nColumns)
{
    float *data = NULL;
    ifstream file;
    file.open(inFilename.c_str());
    string line;
    data = new float[nRows*nColumns];
    for(int i=0;i<nRows*nColumns;i++){
    	file>>data[i];
    }
    file.close();
    return data;
}

int main(int argc, char *argv[])
{
	if(argc > 3)
	{
		string inFile(argv[1]);
		string rowStr(argv[2]);
		string colStr(argv[3]);
		unsigned int row=atoi(rowStr.c_str());
		unsigned int col=atoi(colStr.c_str());
		float *hostData = readMatrix(inFile, row, col);
		float *p_deviceData, *p_reduced_host, *p_reduced_device;
		thrust::device_vector<float> deviceData;
		thrust::device_vector<float> deviceDataNorms;

		cudaGetDevice(&device);
		cudaGetDeviceProperties(&prop, device);
		maxBlocks = prop.maxGridSize[0];
		maxBlocks_y = prop.maxGridSize[1];
		maxBlocks_z = prop.maxGridSize[2];
		maxThreads = prop.maxThreadsPerBlock;
		cout<<"maxBlocks:"<<maxBlocks<<endl;
		cout<<"maxBlocks.y:"<< maxBlocks_y<<endl;
		cout<<"maxBlocks.z:"<< maxBlocks_z<<endl;
		cout<<"maxThreads:"<<maxThreads<<endl;

		cudaMalloc((void **)& p_deviceData, sizeof(float) * row *col);
		cudaMemcpy(p_deviceData, hostData, sizeof(float)*row*col, cudaMemcpyHostToDevice);
		cudaMalloc((void **)& p_reduced_device, sizeof(float) * row);
		p_reduced_host = new float[row];
		deviceData = thrust::device_vector<float>(hostData, hostData+row * col);
		std::clock_t start = std::clock();
		deviceDataNorms = normsOfRowSpace<float>(deviceData, row, col);
		std::cout << "Time old: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
		std::clock_t start2 = std::clock();
		normsOfRowSpaceKernel(p_deviceData, p_reduced_device, row, col);
		std::cout << "Time new: " << (std::clock() - start2) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
		cudaMemcpy(p_reduced_host, p_reduced_device, sizeof(float)*row, cudaMemcpyDeviceToHost);
		ofstream outfile1,outfile2,outfile3;
		outfile1.open("old.txt", ios::out);
		outfile2.open("new.txt", ios::out);
//		outfile3.open("data.txt", ios::out);
		for(int i=0; i< deviceDataNorms.size() ; i++)
		{
			outfile1<<deviceDataNorms[i]<<endl;
		}
		for(int i=0; i< row ; i++)
		{
			outfile2<<p_reduced_host[i]<<endl;
		}
//		for(int i=0; i< row*col ; i++)
//		{
//			outfile3<<hostData[i]<<endl;
//		}
		outfile1.close();
		outfile2.close();
		delete[] p_reduced_host;
		delete[] hostData;
		cudaFree(p_reduced_device);
		cudaFree(p_deviceData);
	}

	return 0;
}

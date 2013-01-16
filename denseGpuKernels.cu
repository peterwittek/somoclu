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
#include <iostream>
#include <cublas_v2.h>
#include <vector>
#include <mpi.h>

#include "somoclu.h"

using namespace std;

#define BLOCK_DIM 16 

// Error handling macro
#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        cudaError_t err = cudaGetLastError(); \
        cerr << "CUDA error calling \""#call"\", code is " << err << endl; \
        my_abort(err); }

//Globals
cublasHandle_t handle;
float* data_d = NULL;
float* dataNorm2_d = NULL;

__global__ void columnArgMin(float *g_idata, int *g_odata, int height, int width)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  float min=g_idata[i];
  int argmin=0;
  if (i<width){
    for (int j=1; j < height; j++){ 
      float element=g_idata[j*width+i];
      if (element<min){
        argmin=j;
        min=element;    
      }
    } 
    g_odata[i]=argmin;
  }
}

__global__ void columnSquareSum(float *g_idata, float *g_odata, int height, int width)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  float element= (i < width) ? g_idata[i] : 0;
  float sum = element*element;
  float c = 0.0;              
  for (int j = 1; j < height; j++){
    element=(i < width) ? g_idata[j*width+i] : 0;
    float y = element*element - c;  
    float t = sum + y;      
    c = (t - sum) - y;  
    sum = t;            
  }
  g_odata[i]=sum;
}

__global__ void transpose(float *odata, float *idata,  int height, int width)
{
  __shared__ float block[BLOCK_DIM][BLOCK_DIM+1];
  
  // read the matrix tile into shared memory
  unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
  unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
  if((xIndex < width) && (yIndex < height)){
    unsigned int index_in = yIndex * width + xIndex;
    block[threadIdx.y][threadIdx.x] = idata[index_in];
  }

  __syncthreads();

  // write the transposed matrix tile to global memory
  xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
  yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
  if((xIndex < height) && (yIndex < width)){
    unsigned int index_out = yIndex * height + xIndex;
    odata[index_out] = block[threadIdx.x][threadIdx.y];
  }
}

void calculateNorm2(float *d_Anorm2, float *d_A, int height, int width)
{
  dim3 grid((width+511)/512, 1, 1);
  dim3 threads(512, 1, 1);
  columnSquareSum<<<grid, threads>>>(d_A, d_Anorm2, height, width);
}

void checkMatrix(float *d, int height, int width){
  float *data=new float[height*width];
  CUDA_CHECK( cudaMemcpy(data, d, height*width*sizeof(float), cudaMemcpyDeviceToHost) );
  for (int i=0; i<height; ++i){
    for (int j=0; j<width; ++j){
      cout << data[i*width+j] << " ";
    }
    cout << endl;
  }
  delete [] data;
}

/** Clear the device memory and shut down CUBLAS
 * 
 */
void shutdownGpu()
{
  CUDA_CHECK(cudaFree(data_d)); 
  CUDA_CHECK(cudaFree(dataNorm2_d));  
  // Shut down CUBLAS
  cublasStatus_t status = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cerr << "!!!! shutdown error (A)\n";
    my_abort(-1);
  }
}

//M is not transposed after CUBLAS matrix multiplication
__global__ void euclidean(float *odata, float *anorm2, float *bnorm2, float *M, int height, int width)
{
  // read the matrix tile into shared memory
  unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
  unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
  if((xIndex < width) && (yIndex < height))
  {
    unsigned int index=yIndex*width+xIndex;
    odata[index] = anorm2[xIndex]-2*M[index]+bnorm2[yIndex];
  }
}

void setCodebookOnDevice(float *hostData, float *deviceData, int nSomX, int nSomY, int nDimensions){
  float *tmpCodebook=NULL;
  CUDA_CHECK(cudaMalloc((void**)&tmpCodebook, nSomX*nSomY*nDimensions * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(tmpCodebook, hostData, nSomX*nSomY*nDimensions * sizeof(float), cudaMemcpyHostToDevice));
  dim3 grid((nDimensions+BLOCK_DIM-1)/BLOCK_DIM, (nSomX*nSomY+BLOCK_DIM-1)/BLOCK_DIM,1), threads(BLOCK_DIM,BLOCK_DIM,1);
  transpose<<<grid, threads>>>(deviceData, tmpCodebook, nSomX*nSomY, nDimensions);
  CUDA_CHECK(cudaFree(tmpCodebook));  
}

/** Find the best matching units -- called from the map function
 * @param bmus - array of best matching units
 * @param codebook - the codebook to save
 * @param nSomX - dimensions of SOM map in the x direction
 * @param nSomY - dimensions of SOM map in the y direction
 * @param nDimensions - dimensions of a data instance
 * @param nVecsPerRank - the number of data points assigned to this GPU
 */

void getBmusOnGpu(int *bmus, float *codebook, int nSomX, int nSomY, int nDimensions, int nVecsPerRank)
{
  float *deviceCodebook=NULL;
  float *codebookNorm2;
  float *d_C;
  float *d_D;

  CUDA_CHECK(cudaMalloc((void**)&deviceCodebook, nSomX*nSomY*nDimensions * sizeof(float)));    
  setCodebookOnDevice(codebook, deviceCodebook, nSomX, nSomY, nDimensions);

  //Calculate the norms of the codebook weight vectors
  CUDA_CHECK(cudaMalloc((void**)&codebookNorm2, nSomX*nSomY * sizeof(float)));
  dim3 grid((nSomX*nSomY+511)/512, 1, 1); 
  dim3 threads(512, 1, 1);
  columnSquareSum<<<grid, threads>>>(deviceCodebook, codebookNorm2, nDimensions, nSomX*nSomY);

  //Calculate the inner products of the data vectors and the weight vectors
  CUDA_CHECK( cudaMalloc((void**)&d_C, nVecsPerRank*nSomX*nSomY*sizeof(float)) );    
  float alpha = 1.0f;float beta = 0.0f;
  cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                                      nVecsPerRank, nSomX*nSomY, nDimensions, 
                                      &alpha, data_d, nVecsPerRank, 
                                              deviceCodebook, nSomX*nSomY, 
                                      &beta, d_C, nVecsPerRank);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cerr << "!!!! kernel execution error.\n";
    my_abort(-1);
  }

  //All components of the vectorized Euclidean distance are available
  CUDA_CHECK( cudaMalloc((void**)&d_D, nVecsPerRank*nSomX*nSomY*sizeof(float)) );
  dim3 grid2((nVecsPerRank+BLOCK_DIM-1)/BLOCK_DIM, (nSomX*nSomY+BLOCK_DIM-1)/BLOCK_DIM,1);
  dim3 threads2(BLOCK_DIM,BLOCK_DIM,1);
  euclidean<<<grid2, threads2>>>(d_D, dataNorm2_d, codebookNorm2, d_C, nSomX*nSomY, nVecsPerRank);
    
  //Finding minimums
  int *d_mins;
  CUDA_CHECK( cudaMalloc((void**)&d_mins, nVecsPerRank*sizeof(int)) );
  dim3 grid3((nVecsPerRank+511)/512, 1, 1);
  dim3 threads3(512, 1, 1);
  columnArgMin<<<grid3, threads3>>>(d_D, d_mins, nSomX*nSomY, nVecsPerRank);
  int *mins=new int[nVecsPerRank];
  CUDA_CHECK( cudaMemcpy(mins, d_mins, nVecsPerRank*sizeof(int), cudaMemcpyDeviceToHost) );
  CUDA_CHECK(cudaFree(d_mins)); 
    
  //Getting back SOM coordinates from minimums
  for (int i=0; i<nVecsPerRank; i++){
    bmus[i*2] = mins[i] % nSomX;
    bmus[i*2+1] = mins[i] / nSomX;
  }        
  CUDA_CHECK(cudaFree(d_C));  
  CUDA_CHECK(cudaFree(d_D));  
  CUDA_CHECK(cudaFree(deviceCodebook)); 
  CUDA_CHECK(cudaFree(codebookNorm2));  
}

/** Initialize CUBLAS and device data
 * @param hostData - the data in the main memory
 * @param height - number of data points assigned to this GPU
 * @param width - dimensions of a data instance
 */

void initializeGpu(float *hostData, int height, int width)
{   
    /* Initialize CUBLAS */
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
      cerr << "!!!! CUBLAS initialization error\n";
      my_abort(-1);
  }
  CUDA_CHECK(cudaMalloc((void**)&data_d, height*width*sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dataNorm2_d, height*sizeof(float)));

  float *tmpDeviceData;
  CUDA_CHECK(cudaMalloc((void**)&tmpDeviceData, height*width* sizeof(float)));
  CUDA_CHECK(cudaMemcpy(tmpDeviceData, hostData, height*width * sizeof(float), cudaMemcpyHostToDevice));
  dim3 grid((width+BLOCK_DIM-1)/BLOCK_DIM, (height+BLOCK_DIM-1)/BLOCK_DIM,1), threads(BLOCK_DIM,BLOCK_DIM,1);
  transpose<<<grid, threads>>>(data_d, tmpDeviceData, height, width);
  CUDA_CHECK(cudaFree(tmpDeviceData));  
  calculateNorm2(dataNorm2_d, data_d, width, height);
}

/** Check and initialize a device attached to a node
 * @param commRank - the MPI rank of this process
 * @param commSize - the size of MPI comm world
 */

void setDevice(int commRank, int commSize)
{
  FILE * fp = popen("/bin/hostname", "r");
  char buf[1024];
  if (fgets(buf, 1023, fp) == NULL) strcpy(buf, "localhost");
  pclose(fp);
  string host = buf;
  host = host.substr(0, host.size() - 1);
  strcpy(buf, host.c_str());

  int devCount;
  int deviceNum=-1;
  CUDA_CHECK(cudaGetDeviceCount(&devCount));

  if (commRank == 0)
  {
    map<string, vector<int> > hosts;
    map<string, int> devCounts;
    MPI_Status stat;
    MPI_Request req;

    hosts[buf].push_back(0);
    devCounts[buf] = devCount;
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
          printf("Error, device count mismatch %d != %d on %s\n", devCounts[buf], devCount, buf); fflush(stdout);
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
        MPI_Abort(MPI_COMM_WORLD, 1);
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
  CUDA_CHECK(cudaSetDevice(deviceNum));
  MPI_Barrier(MPI_COMM_WORLD);
}

/** One epoch on the GPU, dense variant
 */
void trainOneEpochDenseGPU(int itask, float *data, float *numerator, 
                           float *denominator, float *codebook, 
                           unsigned int nSomX, unsigned int nSomY, 
                           unsigned int nDimensions, unsigned int nVectors,
                           unsigned int nVectorsPerRank, float radius)
{           
  int p1[2];
  int p2[2];
  float *localNumerator = new float[nSomY*nSomX*nDimensions];
  float *localDenominator = new float[nSomY*nSomX];
    
  for (unsigned int som_y = 0; som_y < nSomY; som_y++) {
      for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
          localDenominator[som_y*nSomX + som_x] = 0.0;
          for (unsigned int d = 0; d < nDimensions; d++) 
              localNumerator[som_y*nSomX*nDimensions + som_x*nDimensions + d] = 0.0;
      }
  }
  int bmus[nVectorsPerRank*2];
  getBmusOnGpu(bmus, codebook, nSomX, nSomY, nDimensions, nVectorsPerRank);
  for (unsigned int n = 0; n < nVectorsPerRank; n++) {
    if (itask*nVectorsPerRank+n<nVectors){    
      /// get the best matching unit
      p1[0]=bmus[n*2];
      p1[1]=bmus[n*2+1];
          
      /// Accumulate denoms and numers
      for (unsigned int som_y = 0; som_y < nSomY; som_y++) { 
        for (unsigned int som_x = 0; som_x < nSomX; som_x++) {
          p2[0] = som_x;
          p2[1] = som_y;
          float dist = 0.0f;
          for (unsigned int p = 0; p < 2; p++)
            dist += (p1[p] - p2[p]) * (p1[p] - p2[p]);
          dist = sqrt(dist);
          
          float neighbor_fuct = 0.0f;
          neighbor_fuct = exp(-(1.0f * dist * dist) / (radius * radius));
          
          for (unsigned int d = 0; d < nDimensions; d++) {
            localNumerator[som_y*nSomX*nDimensions + som_x*nDimensions + d] += 
              1.0f * neighbor_fuct 
              * (*((data) + n*nDimensions + d));
          }
          localDenominator[som_y*nSomX + som_x] += neighbor_fuct;
        }
      }
    }
  }     
      
  MPI_Reduce(localNumerator, numerator, 
          nSomY*nSomX*nDimensions, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(localDenominator, denominator, 
          nSomY*nSomX, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);  
  delete [] localNumerator;
  delete [] localDenominator;
}

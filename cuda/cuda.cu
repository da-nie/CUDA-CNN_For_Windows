#include "cuda.cu.h"

#include <cstdio>
#include <ctime>
#include <stdio.h>
#include <stdint.h>

#include "handle_error.cu.h"
#include "ccudacnn.cu.h"

//----------------------------------------------------------------------------------------------------
//главная функция программы на CUDA
//----------------------------------------------------------------------------------------------------
void CUDA_Start(void)
{
 int deviceCount;
 cudaDeviceProp devProp;

 HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));
 printf("Found %d devices\n",deviceCount);
 for(int device=0;device<deviceCount;device++)
 {
  HANDLE_ERROR(cudaGetDeviceProperties(&devProp,device));
  printf("Device %d\n", device );
  printf("Compute capability     : %lu.%lu\n",devProp.major,devProp.minor);
  printf("Name                   : %s\n",devProp.name);
  printf("Total Global Memory    : %lu\n",devProp.totalGlobalMem);
  printf("Shared memory per block: %lu\n",devProp.sharedMemPerBlock);
  printf("Registers per block    : %lu\n",devProp.regsPerBlock);
  printf("Warp size              : %lu\n",devProp.warpSize);
  printf("Max threads per block  : %lu\n",devProp.maxThreadsPerBlock);
  printf("Total constant memory  : %lu\n",devProp.totalConstMem);
 }
 HANDLE_ERROR(cudaSetDevice(0));
 HANDLE_ERROR(cudaDeviceReset());

 CCUDACNN<float> cCUDACNN;
 cCUDACNN.Execute();
}


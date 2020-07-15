#ifndef C_CUDA_FUNCTION_H
#define C_CUDA_FUNCTION_H

//****************************************************************************************************
//Класс применения функции в CUDA
//****************************************************************************************************

#include <stdio.h>
#include <stdint.h>

#include "../handle_error.cu.h"
#include "../ccudamatrixstorage.cu.h"
#include "../../system/system.h"

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************

//****************************************************************************************************
//макроопределения
//****************************************************************************************************

//****************************************************************************************************
//константы
//****************************************************************************************************

//****************************************************************************************************
//предварительные объявления
//****************************************************************************************************
template<class type_t>
class CCUDAFunction;

template<class type_t>
__global__ void CUDASigmoidFunction(CCUDAFunction<type_t> cCUDAFunction);//функция CUDA для применения функции сигмоид

template<class type_t>
__global__ void CUDALinearFunction(CCUDAFunction<type_t> cCUDAFunction);//функция CUDA для применения линейной функции

template<class type_t>
__global__ void CUDAReLUFunction(CCUDAFunction<type_t> cCUDAFunction);//функция CUDA для применения функции ReLU

template<class type_t>
__global__ void CUDALeakyReLUFunction(CCUDAFunction<type_t> cCUDAFunction);//функция CUDA для применения функции Leaky ReLU

template<class type_t>
__global__ void CUDATangenceFunction(CCUDAFunction<type_t> cCUDAFunction);//функция CUDA для применения функции гиперболический тангенс

template<class type_t>
__global__ void CUDADifferentialSigmoidFunction(CCUDAFunction<type_t> cCUDAFunction);//функция CUDA для применения производной функции сигмоид

template<class type_t>
__global__ void CUDADifferentialLinearFunction(CCUDAFunction<type_t> cCUDAFunction);//функция CUDA для применения производной линейной функции

template<class type_t>
__global__ void CUDADifferentialReLUFunction(CCUDAFunction<type_t> cCUDAFunction);//функция CUDA для применения производной функции ReLU

template<class type_t>
__global__ void CUDADifferentialLeakyReLUFunction(CCUDAFunction<type_t> cCUDAFunction);//функция CUDA для применения производной функции Leaky ReLU

template<class type_t>
__global__ void CUDADifferentialTangenceFunction(CCUDAFunction<type_t> cCUDAFunction);//функция CUDA для применения производной функции гиперболический тангенс

//****************************************************************************************************
//класс применения функции сигмоид в CUDA
//****************************************************************************************************
template<class type_t>
class CCUDAFunction
{
 //-дружественные функции-------------------------------------------------------------------------------
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 public:
  //-переменные-----------------------------------------------------------------------------------------
  CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Input;//набор ходных данных
  CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Output;//набор выходных данных
  size_t MatrixAmount;//на сколько матриц создан набор
 private:
  public:
  //-конструктор----------------------------------------------------------------------------------------
  __host__ CCUDAFunction(size_t matrix_amount=0);
  //-деструктор-----------------------------------------------------------------------------------------
  __host__ ~CCUDAFunction();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  __host__ void Release(void);//очистить память

  __host__ void SetMatrixAmount(size_t matrix_amount);//задать количество матриц в наборе

  __host__ void ApplySigmoid(void);//применить функцию сигмоид
  __host__ void ApplyReLU(void);//применить функцию ReLU
  __host__ void ApplyLeakyReLU(void);//применить функцию Leaky ReLU
  __host__ void ApplyLinear(void);//применить линейную функцию
  __host__ void ApplyTangence(void);//применить функцию гиперболический тангенс

  __host__ void ApplyDifferentialSigmoid(void);//применить функцию производной от сигмоида
  __host__ void ApplyDifferentialReLU(void);//применить функцию производной от ReLU
  __host__ void ApplyDifferentialLeakyReLU(void);//применить функцию производной от Leaky ReLU
  __host__ void ApplyDifferentialLinear(void);//применить производную линейной функций
  __host__ void ApplyDifferentialTangence(void);//применить функцию производной от гиперболического тангенса

  __device__ void ApplySigmoidProcessing(size_t index);//процесс применения функции сигмоид
  __device__ void ApplyReLUProcessing(size_t index);//процесс применения функции ReLU
  __device__ void ApplyLeakyReLUProcessing(size_t index);//процесс применения функции Leaky ReLU
  __device__ void ApplyLinearProcessing(size_t index);//процесс применения линейной функции
  __device__ void ApplyTangenceProcessing(size_t index);//процесс применения функции гиперболический тангенс

  __device__ void ApplyDifferentialSigmoidProcessing(size_t index);//процесс применения производной функции сигмоид
  __device__ void ApplyDifferentialReLUProcessing(size_t index);//процесс применения производной функции ReLU
  __device__ void ApplyDifferentialLeakyReLUProcessing(size_t index);//процесс применения производной функции Leaky ReLU
  __device__ void ApplyDifferentialLinearProcessing(size_t index);//процесс применения производной линейной функции
  __device__ void ApplyDifferentialTangenceProcessing(size_t index);//процесс применения производной функции гиперболический тангенс
 private:
  //-закрытые функции-----------------------------------------------------------------------------------
  __host__ __device__ type_t Sigmoid(type_t v);//сигмоид
  __host__ __device__ type_t ReLU(type_t v);//ReLU
  __host__ __device__ type_t LeakyReLU(type_t v);//Leaky ReLU
  __host__ __device__ type_t Linear(type_t v);//линейная
  __host__ __device__ type_t Tangence(type_t v);//гиперболический тангенс

  __host__ __device__ type_t dSigmoid(type_t v);//производная сигмоида
  __host__ __device__ type_t dReLU(type_t v);//производная ReLU
  __host__ __device__ type_t dLeakyReLU(type_t v);//производная Leaky ReLU
  __host__ __device__ type_t dLinear(type_t v);//производная линейной функции
  __host__ __device__ type_t dTangence(type_t v);//производная гиперболического тангенса

  __host__ void Init(void);//инициализация
};
//****************************************************************************************************
//конструктор и деструктор класса
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ CCUDAFunction<type_t>::CCUDAFunction(size_t matrix_amount)
{
 MatrixAmount=matrix_amount;
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ CCUDAFunction<type_t>::~CCUDAFunction()
{
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//сигмоид
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CCUDAFunction<type_t>::Sigmoid(type_t v)
{
 if (v>20) v=19.9;
 if (v<-20) v=-19.9;
 return(1.0/(1.0+exp(-v)));
}
//----------------------------------------------------------------------------------------------------
//ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CCUDAFunction<type_t>::ReLU(type_t v)
{
 if (v>0) return(v);
 return(0);
}
//----------------------------------------------------------------------------------------------------
//Leaky ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CCUDAFunction<type_t>::LeakyReLU(type_t v)
{
 if (v>0) return(v);
 return(0.1*v);
}
//----------------------------------------------------------------------------------------------------
//линейная
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CCUDAFunction<type_t>::Linear(type_t v)
{
 return(v);
}
//----------------------------------------------------------------------------------------------------
//гиперболический тангенс
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CCUDAFunction<type_t>::Tangence(type_t v)
{
 if (v>20) v=19.9;
 if (v<-20) v=-19.9;
 type_t e=exp(2*v);
 return((e-1)/(e+1));
}
//----------------------------------------------------------------------------------------------------
//производная сигмоида
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CCUDAFunction<type_t>::dSigmoid(type_t v)
{
 type_t s=Sigmoid(v);
 return((1.0-s)*s);
}
//----------------------------------------------------------------------------------------------------
//производная ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CCUDAFunction<type_t>::dReLU(type_t v)
{
 if (v>=0) return(1);
 return(0);
}
//----------------------------------------------------------------------------------------------------
//производная Leaky ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CCUDAFunction<type_t>::dLeakyReLU(type_t v)
{
 if (v>=0) return(1);
 return(0.1);
}
//----------------------------------------------------------------------------------------------------
//производная линейной функции
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CCUDAFunction<type_t>::dLinear(type_t v)
{
 return(1);
}
//----------------------------------------------------------------------------------------------------
//производная гиперболического тангенса
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t CCUDAFunction<type_t>::dTangence(type_t v)
{
 type_t t=Tangence(v);
 return(1-t*t);
}

//----------------------------------------------------------------------------------------------------
//процесс применения функции сигмоид
//----------------------------------------------------------------------------------------------------
template<class type_t>
__device__ void CCUDAFunction<type_t>::ApplySigmoidProcessing(size_t index)
	{
 size_t input_x=cCUDAMatrixStorage_Input.GetSizeX();
 size_t input_y=cCUDAMatrixStorage_Input.GetSizeY();

 type_t *m_output_ptr=cCUDAMatrixStorage_Output.GetItemPtr(index);
 type_t *m_input_ptr=cCUDAMatrixStorage_Input.GetItemPtr(index);

 for(size_t y=0;y<input_y;y++)
 {
  for(size_t x=0;x<input_x;x++,m_output_ptr++,m_input_ptr++)
  {
   type_t value=*m_input_ptr;
   *m_output_ptr=Sigmoid(value);
  }
 }
}
//----------------------------------------------------------------------------------------------------
//процесс применения функции ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__device__ void CCUDAFunction<type_t>::ApplyReLUProcessing(size_t index)
{
 size_t input_x=cCUDAMatrixStorage_Input.GetSizeX();
 size_t input_y=cCUDAMatrixStorage_Input.GetSizeY();

 type_t *m_output_ptr=cCUDAMatrixStorage_Output.GetItemPtr(index);
 type_t *m_input_ptr=cCUDAMatrixStorage_Input.GetItemPtr(index);

 for(size_t y=0;y<input_y;y++)
 {
  for(size_t x=0;x<input_x;x++,m_output_ptr++,m_input_ptr++)
  {
   type_t value=*m_input_ptr;
   *m_output_ptr=ReLU(value);
  }
 }
}

//----------------------------------------------------------------------------------------------------
//процесс применения функции Leaky ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__device__ void CCUDAFunction<type_t>::ApplyLeakyReLUProcessing(size_t index)
{
 size_t input_x=cCUDAMatrixStorage_Input.GetSizeX();
 size_t input_y=cCUDAMatrixStorage_Input.GetSizeY();

 type_t *m_output_ptr=cCUDAMatrixStorage_Output.GetItemPtr(index);
 type_t *m_input_ptr=cCUDAMatrixStorage_Input.GetItemPtr(index);

 for(size_t y=0;y<input_y;y++)
 {
  for(size_t x=0;x<input_x;x++,m_output_ptr++,m_input_ptr++)
  {
   type_t value=*m_input_ptr;
   *m_output_ptr=LeakyReLU(value);
  }
 }
}

//----------------------------------------------------------------------------------------------------
//процесс применения линейной функции
//----------------------------------------------------------------------------------------------------
template<class type_t>
__device__ void CCUDAFunction<type_t>::ApplyLinearProcessing(size_t index)
{
 size_t input_x=cCUDAMatrixStorage_Input.GetSizeX();
 size_t input_y=cCUDAMatrixStorage_Input.GetSizeY();

 type_t *m_output_ptr=cCUDAMatrixStorage_Output.GetItemPtr(index);
 type_t *m_input_ptr=cCUDAMatrixStorage_Input.GetItemPtr(index);

 for(size_t y=0;y<input_y;y++)
 {
  for(size_t x=0;x<input_x;x++,m_output_ptr++,m_input_ptr++)
  {
   type_t value=*m_input_ptr;
   *m_output_ptr=Linear(value);
  }
 }
}
//----------------------------------------------------------------------------------------------------
//процесс применения функции гиперболический тангенс
//----------------------------------------------------------------------------------------------------
template<class type_t>
__device__ void CCUDAFunction<type_t>::ApplyTangenceProcessing(size_t index)
{
 size_t input_x=cCUDAMatrixStorage_Input.GetSizeX();
 size_t input_y=cCUDAMatrixStorage_Input.GetSizeY();

 type_t *m_output_ptr=cCUDAMatrixStorage_Output.GetItemPtr(index);
 type_t *m_input_ptr=cCUDAMatrixStorage_Input.GetItemPtr(index);

 for(size_t y=0;y<input_y;y++)
 {
  for(size_t x=0;x<input_x;x++,m_output_ptr++,m_input_ptr++)
  {
   type_t value=*m_input_ptr;
   *m_output_ptr=Tangence(value);
  }
 }
}
//----------------------------------------------------------------------------------------------------
//процесс применения производной функции сигмоид
//----------------------------------------------------------------------------------------------------
template<class type_t>
__device__ void CCUDAFunction<type_t>::ApplyDifferentialSigmoidProcessing(size_t index)
{
 size_t input_x=cCUDAMatrixStorage_Input.GetSizeX();
 size_t input_y=cCUDAMatrixStorage_Input.GetSizeY();

 type_t *m_output_ptr=cCUDAMatrixStorage_Output.GetItemPtr(index);
 type_t *m_input_ptr=cCUDAMatrixStorage_Input.GetItemPtr(index);

 for(size_t y=0;y<input_y;y++)
 {
  for(size_t x=0;x<input_x;x++,m_output_ptr++,m_input_ptr++)
  {
   type_t value=*m_input_ptr;
   *m_output_ptr=dSigmoid(value);
  }
 }
}
//----------------------------------------------------------------------------------------------------
//процесс применения производной функции ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__device__ void CCUDAFunction<type_t>::ApplyDifferentialReLUProcessing(size_t index)
{
 size_t input_x=cCUDAMatrixStorage_Input.GetSizeX();
 size_t input_y=cCUDAMatrixStorage_Input.GetSizeY();

 type_t *m_output_ptr=cCUDAMatrixStorage_Output.GetItemPtr(index);
 type_t *m_input_ptr=cCUDAMatrixStorage_Input.GetItemPtr(index);

 for(size_t y=0;y<input_y;y++)
 {
  for(size_t x=0;x<input_x;x++,m_output_ptr++,m_input_ptr++)
  {
   type_t value=*m_input_ptr;
   *m_output_ptr=dReLU(value);
  }
 }
}
//----------------------------------------------------------------------------------------------------
//процесс применения производной функции Leaky ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__device__ void CCUDAFunction<type_t>::ApplyDifferentialLeakyReLUProcessing(size_t index)
{
 size_t input_x=cCUDAMatrixStorage_Input.GetSizeX();
 size_t input_y=cCUDAMatrixStorage_Input.GetSizeY();

 type_t *m_output_ptr=cCUDAMatrixStorage_Output.GetItemPtr(index);
 type_t *m_input_ptr=cCUDAMatrixStorage_Input.GetItemPtr(index);

 for(size_t y=0;y<input_y;y++)
 {
  for(size_t x=0;x<input_x;x++,m_output_ptr++,m_input_ptr++)
  {
   type_t value=*m_input_ptr;
   *m_output_ptr=dLeakyReLU(value);
  }
 }
}
//----------------------------------------------------------------------------------------------------
//процесс применения производной линейной функции
//----------------------------------------------------------------------------------------------------
template<class type_t>
__device__ void CCUDAFunction<type_t>::ApplyDifferentialLinearProcessing(size_t index)
	{
 size_t input_x=cCUDAMatrixStorage_Input.GetSizeX();
 size_t input_y=cCUDAMatrixStorage_Input.GetSizeY();

 type_t *m_output_ptr=cCUDAMatrixStorage_Output.GetItemPtr(index);
 type_t *m_input_ptr=cCUDAMatrixStorage_Input.GetItemPtr(index);

 for(size_t y=0;y<input_y;y++)
 {
  for(size_t x=0;x<input_x;x++,m_output_ptr++,m_input_ptr++)
  {
   type_t value=*m_input_ptr;
   *m_output_ptr=dLinear(value);
  }
 }
}
//----------------------------------------------------------------------------------------------------
//процесс применения производной функции гиперболический тангенс
//----------------------------------------------------------------------------------------------------
template<class type_t>
__device__ void CCUDAFunction<type_t>::ApplyDifferentialTangenceProcessing(size_t index)
{
 size_t input_x=cCUDAMatrixStorage_Input.GetSizeX();
 size_t input_y=cCUDAMatrixStorage_Input.GetSizeY();

 type_t *m_output_ptr=cCUDAMatrixStorage_Output.GetItemPtr(index);
 type_t *m_input_ptr=cCUDAMatrixStorage_Input.GetItemPtr(index);

 for(size_t y=0;y<input_y;y++)
 {
  for(size_t x=0;x<input_x;x++,m_output_ptr++,m_input_ptr++)
  {
   type_t value=*m_input_ptr;
   *m_output_ptr=dTangence(value);
  }
 }
}
//----------------------------------------------------------------------------------------------------
//инициализация
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::Init(void)
{
 if (cCUDAMatrixStorage_Input.GetAmount()!=MatrixAmount) throw "CUDAFunction<type_t>::Init: количество матриц в наборе изображений должно быть равно количеству матриц, для которого создавался класс";
 //задаём выходную матрицу
 cCUDAMatrixStorage_Output.Release();
 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage(cCUDAMatrixStorage_Input.GetSizeY(),cCUDAMatrixStorage_Input.GetSizeX(),MatrixAmount);
 cCUDAMatrixStorage.Create();
 cCUDAMatrixStorage_Output.Move(cCUDAMatrixStorage);
}
 //****************************************************************************************************
//открытые функции класса
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//очистить память
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::Release(void)
{
 cCUDAMatrixStorage_Input.Release();
 cCUDAMatrixStorage_Output.Release();
 MatrixAmount=0;
}

//----------------------------------------------------------------------------------------------------
//задать количество матриц в наборе
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::SetMatrixAmount(size_t matrix_amount)
{
 MatrixAmount=matrix_amount;
}

//----------------------------------------------------------------------------------------------------
//применить функцию сигмоид
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::ApplySigmoid(void)
{
 Init();
 CUDASigmoidFunction<<<MatrixAmount,1>>>(*this);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}
//----------------------------------------------------------------------------------------------------
//применить функцию ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::ApplyReLU(void)
{
 Init();
 CUDAReLUFunction<<<MatrixAmount,1>>>(*this);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}
//----------------------------------------------------------------------------------------------------
//применить функцию Leaky ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::ApplyLeakyReLU(void)
{
 Init();
 CUDALeakyReLUFunction<<<MatrixAmount,1>>>(*this);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}
//----------------------------------------------------------------------------------------------------
//применить линейную функцию
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::ApplyLinear(void)
{
 Init();
 CUDALinearFunction<<<MatrixAmount,1>>>(*this);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}
//----------------------------------------------------------------------------------------------------
//применить функцию гиперболический тангенс
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::ApplyTangence(void)
{
 Init();
 CUDATangenceFunction<<<MatrixAmount,1>>>(*this);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}
//----------------------------------------------------------------------------------------------------
//применить функцию производной от сигмоида
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::ApplyDifferentialSigmoid(void)
{
 Init();
 CUDADifferentialSigmoidFunction<<<MatrixAmount,1>>>(*this);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}
//----------------------------------------------------------------------------------------------------
//применить функцию производной от ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::ApplyDifferentialReLU(void)
{
 Init();
 CUDADifferentialReLUFunction<<<MatrixAmount,1>>>(*this);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}
//----------------------------------------------------------------------------------------------------
//применить функцию производной от Leaky ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::ApplyDifferentialLeakyReLU(void)
{
 Init();
 CUDADifferentialLeakyReLUFunction<<<MatrixAmount,1>>>(*this);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}
//----------------------------------------------------------------------------------------------------
//применить производную линейной функций
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::ApplyDifferentialLinear(void)
{
 Init();
 CUDADifferentialLinearFunction<<<MatrixAmount,1>>>(*this);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}
//----------------------------------------------------------------------------------------------------
//применить функцию производной от гиперболического тангенса
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAFunction<type_t>::ApplyDifferentialTangence(void)
{
 Init();
 CUDADifferentialTangenceFunction<<<MatrixAmount,1>>>(*this);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}
//****************************************************************************************************
//прочее
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//функция CUDA для применения функции сигмоид
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDASigmoidFunction(CCUDAFunction<type_t> cCUDAFunction)
{
 size_t y=blockIdx.x;
// size_t x=threadIdx.x;
 size_t index=y;
 cCUDAFunction.ApplySigmoidProcessing(index);
}
//----------------------------------------------------------------------------------------------------
//функция CUDA для применения линейной функции
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDALinearFunction(CCUDAFunction<type_t> cCUDAFunction)
{
 size_t y=blockIdx.x;
// size_t x=threadIdx.x;
 size_t index=y;
 cCUDAFunction.ApplySigmoidProcessing(index);
}
//----------------------------------------------------------------------------------------------------
//функция CUDA для применения функции ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAReLUFunction(CCUDAFunction<type_t> cCUDAFunction)
{
 size_t y=blockIdx.x;
// size_t x=threadIdx.x;
 size_t index=y;
 cCUDAFunction.ApplyReLUProcessing(index);
}
//----------------------------------------------------------------------------------------------------
//функция CUDA для применения функции Leaky ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDALeakyReLUFunction(CCUDAFunction<type_t> cCUDAFunction)
{
 size_t y=blockIdx.x;
// size_t x=threadIdx.x;
 size_t index=y;
 cCUDAFunction.ApplyLeakyReLUProcessing(index);
}
//----------------------------------------------------------------------------------------------------
//функция CUDA для применения функции гиперболический тангенс
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATangenceFunction(CCUDAFunction<type_t> cCUDAFunction)
{
 size_t y=blockIdx.x;
// size_t x=threadIdx.x;
 size_t index=y;
 cCUDAFunction.ApplyTangenceProcessing(index);
}
//----------------------------------------------------------------------------------------------------
//функция CUDA для применения производной функции сигмоид
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDADifferentialSigmoidFunction(CCUDAFunction<type_t> cCUDAFunction)
{
 size_t y=blockIdx.x;
// size_t x=threadIdx.x;
 size_t index=y;
 cCUDAFunction.ApplyDifferentialSigmoidProcessing(index);
}
//----------------------------------------------------------------------------------------------------
//функция CUDA для применения производной линейной функции
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDADifferentialLinearFunction(CCUDAFunction<type_t> cCUDAFunction)
{
 size_t y=blockIdx.x;
// size_t x=threadIdx.x;
 size_t index=y;
 cCUDAFunction.ApplyDifferentialLinearProcessing(index);
}
//----------------------------------------------------------------------------------------------------
//функция CUDA для применения производной функции ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDADifferentialReLUFunction(CCUDAFunction<type_t> cCUDAFunction)
{
 size_t y=blockIdx.x;
// size_t x=threadIdx.x;
 size_t index=y;
 cCUDAFunction.ApplyDifferentialReLUProcessing(index);
}
//----------------------------------------------------------------------------------------------------
//функция CUDA для применения производной функции Leaky ReLU
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDADifferentialLeakyReLUFunction(CCUDAFunction<type_t> cCUDAFunction)
{
 size_t y=blockIdx.x;
// size_t x=threadIdx.x;
 size_t index=y;
 cCUDAFunction.ApplyDifferentialLeakyReLUProcessing(index);
}
//----------------------------------------------------------------------------------------------------
//функция CUDA для применения производной функции гиперболический тангенс
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDADifferentialTangenceFunction(CCUDAFunction<type_t> cCUDAFunction)
{
 size_t y=blockIdx.x;
// size_t x=threadIdx.x;
 size_t index=y;
 cCUDAFunction.ApplyDifferentialTangenceProcessing(index);
}

#endif

#ifndef C_CUDA_MAX_DE_POOLING_H
#define C_CUDA_MAX_DE_POOLING_H

//****************************************************************************************************
//Класс выполнения обратной обратной субдискретизации в CUDA
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
class CCUDAMaxDePooling;

template<class type_t>
__global__ void CUDAMaxDePoolingFunction(CCUDAMaxDePooling<type_t> cCUDAMaxDePooling,size_t image_width,size_t image_height,size_t output_width,size_t output_height);//функция CUDA для вычисления обратной субдискретизации

//****************************************************************************************************
//класс выполнения обратной обратной субдискретизации в CUDA
//****************************************************************************************************
template<class type_t>
class CCUDAMaxDePooling
{
 //-дружественные функции-------------------------------------------------------------------------------
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 public:
  //-переменные-----------------------------------------------------------------------------------------
  CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Input;//входное изображение
  CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Output;//выход
  CCUDAMatrixStorage<size_t> cCUDAMatrixStorage_InputIndex;//индекс входного изображения
  size_t MatrixAmount;//на сколько матриц создан набор
 private:
  public:
  //-конструктор----------------------------------------------------------------------------------------
  __host__ CCUDAMaxDePooling(size_t matrix_amount=0);
  //-деструктор-----------------------------------------------------------------------------------------
  __host__ ~CCUDAMaxDePooling();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  __host__ void Release(void);//очистить память
  __host__ void SetMatrixAmount(size_t matrix_amount);//задать количество матриц в наборе
  __host__ void MaxDePooling(size_t image_width,size_t image_height,size_t output_width,size_t output_height);//выполнить обратное прореживание
  __device__ void MaxDePoolingProcessing(size_t image_index,size_t image_width,size_t image_height,size_t output_width,size_t output_height);//процесс прореживания
 private:
  //-закрытые функции-----------------------------------------------------------------------------------
};
//****************************************************************************************************
//конструктор и деструктор класса
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ CCUDAMaxDePooling<type_t>::CCUDAMaxDePooling(size_t matrix_amount)
{
 MatrixAmount=matrix_amount;
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ CCUDAMaxDePooling<type_t>::~CCUDAMaxDePooling()
{
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//процесс обратного прореживания
//----------------------------------------------------------------------------------------------------
template<class type_t>
__device__ void CCUDAMaxDePooling<type_t>::MaxDePoolingProcessing(size_t image_index,size_t image_width,size_t image_height,size_t output_width,size_t output_height)
{
 //для каждого изображения применяем обратное прореживание
 size_t output_index=image_index;
 size_t input_index=image_index;

 type_t *output_ptr=cCUDAMatrixStorage_Output.GetItemPtr(output_index);//определяем куда поместить результат
 type_t *image_ptr=cCUDAMatrixStorage_Input.GetItemPtr(image_index);//выбираем строку с изображением
 size_t *input_index_ptr=cCUDAMatrixStorage_InputIndex.GetItemPtr(input_index);//определяем откуда взять индекс выбранных точек
 //очищаем матрицу
 type_t *o_ptr=output_ptr;
 for(size_t n=0;n<output_width*output_height;n++,o_ptr++) *o_ptr=0;
 //задаём значения
 for(size_t n=0;n<image_width*image_height;n++,input_index_ptr++,image_ptr++)
 {
  size_t index=*input_index_ptr;
  type_t value=*image_ptr;
  *(output_ptr+index)=value;
 }
}

//****************************************************************************************************
//открытые функции класса
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//очистить память
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAMaxDePooling<type_t>::Release(void)
{
 cCUDAMatrixStorage_Input.Release();
 cCUDAMatrixStorage_Output.Release();
 MatrixAmount=0;
}

//----------------------------------------------------------------------------------------------------
//задать количество матриц в наборе
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAMaxDePooling<type_t>::SetMatrixAmount(size_t matrix_amount)
{
 MatrixAmount=matrix_amount;
}

//----------------------------------------------------------------------------------------------------
//выполнить обратную субдискретизацию
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAMaxDePooling<type_t>::MaxDePooling(size_t image_width,size_t image_height,size_t output_width,size_t output_height)
{
 double begin_time=GetSecondCounter();

 if (cCUDAMatrixStorage_Input.GetAmount()!=MatrixAmount) throw "CCUDAMaxDePooling<type_t>::MaxDePooling: количество матриц в наборе изображений должно быть равно количеству матриц, для которого создавался класс";
 if (cCUDAMatrixStorage_Input.GetAmount()!=cCUDAMatrixStorage_InputIndex.GetAmount()) throw "CCUDAMaxDePooling<type_t>::MaxDePooling: количество матриц должно быть одинаково";
 if (cCUDAMatrixStorage_Input.GetSizeX()!=cCUDAMatrixStorage_InputIndex.GetSizeX() || cCUDAMatrixStorage_Input.GetSizeY()!=cCUDAMatrixStorage_InputIndex.GetSizeY()) throw "CCUDAMaxDePooling<type_t>::MaxDePooling: размеры матриц индексов и входных данных должны быть одинаковы";
 if (cCUDAMatrixStorage_Input.GetSizeY()!=1) throw "CCUDAMaxDePooling<type_t>::MaxDePooling: высота матриц должна быть равна 1";

 if (image_width*image_height!=cCUDAMatrixStorage_Input.GetSizeX()) throw "CCUDAMaxDePooling<type_t>::MaxDePooling: размеры входного изображения должны соответствовать матрицам изображения и индексов";

 size_t image_amount=cCUDAMatrixStorage_Input.GetSizeY();//количество изображений в одной матрице
 //задаём выходную матрицу
 cCUDAMatrixStorage_Output.Release();
 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage(1,output_height*output_width,MatrixAmount*image_amount);
 cCUDAMatrixStorage.Create();
 cCUDAMatrixStorage_Output.Move(cCUDAMatrixStorage);
 //выполняем обратное прореживание
 CUDAMaxDePoolingFunction<<<MatrixAmount*image_amount,1>>>(*this,image_width,image_height,output_width,output_height);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 double delta_t=GetSecondCounter()-begin_time;
 //printf("MaxDePooling: %.4f second\r\n",delta_t);
}

//****************************************************************************************************
//прочее
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//функция CUDA для вычисления обратной субдискретизации
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAMaxDePoolingFunction(CCUDAMaxDePooling<type_t> cCUDAMaxDePooling,size_t image_width,size_t image_height,size_t output_width,size_t output_height)
{
 size_t image_index=blockIdx.x;
 cCUDAMaxDePooling.MaxDePoolingProcessing(image_index,image_width,image_height,output_width,output_height);
}

#endif

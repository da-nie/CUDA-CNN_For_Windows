#ifndef C_CUDA_MAX_POOLING_H
#define C_CUDA_MAX_POOLING_H

//****************************************************************************************************
//Класс выполнения субдискретизации в CUDA
//****************************************************************************************************

#include <stdio.h>
#include <stdint.h>

#include "../handle_error.cu.h"
#include "../ccudamatrixstorage.cu.h"
#include "../ccudatimespent.cu.h"
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
class CCUDAMaxPooling;

template<class type_t>
__global__ void CUDAMaxPoolingFunction(CCUDAMaxPooling<type_t> cCUDAMaxPooling,size_t image_width,size_t image_height,size_t pooling_width,size_t pooling_height);//функция CUDA для вычисления субдискретизации

//****************************************************************************************************
//класс выполнения субдискретизации в CUDA
//****************************************************************************************************
template<class type_t>
class CCUDAMaxPooling
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
  CCUDAMatrixStorage<size_t> cCUDAMatrixStorage_OutputIndex;//индекс входного изображения
  size_t MatrixAmount;//на сколько матриц создан набор
 private:
  public:
  //-конструктор----------------------------------------------------------------------------------------
  __host__ CCUDAMaxPooling(size_t matrix_amount=0);
  //-деструктор-----------------------------------------------------------------------------------------
  __host__ ~CCUDAMaxPooling();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  __host__ void Release(void);//очистить память
  __host__ void SetMatrixAmount(size_t matrix_amount);//задать количество матриц в наборе
  __host__ void MaxPooling(size_t image_width,size_t image_height,size_t pooling_width,size_t pooling_height,size_t &output_width,size_t &output_height);//выполнить прореживание
  __device__ void MaxPoolingProcessing(size_t image_index,size_t image_width,size_t image_height,size_t pooling_width,size_t pooling_height);//процесс прореживания
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
__host__ CCUDAMaxPooling<type_t>::CCUDAMaxPooling(size_t matrix_amount)
{
 MatrixAmount=matrix_amount;
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ CCUDAMaxPooling<type_t>::~CCUDAMaxPooling()
{
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//процесс прореживания
//----------------------------------------------------------------------------------------------------
template<class type_t>
__device__ void CCUDAMaxPooling<type_t>::MaxPoolingProcessing(size_t image_index,size_t image_width,size_t image_height,size_t pooling_width,size_t pooling_height)
{
 size_t output_width=image_width/pooling_width;
 size_t output_height=image_height/pooling_height;

 //для каждого изображения применяем прореживание
 size_t output_index=image_index;
 type_t *output_ptr=cCUDAMatrixStorage_Output.GetItemPtr(output_index);//определяем куда поместить результат
 type_t *image_ptr=cCUDAMatrixStorage_Input.GetItemPtr(image_index);//выбираем строку с изображением
 size_t *output_index_ptr=cCUDAMatrixStorage_OutputIndex.GetItemPtr(output_index);//определяем куда поместить индекс выбранных точек
 //применяем слой субдискретизации (pooling) по максимальному сигналу
 type_t *p_z_ptr=output_ptr;
 type_t *c_h_y_ptr=image_ptr;
 size_t *p_i_ptr=output_index_ptr;
 size_t c_y_offset=0;
 for(size_t y=0;y<output_height;y++,c_h_y_ptr+=image_width*pooling_height,c_y_offset+=image_width*pooling_height)
 {
  type_t *c_h_x_ptr=c_h_y_ptr;
  size_t c_x_offset=c_y_offset;
  for(size_t x=0;x<output_width;x++,p_z_ptr++,c_h_x_ptr+=pooling_width,p_i_ptr++,c_x_offset+=pooling_width)
  {
   //ищем максимум в блоке субдискретизации
   type_t *c_h_window_y_ptr=c_h_x_ptr;
   size_t c_window_y_offset=c_x_offset;
   type_t max_h=*c_h_window_y_ptr;//максимальное значение в блоке субдискретизации
   size_t max_pos=c_window_y_offset;//индекс максимального элемента
   for(size_t yp=0;yp<pooling_height;yp++,c_h_window_y_ptr+=image_width,c_window_y_offset+=image_width)
   {
    type_t *c_h_window_x_ptr=c_h_window_y_ptr;
	size_t c_window_x_offset=c_window_y_offset;
    for(size_t xp=0;xp<pooling_width;xp++,c_h_window_x_ptr++,c_window_x_offset++)
    {
     type_t value_h=*c_h_window_x_ptr;
     if (max_h<value_h)
	 {
	  max_h=value_h;
	  max_pos=c_window_x_offset;
	 }
    }
   }
   //сохраняем элемент
   *p_z_ptr=max_h;
   *p_i_ptr=max_pos;
  }
 }
}

//****************************************************************************************************
//открытые функции класса
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//очистить память
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAMaxPooling<type_t>::Release(void)
{
 cCUDAMatrixStorage_Input.Release();
 cCUDAMatrixStorage_Output.Release();
 MatrixAmount=0;
}

//----------------------------------------------------------------------------------------------------
//задать количество матриц в наборе
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAMaxPooling<type_t>::SetMatrixAmount(size_t matrix_amount)
{
 MatrixAmount=matrix_amount;
}

//----------------------------------------------------------------------------------------------------
//выполнить субдискретизацию
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAMaxPooling<type_t>::MaxPooling(size_t image_width,size_t image_height,size_t pooling_width,size_t pooling_height,size_t &output_width,size_t &output_height)
{
 if (cCUDAMatrixStorage_Input.GetAmount()!=MatrixAmount) throw "CCUDAMaxPooling<type_t>::MaxPooling: количество матриц в наборе изображений должно быть равно количеству матриц, для которого создавался класс";
 if (pooling_width==0 || pooling_height==0) throw "CCUDAMaxPooling<type_t>::MaxPooling: ширина и высота субдискретизации не могут быть нулевыми";
 if (cCUDAMatrixStorage_Input.GetSizeY()!=1) throw "CCUDAMaxPooling<type_t>::MaxPooling: высота входного изображения должна быть равна 1";
 if (cCUDAMatrixStorage_Input.GetSizeX()*cCUDAMatrixStorage_Input.GetSizeY()!=image_width*image_height) throw "CCUDAMaxPooling<type_t>::MaxPooling: количество элементов изображения должно быть равно размеру изображения";

 output_width=image_width/pooling_width;
 output_height=image_height/pooling_height;
 if (output_width==0 || output_height==0) throw "CCUDAMaxPooling<type_t>::MaxPooling: с заданными параметрами субдискретизации выходное изображение имеет нулевые размеры";

 size_t image_amount=cCUDAMatrixStorage_Input.GetSizeY();//количество изображений в одной матрице
 //задаём выходную матрицу
 cCUDAMatrixStorage_Output.Release();
 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage(1,output_height*output_width,MatrixAmount*image_amount);
 cCUDAMatrixStorage.Create();
 cCUDAMatrixStorage_Output.Move(cCUDAMatrixStorage);

 cCUDAMatrixStorage_OutputIndex.Release();
 CCUDAMatrixStorage<size_t> cCUDAMatrixStorage_Index(1,output_height*output_width,MatrixAmount*image_amount);
 cCUDAMatrixStorage_Index.Create();
 cCUDAMatrixStorage_OutputIndex.Move(cCUDAMatrixStorage_Index);

 CCUDATimeSpent cCUDATimeSpent;
 cCUDATimeSpent.Start();

 //выполняем субдискретизацию
 CUDAMaxPoolingFunction<<<MatrixAmount*image_amount,1>>>(*this,image_width,image_height,pooling_width,pooling_height);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 float gpu_time=cCUDATimeSpent.Stop();
 char str[255];
 sprintf(str,"MaxPooling: %.2f millisecond\r\n",gpu_time);
 //PutMessageToConsole(str);
}

//****************************************************************************************************
//прочее
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//функция CUDA для вычисления субдискретизации
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAMaxPoolingFunction(CCUDAMaxPooling<type_t> cCUDAMaxPooling,size_t image_width,size_t image_height,size_t pooling_width,size_t pooling_height)
{
 size_t image_index=blockIdx.x;
 cCUDAMaxPooling.MaxPoolingProcessing(image_index,image_width,image_height,pooling_width,pooling_height);
}

#endif

#ifndef C_CUDA_BACK_DE_CONVOLUTION_H
#define C_CUDA_BACK_DE_CONVOLUTION_H

//****************************************************************************************************
//Класс выполнения обратной свёртки в CUDA
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
class CCUDABackDeConvolution;

template<class type_t>
__global__ void CUDABackDeConvolutionFunction(CCUDABackDeConvolution<type_t> cCUDABackDeConvolution,size_t delta_width,size_t delta_height,size_t kernel_width,size_t kernel_height,size_t delta_depth,size_t kernel_depth,size_t kernel_amount);//функция CUDA для вычисления свёртки

//****************************************************************************************************
//класс выполнения обратной свёртки в CUDA
//****************************************************************************************************
template<class type_t>
class CCUDABackDeConvolution
{
 //-дружественные функции-------------------------------------------------------------------------------
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 public:
  //-переменные-----------------------------------------------------------------------------------------
  CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Kernel;//набор ядер
  CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Delta;//набор образов
  CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Output;//набор выходных данных
 private:
  public:
  //-конструктор----------------------------------------------------------------------------------------
  __host__ CCUDABackDeConvolution(void);
  //-деструктор-----------------------------------------------------------------------------------------
  __host__ ~CCUDABackDeConvolution();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  __host__ void Release(void);//очистить память
  __host__ void BackDeConvolution(size_t delta_width,size_t delta_height,size_t kernel_width,size_t kernel_height,size_t &output_width,size_t &output_height);//выполнить свёртку
  __device__ void BackDeConvolutionProcessing(size_t delta_index,size_t kernel_depth_index,size_t delta_width,size_t delta_height,size_t kernel_width,size_t kernel_height,size_t delta_depth,size_t kernel_depth,size_t kernel_amount,size_t x,size_t y);//процесс рассчёта свёртки
  __host__ static void Test(void);//протестировать класс
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
__host__ CCUDABackDeConvolution<type_t>::CCUDABackDeConvolution(void)
{
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ CCUDABackDeConvolution<type_t>::~CCUDABackDeConvolution()
{
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//процесс рассчёта свёртки
//----------------------------------------------------------------------------------------------------
template<class type_t>
__device__ void CCUDABackDeConvolution<type_t>::BackDeConvolutionProcessing(size_t delta_index,size_t kernel_depth_index,size_t delta_width,size_t delta_height,size_t kernel_width,size_t kernel_height,size_t delta_depth,size_t kernel_depth,size_t kernel_amount,size_t x,size_t y)
{
 size_t output_width=delta_width+kernel_width-1;
 size_t output_height=delta_height+kernel_height-1;

 type_t *delta_ptr=cCUDAMatrixStorage_Delta.GetItemPtr(delta_index);
 type_t *output_ptr=cCUDAMatrixStorage_Output.GetItemPtr(delta_index)+kernel_depth_index*output_width*output_height+y*output_width+x;

 size_t padding=0;
 size_t step=1;
 padding=kernel_width-1-padding;

 size_t kernel_depth_offset=kernel_depth_index*kernel_width*kernel_height;
 type_t sum=0;//сумма для градиента
 //идём по всем весовым коэффициентам фильтров
 for(size_t i=0;i<kernel_height;i++)
 {
  int32_t i0=static_cast<int32_t>(y+i);
  i0-=static_cast<int32_t>(padding);
  if (i0<0 || i0>=delta_height) continue;
  for(size_t j=0;j<kernel_width;j++)
  {
   size_t j0=static_cast<int32_t>(x+j);
   j0-=static_cast<int32_t>(padding);
   //игнорируем выходящие за границы элементы
   if (j0<0 || j0>=delta_width) continue;
   //суммируем по всем фильтрам
   size_t offset_k_ptr=(kernel_height-1-i)*kernel_width+(kernel_width-1-j)+kernel_depth_offset;
   size_t offset_d_ptr=i0*delta_width+j0;
   for(size_t k=0;k<kernel_amount;k++)
   {
    type_t *d_ptr=delta_ptr+k*delta_width*delta_height+offset_d_ptr;
	type_t *kernel_ptr=cCUDAMatrixStorage_Kernel.GetItemPtr(k);
    type_t *k_ptr=kernel_ptr+offset_k_ptr;
    sum+=(*k_ptr)*(*d_ptr);//добавляем произведение повёрнутых фильтров на дельты
   }
  }
 }
 *output_ptr=sum;//записываем результат в тензор градиента
}

//****************************************************************************************************
//открытые функции класса
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//очистить память
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDABackDeConvolution<type_t>::Release(void)
{
 cCUDAMatrixStorage_Kernel.Release();
 cCUDAMatrixStorage_Delta.Release();
 cCUDAMatrixStorage_Output.Release();
}

//----------------------------------------------------------------------------------------------------
//выполнить свёртку
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDABackDeConvolution<type_t>::BackDeConvolution(size_t delta_width,size_t delta_height,size_t kernel_width,size_t kernel_height,size_t &output_width,size_t &output_height)
{
 double begin_time=GetSecondCounter();

 if (cCUDAMatrixStorage_Kernel.GetSizeX()!=kernel_width*kernel_height) throw "CCUDABackDeConvolution<type_t>::DeConvolution: ширина матрицы ядер должна соответствовать количеству элементов одного ядра";
 if (cCUDAMatrixStorage_Delta.GetSizeX()!=delta_width*delta_height) throw "CCUDABackDeConvolution<type_t>::DeConvolution: ширина матрицы дельт должна соответствовать количеству элементов одной дельты";
 //параметры свёртки
 size_t delta_depth=cCUDAMatrixStorage_Delta.GetSizeY();//глубина дельт в одной матрице (соответствует количеству ядер)
 size_t delta_amount=cCUDAMatrixStorage_Delta.GetAmount();//количество дельт (соответствует количеству изображений)
 size_t kernel_depth=cCUDAMatrixStorage_Kernel.GetSizeY();//глубина ядер в одной матрице
 size_t kernel_amount=cCUDAMatrixStorage_Kernel.GetAmount();//количество ядер

 if (delta_depth!=kernel_amount) throw "CCUDABackDeConvolution<type_t>::DeConvolution: глубина матрицы дельт должна быть равна количеству ядер";

 output_width=delta_width+kernel_width-1;
 output_height=delta_height+kernel_height-1;

 //задаём выходную матрицу
 cCUDAMatrixStorage_Output.Release();
 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_A(kernel_depth,output_height*output_width,delta_amount);
 cCUDAMatrixStorage_A.Create();
 cCUDAMatrixStorage_Output.Move(cCUDAMatrixStorage_A);

 CCUDATimeSpent cCUDATimeSpent;
 cCUDATimeSpent.Start();

 //выполняем свёртку
 dim3 grid(delta_amount*kernel_depth,output_height);
 CUDABackDeConvolutionFunction<<<grid,output_width>>>(*this,delta_width,delta_height,kernel_width,kernel_height,delta_depth,kernel_depth,kernel_amount);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 float gpu_time=cCUDATimeSpent.Stop();
 char str[255];
 sprintf(str,"BackDeConvolution: %.4f millisecond\r\n",gpu_time);
 //PutMessageToConsole(str);
}

//----------------------------------------------------------------------------------------------------
//протестировать класс
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDABackDeConvolution<type_t>::Test(void)
{
 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Delta(2,2*2,2);
 cCUDAMatrixStorage_Delta.Create();
 type_t delta_1[]={2,1,4,4, 2*2,1*2,4*2,4*2};
 type_t delta_2[]={2*3,1*3,4*3,4*3, 2*2*3,1*2*3,4*2*3,4*2*3};
 cCUDAMatrixStorage_Delta.Set(0,delta_1);
 cCUDAMatrixStorage_Delta.Set(1,delta_2);

 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Kernel(2,3*3,2);
 cCUDAMatrixStorage_Kernel.Create();
 type_t kernel_1[]={1,4,1, 1,4,3, 3,3,1,  1*3,4*3,1*3, 1*3,4*3,3*3, 3*3,3*3,1*3};
 type_t kernel_2[]={1*7,4*7,1*7, 1*7,4*7,3*7, 3*7,3*7,1*7,  1*3*7,4*3*7,1*3*7, 1*3*7,4*3*7,3*3*7, 3*3*7,3*3*7,1*3*7};
 cCUDAMatrixStorage_Kernel.Set(0,kernel_1);
 cCUDAMatrixStorage_Kernel.Set(1,kernel_2);

 CCUDABackDeConvolution<type_t> cCUDABackDeConvolution_A;
 //подключаемся к ядрам
 cCUDABackDeConvolution_A.cCUDAMatrixStorage_Delta.Connect(cCUDAMatrixStorage_Delta);
 //подключаемся к части исходных данных
 cCUDABackDeConvolution_A.cCUDAMatrixStorage_Kernel.Connect(cCUDAMatrixStorage_Kernel);
 //выполняем свёртку
 size_t backward_de_conv_a_width;
 size_t backward_de_conv_a_height;
 cCUDABackDeConvolution_A.BackDeConvolution(2,2,3,3,backward_de_conv_a_width,backward_de_conv_a_height);

 //проверяем результат
 if (cCUDABackDeConvolution_A.cCUDAMatrixStorage_Output.GetAmount()!=2) throw "Класс CCUDABackDeConvolution провалил тестирование!";
 CMatrix<type_t> cMatrix_1(cCUDABackDeConvolution_A.cCUDAMatrixStorage_Output.GetSizeY(),cCUDABackDeConvolution_A.cCUDAMatrixStorage_Output.GetSizeX());
 CMatrix<type_t> cMatrix_2(cCUDABackDeConvolution_A.cCUDAMatrixStorage_Output.GetSizeY(),cCUDABackDeConvolution_A.cCUDAMatrixStorage_Output.GetSizeX());
 cCUDABackDeConvolution_A.cCUDAMatrixStorage_Output.Copy(0,cMatrix_1.GetColumnPtr(0));
 cCUDABackDeConvolution_A.cCUDAMatrixStorage_Output.Copy(1,cMatrix_2.GetColumnPtr(0));

 if (cMatrix_1.GetSizeX()!=16 || cMatrix_1.GetSizeY()!=2) throw "Класс CCUDABackDeConvolution провалил тестирование!";
 if (cMatrix_2.GetSizeX()!=16 || cMatrix_2.GetSizeY()!=2) throw "Класс CCUDABackDeConvolution провалил тестирование!";

 type_t test_1[]={30,135,90,15,90,435,450,105,150,435,495,195,180,360,240,60,90,405,270,45,270,1305,1350,315,450,1305,1485,585,540,1080,720,180};
 type_t test_2[]={90,405,270,45,270,1305,1350,315,450,1305,1485,585,540,1080,720,180,270,1215,810,135,810,3915,4050,945,1350,3915,4455,1755,1620,3240,2160,540};

 type_t *ptr_1=cMatrix_1.GetColumnPtr(0);
 type_t *ptr_2=cMatrix_2.GetColumnPtr(0);

 static const type_t EPS=0.0001;

 for(size_t n=0;n<cMatrix_1.GetSizeX()*cMatrix_1.GetSizeY();n++,ptr_1++,ptr_2++)
 {
  type_t v1=*ptr_1;
  type_t v2=*ptr_2;

  type_t d1=fabs(v1-test_1[n]);
  type_t d2=fabs(v2-test_2[n]);

  if (d1>EPS) throw "Класс CCUDABackDeConvolution провалил тестирование!";
  if (d2>EPS) throw "Класс CCUDABackDeConvolution провалил тестирование!";
 }
}

//****************************************************************************************************
//прочее
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//функция CUDA для вычисления свёртки
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDABackDeConvolutionFunction(CCUDABackDeConvolution<type_t> cCUDABackDeConvolution,size_t delta_width,size_t delta_height,size_t kernel_width,size_t kernel_height,size_t delta_depth,size_t kernel_depth,size_t kernel_amount)
{
 size_t s=blockIdx.x;
 size_t delta_index=s/kernel_depth;
 size_t kernel_depth_index=s%kernel_depth;
 size_t x=threadIdx.x;
 size_t y=blockIdx.y;
 cCUDABackDeConvolution.BackDeConvolutionProcessing(delta_index,kernel_depth_index,delta_width,delta_height,kernel_width,kernel_height,delta_depth,kernel_depth,kernel_amount,x,y);
}

#endif

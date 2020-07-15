#ifndef C_CUDA_FORWARD_CONVOLUTION_H
#define C_CUDA_FORWARD_CONVOLUTION_H

//****************************************************************************************************
//Класс выполнения прямой свёртки в CUDA
//****************************************************************************************************

#include <stdio.h>
#include <stdint.h>

#include "../../common/cmatrix.h"
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
class CCUDAForwardConvolution;

template<class type_t>
__global__ void CUDAForwardConvolutionFunction(CCUDAForwardConvolution<type_t> cCUDAForwardConvolution,size_t image_width,size_t image_height,size_t kernel_width,size_t kernel_height,size_t image_depth);//функция CUDA для вычисления свёртки

//****************************************************************************************************
//класс выполнения прямой свёртки в CUDA
//****************************************************************************************************
template<class type_t>
class CCUDAForwardConvolution
{
 //-дружественные функции-------------------------------------------------------------------------------
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 public:
  //-переменные-----------------------------------------------------------------------------------------
  CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Kernel;//набор ядер
  CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Bias;//набор смещений
  CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Image;//набор образов
  CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Output;//набор выходных данных
 private:
  public:
  //-конструктор----------------------------------------------------------------------------------------
  __host__ CCUDAForwardConvolution(void);
  //-деструктор-----------------------------------------------------------------------------------------
  __host__ ~CCUDAForwardConvolution();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  __host__ void Release(void);//очистить память
  __host__ void ForwardConvolution(size_t image_width,size_t image_height,size_t kernel_width,size_t kernel_height,size_t &output_width,size_t &output_height);//выполнить свёртку
  __device__ void ForwardConvolutionProcessing(size_t image_index,size_t kernel_index,size_t image_width,size_t image_height,size_t kernel_width,size_t kernel_height,size_t image_depth,size_t y);//процесс рассчёта свёртки
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
__host__ CCUDAForwardConvolution<type_t>::CCUDAForwardConvolution(void)
{
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ CCUDAForwardConvolution<type_t>::~CCUDAForwardConvolution()
{
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//процесс рассчёта свёртки
//----------------------------------------------------------------------------------------------------
template<class type_t>
__device__ void CCUDAForwardConvolution<type_t>::ForwardConvolutionProcessing(size_t image_index,size_t kernel_index,size_t image_width,size_t image_height,size_t kernel_width,size_t kernel_height,size_t image_depth,size_t y)
{
 size_t output_width=image_width-kernel_width+1;
 size_t output_height=image_height-kernel_height+1;

 size_t output_index=image_index;
 type_t *kernel_ptr=cCUDAMatrixStorage_Kernel.GetItemPtr(kernel_index);
 type_t *bias_ptr=cCUDAMatrixStorage_Bias.GetItemPtr(kernel_index);
 type_t *image_ptr=cCUDAMatrixStorage_Image.GetItemPtr(image_index);
 type_t *output_ptr=cCUDAMatrixStorage_Output.GetItemPtr(output_index)+kernel_index*output_width*output_height;

 size_t padding=0;
 size_t step=1;

 //for(size_t y=0;y<output_height;y++)
 {
  for(size_t x=0;x<output_width;x++)
  {
   type_t sum=*bias_ptr;//сразу прибавляем смещение
   //проходимся фильтрами
   for(size_t i=0;i<kernel_height;i++)
   {
    int32_t i0=static_cast<int32_t>(step*y+i);
    i0-=static_cast<int32_t>(padding);
    if (i0<0 || i0>=image_height) continue;
    for(size_t j=0;j<kernel_width;j++)
	{
     int32_t j0=static_cast<int32_t>(step*x+j);
	 j0-=static_cast<int32_t>(padding);
     //поскольку вне границ входного тензора элементы нулевые, то просто игнорируем их
     if (j0<0 || j0>=image_width) continue;
     //проходимся по всей глубине тензора и считаем сумму
     for(size_t c=0;c<image_depth;c++)
	 {
      type_t *i_ptr=image_ptr+c*image_width*image_height+i0*image_width+j0;
	  type_t *k_ptr=kernel_ptr+c*kernel_width*kernel_height+i*kernel_width+j;
	  sum+=(*i_ptr)*(*k_ptr);
	 }
    }
   }
   type_t *o_ptr=output_ptr+y*output_width+x;
   *o_ptr=sum;//записываем результат свёртки в выходной тензор
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
__host__ void CCUDAForwardConvolution<type_t>::Release(void)
{
 cCUDAMatrixStorage_Kernel.Release();
 cCUDAMatrixStorage_Image.Release();
 cCUDAMatrixStorage_Output.Release();
}

//----------------------------------------------------------------------------------------------------
//выполнить свёртку
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAForwardConvolution<type_t>::ForwardConvolution(size_t image_width,size_t image_height,size_t kernel_width,size_t kernel_height,size_t &output_width,size_t &output_height)
{
 double begin_time=GetSecondCounter();

 if (cCUDAMatrixStorage_Kernel.GetSizeX()!=kernel_width*kernel_height) throw "CCUDAForwardConvolution<type_t>::ForwardConvolution: ширина матрицы ядер должна соответствовать количеству элементов одного ядра";
 if (cCUDAMatrixStorage_Kernel.GetSizeY()!=cCUDAMatrixStorage_Image.GetSizeY()) throw "CCUDAForwardConvolution<type_t>::ForwardConvolution: высота матрицы ядер должна совпадать с высотой матрицы входного изображения";
 if (cCUDAMatrixStorage_Bias.GetSizeY()*cCUDAMatrixStorage_Bias.GetSizeX()!=1) throw "CCUDAForwardConvolution<type_t>::ForwardConvolution: размер матрицы смещений должен быть 1x1";
 if (cCUDAMatrixStorage_Bias.GetAmount()!=cCUDAMatrixStorage_Kernel.GetAmount()) throw "CCUDAForwardConvolution<type_t>::ForwardConvolution: количество матриц смещений должно быть равно количетсву матриц ядер";

 //параметры свёртки
 size_t image_depth=cCUDAMatrixStorage_Image.GetSizeY();//глубина изображений в одной матрице
 size_t image_amount=cCUDAMatrixStorage_Image.GetAmount();//количество изображений
 size_t kernel_depth=cCUDAMatrixStorage_Kernel.GetSizeY();//глубина ядер в одной матрице
 size_t kernel_amount=cCUDAMatrixStorage_Kernel.GetAmount();//количество ядер

 output_width=image_width-kernel_width+1;
 output_height=image_height-kernel_height+1;
 //задаём выходную матрицу
 cCUDAMatrixStorage_Output.Release();
 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage(kernel_amount,output_height*output_width,image_amount);
 cCUDAMatrixStorage.Create();
 cCUDAMatrixStorage_Output.Move(cCUDAMatrixStorage);
 //выполняем свёртку
 dim3 grid(image_amount,output_height);
 CUDAForwardConvolutionFunction<<<grid,kernel_amount>>>(*this,image_width,image_height,kernel_width,kernel_height,image_depth);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 double delta_t=GetSecondCounter()-begin_time;
 char str[255];
 sprintf(str,"ForwardConvolution: %.4f second\r\n",delta_t);
 //PutMessageToConsole(str);
}

//----------------------------------------------------------------------------------------------------
//протестировать класс
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAForwardConvolution<type_t>::Test(void)
{
 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Kernel(2,3*3,2);
 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Bias(1,1,2);
 cCUDAMatrixStorage_Kernel.Create();
 cCUDAMatrixStorage_Bias.Create();
 type_t kernel_1[]={1,4,1, 1,4,3, 3,3,1, 1*3,4*3,1*3, 1*3,4*3,3*3, 3*3,3*3,1*3};
 type_t kernel_2[]={1*7,4*7,1*7, 1*7,4*7,3*7, 3*7,3*7,1*7, 1*7*3*7,4*3*7,1*3*7, 1*3*7,4*3*7,3*3*7, 3*3*7,3*3*7,1*3*7};
 type_t bias_1[]={0};
 type_t bias_2[]={0};
 cCUDAMatrixStorage_Kernel.Set(0,kernel_1);
 cCUDAMatrixStorage_Kernel.Set(1,kernel_2);

 cCUDAMatrixStorage_Bias.Set(0,bias_1);
 cCUDAMatrixStorage_Bias.Set(1,bias_2);

 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Image(2,4*4,2);
 cCUDAMatrixStorage_Image.Create();
 type_t image[]={4,5,8,7, 1,8,8,8, 3,6,6,4, 6,5,7,8, 4*5,5*5,8*5,7*5, 1*5,8*5,8*5,8*5, 3*5,6*5,6*5,4*5, 6*5,5*5,7*5,8*5};
 cCUDAMatrixStorage_Image.Set(0,image);
 cCUDAMatrixStorage_Image.Set(1,image);

 CCUDAForwardConvolution<type_t> cCUDAForwardConvolution_A;
 //подключаемся к ядрам
 cCUDAForwardConvolution_A.cCUDAMatrixStorage_Kernel.Connect(cCUDAMatrixStorage_Kernel);
 //подключаемся к смещениям
 cCUDAForwardConvolution_A.cCUDAMatrixStorage_Bias.Connect(cCUDAMatrixStorage_Bias);
 //подключаемся к части исходных данных
 cCUDAForwardConvolution_A.cCUDAMatrixStorage_Image.Connect(cCUDAMatrixStorage_Image);
 //выполняем свёртку
 size_t forward_conv_a_width;
 size_t forward_conv_a_height;
 cCUDAForwardConvolution_A.ForwardConvolution(4,4,3,3,forward_conv_a_width,forward_conv_a_height);


 //проверяем результат
 if (cCUDAForwardConvolution_A.cCUDAMatrixStorage_Output.GetAmount()!=2) throw "Класс CCUDAForwardConvolution провалил тестирование!";
 CMatrix<type_t> cMatrix_1(cCUDAForwardConvolution_A.cCUDAMatrixStorage_Output.GetSizeY(),cCUDAForwardConvolution_A.cCUDAMatrixStorage_Output.GetSizeX());
 CMatrix<type_t> cMatrix_2(cCUDAForwardConvolution_A.cCUDAMatrixStorage_Output.GetSizeY(),cCUDAForwardConvolution_A.cCUDAMatrixStorage_Output.GetSizeX());
 cCUDAForwardConvolution_A.cCUDAMatrixStorage_Output.Copy(0,cMatrix_1.GetColumnPtr(0));
 cCUDAForwardConvolution_A.cCUDAMatrixStorage_Output.Copy(1,cMatrix_2.GetColumnPtr(0));

 if (cMatrix_1.GetSizeX()!=4 || cMatrix_1.GetSizeY()!=2) throw "Класс CCUDAForwardConvolution провалил тестирование!";
 if (cMatrix_2.GetSizeX()!=4 || cMatrix_2.GetSizeY()!=2) throw "Класс CCUDAForwardConvolution провалил тестирование!";

 type_t test_1[]={1952,2368,2016,2144,16184,19726,14742,20048};
 type_t test_2[]={1952,2368,2016,2144,16184,19726,14742,20048};

 type_t *ptr_1=cMatrix_1.GetColumnPtr(0);
 type_t *ptr_2=cMatrix_2.GetColumnPtr(0);

 static const type_t EPS=0.0001;

 for(size_t n=0;n<cMatrix_1.GetSizeX()*cMatrix_1.GetSizeY();n++,ptr_1++,ptr_2++)
 {
  type_t v1=*ptr_1;
  type_t v2=*ptr_2;

  type_t d1=fabs(v1-test_1[n]);
  type_t d2=fabs(v2-test_2[n]);

  if (d1>EPS) throw "Класс CCUDAForwardConvolution провалил тестирование!";
  if (d2>EPS) throw "Класс CCUDAForwardConvolution провалил тестирование!";
 }
}

//****************************************************************************************************
//прочее
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//функция CUDA для вычисления свёртки
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAForwardConvolutionFunction(CCUDAForwardConvolution<type_t> cCUDAForwardConvolution,size_t image_width,size_t image_height,size_t kernel_width,size_t kernel_height,size_t image_depth)
{
 size_t kernel_index=threadIdx.x;
 size_t image_index=blockIdx.x;
 size_t y=blockIdx.y;
 cCUDAForwardConvolution.ForwardConvolutionProcessing(image_index,kernel_index,image_width,image_height,kernel_width,kernel_height,image_depth,y);
}

#endif

#ifndef C_CUDA_BACK_CONVOLUTION_H
#define C_CUDA_BACK_CONVOLUTION_H

//****************************************************************************************************
//Класс выполнения обратной свёртки в CUDA
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
class CCUDABackConvolution;

template<class type_t>
__global__ void CUDAZeroFunction(CCUDABackConvolution<type_t> cCUDABackConvolution,size_t image_width,size_t image_height,size_t delta_width,size_t delta_height,size_t image_depth,size_t image_amount);//функция CUDA для очистки свёртки

template<class type_t>
__global__ void CUDABackConvolutionFunction(CCUDABackConvolution<type_t> cCUDABackConvolution,size_t image_width,size_t image_height,size_t delta_width,size_t delta_height,size_t image_depth,size_t image_amount);//функция CUDA для вычисления свёртки

template<class type_t>
__global__ void CUDASummFunction(CCUDABackConvolution<type_t> cCUDABackConvolution,size_t output_width,size_t output_height,size_t image_amount);//функция CUDA для вычисления суммы коэффициентов

template<class type_t>
__global__ void CUDASummBiasFunction(CCUDABackConvolution<type_t> cCUDABackConvolution,size_t image_amount);//функция CUDA для вычисления суммы смещений


//****************************************************************************************************
//класс выполнения обратной свёртки в CUDA
//****************************************************************************************************
template<class type_t>
class CCUDABackConvolution
{
 //-дружественные функции-------------------------------------------------------------------------------
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 public:
  //-переменные-----------------------------------------------------------------------------------------
  CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Delta;//набор ядер
  CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Image;//набор образов
  CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Output;//набор выходных данных
  CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_OutputBias;//набор выходных данных смещений
 private:
  CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_MiddleOutput;//промежуточный результат
  CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_MiddleOutputBias;//промежуточный результат смещений
 private:
  public:
  //-конструктор----------------------------------------------------------------------------------------
  __host__ CCUDABackConvolution(void);
  //-деструктор-----------------------------------------------------------------------------------------
  __host__ ~CCUDABackConvolution();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  __host__ void Release(void);//очистить память
  __host__ void BackConvolution(size_t image_width,size_t image_height,size_t delta_width,size_t delta_height,size_t &output_width,size_t &output_height);//выполнить свёртку

  __device__ void ZeroProcessing(size_t image_index,size_t delta_index,size_t image_width,size_t image_height,size_t delta_width,size_t delta_height,size_t image_depth,size_t image_amount);//процесс очистки свёртки
  __device__ void BackConvolutionProcessing(size_t image_index,size_t delta_index,size_t image_width,size_t image_height,size_t delta_width,size_t delta_height,size_t image_depth,size_t image_amount,size_t offset);//процесс рассчёта свёртки
  __device__ void SummProcessing(size_t depth_index,size_t delta_index,size_t output_width,size_t output_height,size_t image_amount);//процесс сложения результата от разных изображений для коэффициентов
  __device__ void SummBiasProcessing(size_t kernel_index,size_t image_amount);//процесс сложения результата от разных изображений для смещений
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
__host__ CCUDABackConvolution<type_t>::CCUDABackConvolution(void)
{
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ CCUDABackConvolution<type_t>::~CCUDABackConvolution()
{
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//процесс очистки свёртки
//----------------------------------------------------------------------------------------------------
template<class type_t>
__device__ void CCUDABackConvolution<type_t>::ZeroProcessing(size_t image_index,size_t delta_index,size_t image_width,size_t image_height,size_t delta_width,size_t delta_height,size_t image_depth,size_t image_amount)
{
 size_t output_width=image_width-delta_width+1;
 size_t output_height=image_height-delta_height+1;

 size_t output_index=delta_index;
 type_t *output_ptr=cCUDAMatrixStorage_MiddleOutput.GetItemPtr(output_index)+image_index*output_width*output_height;
 type_t *output_bias_ptr=cCUDAMatrixStorage_MiddleOutputBias.GetItemPtr(output_index)+image_index;

 size_t padding=0;
 size_t step=1;
 //обнуляем суммы
 for(size_t d=0;d<image_depth;d++)
 {
  for(size_t y=0;y<output_height;y++)
  {
   for(size_t x=0;x<output_width;x++)
   {
    size_t offset=d*output_width*output_height*image_amount;
	offset+=y*output_width;
	offset+=x;
    *(output_ptr+offset)=0;
   }
  }
 }
 *output_bias_ptr=0;
}



//----------------------------------------------------------------------------------------------------
//процесс рассчёта свёртки
//----------------------------------------------------------------------------------------------------
template<class type_t>
__device__ void CCUDABackConvolution<type_t>::BackConvolutionProcessing(size_t image_index,size_t delta_index,size_t image_width,size_t image_height,size_t delta_width,size_t delta_height,size_t image_depth,size_t image_amount,size_t offset)
{
 size_t output_width=image_width-delta_width+1;
 size_t output_height=image_height-delta_height+1;

 size_t output_index=delta_index;
 type_t *delta_ptr=cCUDAMatrixStorage_Delta.GetItemPtr(image_index)+delta_index*delta_width*delta_height;
 type_t *image_ptr=cCUDAMatrixStorage_Image.GetItemPtr(image_index);
 type_t *output_ptr=cCUDAMatrixStorage_MiddleOutput.GetItemPtr(output_index)+image_index*output_width*output_height;
 type_t *output_bias_ptr=cCUDAMatrixStorage_MiddleOutputBias.GetItemPtr(output_index)+image_index;

 size_t padding=0;
 size_t step=1;

 size_t i=offset/output_width;
 size_t j=offset%output_width;

 //расчитываем градиенты весов фильтров и смещений
 //for(size_t i=0;i<output_height;i++)
 {
  //for(size_t j=0;j<output_width;j++)
  {
   for(size_t y=0;y<delta_height;y++)
   {
    int32_t i0=static_cast<int32_t>(i+y);
    i0-=static_cast<int32_t>(padding);
    if (i0<0 || i0>=image_height) continue;
    for(size_t x=0;x<delta_width;x++)
    {
     int32_t j0=static_cast<int32_t>(j+x);
     j0-=static_cast<int32_t>(padding);
     if (j0<0 || j0>=image_width) continue;

     type_t *d_ptr=delta_ptr+y*delta_width+x;
     type_t delta=*d_ptr;

     //наращиваем градиент фильтра
     for(size_t c=0;c<image_depth;c++)
     {
      type_t *i_ptr=image_ptr+c*image_width*image_height+i0*image_width+j0;
	  type_t *o_ptr=output_ptr+c*output_width*output_height*image_amount+i*output_width+j;
	  (*o_ptr)+=delta*(*i_ptr);
     }
	 if (offset==0) *output_bias_ptr+=delta;//наращиваем градиент смещения
    }
   }
  }
 }
}

//----------------------------------------------------------------------------------------------------
//процесс сложения результата от разных изображений для коэффициентов
//----------------------------------------------------------------------------------------------------
template<class type_t>
__device__ void CCUDABackConvolution<type_t>::SummProcessing(size_t depth_index,size_t delta_index,size_t output_width,size_t output_height,size_t image_amount)
{
 type_t *input_ptr=cCUDAMatrixStorage_MiddleOutput.GetItemPtr(delta_index)+depth_index*output_width*output_height*image_amount;
 type_t *output_ptr=cCUDAMatrixStorage_Output.GetItemPtr(delta_index)+depth_index*output_width*output_height;
 //суммируем
 for(size_t n=0;n<output_width*output_height;n++)
 {
  type_t summ=0;
  for(size_t m=0;m<image_amount;m++)
  {
   size_t offset=n+m*output_width*output_height;
   type_t v=*(input_ptr+offset);
   summ+=v;
  }
  *(output_ptr+n)=summ;
 }
}

//----------------------------------------------------------------------------------------------------
//процесс сложения результата от разных изображений для смещений
//----------------------------------------------------------------------------------------------------
template<class type_t>
__device__ void CCUDABackConvolution<type_t>::SummBiasProcessing(size_t kernel_index,size_t image_amount)
{
 type_t *input_ptr=cCUDAMatrixStorage_MiddleOutputBias.GetItemPtr(kernel_index);
 type_t *output_ptr=cCUDAMatrixStorage_OutputBias.GetItemPtr(kernel_index);
 //суммируем
 type_t summ=0;
 for(size_t m=0;m<image_amount;m++,input_ptr++) summ+=(*input_ptr);
 *(output_ptr)=summ;
}

//****************************************************************************************************
//открытые функции класса
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//очистить память
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDABackConvolution<type_t>::Release(void)
{
 cCUDAMatrixStorage_Delta.Release();
 cCUDAMatrixStorage_Image.Release();
 cCUDAMatrixStorage_Output.Release();
 cCUDAMatrixStorage_OutputBias.Release();
 cCUDAMatrixStorage_MiddleOutput.Release();
 cCUDAMatrixStorage_MiddleOutputBias.Release();
}

//----------------------------------------------------------------------------------------------------
//выполнить свёртку
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDABackConvolution<type_t>::BackConvolution(size_t image_width,size_t image_height,size_t delta_width,size_t delta_height,size_t &output_width,size_t &output_height)
{
 double begin_time=GetSecondCounter();

 if (cCUDAMatrixStorage_Delta.GetSizeX()!=delta_width*delta_height) throw "CCUDABackConvolution<type_t>::BackConvolution: ширина матрицы дельт должна соответствовать количеству элементов одной дельты";
 if (cCUDAMatrixStorage_Image.GetSizeX()!=image_width*image_height) throw "CCUDABackConvolution<type_t>::BackConvolution: ширина матрицы изображений должна соответствовать количеству элементов одного изображения";
 if (cCUDAMatrixStorage_Delta.GetAmount()!=cCUDAMatrixStorage_Image.GetAmount()) throw "CCUDABackConvolution<type_t>::BackConvolution: количество дельт должно соответствовать количеству изображений";

 //параметры свёртки
 size_t image_depth=cCUDAMatrixStorage_Image.GetSizeY();//глубина изображений в одной матрице
 size_t image_amount=cCUDAMatrixStorage_Image.GetAmount();//количество изображений
 size_t delta_depth=cCUDAMatrixStorage_Delta.GetSizeY();//глубина дельт в одной матрице (соответствует количеству исходных ядер)
 size_t delta_amount=cCUDAMatrixStorage_Delta.GetAmount();//количество дельт (соответствует количеству изображений)

 output_width=image_width-delta_width+1;
 output_height=image_height-delta_height+1;
 //задаём выходную матрицу
 cCUDAMatrixStorage_Output.Release();
 cCUDAMatrixStorage_MiddleOutput.Release();
 //поправки от разных изображений лежат по ширине.
 //то есть, их потребуется просуммировать
 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_A(image_depth,output_height*output_width*image_amount,delta_depth);
 cCUDAMatrixStorage_A.Create();
 cCUDAMatrixStorage_MiddleOutput.Move(cCUDAMatrixStorage_A);

 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_BiasA(1,image_amount,delta_depth);
 cCUDAMatrixStorage_BiasA.Create();
 cCUDAMatrixStorage_MiddleOutputBias.Move(cCUDAMatrixStorage_BiasA);

 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_B(image_depth,output_height*output_width,delta_depth);
 cCUDAMatrixStorage_B.Create();
 cCUDAMatrixStorage_Output.Move(cCUDAMatrixStorage_B);

 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_BiasB(1,1,delta_depth);
 cCUDAMatrixStorage_BiasB.Create();
 cCUDAMatrixStorage_OutputBias.Move(cCUDAMatrixStorage_BiasB);

 //очищаем свёртки
 CUDAZeroFunction<<<image_amount,delta_depth>>>(*this,image_width,image_height,delta_width,delta_height,image_depth,image_amount);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());

 //выполняем свёртку
 PauseInMs(5);
 dim3 grid_a(image_amount,delta_depth);
 //dim3 grid_a(image_amount,output_height*output_width);
 //dim3 grid_a(image_amount,1);
 CUDABackConvolutionFunction<<<grid_a,output_height*output_width>>>(*this,image_width,image_height,delta_width,delta_height,image_depth,image_amount);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
 //выполняем суммирование коэффициентов
 PauseInMs(5);
 dim3 grid_b(image_depth,1);
 CUDASummFunction<<<grid_b,delta_depth>>>(*this,output_width,output_height,image_amount);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
 cCUDAMatrixStorage_MiddleOutput.Release();

 //выполняем суммирование смещений
 PauseInMs(5);
 CUDASummBiasFunction<<<delta_depth,1>>>(*this,image_amount);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
 cCUDAMatrixStorage_MiddleOutputBias.Release();

 double delta_t=GetSecondCounter()-begin_time;

 char str[255];
 sprintf(str,"BackConvolution: %.4f second\r\n",delta_t);
 //PutMessageToConsole(str);
}

//----------------------------------------------------------------------------------------------------
//протестировать класс
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDABackConvolution<type_t>::Test(void)
{
 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Image(2,4*4,2);
 cCUDAMatrixStorage_Image.Create();
 type_t image[]={4,5,8,7, 1,8,8,8, 3,6,6,4, 6,5,7,8, 4*5,5*5,8*5,7*5, 1*5,8*5,8*5,8*5, 3*5,6*5,6*5,4*5, 6*5,5*5,7*5,8*5};
 cCUDAMatrixStorage_Image.Set(0,image);
 cCUDAMatrixStorage_Image.Set(1,image);

 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Delta(2,2*2,2);
 cCUDAMatrixStorage_Delta.Create();
 type_t delta[]={2,1,4,4, 2*2,1*2,4*2,4*2};
 cCUDAMatrixStorage_Delta.Set(0,delta);
 cCUDAMatrixStorage_Delta.Set(1,delta);

 CCUDABackConvolution<type_t> cCUDABackConvolution_A;
 //подключаемся к ядрам
 cCUDABackConvolution_A.cCUDAMatrixStorage_Delta.Connect(cCUDAMatrixStorage_Delta);
 //подключаемся к части исходных данных
 cCUDABackConvolution_A.cCUDAMatrixStorage_Image.Connect(cCUDAMatrixStorage_Image);
 //выполняем свёртку
 size_t backward_conv_a_width;
 size_t backward_conv_a_height;
 cCUDABackConvolution_A.BackConvolution(4,4,2,2,backward_conv_a_width,backward_conv_a_height);


 //проверяем результат
 if (cCUDABackConvolution_A.cCUDAMatrixStorage_Output.GetAmount()!=2) throw "Класс CCUDABackConvolution провалил тестирование!";
 CMatrix<type_t> cMatrix_1(cCUDABackConvolution_A.cCUDAMatrixStorage_Output.GetSizeY(),cCUDABackConvolution_A.cCUDAMatrixStorage_Output.GetSizeX());
 CMatrix<type_t> cMatrix_2(cCUDABackConvolution_A.cCUDAMatrixStorage_Output.GetSizeY(),cCUDABackConvolution_A.cCUDAMatrixStorage_Output.GetSizeX());
 cCUDABackConvolution_A.cCUDAMatrixStorage_Output.Copy(0,cMatrix_1.GetColumnPtr(0));
 cCUDABackConvolution_A.cCUDAMatrixStorage_Output.Copy(1,cMatrix_2.GetColumnPtr(0));

 if (cCUDABackConvolution_A.cCUDAMatrixStorage_OutputBias.GetAmount()!=2) throw "Класс CCUDABackConvolution провалил тестирование!";
 CMatrix<type_t> cMatrix_Bias1(cCUDABackConvolution_A.cCUDAMatrixStorage_OutputBias.GetSizeY(),cCUDABackConvolution_A.cCUDAMatrixStorage_OutputBias.GetSizeX());
 CMatrix<type_t> cMatrix_Bias2(cCUDABackConvolution_A.cCUDAMatrixStorage_OutputBias.GetSizeY(),cCUDABackConvolution_A.cCUDAMatrixStorage_OutputBias.GetSizeX());
 cCUDABackConvolution_A.cCUDAMatrixStorage_OutputBias.Copy(0,cMatrix_Bias1.GetColumnPtr(0));
 cCUDABackConvolution_A.cCUDAMatrixStorage_OutputBias.Copy(1,cMatrix_Bias2.GetColumnPtr(0));

 if (cMatrix_1.GetSizeX()!=9 || cMatrix_1.GetSizeY()!=2) throw "Класс CCUDABackConvolution провалил тестирование!";
 if (cMatrix_2.GetSizeX()!=9 || cMatrix_2.GetSizeY()!=2) throw "Класс CCUDABackConvolution провалил тестирование!";

 if (cMatrix_Bias1.GetSizeX()!=1 || cMatrix_Bias1.GetSizeY()!=1) throw "Класс CCUDABackConvolution провалил тестирование!";
 if (cMatrix_Bias2.GetSizeX()!=1 || cMatrix_Bias2.GetSizeY()!=1) throw "Класс CCUDABackConvolution провалил тестирование!";

 type_t test_1[]={98,164,174,92,144,128,112,132,152,490,820,870,460,720,640,560,660,760};
 type_t test_2[]={196,328,348,184,288,256,224,264,304,980,1640,1740,920,1440,1280,1120,1320,1520};

 type_t *ptr_1=cMatrix_1.GetColumnPtr(0);
 type_t *ptr_2=cMatrix_2.GetColumnPtr(0);

 //проверяем значения свёрток
 static const type_t EPS=0.0001;

 for(size_t n=0;n<cMatrix_1.GetSizeX()*cMatrix_1.GetSizeY();n++,ptr_1++,ptr_2++)
 {
  type_t v1=*ptr_1;
  type_t v2=*ptr_2;

  type_t d1=fabs(v1-test_1[n]);
  type_t d2=fabs(v2-test_2[n]);

  if (d1>EPS) throw "Класс CCUDABackConvolution провалил тестирование!";
  if (d2>EPS) throw "Класс CCUDABackConvolution провалил тестирование!";
 }
 //проверяем смещения
 if (fabs(cMatrix_Bias1.GetElement(0,0)-22)>EPS) throw "Класс CCUDABackConvolution провалил тестирование!";
 if (fabs(cMatrix_Bias2.GetElement(0,0)-44)>EPS) throw "Класс CCUDABackConvolution провалил тестирование!";
}

//****************************************************************************************************
//прочее
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//функция CUDA для очистки свёртки
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAZeroFunction(CCUDABackConvolution<type_t> cCUDABackConvolution,size_t image_width,size_t image_height,size_t delta_width,size_t delta_height,size_t image_depth,size_t image_amount)
{
 size_t delta_index=threadIdx.x;
 size_t image_index=blockIdx.x;
 cCUDABackConvolution.ZeroProcessing(image_index,delta_index,image_width,image_height,delta_width,delta_height,image_depth,image_amount);
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для вычисления свёртки
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDABackConvolutionFunction(CCUDABackConvolution<type_t> cCUDABackConvolution,size_t image_width,size_t image_height,size_t delta_width,size_t delta_height,size_t image_depth,size_t image_amount)
{
	/*
 size_t delta_index=threadIdx.x;
 size_t image_index=blockIdx.x;
 size_t i=blockIdx.y;
 */

 size_t i=threadIdx.x;
 size_t image_index=blockIdx.x;
 size_t delta_index=blockIdx.y;
 cCUDABackConvolution.BackConvolutionProcessing(image_index,delta_index,image_width,image_height,delta_width,delta_height,image_depth,image_amount,i);
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для вычисления суммы коэффициентов
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDASummFunction(CCUDABackConvolution<type_t> cCUDABackConvolution,size_t output_width,size_t output_height,size_t image_amount)
{
 size_t delta_index=threadIdx.x;
 size_t depth_index=blockIdx.x;
 cCUDABackConvolution.SummProcessing(depth_index,delta_index,output_width,output_height,image_amount);
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для вычисления суммы смещений
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDASummBiasFunction(CCUDABackConvolution<type_t> cCUDABackConvolution,size_t image_amount)
{
 size_t kernel_index=blockIdx.x;
 cCUDABackConvolution.SummBiasProcessing(kernel_index,image_amount);
}

#endif

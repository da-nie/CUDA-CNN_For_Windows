#ifndef C_CUDA_MATRIX_STORAGE_H
#define C_CUDA_MATRIX_STORAGE_H

//****************************************************************************************************
//Класс хранения набора матриц в CUDA
//****************************************************************************************************

#include <stdio.h>
#include <stdint.h>

#include "handle_error.cu.h"
#include "../system/system.h"
#include "ccudasharedptr.cu.h"

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

//****************************************************************************************************
//Класс хранения набора матриц в CUDA
//****************************************************************************************************
template<class type_t>
class CCUDAMatrixStorage
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 public:
  //-переменные-----------------------------------------------------------------------------------------
  CCUDASharedPtr<type_t> cCUDASharedPtr_ItemPtr;////массив данных набора матриц
 private:
  size_t Amount;//количество матриц в наборе
  size_t Size_X;//размер по x
  size_t Size_Y;//размер по y
  size_t Size;//размер в элементах
 public:
  //-конструктор----------------------------------------------------------------------------------------
  __host__ CCUDAMatrixStorage(size_t size_y=0,size_t size_x=0,size_t amount=0);
  //-деструктор-----------------------------------------------------------------------------------------
  __host__ ~CCUDAMatrixStorage();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  __host__ void Release(void);//очистить память
  __host__ void Create(void);//выделить память для матриц
  __host__ void Move(CCUDAMatrixStorage &cCUDAMatrixStorage);//переместить
  __host__ void Connect(size_t size_y,size_t size_x,size_t amount,CCUDASharedPtr<type_t> item_ptr);//подключиться к набору данных
  __host__ void Connect(CCUDAMatrixStorage &cCUDAMatrixStorage);//подключиться к набору данных
  __host__ bool Copy(size_t index,type_t *item_ptr);//скопировать матрицу
  __host__ bool Set(size_t index,type_t *item_ptr);//задать матрицу

  __device__ __host__ size_t GetAmount(void) const;//получить количество матриц в наборе
  __device__ __host__ size_t GetSizeX(void) const;//получить размер по x
  __device__ __host__ size_t GetSizeY(void) const;//получить размер по y
  __device__ __host__ size_t GetSize(void) const;//получить размер в элементах

  __device__ __host__ bool Reinterpret(size_t new_size_y,size_t new_size_x,size_t new_amount);//задать новую интерпретацию данных

  __host__ __device__ type_t* GetItemPtr(size_t index);//получить указатель на элементы матрицы

  __host__ static void MatrixAddMatrix(CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Output,CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Left,CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Right);//сложить матрицы
  __host__ static void MatrixSubMatrix(CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Output,CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Left,CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Right);//вычесть матрицы
  __host__ static void MatrixMulMatrix(CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Output,CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Left,CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Right);//умножить матрицы
  __host__ static void MatrixMulValue(CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Output,CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Left,const type_t &value_right);//умножить матрицу на число
  __host__ static void TransponseMatrix(CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Output,CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Input);//транспонировать матрицу

  __host__ static void TransponseMatrixMulMatrix(CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Output,CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Left,CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Right);//умножить транспонированную матрицу на матрицу
  __host__ static void InitMatrix(CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage,type_t value);//задать одинаковое значение в матрице
  __host__ static void MatrixColumnScalarProduction(CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Output,CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Left,CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Right);//посчитать скалярное произведение строк матриц между собой

  __host__ static void MatrixAddValue(CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Output,CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Left,const type_t &value_right);//прибавить к каждому элементу матрицы число

  __host__ void Print(char *name)
  {
   printf("%s: Amount:%i Height:%i Width:%i\r\n",name,Amount,Size_Y,Size_X);
  }

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
__host__ CCUDAMatrixStorage<type_t>::CCUDAMatrixStorage(size_t size_y,size_t size_x,size_t amount)
{
 Size_X=size_x;
 Size_Y=size_y;
 Amount=amount;
 Size=Size_X*Size_Y;
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ CCUDAMatrixStorage<type_t>::~CCUDAMatrixStorage()
{
 Release();
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//
//----------------------------------------------------------------------------------------------------


//****************************************************************************************************
//открытые функции класса
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//очистить память
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAMatrixStorage<type_t>::Release(void)
{
 cCUDASharedPtr_ItemPtr.Release();
 Amount=0;
}
//----------------------------------------------------------------------------------------------------
//выделить память для матриц
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAMatrixStorage<type_t>::Create(void)
{
 if (Amount==0 || Size==0) return;
 cCUDASharedPtr_ItemPtr.Create(Amount*Size);
}
//----------------------------------------------------------------------------------------------------
//переместить
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAMatrixStorage<type_t>::Move(CCUDAMatrixStorage &cCUDAMatrixStorage)
{
 Release();
 cCUDASharedPtr_ItemPtr=cCUDAMatrixStorage.cCUDASharedPtr_ItemPtr;
 Size_X=cCUDAMatrixStorage.Size_X;
 Size_Y=cCUDAMatrixStorage.Size_Y;
 Size=cCUDAMatrixStorage.Size;
 Amount=cCUDAMatrixStorage.Amount;

 cCUDAMatrixStorage.cCUDASharedPtr_ItemPtr.Release();
 cCUDAMatrixStorage.Size_X=0;
 cCUDAMatrixStorage.Size_Y=0;
 cCUDAMatrixStorage.Size=0;
 cCUDAMatrixStorage.Amount=0;
}
//----------------------------------------------------------------------------------------------------
//подключиться к набору данных
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAMatrixStorage<type_t>::Connect(size_t size_y,size_t size_x,size_t amount,CCUDASharedPtr<type_t> item_ptr)
{
 Release();
 Amount=amount;
 cCUDASharedPtr_ItemPtr=item_ptr;
 Size_X=size_x;
 Size_Y=size_y;
 Size=Size_X*Size_Y;
}
//----------------------------------------------------------------------------------------------------
//подключиться к набору данных
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAMatrixStorage<type_t>::Connect(CCUDAMatrixStorage &cCUDAMatrixStorage)
{
 Release();
 Amount=cCUDAMatrixStorage.Amount;
 cCUDASharedPtr_ItemPtr=cCUDAMatrixStorage.cCUDASharedPtr_ItemPtr;
 Size_X=cCUDAMatrixStorage.Size_X;
 Size_Y=cCUDAMatrixStorage.Size_Y;
 Size=Size_X*Size_Y;
}
//----------------------------------------------------------------------------------------------------
//скопировать матрицу
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ bool CCUDAMatrixStorage<type_t>::Copy(size_t index,type_t *item_ptr)
{
 if (index>=Amount) return(false);
 HANDLE_ERROR(cudaMemcpy(item_ptr,cCUDASharedPtr_ItemPtr.Get()+Size*index,sizeof(type_t)*Size,cudaMemcpyDeviceToHost));
 return(true);
}
//----------------------------------------------------------------------------------------------------
//задать матрицу
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ bool CCUDAMatrixStorage<type_t>::Set(size_t index,type_t *item_ptr)
{
 if (index>=Amount) return(false);
 HANDLE_ERROR(cudaMemcpy(cCUDASharedPtr_ItemPtr.Get()+Size*index,item_ptr,sizeof(type_t)*Size,cudaMemcpyHostToDevice));
 return(true);
}
//----------------------------------------------------------------------------------------------------
//получить количество матриц в наборе
//----------------------------------------------------------------------------------------------------
template<class type_t>
__device__ __host__ size_t CCUDAMatrixStorage<type_t>::GetAmount(void) const
{
 return(Amount);
}
//----------------------------------------------------------------------------------------------------
//получить размер по x
//----------------------------------------------------------------------------------------------------
template<class type_t>
__device__ __host__ size_t CCUDAMatrixStorage<type_t>::GetSizeX(void) const
{
 return(Size_X);
}
//----------------------------------------------------------------------------------------------------
//получить размер по y
//----------------------------------------------------------------------------------------------------
template<class type_t>
__device__ __host__ size_t CCUDAMatrixStorage<type_t>::GetSizeY(void) const
{
 return(Size_Y);
}
//----------------------------------------------------------------------------------------------------
//получить размер в элементах
//----------------------------------------------------------------------------------------------------
template<class type_t>
__device__ __host__ size_t CCUDAMatrixStorage<type_t>::GetSize(void) const
{
 return(Size);
}
//----------------------------------------------------------------------------------------------------
//задать новую интерпретацию данных
//----------------------------------------------------------------------------------------------------
template<class type_t>
__device__ __host__ bool CCUDAMatrixStorage<type_t>::Reinterpret(size_t new_size_y,size_t new_size_x,size_t new_amount)
{
 if (new_size_x*new_size_y*new_amount!=Size*Amount) return(false);//такая интерпретация невозможна
 Size_X=new_size_x;
 Size_Y=new_size_y;
 Size=Size_X*Size_Y;
 Amount=new_amount;
 return(true);
}
//----------------------------------------------------------------------------------------------------
//получить указатель на элементы матрицы
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ __device__ type_t* CCUDAMatrixStorage<type_t>::GetItemPtr(size_t index)
{
 if (index>=Amount) return(NULL);
 return(cCUDASharedPtr_ItemPtr.Get()+Size*index);
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для сложения матриц
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAMatrixAddMatrixFunction(CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Output,CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Left,CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Right)
{
 size_t output_index=blockIdx.x;
 size_t left_index=output_index%cCUDAMatrixStorage_Left.GetAmount();
 size_t right_index=output_index%cCUDAMatrixStorage_Right.GetAmount();

 size_t left_x=cCUDAMatrixStorage_Left.GetSizeX();
 size_t left_y=cCUDAMatrixStorage_Left.GetSizeY();

 type_t *m_output_ptr=cCUDAMatrixStorage_Output.GetItemPtr(output_index);
 type_t *m_left_ptr=cCUDAMatrixStorage_Left.GetItemPtr(left_index);
 type_t *m_right_ptr=cCUDAMatrixStorage_Right.GetItemPtr(right_index);

 for(size_t y=0;y<left_y;y++)
 {
  for(size_t x=0;x<left_x;x++,m_output_ptr++,m_left_ptr++,m_right_ptr++)
  {
   *m_output_ptr=(*m_left_ptr)+(*m_right_ptr);
  }
 }
}

//----------------------------------------------------------------------------------------------------
//сложить матрицы
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAMatrixStorage<type_t>::MatrixAddMatrix(CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Output,CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Left,CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Right)
{
 if (cCUDAMatrixStorage_Left.Size_X!=cCUDAMatrixStorage_Right.Size_X || cCUDAMatrixStorage_Left.Size_Y!=cCUDAMatrixStorage_Right.Size_Y) throw "Ошибка функции 'MatrixAddMatrix'! Размерности матриц не совпадают!";
 //запускаем процесс
 size_t amount=cCUDAMatrixStorage_Left.GetAmount();
 if (cCUDAMatrixStorage_Right.GetAmount()>amount) amount=cCUDAMatrixStorage_Right.GetAmount();

 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage(cCUDAMatrixStorage_Left.Size_Y,cCUDAMatrixStorage_Left.Size_X,amount);
 cCUDAMatrixStorage.Create();
 cCUDAMatrixStorage_Output.Move(cCUDAMatrixStorage);
 CUDAMatrixAddMatrixFunction<<<amount,1>>>(cCUDAMatrixStorage_Output,cCUDAMatrixStorage_Left,cCUDAMatrixStorage_Right);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для вычитания матриц
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAMatrixSubMatrixFunction(CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Output,CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Left,CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Right)
{
 size_t output_index=blockIdx.x;
 size_t left_index=output_index%cCUDAMatrixStorage_Left.GetAmount();
 size_t right_index=output_index%cCUDAMatrixStorage_Right.GetAmount();

 size_t left_x=cCUDAMatrixStorage_Left.GetSizeX();
 size_t left_y=cCUDAMatrixStorage_Left.GetSizeY();

 type_t *m_output_ptr=cCUDAMatrixStorage_Output.GetItemPtr(output_index);
 type_t *m_left_ptr=cCUDAMatrixStorage_Left.GetItemPtr(left_index);
 type_t *m_right_ptr=cCUDAMatrixStorage_Right.GetItemPtr(right_index);

 for(size_t y=0;y<left_y;y++)
 {
  for(size_t x=0;x<left_x;x++,m_output_ptr++,m_left_ptr++,m_right_ptr++)
  {
   *m_output_ptr=(*m_left_ptr)-(*m_right_ptr);
  }
 }
}
//----------------------------------------------------------------------------------------------------
//вычесть матрицы
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAMatrixStorage<type_t>::MatrixSubMatrix(CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Output,CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Left,CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Right)
{
 if (cCUDAMatrixStorage_Left.Size_X!=cCUDAMatrixStorage_Right.Size_X || cCUDAMatrixStorage_Left.Size_Y!=cCUDAMatrixStorage_Right.Size_Y) throw "Ошибка функции 'MatrixSubMatrix'! Размерности матриц не совпадают!";
 //запускаем процесс
 size_t amount=cCUDAMatrixStorage_Left.GetAmount();
 if (cCUDAMatrixStorage_Right.GetAmount()>amount) amount=cCUDAMatrixStorage_Right.GetAmount();

 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage(cCUDAMatrixStorage_Left.Size_Y,cCUDAMatrixStorage_Left.Size_X,amount);
 cCUDAMatrixStorage.Create();
 cCUDAMatrixStorage_Output.Move(cCUDAMatrixStorage);
 CUDAMatrixSubMatrixFunction<<<amount,1>>>(cCUDAMatrixStorage_Output,cCUDAMatrixStorage_Left,cCUDAMatrixStorage_Right);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для умножения матриц
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAMatrixMulMatrixFunction(CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Output,CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Left,CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Right)
{
 size_t output_index=blockIdx.x;
 size_t left_index=output_index%cCUDAMatrixStorage_Left.GetAmount();
 size_t right_index=output_index%cCUDAMatrixStorage_Right.GetAmount();

 size_t right_x=cCUDAMatrixStorage_Right.GetSizeX();
 size_t left_x=cCUDAMatrixStorage_Left.GetSizeX();
 size_t left_y=cCUDAMatrixStorage_Left.GetSizeY();

 type_t *m_output_ptr=cCUDAMatrixStorage_Output.GetItemPtr(output_index);
 type_t *m_left_y_ptr=cCUDAMatrixStorage_Left.GetItemPtr(left_index);
 for(size_t y=0;y<left_y;y++,m_left_y_ptr+=left_x)
 {
  const type_t *m_right_x_ptr=cCUDAMatrixStorage_Right.GetItemPtr(right_index);
  for(size_t x=0;x<right_x;x++,m_output_ptr++,m_right_x_ptr++)
  {
   type_t s=0;
   const type_t *m_left_x_ptr=m_left_y_ptr;
   const type_t *m_right_y_ptr=m_right_x_ptr;
   for(size_t n=0;n<left_x;n++,m_left_x_ptr++,m_right_y_ptr+=right_x) s+=(*m_left_x_ptr)*(*m_right_y_ptr);
   *m_output_ptr=s;
  }
 }
}
//----------------------------------------------------------------------------------------------------
//умножить матрицы
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAMatrixStorage<type_t>::MatrixMulMatrix(CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Output,CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Left,CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Right)
{
 if (cCUDAMatrixStorage_Left.Size_X!=cCUDAMatrixStorage_Right.Size_Y) throw "Ошибка функции 'MatrixMulMatrix'! Размерности матриц не совпадают!";
 //запускаем процесс
 size_t amount=cCUDAMatrixStorage_Left.GetAmount();
 if (cCUDAMatrixStorage_Right.GetAmount()>amount) amount=cCUDAMatrixStorage_Right.GetAmount();

 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage(cCUDAMatrixStorage_Left.Size_Y,cCUDAMatrixStorage_Right.Size_X,amount);
 cCUDAMatrixStorage.Create();
 cCUDAMatrixStorage_Output.Move(cCUDAMatrixStorage);
 CUDAMatrixMulMatrixFunction<<<amount,1>>>(cCUDAMatrixStorage_Output,cCUDAMatrixStorage_Left,cCUDAMatrixStorage_Right);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для умножения матрицы на число
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAMatrixMulValueFunction(CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Output,CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Input,type_t value)
{
 size_t output_index=blockIdx.x;
 size_t input_index=output_index;

 size_t input_x=cCUDAMatrixStorage_Input.GetSizeX();
 size_t input_y=cCUDAMatrixStorage_Input.GetSizeY();

 type_t *m_output_ptr=cCUDAMatrixStorage_Output.GetItemPtr(output_index);
 type_t *m_input_ptr=cCUDAMatrixStorage_Input.GetItemPtr(input_index);

 for(size_t y=0;y<input_y;y++)
 {
  for(size_t x=0;x<input_x;x++,m_output_ptr++,m_input_ptr++)
  {
   *m_output_ptr=(*m_input_ptr)*value;
  }
 }
}
//----------------------------------------------------------------------------------------------------
//умножить матрицу на число
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAMatrixStorage<type_t>::MatrixMulValue(CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Output,CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Left,const type_t &value_right)
{
 //запускаем процесс
 size_t amount=cCUDAMatrixStorage_Left.GetAmount();

 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage(cCUDAMatrixStorage_Left.Size_Y,cCUDAMatrixStorage_Left.Size_X,amount);
 cCUDAMatrixStorage.Create();
 cCUDAMatrixStorage_Output.Move(cCUDAMatrixStorage);
 CUDAMatrixMulValueFunction<<<amount,1>>>(cCUDAMatrixStorage_Output,cCUDAMatrixStorage_Left,value_right);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для транспонирования матрицы
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAMatrixTransponseFunction(CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Output,CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Input)
{
 size_t output_index=blockIdx.x;
 size_t input_index=output_index;

 size_t input_x=cCUDAMatrixStorage_Input.GetSizeX();
 size_t input_y=cCUDAMatrixStorage_Input.GetSizeY();

 type_t *m_output_ptr=cCUDAMatrixStorage_Output.GetItemPtr(output_index);
 type_t *m_input_ptr=cCUDAMatrixStorage_Input.GetItemPtr(input_index);

 for(size_t y=0;y<input_y;y++,m_output_ptr++)
 {
  type_t *m_output_ptr_local=m_output_ptr;
  for(size_t x=0;x<input_x;x++,m_output_ptr_local+=input_y,m_input_ptr++)
  {
   *m_output_ptr_local=*m_input_ptr;
  }
 }
}
//----------------------------------------------------------------------------------------------------
//транспонировать матрицу
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAMatrixStorage<type_t>::TransponseMatrix(CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Output,CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Input)
{
 //запускаем процесс
 size_t amount=cCUDAMatrixStorage_Input.GetAmount();

 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage(cCUDAMatrixStorage_Input.Size_X,cCUDAMatrixStorage_Input.Size_Y,amount);
 cCUDAMatrixStorage.Create();
 cCUDAMatrixStorage_Output.Move(cCUDAMatrixStorage);
 CUDAMatrixTransponseFunction<<<amount,1>>>(cCUDAMatrixStorage_Output,cCUDAMatrixStorage_Input);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для умножения транспонированной матрицы на матрицу
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDATransponseMatrixMulMatrixFunction(CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Output,CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Left,CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Right)
{
 size_t output_index=blockIdx.x;
 size_t left_index=output_index%cCUDAMatrixStorage_Left.GetAmount();
 size_t right_index=output_index%cCUDAMatrixStorage_Right.GetAmount();

 size_t right_x=cCUDAMatrixStorage_Right.GetSizeX();
 size_t left_x=cCUDAMatrixStorage_Left.GetSizeX();
 size_t left_y=cCUDAMatrixStorage_Left.GetSizeY();

 type_t *m_output_ptr=cCUDAMatrixStorage_Output.GetItemPtr(output_index);
 type_t *m_left_y_ptr=cCUDAMatrixStorage_Left.GetItemPtr(left_index);
 for(size_t y=0;y<left_x;y++,m_left_y_ptr++)
 {
  const type_t *m_right_x_ptr=cCUDAMatrixStorage_Right.GetItemPtr(right_index);
  for(size_t x=0;x<right_x;x++,m_output_ptr++,m_right_x_ptr++)
  {
   type_t s=0;
   const type_t *m_left_x_ptr=m_left_y_ptr;
   const type_t *m_right_y_ptr=m_right_x_ptr;
   for(size_t n=0;n<left_y;n++,m_left_x_ptr+=left_x,m_right_y_ptr+=right_x) s+=(*m_left_x_ptr)*(*m_right_y_ptr);
   *m_output_ptr=s;
  }
 }
}
//----------------------------------------------------------------------------------------------------
//умножить транспонированную матрицу на матрицу
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAMatrixStorage<type_t>::TransponseMatrixMulMatrix(CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Output,CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Left,CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Right)
{
 if (cCUDAMatrixStorage_Left.GetSizeY()!=cCUDAMatrixStorage_Right.GetSizeY()) throw "Ошибка функции 'TransponseMatrixMulMatrix'! Размерности матриц не совпадают!";
 //запускаем процесс
 size_t amount=cCUDAMatrixStorage_Left.GetAmount();
 if (cCUDAMatrixStorage_Right.GetAmount()>amount) amount=cCUDAMatrixStorage_Right.GetAmount();

 cCUDAMatrixStorage_Output.Release();
 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage(cCUDAMatrixStorage_Left.GetSizeX(),cCUDAMatrixStorage_Right.GetSizeX(),amount);
 cCUDAMatrixStorage.Create();
 cCUDAMatrixStorage_Output.Move(cCUDAMatrixStorage);
 CUDATransponseMatrixMulMatrixFunction<<<amount,1>>>(cCUDAMatrixStorage_Output,cCUDAMatrixStorage_Left,cCUDAMatrixStorage_Right);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}

//----------------------------------------------------------------------------------------------------
//функция CUDA для задания одинакового значения в матрице
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAInitMatrixFunction(CCUDAMatrixStorage<type_t> cCUDAMatrixStorage,type_t value)
{
 size_t index=blockIdx.x;

 size_t input_x=cCUDAMatrixStorage.GetSizeX();
 size_t input_y=cCUDAMatrixStorage.GetSizeY();

 type_t *m_ptr=cCUDAMatrixStorage.GetItemPtr(index);

 for(size_t y=0;y<input_y;y++)
 {
  for(size_t x=0;x<input_x;x++,m_ptr++) *m_ptr=value;
 }
}
//----------------------------------------------------------------------------------------------------
//задать одинаковое значение в матрице
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAMatrixStorage<type_t>::InitMatrix(CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage,type_t value)
{
 //запускаем процесс
 size_t amount=cCUDAMatrixStorage.GetAmount();
 CUDAInitMatrixFunction<<<amount,1>>>(cCUDAMatrixStorage,value);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для вычисления скалярного произведения строк матриц между собой
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAMatrixColumnScalarProductionFunction(CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Output,CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Left,CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Right)
{
 size_t output_index=blockIdx.x;
 size_t left_index=output_index%cCUDAMatrixStorage_Left.GetAmount();
 size_t right_index=output_index%cCUDAMatrixStorage_Right.GetAmount();

 type_t *right_ptr=cCUDAMatrixStorage_Right.GetItemPtr(right_index);
 type_t *output_ptr=cCUDAMatrixStorage_Output.GetItemPtr(output_index);
 type_t *left_ptr=cCUDAMatrixStorage_Left.GetItemPtr(left_index);

 size_t size_y=cCUDAMatrixStorage_Left.GetSizeY();
 size_t size_x=cCUDAMatrixStorage_Left.GetSizeX();
 for(size_t y=0;y<size_y;y++,output_ptr++)
 {
  type_t sc=0;
  for(size_t x=0;x<size_x;x++,left_ptr++,right_ptr++)
  {
   type_t a=*left_ptr;
   type_t b=*right_ptr;
   sc+=a*b;
  }
  *output_ptr=sc;
 }
}
//----------------------------------------------------------------------------------------------------
//посчитать скалярное произведение строк матриц между собой
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ void CCUDAMatrixStorage<type_t>::MatrixColumnScalarProduction(CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Output,CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Left,CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Right)
{
 if (cCUDAMatrixStorage_Right.GetSizeX()!=cCUDAMatrixStorage_Left.GetSizeX()) throw "Ошибка функции 'MatrixColumnScalarProduction'! Размерности матриц не совпадают!";
 if (cCUDAMatrixStorage_Right.GetSizeY()!=cCUDAMatrixStorage_Left.GetSizeY()) throw "Ошибка функции 'MatrixColumnScalarProduction'! Размерности матриц не совпадают!";
 //задаём выходную матрицу
 size_t amount=cCUDAMatrixStorage_Left.GetAmount();
 if (cCUDAMatrixStorage_Right.GetAmount()>amount) amount=cCUDAMatrixStorage_Right.GetAmount();

 cCUDAMatrixStorage_Output.Release();
 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage(1,cCUDAMatrixStorage_Left.GetSizeY(),amount);
 cCUDAMatrixStorage.Create();
 cCUDAMatrixStorage_Output.Move(cCUDAMatrixStorage);
 //выполняем свёртку
 CUDAMatrixColumnScalarProductionFunction<<<amount,1>>>(cCUDAMatrixStorage_Output,cCUDAMatrixStorage_Left,cCUDAMatrixStorage_Right);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}


//----------------------------------------------------------------------------------------------------
//функция CUDA для прибавления числа к каждому элементу матрицы
//----------------------------------------------------------------------------------------------------
template<class type_t>
__global__ void CUDAMatrixAddValueFunction(CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Output,CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Input,type_t value)
{
 size_t output_index=blockIdx.x;
 size_t input_index=output_index;

 size_t input_x=cCUDAMatrixStorage_Input.GetSizeX();
 size_t input_y=cCUDAMatrixStorage_Input.GetSizeY();

 type_t *m_output_ptr=cCUDAMatrixStorage_Output.GetItemPtr(output_index);
 type_t *m_input_ptr=cCUDAMatrixStorage_Input.GetItemPtr(input_index);

 for(size_t y=0;y<input_y;y++)
 {
  for(size_t x=0;x<input_x;x++,m_output_ptr++,m_input_ptr++)
  {
   *m_output_ptr=(*m_input_ptr)+value;
  }
 }
}
//----------------------------------------------------------------------------------------------------
//прибавить к каждому элементу матрицы число
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ static void MatrixAddValue(CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Output,CCUDAMatrixStorage<type_t> &cCUDAMatrixStorage_Left,const type_t &value_right)
{
 //запускаем процесс
 size_t amount=cCUDAMatrixStorage_Left.GetAmount();

 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage(cCUDAMatrixStorage_Left.Size_Y,cCUDAMatrixStorage_Left.Size_X,amount);
 cCUDAMatrixStorage.Create();
 cCUDAMatrixStorage_Output.Move(cCUDAMatrixStorage);
 CUDAMatrixAddValueFunction<<<amount,1>>>(cCUDAMatrixStorage_Output,cCUDAMatrixStorage_Left,value_right);
 HANDLE_ERROR(cudaGetLastError());
 HANDLE_ERROR(cudaDeviceSynchronize());
}

#endif

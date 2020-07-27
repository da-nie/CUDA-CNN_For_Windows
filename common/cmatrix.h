#ifndef C_MATRIX_H
#define C_MATRIX_H

//****************************************************************************************************
//Класс матриц произвольной размерности
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include <vector>
#include "cvector.h"
#include "idatastream.h"
#include "tga.h"

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
class CMatrix;

//****************************************************************************************************
//прототипы функций
//****************************************************************************************************

template<class type_t>
CMatrix<type_t> operator+(const CMatrix<type_t>& cMatrix_Left,const CMatrix<type_t>& cMatrix_Right);//оператор "+"
template<class type_t>
CMatrix<type_t> operator-(const CMatrix<type_t>& cMatrix_Left,const CMatrix<type_t>& cMatrix_Right);//оператор "-"
template<class type_t>
CMatrix<type_t> operator*(const CMatrix<type_t>& cMatrix_Left,const CMatrix<type_t>& cMatrix_Right);//оператор "*"
template<class type_t>
CMatrix<type_t> operator*(const CMatrix<type_t>& cMatrix_Left,const type_t& value_right);//оператор "*"
template<class type_t>
CMatrix<type_t> operator*(const type_t& value_left,const CMatrix<type_t>& cMatrix_Right);//оператор "*"
template<class type_t>
CMatrix<type_t> operator*(const CMatrix<type_t>& cMatrix_Left,const CMatrix<type_t>& cMatrix_Right);//оператор "*"
template<class type_t>
CVector<type_t> operator*(const CMatrix<type_t> &cMatrix_Left,const CVector<type_t> &cVector_Right);//оператор "*"
template<class type_t>
CVector<type_t> operator*(const CVector<type_t> &cVector_Left,const CMatrix<type_t> &cMatrix_Right);//оператор "*"
template<class type_t>
CMatrix<type_t> operator&(const CVector<type_t>& cVector_Left,const CVector<type_t>& cVector_Right);//оператор "&" (умножение вектора столбца на вектор строку - результатом будет матрица)

//****************************************************************************************************
//Класс матриц произвольной размерности
//****************************************************************************************************
template<class type_t>
class CMatrix
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 private:
  //-переменные-----------------------------------------------------------------------------------------
  std::vector<type_t> Item;//массив компонентов матрицы
  size_t Size_X;//размер по X
  size_t Size_Y;//размер по Y
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CMatrix<type_t>(size_t size_y=1,size_t size_x=1);
  //-конструктор копирования----------------------------------------------------------------------------
  CMatrix<type_t>(const CMatrix<type_t> &cMatrix);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CMatrix<type_t>();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  size_t GetSizeX(void) const;//получить размер по x
  size_t GetSizeY(void) const;//получить размер по y
  type_t GetElement(size_t y,size_t x) const;//получить элемент матрицы
  void SetElement(size_t y,size_t x,type_t value);//задать элемент матрицы
  type_t* GetColumnPtr(size_t y);//получить указатель на строку матрицы
  void Unitary(void);//привести к единичному виду
  void Zero(void);//обнулить матрицу
  CMatrix<type_t> Transpose(void);//получить транспонированную матрицу
  void Move(CMatrix<type_t> &cMatrix);//переместить матрицу

  CMatrix<type_t>& operator=(const CMatrix<type_t> &cMatrix);//оператор "="

  friend CMatrix<type_t> operator+<type_t>(const CMatrix<type_t> &cMatrix_Left,const CMatrix<type_t> &cMatrix_Right);//оператор "+"
  friend CMatrix<type_t> operator-<type_t>(const CMatrix<type_t> &cMatrix_Left,const CMatrix<type_t> &cMatrix_Right);//оператор "-"

  friend CMatrix<type_t> operator*<type_t>(const CMatrix<type_t> &cMatrix_Left,const CMatrix<type_t> &cMatrix_Right);//оператор "*"

  friend CMatrix<type_t> operator*<type_t>(const CMatrix<type_t> &cMatrix_Left,const type_t &value_right);//оператор "*"
  friend CMatrix<type_t> operator*<type_t>(const type_t &value_left,const CMatrix<type_t> &cMatrix_Right);//оператор "*"

  friend CVector<type_t> operator*<type_t>(const CMatrix<type_t> &cMatrix_Left,const CVector<type_t> &cVector_Right);//оператор "*"
  friend CVector<type_t> operator*<type_t>(const CVector<type_t> &cVector_Left,const CMatrix<type_t> &cMatrix_Right);//оператор "*"

  friend CMatrix<type_t> operator&<type_t>(const CVector<type_t> &cVector_Left,const CVector<type_t> &cVector_Right);//оператор "&" (умножение вектора столбца на вектор строку - результатом будет матрица)

  static void Add(CMatrix<type_t> &cMatrix_Output,const CMatrix<type_t> &cMatrix_Left,const CMatrix<type_t> &cMatrix_Right);//сложить матрицы
  static void Sub(CMatrix<type_t> &cMatrix_Output,const CMatrix<type_t> &cMatrix_Left,const CMatrix<type_t> &cMatrix_Right);//вычесть матрицы
  static void Mul(CMatrix<type_t> &cMatrix_Output,const CMatrix<type_t> &cMatrix_Left,const CMatrix<type_t> &cMatrix_Right);//умножить матрицы
  static void Mul(CMatrix<type_t> &cMatrix_Output,const CMatrix<type_t> &cMatrix_Left,const type_t &value_right);//умножить матрицу на число
  static void Mul(CMatrix<type_t> &cMatrix_Output,const type_t &value_left,const CMatrix<type_t> &cMatrix_Right);//умножить матрицу на число
  static void Mul(CVector<type_t> &cVector_Output,const CMatrix<type_t> &cMatrix_Left,const CVector<type_t> &cVector_Right);//умножить матрицу на вектор
  static void Mul(CVector<type_t> &cVector_Output,const CVector<type_t> &cVector_Left,const CMatrix<type_t> &cMatrix_Right);//умножить вектор на матрицу
  static void Mul(CMatrix<type_t> &cMatrix_Output,const CVector<type_t> &cVector_Left,const CVector<type_t> &cVector_Right);//умножить вектора и получить матрицу
  static void Transponse(CMatrix<type_t> &cMatrix_Output,const CMatrix<type_t> &cMatrix_Input);//транспонировать матрицу

  bool Save(IDataStream *iDataStream_Ptr);//сохранить матрицу
  bool Load(IDataStream *iDataStream_Ptr);//загрузить матрицу

  static bool Test(void);//проестировать класс матриц

  void SaveImage(const std::string &file_name,size_t repeat_width,size_t repeat_height);//записать как картинку

 private:
  //-закрытые функции-----------------------------------------------------------------------------------
};


//****************************************************************************************************
//конструктор и деструктор
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CMatrix<type_t>::CMatrix(size_t size_y,size_t size_x)
{
 Size_X=size_x;
 Size_Y=size_y;
 Item.resize(Size_X*Size_Y);
}
//----------------------------------------------------------------------------------------------------
//конструктор копирования
//----------------------------------------------------------------------------------------------------
template<class type_t>
CMatrix<type_t>::CMatrix(const CMatrix<type_t> &cMatrix)
{
 if (&cMatrix==this) return;
 Item=cMatrix.Item;
 Size_X=cMatrix.Size_X;
 Size_Y=cMatrix.Size_Y;
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CMatrix<type_t>::~CMatrix()
{
 Item.clear();
 Size_X=0;
 Size_Y=0;
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//
//----------------------------------------------------------------------------------------------------

//****************************************************************************************************
//статические функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//сложить матрицы
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMatrix<type_t>::Add(CMatrix<type_t> &cMatrix_Output,const CMatrix<type_t> &cMatrix_Left,const CMatrix<type_t> &cMatrix_Right)
{
 if (cMatrix_Left.Size_X!=cMatrix_Right.Size_X || cMatrix_Left.Size_Y!=cMatrix_Right.Size_Y)
 {
  throw "Ошибка оператора '+'! Размерности матриц не совпадают!";
 }
 const type_t *left_ptr=&cMatrix_Left.Item[0];
 const type_t *right_ptr=&cMatrix_Right.Item[0];
 type_t *o_ptr=&cMatrix_Output.Item[0];

 for(size_t y=0;y<cMatrix_Left.Size_Y;y++)
 {
  for(size_t x=0;x<cMatrix_Left.Size_X;x++,o_ptr++,left_ptr++,right_ptr++)
  {
   *o_ptr=(*left_ptr)+(*right_ptr);
   //cMatrix_Output.Item[cMatrix_Left.Size_X*y+x]=cMatrix_Left.Item[cMatrix_Left.Size_X*y+x]+cMatrix_Right.Item[cMatrix_Right.Size_X*y+x];
  }
 }
}
//----------------------------------------------------------------------------------------------------
//вычесть матрицы
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMatrix<type_t>::Sub(CMatrix<type_t> &cMatrix_Output,const CMatrix<type_t> &cMatrix_Left,const CMatrix<type_t> &cMatrix_Right)
{
 if (cMatrix_Left.Size_X!=cMatrix_Right.Size_X || cMatrix_Left.Size_Y!=cMatrix_Right.Size_Y)
 {
  throw "Ошибка оператора '-'! Размерности матриц не совпадают!";
 }
 const type_t *left_ptr=&cMatrix_Left.Item[0];
 const type_t *right_ptr=&cMatrix_Right.Item[0];
 type_t *o_ptr=&cMatrix_Output.Item[0];

 for(size_t y=0;y<cMatrix_Left.Size_Y;y++)
 {
  for(size_t x=0;x<cMatrix_Left.Size_X;x++,o_ptr++,left_ptr++,right_ptr++)
  {
   //cMatrix.Item[cMatrix_Left.Size_X*y+x]=cMatrix_Left.Item[cMatrix_Left.Size_X*y+x]-cMatrix_Right.Item[cMatrix_Right.Size_X*y+x];
   *o_ptr=(*left_ptr)-(*right_ptr);
  }
 }
}
//----------------------------------------------------------------------------------------------------
//умножить матрицы
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMatrix<type_t>::Mul(CMatrix<type_t> &cMatrix_Output,const CMatrix<type_t> &cMatrix_Left,const CMatrix<type_t> &cMatrix_Right)
{
 if (cMatrix_Left.Size_X!=cMatrix_Right.Size_Y || cMatrix_Output.Size_Y!=cMatrix_Left.Size_Y || cMatrix_Output.Size_X!=cMatrix_Right.Size_X)
 {
  throw "Ошибка оператора '*'! Размерности матриц не совпадают!";
 }

 type_t *m=&cMatrix_Output.Item[0];
 for(size_t y=0;y<cMatrix_Left.Size_Y;y++)
 {
  const type_t *m1_begin=&cMatrix_Left.Item[y*cMatrix_Left.Size_X];
  for(size_t x=0;x<cMatrix_Right.Size_X;x++,m++)
  {
   type_t s=0;
   const type_t *m2=&cMatrix_Right.Item[x];
   const type_t *m1=m1_begin;
   for(size_t n=0;n<cMatrix_Left.Size_X;n++,m1++,m2+=cMatrix_Right.Size_X) s+=(*m1)*(*m2);
   *m=s;
  }
 }
}
//----------------------------------------------------------------------------------------------------
//умножить матрицу на число
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMatrix<type_t>::Mul(CMatrix<type_t> &cMatrix_Output,const CMatrix<type_t> &cMatrix_Left,const type_t &value_right)
{
 if (cMatrix_Output.Size_X!=cMatrix_Left.Size_X || cMatrix_Output.Size_Y!=cMatrix_Left.Size_Y)
 {
  throw "Ошибка оператора '*'! Размерность матриц не совпадают!";
 }

 const type_t *left_ptr=&cMatrix_Left.Item[0];
 type_t *o_ptr=&cMatrix_Output.Item[0];

 for(size_t y=0;y<cMatrix_Left.Size_Y;y++)
 {
  for(size_t x=0;x<cMatrix_Left.Size_X;x++,o_ptr++,left_ptr++)
  {
   *o_ptr=(*left_ptr)*value_right;
   //cMatrix.Item[cMatrix_Left.Size_X*y+x]=cMatrix_Left.Item[cMatrix_Left.Size_X*y+x]*value_right;
  }
 }
}
//----------------------------------------------------------------------------------------------------
//умножить матрицу на число
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMatrix<type_t>::Mul(CMatrix<type_t> &cMatrix_Output,const type_t &value_left,const CMatrix<type_t> &cMatrix_Right)
{
 if (cMatrix_Output.Size_X!=cMatrix_Right.Size_X || cMatrix_Output.Size_Y!=cMatrix_Right.Size_Y)
 {
  throw "Ошибка оператора '*'! Размерность матриц не совпадают!";
 }

 const type_t *right_ptr=&cMatrix_Right.Item[0];
 type_t *o_ptr=&cMatrix_Output.Item[0];

 for(size_t y=0;y<cMatrix_Right.Size_Y;y++)
 {
  for(size_t x=0;x<cMatrix_Right.Size_X;x++,o_ptr++,right_ptr++)
  {
   *o_ptr=(*right_ptr)*value_left;
   //cMatrix.Item[cMatrix_Right.Size_X*y+x]=cMatrix_Right.Item[cMatrix_Right.Size_X*y+x]*value_left;
  }
 }
}
//----------------------------------------------------------------------------------------------------
//умножить матрицу на вектор
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMatrix<type_t>::Mul(CVector<type_t> &cVector_Output,const CMatrix<type_t> &cMatrix_Left,const CVector<type_t> &cVector_Right)
{
 //CVector<type_t> cVector(cMatrix_Left.Size_Y);

 if (cMatrix_Left.Size_X!=cVector_Right.GetSize() || cVector_Output.GetSize()!=cMatrix_Left.Size_Y)
 {
  throw "Ошибка оператора '*'! Размерность матрицы и вектора не совпадают!";
 }

 const type_t *left_ptr=&cMatrix_Left.Item[0];
 const type_t *right_ptr=const_cast<CVector<type_t>&>(cVector_Right).GetItemPtr();
 type_t *o_ptr=cVector_Output.GetItemPtr();

 //умножается строка на столбец
 for(size_t y=0;y<cMatrix_Left.Size_Y;y++,left_ptr+=cMatrix_Left.Size_X,o_ptr++)
 {
  type_t value=0;
  const type_t *left_ptr_local=left_ptr;
  const type_t *right_ptr_local=right_ptr;
  for(size_t x=0;x<cMatrix_Left.Size_X;x++,left_ptr_local++,right_ptr_local++)
  {
   type_t m_value=*left_ptr_local;
   type_t v_value=*right_ptr_local;

   value+=m_value*v_value;
  }
  *o_ptr=value;
  //cVector.SetElement(y,value);
 }
}
//----------------------------------------------------------------------------------------------------
//умножить вектор на матрицу
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMatrix<type_t>::Mul(CVector<type_t> &cVector_Output,const CVector<type_t> &cVector_Left,const CMatrix<type_t> &cMatrix_Right)
{
 //CVector<type_t> cVector(cMatrix_Right.Size_X);

 if (cMatrix_Right.Size_Y!=cVector_Left.GetSize() || cVector_Output.GetSize()!=cMatrix_Right.Size_X)
 {
  throw "Ошибка оператора '*'! Размерность матрицы и вектора не совпадают!";
 }

 const type_t *left_ptr=const_cast<CVector<type_t>&>(cVector_Left).GetItemPtr();
 const type_t *right_ptr=&cMatrix_Right.Item[0];
 type_t *o_ptr=cVector_Output.GetItemPtr();

 //умножается строка на столбец
 for(size_t x=0;x<cMatrix_Right.Size_X;x++,o_ptr++,right_ptr++)
 {
  type_t value=0;
  const type_t *left_ptr_local=left_ptr;
  const type_t *right_ptr_local=right_ptr;
  for(size_t y=0;y<cMatrix_Right.Size_Y;y++,right_ptr_local+=cMatrix_Right.Size_X,left_ptr_local++)
  {
   type_t m_value=*right_ptr_local;
   type_t v_value=*left_ptr_local;

   value+=m_value*v_value;
  }
  *o_ptr=value;
  //cVector.SetElement(x,value);
 }
}
//----------------------------------------------------------------------------------------------------
//умножить вектора и получить матрицу
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMatrix<type_t>::Mul(CMatrix<type_t> &cMatrix_Output,const CVector<type_t> &cVector_Left,const CVector<type_t> &cVector_Right)
{
 //CMatrix<type_t> cMatrix(cVector_Left.GetSize(),cVector_Right.GetSize());

 if (cMatrix_Output.Size_Y!=cVector_Left.GetSize() || cMatrix_Output.Size_X!=cVector_Right.GetSize())
 {
  throw "Ошибка оператора '&'! Размерности матрицы и векторов не совпадают!";
 }
 type_t *m=&cMatrix_Output.Item[0];

 const type_t *left_ptr=const_cast<CVector<type_t>&>(cVector_Left).GetItemPtr();
 const type_t *right_ptr=const_cast<CVector<type_t>&>(cVector_Right).GetItemPtr();

 size_t size_left=cVector_Left.GetSize();
 size_t size_right=cVector_Right.GetSize();
 for(size_t y=0;y<size_left;y++,left_ptr++)
 {
  const type_t *right_ptr_local=right_ptr;
  for(size_t x=0;x<size_right;x++,m++,right_ptr_local++)
  {
   *m=(*left_ptr)*(*right_ptr_local);
  }
 }
}
//----------------------------------------------------------------------------------------------------
//транспонировать матрицу
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMatrix<type_t>::Transponse(CMatrix<type_t> &cMatrix_Output,const CMatrix<type_t> &cMatrix_Input)
{
 if (cMatrix_Output.Size_Y!=cMatrix_Input.Size_X || cMatrix_Output.Size_X!=cMatrix_Input.Size_Y)
 {
  throw "Ошибка транспонирования! Размерности матриц не совпадают!";
 }
 const type_t *i_ptr=&cMatrix_Input.Item[0];
 type_t *o_ptr=&cMatrix_Output.Item[0];
 for(size_t y=0;y<cMatrix_Input.Size_Y;y++,o_ptr++)
 {
  type_t *o_ptr_local=o_ptr;
  for(size_t x=0;x<cMatrix_Input.Size_X;x++,o_ptr_local+=cMatrix_Input.Size_Y,i_ptr++)
  {
   //cMatrix_Output.Item[cMatrix_Input.Size_Y*x+y]=cMatrix_Input.Item[cMatrix_Input.Size_X*y+x];
   *o_ptr_local=*i_ptr;
  }
 }
}

//****************************************************************************************************
//открытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//получить размер по x
//----------------------------------------------------------------------------------------------------
template<class type_t>
size_t CMatrix<type_t>::GetSizeX(void) const
{
 return(Size_X);
}
//----------------------------------------------------------------------------------------------------
//получить размер по y
//----------------------------------------------------------------------------------------------------
template<class type_t>
size_t CMatrix<type_t>::GetSizeY(void) const
{
 return(Size_Y);
}
//----------------------------------------------------------------------------------------------------
//получить элемент матрицы
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CMatrix<type_t>::GetElement(size_t y,size_t x) const
{
 if (x>=Size_X)
 {
  throw("Ошибка доступа к элементу матрицы для чтения!");
 }
 if (y>=Size_Y)
 {
  throw("Ошибка доступа к элементу матрицы для чтения!");
 }
 return(Item[Size_X*y+x]);
}
//----------------------------------------------------------------------------------------------------
//задать элемент матрицы
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMatrix<type_t>::SetElement(size_t y,size_t x,type_t value)
{
 if (x>=Size_X)
 {
  throw("Ошибка доступа к элементу матрицы для записи!");
 }
 if (y>=Size_Y)
 {
  throw("Ошибка доступа к элементу матрицы для записи!");
 }
 Item[Size_X*y+x]=value;
}

//----------------------------------------------------------------------------------------------------
//получить указатель на строку матрицы
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t* CMatrix<type_t>::GetColumnPtr(size_t y)
{
 if (y>=Size_Y)
 {
  throw("Ошибка получения указателя на строку матрицы!");
 }
 return(&Item[Size_X*y]);
}

//----------------------------------------------------------------------------------------------------
//привести к единичному виду
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMatrix<type_t>::Unitary(void)
{
 type_t *o_ptr=&Item[0];
 for(size_t y=0;y<Size_Y;y++)
 {
  for(size_t x=0;x<Size_X;x++,o_ptr++)
  {
   if (x==y) *o_ptr=1;
        else *o_ptr=0;
  }
 }
}
//----------------------------------------------------------------------------------------------------
//обнулить матрицу
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMatrix<type_t>::Zero(void)
{
 type_t *o_ptr=&Item[0];
 for(size_t y=0;y<Size_Y;y++)
 {
  for(size_t x=0;x<Size_X;x++,o_ptr++)
  {
   *o_ptr=0;
  }
 }
}
//----------------------------------------------------------------------------------------------------
//получить транспонированную матрицу
//----------------------------------------------------------------------------------------------------
template<class type_t>
CMatrix<type_t> CMatrix<type_t>::Transpose(void)
{
 CMatrix<type_t> cMatrix(Size_X,Size_Y);
 CMatrix<type_t>::Transponse(cMatrix,*this);
 return(cMatrix);
}
//----------------------------------------------------------------------------------------------------
//переместить матрицу
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMatrix<type_t>::Move(CMatrix<type_t> &cMatrix)
{
 if (this==&cMatrix) return;
 Item=std::move(cMatrix.Item);
 Size_X=cMatrix.Size_X;
 Size_Y=cMatrix.Size_Y;
 cMatrix.Size_X=0;
 cMatrix.Size_Y=0;
}

//----------------------------------------------------------------------------------------------------
//оператор "="
//----------------------------------------------------------------------------------------------------
template<class type_t>
CMatrix<type_t>& CMatrix<type_t>::operator=(const CMatrix<type_t> &cMatrix)
{
 if (this!=&cMatrix)
 {
  Item=cMatrix.Item;
  Size_X=cMatrix.Size_X;
  Size_Y=cMatrix.Size_Y;
 }
 return(*this);
}
//----------------------------------------------------------------------------------------------------
//оператор "+"
//----------------------------------------------------------------------------------------------------
template<class type_t>
CMatrix<type_t> operator+(const CMatrix<type_t> &cMatrix_Left,const CMatrix<type_t> &cMatrix_Right)
{
 CMatrix<type_t> cMatrix(cMatrix_Left.Size_Y,cMatrix_Left.Size_X);
 CMatrix<type_t>::Add(cMatrix,cMatrix_Left,cMatrix_Right);
 return(cMatrix);
}
//----------------------------------------------------------------------------------------------------
//оператор "-"
//----------------------------------------------------------------------------------------------------
template<class type_t>
CMatrix<type_t> operator-(const CMatrix<type_t> &cMatrix_Left,const CMatrix<type_t> &cMatrix_Right)
{
 CMatrix<type_t> cMatrix(cMatrix_Left.Size_Y,cMatrix_Left.Size_X);
 CMatrix<type_t>::Sub(cMatrix,cMatrix_Left,cMatrix_Right);
 return(cMatrix);
}
//----------------------------------------------------------------------------------------------------
//оператор "*"
//----------------------------------------------------------------------------------------------------
template<class type_t>
CMatrix<type_t> operator*(const CMatrix<type_t> &cMatrix_Left,const CMatrix<type_t> &cMatrix_Right)
{
 CMatrix<type_t> cMatrix(cMatrix_Left.Size_Y,cMatrix_Right.Size_X);
 CMatrix<type_t>::Mul(cMatrix,cMatrix_Left,cMatrix_Right);
 return(cMatrix);
}
//----------------------------------------------------------------------------------------------------
//оператор "*"
//----------------------------------------------------------------------------------------------------
template<class type_t>
CMatrix<type_t> operator*(const CMatrix<type_t> &cMatrix_Left,const type_t &value_right)
{
 CMatrix<type_t> cMatrix(cMatrix_Left.Size_Y,cMatrix_Left.Size_X);
 CMatrix<type_t>::Mul(cMatrix,cMatrix_Left,value_right);
 return(cMatrix);
}
//----------------------------------------------------------------------------------------------------
//оператор "*"
//----------------------------------------------------------------------------------------------------
template<class type_t>
CMatrix<type_t> operator*(const type_t &value_left,const CMatrix<type_t> &cMatrix_Right)
{
 CMatrix<type_t> cMatrix(cMatrix_Right.Size_Y,cMatrix_Right.Size_X);
 CMatrix<type_t>::Mul(cMatrix,value_left,cMatrix_Right);
 return(cMatrix);
}
//----------------------------------------------------------------------------------------------------
//оператор "*"
//----------------------------------------------------------------------------------------------------
template<class type_t>
CVector<type_t> operator*(const CMatrix<type_t> &cMatrix_Left,const CVector<type_t> &cVector_Right)
{
 CVector<type_t> cVector(cMatrix_Left.Size_Y);
 CMatrix<type_t>::Mul(cVector,cMatrix_Left,cVector_Right);
 return(cVector);
}
//----------------------------------------------------------------------------------------------------
//оператор "*"
//----------------------------------------------------------------------------------------------------
template<class type_t>
CVector<type_t> operator*(const CVector<type_t> &cVector_Left,const CMatrix<type_t> &cMatrix_Right)
{
 CVector<type_t> cVector(cMatrix_Right.Size_X);
 CMatrix<type_t>::Mul(cVector,cVector_Left,cMatrix_Right);
 return(cVector);
}

//----------------------------------------------------------------------------------------------------
//оператор "&" (умножение вектора столбца на вектор строку - результатом будет матрица)
//----------------------------------------------------------------------------------------------------
template<class type_t>
CMatrix<type_t> operator&(const CVector<type_t> &cVector_Left,const CVector<type_t> &cVector_Right)
{
 CMatrix<type_t> cMatrix(cVector_Left.GetSize(),cVector_Right.GetSize());
 CMatrix<type_t>::Mul(cMatrix,cVector_Left,cVector_Right);
 return(cMatrix);
}
//----------------------------------------------------------------------------------------------------
//сохранить матрицу
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CMatrix<type_t>::Save(IDataStream *iDataStream_Ptr)
{
 //сохраняем размерность матрицы
 iDataStream_Ptr->SaveUInt32(Size_Y);
 iDataStream_Ptr->SaveUInt32(Size_X);
 //сохраняем данные матрицы
 for(size_t n=0;n<Size_X*Size_Y;n++) iDataStream_Ptr->SaveDouble(Item[n]);
 return(true);
}
//----------------------------------------------------------------------------------------------------
//загрузить матрицу
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CMatrix<type_t>::Load(IDataStream *iDataStream_Ptr)
{
 //загружаем размерность матрицы
 Size_Y=iDataStream_Ptr->LoadUInt32();
 Size_X=iDataStream_Ptr->LoadUInt32();

 std::vector<type_t> item(Size_X*Size_Y);
 Item.clear();
 std::swap(Item,item);

 //загружаем данные матрицы
 for(size_t n=0;n<Size_X*Size_Y;n++) Item[n]=static_cast<type_t>(iDataStream_Ptr->LoadDouble());
 return(true);
}
//----------------------------------------------------------------------------------------------------
//протестировать класс матриц
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CMatrix<type_t>::Test(void)
{
 CMatrix<type_t> cMatrixA(2,2);
 CMatrix<type_t> cMatrixB(2,2);
 CMatrix<type_t> cMatrixC(2,2);

 cMatrixA.SetElement(0,0,1);
 cMatrixA.SetElement(0,1,2);
 cMatrixA.SetElement(1,0,3);
 cMatrixA.SetElement(1,1,4);

 cMatrixB.SetElement(0,0,1);
 cMatrixB.SetElement(0,1,2);
 cMatrixB.SetElement(1,0,3);
 cMatrixB.SetElement(1,1,4);

 //проверка на заполнение матрицы
 if (cMatrixA.GetElement(0,0)!=1) return(false);
 if (cMatrixA.GetElement(0,1)!=2) return(false);
 if (cMatrixA.GetElement(1,0)!=3) return(false);
 if (cMatrixA.GetElement(1,1)!=4) return(false);

 //умножение на число справа
 cMatrixC=cMatrixA*2.0f;
 if (cMatrixC.GetElement(0,0)!=2) return(false);
 if (cMatrixC.GetElement(0,1)!=4) return(false);
 if (cMatrixC.GetElement(1,0)!=6) return(false);
 if (cMatrixC.GetElement(1,1)!=8) return(false);
 //умножение на число слева
 cMatrixC=2.0f*cMatrixA;
 if (cMatrixC.GetElement(0,0)!=2) return(false);
 if (cMatrixC.GetElement(0,1)!=4) return(false);
 if (cMatrixC.GetElement(1,0)!=6) return(false);
 if (cMatrixC.GetElement(1,1)!=8) return(false);
 //умножение матриц
 cMatrixC=cMatrixA*cMatrixB;
 if (cMatrixC.GetElement(0,0)!=7) return(false);
 if (cMatrixC.GetElement(0,1)!=10) return(false);
 if (cMatrixC.GetElement(1,0)!=15) return(false);
 if (cMatrixC.GetElement(1,1)!=22) return(false);
 //вычитание матриц
 cMatrixC=cMatrixA-cMatrixB;
 if (cMatrixC.GetElement(0,0)!=0) return(false);
 if (cMatrixC.GetElement(0,1)!=0) return(false);
 if (cMatrixC.GetElement(1,0)!=0) return(false);
 if (cMatrixC.GetElement(1,1)!=0) return(false);
  //сложение матриц
 cMatrixC=cMatrixA+cMatrixB;
 if (cMatrixC.GetElement(0,0)!=2) return(false);
 if (cMatrixC.GetElement(0,1)!=4) return(false);
 if (cMatrixC.GetElement(1,0)!=6) return(false);
 if (cMatrixC.GetElement(1,1)!=8) return(false);

 //умножение матрицы на вектор справа
 CVector<type_t> cVectorA(2);
 CVector<type_t> cVectorB(2);
 cVectorA.Set(10,20);

 cVectorB=cMatrixA*cVectorA;
 if (cVectorB.GetElement(0)!=50) return(false);
 if (cVectorB.GetElement(1)!=110) return(false);

 //умножение матрицы на вектор слева
 cVectorB=cVectorA*cMatrixA;
 if (cVectorB.GetElement(0)!=70) return(false);
 if (cVectorB.GetElement(1)!=100) return(false);

 //получение матрицы из векторов
 cVectorA.Set(1,2);
 cVectorB.Set(3,4);
 cMatrixC=cVectorA&cVectorB;

 if (cMatrixC.GetElement(0,0)!=3) return(false);
 if (cMatrixC.GetElement(0,1)!=4) return(false);
 if (cMatrixC.GetElement(1,0)!=6) return(false);
 if (cMatrixC.GetElement(1,1)!=8) return(false);

 //транспонирование матрицы
 cMatrixC=cMatrixA.Transpose();
 if (cMatrixC.GetElement(0,0)!=1) return(false);
 if (cMatrixC.GetElement(0,1)!=3) return(false);
 if (cMatrixC.GetElement(1,0)!=2) return(false);
 if (cMatrixC.GetElement(1,1)!=4) return(false);

 return(true);
}
//----------------------------------------------------------------------------------------------------
//записать как картинку
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CMatrix<type_t>::SaveImage(const std::string &file_name,size_t repeat_width,size_t repeat_height)
{
 //сохраним ядро как картинку
 uint32_t *image=new uint32_t[Size_X*Size_Y*repeat_width*repeat_height];
 double max=GetElement(0,0);
 double min=max;
 for(size_t y=0;y<Size_Y;y++)
 {
  for(size_t x=0;x<Size_X;x++)
  {
   double v=GetElement(y,x);
   if (v>max) max=v;
   if (v<min) min=v;
  }
 }
 double delta=(max-min);
 if (delta==0) delta=1;
 for(size_t y=0;y<Size_Y;y++)
 {
  for(size_t x=0;x<Size_X;x++)
  {
   double v=GetElement(y,x);
   double b=(v-min)/delta;
   uint8_t c=(uint8_t)(b*255.0f);
   uint32_t color=0xff;
   color<<=8;
   color|=c;
   color<<=8;
   color|=c;
   color<<=8;
   color|=c;

   for(size_t ky=0;ky<repeat_height;ky++)
   {
    for(size_t kx=0;kx<repeat_width;kx++)
    {
     size_t ix=x+kx*Size_X;
     size_t iy=y+ky*Size_Y;
     image[ix+iy*Size_X*repeat_width]=color;
    }
   }
  }
 }
 SaveTGA(file_name.c_str(),Size_X*repeat_width,Size_Y*repeat_height,(uint8_t*)image);
 delete[](image);
}
#endif

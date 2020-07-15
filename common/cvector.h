#ifndef C_VECTOR_H
#define C_VECTOR_H

//****************************************************************************************************
//Класс векторов произвольной размерности
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include <vector>
#include <math.h>
#include "idatastream.h"

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
class CVector;

//****************************************************************************************************
//прототипы функций
//****************************************************************************************************
template<class type_t>
CVector<type_t> operator+(const CVector<type_t>& cVector_Left,const CVector<type_t>& cVector_Right);//оператор "+"
template<class type_t>
CVector<type_t> operator-(const CVector<type_t>& cVector_Left,const CVector<type_t>& cVector_Right);//оператор "-"
template<class type_t>
type_t operator*(const CVector<type_t>& cVector_Left,const CVector<type_t>& cVector_Right);//оператор "*" (скалярное произведение)
template<class type_t>
CVector<type_t> operator^(const CVector<type_t>& cVector_Left,const CVector<type_t>& cVector_Right);//оператор "^" (векторное произведение)
template<class type_t>
CVector<type_t> operator*(const CVector<type_t>& cVector_Left,const type_t &value_right);//оператор "*"
template<class type_t>
CVector<type_t> operator*(const type_t &value_left,const CVector<type_t>& cVector_Right);//оператор "*"
template<class type_t>
CVector<type_t> operator/(const CVector<type_t>& cVector_Left,const type_t &value_right);//оператор "/"

//****************************************************************************************************
//Класс векторов произвольной размерности
//****************************************************************************************************
template<class type_t>
class CVector
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 private:
  //-переменные-----------------------------------------------------------------------------------------
  size_t Size;//размерность вектора
  std::vector<type_t> Item;//массив компонентов вектора
 public:
  //-конструктор----------------------------------------------------------------------------------------
  CVector<type_t>(size_t size=1);
  //-конструктор копирования----------------------------------------------------------------------------
  CVector<type_t>(const CVector<type_t> &cVector);
  //-деструктор-----------------------------------------------------------------------------------------
  ~CVector<type_t>();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  type_t *GetItemPtr(void);//получить указатель на данные
  size_t GetSize(void) const;//получить размер вектора
  void Normalize(void);//нормировка вектора
  type_t GetNorma(void) const;//получить норму вектора
  type_t GetElement(size_t index) const;//получить элемент вектора
  void SetElement(size_t index,type_t value);//задать элемент вектора
  void Set(type_t x);//задать одномерный вектор
  void Set(type_t x,type_t y);//задать двухмерный вектор
  void Set(type_t x,type_t y,type_t z);//задать трёхмерный вектор
  void Set(type_t x,type_t y,type_t z,type_t a);//задать четырёхмерный вектор
  void Zero(void);//обнулить вектор
  void Move(CVector<type_t> &cVector);//переместить вектор

  CVector<type_t>& operator=(const CVector<type_t>& cVector);//оператор "="
  friend CVector<type_t> operator+<type_t>(const CVector<type_t>& cVector_Left,const CVector<type_t>& cVector_Right);//оператор "+"
  friend CVector<type_t> operator-<type_t>(const CVector<type_t>& cVector_Left,const CVector<type_t>& cVector_Right);//оператор "-"
  friend type_t operator*<type_t>(const CVector<type_t>& cVector_Left,const CVector<type_t>& cVector_Right);//оператор "*" (скалярное произведение)
  friend CVector<type_t> operator^<type_t>(const CVector<type_t>& cVector_Left,const CVector<type_t>& cVector_Right);//оператор "^" (векторное произведение)
  friend CVector<type_t> operator*<type_t>(const CVector<type_t>& cVector_Left,const type_t &value_right);//оператор "*"
  friend CVector<type_t> operator*<type_t>(const type_t &value_left,const CVector<type_t>& cVector_Right);//оператор "*"
  friend CVector<type_t> operator/<type_t>(const CVector<type_t>& cVector_Left,const type_t &value_right);//оператор "/"

  static void Add(CVector<type_t> &cVector_Output,const CVector<type_t>& cVector_Left,const CVector<type_t>& cVector_Right);//сложить вектора
  static void Sub(CVector<type_t> &cVector_Output,const CVector<type_t>& cVector_Left,const CVector<type_t>& cVector_Right);//вычесть вектора
  static type_t Mul(const CVector<type_t>& cVector_Left,const CVector<type_t>& cVector_Right);//скалярное произведение векторов
  static void Mul(CVector<type_t> &cVector_Output,const CVector<type_t>& cVector_Left,const CVector<type_t>& cVector_Right);//векторное произведение
  static void Mul(CVector<type_t> &cVector_Output,const CVector<type_t>& cVector_Left,const type_t &value_right);//умножение на число справа
  static void Mul(CVector<type_t> &cVector_Output,const type_t &value_left,const CVector<type_t>& cVector_Right);//умножение на число слева
  static void Div(CVector<type_t> &cVector_Output,const CVector<type_t>& cVector_Left,const type_t &value_right);//деление на число

  bool Save(IDataStream *iDataStream_Ptr);//сохранить вектор
  bool Load(IDataStream *iDataStream_Ptr);//загрузить вектор

  static bool Test(void);//протестировать класс векторов
 private:
  //-закрытые функции-----------------------------------------------------------------------------------
};

//****************************************************************************************************
//константы
//****************************************************************************************************

static const double CVECTOR_EPS=0.0000000001;

//****************************************************************************************************
//макроопределения
//****************************************************************************************************

//****************************************************************************************************
//конструктор и деструктор
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CVector<type_t>::CVector(size_t size)
{
 Size=size;
 Item.resize(Size);
}
//----------------------------------------------------------------------------------------------------
//конструктор коипирования
//----------------------------------------------------------------------------------------------------
template<class type_t>
CVector<type_t>::CVector(const CVector<type_t> &cVector)
{
 if (&cVector==this) return;
 Size=cVector.Size;
 Item=cVector.Item;
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
CVector<type_t>::~CVector()
{
 Size=0;
 Item.clear();
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************


//****************************************************************************************************
//статические функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//сложить вектора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CVector<type_t>::Add(CVector<type_t> &cVector_Output,const CVector<type_t>& cVector_Left,const CVector<type_t>& cVector_Right)
{
 if (cVector_Left.Size!=cVector_Right.Size || cVector_Output.Size!=cVector_Left.Size)
 {
  throw "Ошибка оператора '+'! Размерности векторов не совпадают!";
 }
 const type_t *left_ptr=&cVector_Left.Item[0];
 const type_t *right_ptr=&cVector_Right.Item[0];
 type_t *output_ptr=&cVector_Output.Item[0];
 for(size_t n=0;n<cVector_Left.Size;n++,left_ptr++,right_ptr++,output_ptr++) *output_ptr=(*left_ptr)+(*right_ptr);
}
//----------------------------------------------------------------------------------------------------
//вычесть вектора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CVector<type_t>::Sub(CVector<type_t> &cVector_Output,const CVector<type_t>& cVector_Left,const CVector<type_t>& cVector_Right)
{
 if (cVector_Left.Size!=cVector_Right.Size || cVector_Output.Size!=cVector_Left.Size)
 {
  throw "Ошибка оператора '-'! Размерности векторов не совпадают!";
 }
 const type_t *left_ptr=&cVector_Left.Item[0];
 const type_t *right_ptr=&cVector_Right.Item[0];
 type_t *output_ptr=&cVector_Output.Item[0];
 for(size_t n=0;n<cVector_Left.Size;n++,left_ptr++,right_ptr++,output_ptr++) *output_ptr=(*left_ptr)-(*right_ptr);
}
//----------------------------------------------------------------------------------------------------
//скалярное произведение векторов
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CVector<type_t>::Mul(const CVector<type_t>& cVector_Left,const CVector<type_t>& cVector_Right)
{
 if (cVector_Left.Size!=cVector_Right.Size)
 {
  throw "Ошибка оператора '*'! Размерности векторов не совпадают!";
 }
 type_t ret=0;
 const type_t *left_ptr=&cVector_Left.Item[0];
 const type_t *right_ptr=&cVector_Right.Item[0];
 for(size_t n=0;n<cVector_Left.Size;n++,left_ptr++,right_ptr++) ret+=(*left_ptr)*(*right_ptr);
 return(ret);
}
//----------------------------------------------------------------------------------------------------
//векторное произведение
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CVector<type_t>::Mul(CVector<type_t> &cVector_Output,const CVector<type_t>& cVector_Left,const CVector<type_t>& cVector_Right)
{
 if (cVector_Left.Size!=3 || cVector_Right.Size!=3)//только для векторов размерности 3
 {
  throw "Ошибка! Не определено векторное произведение векторов размерности отличной от 3.";
 }
 cVector_Output.Item[0]=cVector_Left.Item[1]*cVector_Right.Item[2]-cVector_Right.Item[1]*cVector_Left.Item[2];
 cVector_Output.Item[1]=-(cVector_Left.Item[0]*cVector_Right.Item[2]-cVector_Right.Item[0]*cVector_Left.Item[2]);
 cVector_Output.Item[2]=cVector_Left.Item[0]*cVector_Right.Item[1]-cVector_Right.Item[0]*cVector_Left.Item[1];
}
//----------------------------------------------------------------------------------------------------
//умножение на число справа
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CVector<type_t>::Mul(CVector<type_t> &cVector_Output,const CVector<type_t>& cVector_Left,const type_t &value_right)
{
 if (cVector_Output.Size!=cVector_Left.Size)
 {
  throw "Ошибка оператора '*'! Размерности векторов не совпадают!";
 }
 const type_t *left_ptr=&cVector_Left.Item[0];
 type_t *output_ptr=&cVector_Output.Item[0];
 for(size_t n=0;n<cVector_Left.Size;n++,left_ptr++,output_ptr++) *output_ptr=(*left_ptr)*value_right;
}
//----------------------------------------------------------------------------------------------------
//умножение на число слева
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CVector<type_t>::Mul(CVector<type_t> &cVector_Output,const type_t &value_left,const CVector<type_t>& cVector_Right)
{
 if (cVector_Output.Size!=cVector_Right.Size)
 {
  throw "Ошибка оператора '*'! Размерности векторов не совпадают!";
 }
 const type_t *right_ptr=&cVector_Right.Item[0];
 type_t *output_ptr=&cVector_Output.Item[0];
 for(size_t n=0;n<cVector_Right.Size;n++,right_ptr++,output_ptr++) *output_ptr=value_left*(*right_ptr);
}
//----------------------------------------------------------------------------------------------------
//деление на число
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CVector<type_t>::Div(CVector<type_t> &cVector_Output,const CVector<type_t>& cVector_Left,const type_t &value_right)
{
 if (cVector_Output.Size!=cVector_Left.Size)
 {
  throw "Ошибка оператора '/'! Размерности векторов не совпадают!";
 }
 const type_t *left_ptr=&cVector_Left.Item[0];
 type_t *output_ptr=&cVector_Output.Item[0];
 for(size_t n=0;n<cVector_Left.Size;n++,left_ptr++,output_ptr++) *output_ptr=(*left_ptr)/value_right;
}

//****************************************************************************************************
//открытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//получить указатель на данные
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t* CVector<type_t>::GetItemPtr(void)
{
 return(&Item[0]);
}
//----------------------------------------------------------------------------------------------------
//получить размер вектора
//----------------------------------------------------------------------------------------------------
template<class type_t>
size_t CVector<type_t>::GetSize(void) const
{
 return(Size);
}
//----------------------------------------------------------------------------------------------------
//нормировка вектора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CVector<type_t>::Normalize(void)
{
 type_t norma=GetNorma();
 if (norma<CVECTOR_EPS) return;
 for(size_t n=0;n<Size;n++) Item[n]/=norma;
}
//----------------------------------------------------------------------------------------------------
//получить норму вектора
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CVector<type_t>::GetNorma(void) const
{
 type_t norma=0;
 for(size_t n=0;n<Size;n++) norma+=Item[n]*Item[n];
 norma=sqrt(norma);
 return(norma);
}
//----------------------------------------------------------------------------------------------------
//получить элемент вектора
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CVector<type_t>::GetElement(size_t index) const
{
 if (index>=Size)
 {
  throw("Ошибка доступа к элементу вектора для чтения!");
 }
 return(Item[index]);
}
//----------------------------------------------------------------------------------------------------
//задать элемент вектора
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CVector<type_t>::SetElement(size_t index,type_t value)
{
 if (index>=Size)
 {
  throw("Ошибка доступа к элементу вектора для записи!");
 }
 Item[index]=value;
}
//----------------------------------------------------------------------------------------------------
//задать одномерный вектор
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CVector<type_t>::Set(type_t x)
{
 if (Size<1) return;
 Item[0]=x;
}
//----------------------------------------------------------------------------------------------------
//задать двухмерный вектор
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CVector<type_t>::Set(type_t x,type_t y)
{
 if (Size<2) return;
 Item[0]=x;
 Item[1]=y;
}
//----------------------------------------------------------------------------------------------------
//задать трёхмерный вектор
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CVector<type_t>::Set(type_t x,type_t y,type_t z)
{
 if (Size<3) return;
 Item[0]=x;
 Item[1]=y;
 Item[2]=z;
}
//----------------------------------------------------------------------------------------------------
//задать четырёхмерный вектор
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CVector<type_t>::Set(type_t x,type_t y,type_t z,type_t a)
{
 if (Size<4) return;
 Item[0]=x;
 Item[1]=y;
 Item[2]=z;
 Item[3]=a;
}
//----------------------------------------------------------------------------------------------------
//обнулить вектор
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CVector<type_t>::Zero(void)
{
 for(size_t n=0;n<Size;n++) Item[n]=0;
}
//----------------------------------------------------------------------------------------------------
//переместить вектор
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CVector<type_t>::Move(CVector<type_t> &cVector)
{
 if (this==&cVector) return;
 Item=std::move(cVector.Item);
 Size=cVector.Size;
 cVector.Size=0;
}
//----------------------------------------------------------------------------------------------------
//оператор "="
//----------------------------------------------------------------------------------------------------
template<class type_t>
CVector<type_t>& CVector<type_t>::operator=(const CVector<type_t> &cVector)
{
 if (this!=&cVector)
 {
  Size=cVector.Size;
  Item=cVector.Item;
 }
 return(*this);
}

//----------------------------------------------------------------------------------------------------
//оператор "+"
//----------------------------------------------------------------------------------------------------
template<class type_t>
CVector<type_t> operator+(const CVector<type_t>& cVector_Left,const CVector<type_t>& cVector_Right)
{
 CVector<type_t> cVector(cVector_Left.Size);
 CVector<type_t>::Add(cVector,cVector_Left,cVector_Right);
 return(cVector);
}
//----------------------------------------------------------------------------------------------------
//оператор "-"
//----------------------------------------------------------------------------------------------------
template<class type_t>
CVector<type_t> operator-(const CVector<type_t>& cVector_Left,const CVector<type_t>& cVector_Right)
{
 CVector<type_t> cVector(cVector_Left.Size);
 CVector<type_t>::Sub(cVector,cVector_Left,cVector_Right);
 return(cVector);
}
//----------------------------------------------------------------------------------------------------
//оператор "*" (скалярное произведение)
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t operator*(const CVector<type_t>& cVector_Left,const CVector<type_t>& cVector_Right)
{
 return(CVector<type_t>::Mul(cVector_Left,cVector_Right));
}
//----------------------------------------------------------------------------------------------------
//оператор "^" (векторное произведение)
//----------------------------------------------------------------------------------------------------
template<class type_t>
CVector<type_t> operator^(const CVector<type_t>& cVector_Left,const CVector<type_t>& cVector_Right)
{
 CVector<type_t> cVector(cVector_Left.Size);
 CVector<type_t>::Mul(cVector,cVector_Left,cVector_Right);
 return(cVector);
}
//----------------------------------------------------------------------------------------------------
//оператор "*"
//----------------------------------------------------------------------------------------------------
template<class type_t>
CVector<type_t> operator*(const CVector<type_t>& cVector_Left,const type_t &value_right)
{
 CVector<type_t> cVector(cVector_Left.Size);
 CVector<type_t>::Mul(cVector,cVector_Left,value_right);
 return(cVector);
}
//----------------------------------------------------------------------------------------------------
//оператор "*"
//----------------------------------------------------------------------------------------------------
template<class type_t>
CVector<type_t> operator*(const type_t &value_left,const CVector<type_t>& cVector_Right)
{
 CVector<type_t> cVector(cVector_Right.Size);
 CVector<type_t>::Mul(cVector,value_left,cVector_Right);
 return(cVector);
}
//----------------------------------------------------------------------------------------------------
//оператор "/"
//----------------------------------------------------------------------------------------------------
template<class type_t>
CVector<type_t> operator/(const CVector<type_t>& cVector_Left,const type_t &value_right)
{
 CVector<type_t> cVector(cVector_Left.Size);
 CVector<type_t>::Div(cVector,cVector_Left,value_right);
 return(cVector);
}
//----------------------------------------------------------------------------------------------------
//сохранить вектор
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CVector<type_t>::Save(IDataStream *iDataStream_Ptr)
{
 //сохраняем размерность вектора
 iDataStream_Ptr->SaveUInt32(Size);
 //сохраняем данные вектора
 for(size_t n=0;n<Size;n++) iDataStream_Ptr->SaveDouble(Item[n]);
 return(true);
}
//----------------------------------------------------------------------------------------------------
//загрузить вектор
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CVector<type_t>::Load(IDataStream *iDataStream_Ptr)
{
 //загружаем размерность вектора
 Size=iDataStream_Ptr->LoadUInt32();

 std::vector<type_t> item(Size);
 Item.clear();
 std::swap(Item,item);

 //загружаем данные вектора
 for(size_t n=0;n<Size;n++) Item[n]=iDataStream_Ptr->LoadDouble();
 return(true);
}
//----------------------------------------------------------------------------------------------------
//протестировать класс векторов
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CVector<type_t>::Test(void)
{
 CVector<type_t> cVector_A(3);
 CVector<type_t> cVector_B(3);
 CVector<type_t> cVector_C(3);

 cVector_A.Set(1,3,5);
 cVector_B.Set(3,7,11);
 //проверяем вычитание
 cVector_C=cVector_A-cVector_B;
 if (cVector_C.GetElement(0)!=-2) return(false);
 if (cVector_C.GetElement(1)!=-4) return(false);
 if (cVector_C.GetElement(2)!=-6) return(false);
 //проверяем сложение
 cVector_C=cVector_A+cVector_B;
 if (cVector_C.GetElement(0)!=4) return(false);
 if (cVector_C.GetElement(1)!=10) return(false);
 if (cVector_C.GetElement(2)!=16) return(false);
 //проверяем скалярное произведение
 type_t v=cVector_A*cVector_B;
 if (v!=(3+21+55)) return(false);
 //проверяем векторное поизведение
 cVector_C=cVector_A^cVector_B;
 //проверяем умножение на число справа
 cVector_C=cVector_A*2;
 if (cVector_C.GetElement(0)!=2) return(false);
 if (cVector_C.GetElement(1)!=6) return(false);
 if (cVector_C.GetElement(2)!=10) return(false);
 //проверяем умножение на число слева
 cVector_C=2*cVector_A;
 if (cVector_C.GetElement(0)!=2) return(false);
 if (cVector_C.GetElement(1)!=6) return(false);
 if (cVector_C.GetElement(2)!=10) return(false);
 //проверяем деление на число
 cVector_C=cVector_A/0.5;
 if (cVector_C.GetElement(0)!=2) return(false);
 if (cVector_C.GetElement(1)!=6) return(false);
 if (cVector_C.GetElement(2)!=10) return(false);
 //проверяем норма
 type_t norma=cVector_A.GetNorma();
 if (norma!=sqrt(1.0*1.0+3.0*3.0+5.0*5.0)) return(false);
 //проверяем обнуление
 cVector_A.Zero();
 if (cVector_A.GetElement(0)!=0) return(false);
 if (cVector_A.GetElement(1)!=0) return(false);
 if (cVector_A.GetElement(2)!=00) return(false);

 return(true);
}

#endif



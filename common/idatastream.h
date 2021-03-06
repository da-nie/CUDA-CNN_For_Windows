#ifndef I_DATA_STREAM_H
#define I_DATA_STREAM_H

//****************************************************************************************************
//интерфейс класса потока ввода-вывода данных
//****************************************************************************************************

//****************************************************************************************************
//подключаемые библиотеки
//****************************************************************************************************
#include <stdint.h>
#include <string>

//****************************************************************************************************
//интерфейс класса потока ввода-вывода данных
//****************************************************************************************************
class IDataStream
{
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
 private:
  //-переменные-----------------------------------------------------------------------------------------
 public:
  //-конструктор----------------------------------------------------------------------------------------
  //-деструктор-----------------------------------------------------------------------------------------
  virtual ~IDataStream() {};
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  virtual int8_t LoadInt8(void)=0;//загрузка числа типа int8_t
  virtual int16_t LoadInt16(void)=0;//загрузка числа типа int16_t
  virtual int32_t LoadInt32(void)=0;//загрузка числа типа int32_t

  virtual uint8_t LoadUInt8(void)=0;//загрузка числа типа uint8_t
  virtual uint16_t LoadUInt16(void)=0;//загрузка числа типа uint16_t
  virtual uint32_t LoadUInt32(void)=0;//загрузка числа типа uint32_t

  virtual float LoadFloat(void)=0;//загрузка числа типа float
  virtual double LoadDouble(void)=0;//загрузка числа типа double

  virtual void SaveFloat(float value)=0;//запись числа типа float
  virtual void SaveDouble(double value)=0;//запись числа типа double

  //сохранить массив
  template<class Type>
  void LoadArray(Type *ptr,size_t size)
  {
   uint8_t *v=reinterpret_cast<uint8_t*>(ptr);
   for(size_t n=0;n<size*sizeof(Type);n++,v++) *v=LoadUInt8();
  }

  virtual void SaveInt8(int8_t value)=0;//запись числа типа int8_t
  virtual void SaveInt16(int16_t value)=0;//запись числа типа int16_t
  virtual void SaveInt32(int32_t value)=0;//запись числа типа int32_t

  virtual void SaveUInt8(int8_t value)=0;//запись числа типа uint8_t
  virtual void SaveUInt16(int16_t value)=0;//запись числа типа uint16_t
  virtual void SaveUInt32(int32_t value)=0;//запись числа типа uint32_t

  template<class Type>//сохранить массив
  void SaveArray(const Type *ptr,size_t size)
  {
   const uint8_t *v=reinterpret_cast<const uint8_t*>(ptr);
   for(size_t n=0;n<size*sizeof(Type);n++,v++) SaveUInt8(*v);
  }

  virtual bool IsFail(void) const=0;//получить, была ли ошибка
  //-статические функции--------------------------------------------------------------------------------  
  static IDataStream* CreateNewDataStreamFile(std::string file_name,bool write=false);//создать новый объект загрузки из файла
 private:
  //-закрытые функции-----------------------------------------------------------------------------------  
};

#endif
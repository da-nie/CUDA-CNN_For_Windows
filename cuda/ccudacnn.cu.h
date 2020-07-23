#ifndef C_CUDA_CNN_H
#define C_CUDA_CNN_H

//****************************************************************************************************
//класс свёрточной нейросети в CUDA
//****************************************************************************************************

#include <stdio.h>
#include <stdint.h>
#include <memory>
#include <string>

#include "handle_error.cu.h"
#include "ccudamatrixstorage.cu.h"
#include "../system/system.h"
#include "../common/cmatrix.h"
#include "../common/cimage.h"

#include "math/ccudaforwardconvolution.cu.h"

#include "math/ccudabackconvolution.cu.h"
#include "math/ccudabackdeconvolution.cu.h"
#include "math/ccudamaxpooling.cu.h"
#include "math/ccudamaxdepooling.cu.h"
#include "math/ccudafunction.cu.h"

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
//класс свёрточной нейросети в CUDA
//****************************************************************************************************
template<class type_t>
class CCUDACNN
{
 //-дружественные функции-------------------------------------------------------------------------------
 public:
  //-перечисления---------------------------------------------------------------------------------------
  //-структуры------------------------------------------------------------------------------------------
  //-константы------------------------------------------------------------------------------------------
  static const size_t STRING_BUFFER_SIZE=1024;//размер буфера строки

  static const size_t CUDA_MAX_INPUT_IMAGE_AMOUNT=55;//количество одновременно обрабатываемых CUDA изображений

  static const size_t IMAGE_WIDTH=224; //ширина входных изображений
  static const size_t IMAGE_HEIGHT=224; //высота входных изображений

  static const size_t KERNEL_A_AMOUNT=8;//количество ядер первого слоя
  static const size_t KERNEL_A_WIDTH=5;//ширина ядер первого слоя
  static const size_t KERNEL_A_HEIGHT=5;//высота ядер первого слоя

  static const size_t POOLING_A_WIDTH=4;//коэффициент дискретизации по ширине первого слоя
  static const size_t POOLING_A_HEIGHT=4;//коэффициент дискретизации по высоте первого слоя

  static const size_t KERNEL_B_AMOUNT=2;//количество ядер второго слоя
  static const size_t KERNEL_B_WIDTH=5;//ширина ядер второго слоя
  static const size_t KERNEL_B_HEIGHT=5;//высота ядер второго слоя

  static const size_t POOLING_B_WIDTH=4;//коэффициент дискретизации по ширине второго слоя
  static const size_t POOLING_B_HEIGHT=4;//коэффициент дискретизации по высоте второго слоя

  static const size_t NN_LAYER_AMOUNT=2;//количество слоёв полносвязной нейросети

  static const size_t PAUSE_IN_MS=5;//пауза в миллисекундах

  static const size_t OUTPUT_LAYER_SIZE=4;//размер выходного образа
 public:
  //-переменные-----------------------------------------------------------------------------------------
  std::vector<std::pair<CMatrix<type_t>,CMatrix<type_t> > > InputImage;//набор входных образов: изображение->ответ нейросети

  CMatrix<type_t> KernelA[KERNEL_A_AMOUNT];//набор ядер первого слоя
  CMatrix<type_t> KernelBiasA[KERNEL_A_AMOUNT];//смещения ядер первого слоя
  CMatrix<type_t> KernelB[KERNEL_B_AMOUNT];//набор ядер второго слоя
  CMatrix<type_t> KernelBiasB[KERNEL_B_AMOUNT];//смещения ядер второго слоя

  CMatrix<type_t> LayerWeigh[NN_LAYER_AMOUNT];//веса полносвязной нейросети
  CMatrix<type_t> LayerBias[NN_LAYER_AMOUNT];//набор смещений полносвязной сети

  CMatrix<type_t> dKernelA[KERNEL_A_AMOUNT];//набор поправок ядер первого слоя
  CMatrix<type_t> dKernelBiasA[KERNEL_A_AMOUNT];//поправки смещений ядер первого слоя
  CMatrix<type_t> dKernelB[KERNEL_B_AMOUNT];//набор поправок ядер второго слоя
  CMatrix<type_t> dKernelBiasB[KERNEL_B_AMOUNT];//поправки смещений ядер второго слоя

  CMatrix<type_t> dLayerWeigh[NN_LAYER_AMOUNT];//поправки весов полносвязной нейросети
  CMatrix<type_t> dLayerBias[NN_LAYER_AMOUNT];//набор поправок смещений полносвязной сети

  CMatrix<type_t> LastdKernelA[KERNEL_A_AMOUNT];//набор поправок ядер первого слоя
  CMatrix<type_t> LastdKernelBiasA[KERNEL_A_AMOUNT];//поправки смещений ядер первого слоя
  CMatrix<type_t> LastdKernelB[KERNEL_B_AMOUNT];//набор поправок ядер второго слоя
  CMatrix<type_t> LastdKernelBiasB[KERNEL_B_AMOUNT];//поправки смещений ядер второго слоя

  CMatrix<type_t> LastdLayerWeigh[NN_LAYER_AMOUNT];//поправки весов полносвязной нейросети
  CMatrix<type_t> LastdLayerBias[NN_LAYER_AMOUNT];//набор поправок смещений полносвязной сети


  CCUDAMatrixStorage<type_t> CUDA_KernelA;//набор ядер первого слоя для CUDA
  CCUDAMatrixStorage<type_t> CUDA_KernelB;//набор ядер второго слоя для CUDA

  CCUDAMatrixStorage<type_t> CUDA_KernelBiasA;//набор смещений ядер первого слоя для CUDA
  CCUDAMatrixStorage<type_t> CUDA_KernelBiasB;//набор смещений второго слоя для CUDA

  CCUDAMatrixStorage<type_t> CUDA_LayerWeigh[NN_LAYER_AMOUNT];//набор весов полносвязной нейросети
  CCUDAMatrixStorage<type_t> CUDA_LayerBias[NN_LAYER_AMOUNT];//набор смещений полносвязной сети
 private:
  public:
  //-конструктор----------------------------------------------------------------------------------------
  __host__ CCUDACNN(void);
  //-деструктор-----------------------------------------------------------------------------------------
  __host__ ~CCUDACNN();
 public:
  //-открытые функции-----------------------------------------------------------------------------------
  void Execute(void);//запустить на выполнение
 private:
  //-закрытые функции-----------------------------------------------------------------------------------
  type_t GetRandValue(type_t max_value);//случайное число
  void SaveMatrix(char *file_name,const CMatrix<type_t> &cMatrix);//записать матрицу
  void SaveCUDAMatrixStorage(char *file_name,size_t first_index,CCUDAMatrixStorage<type_t> &matrix,bool save_image=false);//записать набор матриц CUDA

  void LoadImage(void);//загрузка входных изображений
  void FindFile(const std::string &path,size_t output_neuron,std::vector<std::pair<CMatrix<type_t>,CMatrix<type_t> > > &training_image);//обработка файлов
  void ProcessingFile(const std::string &file_name,size_t output_neuron,std::vector<std::pair<CMatrix<type_t>,CMatrix<type_t> > > &training_image);//обработка файла
  void InitKernel(CMatrix<type_t> &cMatrixKernel,CMatrix<type_t> &cMatrixKernelBias,size_t kernel_width,size_t kernel_height,size_t kernel_depth,size_t image_width,size_t image_height);//инициализация ядер
  void InitWeight(CMatrix<type_t> &cMatrix);//инициализация весов
  bool NetProcessing(bool only_cost,double max_cost,const std::vector<size_t> &image_kit,double &cost);//выполнить один проход обучения для набора с заданными номерами образов или только посчитать ошибку
  void ExchangeTrainingImage(std::vector<size_t> &array_index_of_image);//перемешать обучающие образы
  void SaveKernelImage(void);//сохранить изображения ядер
  bool LoadNet(const std::string &file_name);//загрузить нейросеть
  bool SaveNet(const std::string &file_name);//сохранить нейросеть
  void InitNet(void);//инициализировать нейросеть
  void ResetDeltaWeighAndBias(void);//обнулить поправки к весам и смещениям
  void UpdateWeighAndBias(double speed);//обновить веса и смещения
  void TrainingNet(void);//обучить нейросеть
};
//****************************************************************************************************
//конструктор и деструктор класса
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//конструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ CCUDACNN<type_t>::CCUDACNN(void)
{
}
//----------------------------------------------------------------------------------------------------
//деструктор
//----------------------------------------------------------------------------------------------------
template<class type_t>
__host__ CCUDACNN<type_t>::~CCUDACNN()
{
}
//****************************************************************************************************
//закрытые функции
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//случайное число
//----------------------------------------------------------------------------------------------------
template<class type_t>
type_t CCUDACNN<type_t>::GetRandValue(type_t max_value)
{
 return((static_cast<type_t>(rand())*max_value)/static_cast<type_t>(RAND_MAX));
}

//----------------------------------------------------------------------------------------------------
//записать матрицу
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CCUDACNN<type_t>::SaveMatrix(char *file_name,const CMatrix<type_t> &cMatrix)
{
 FILE *file=fopen(file_name,"wb");
 fprintf(file,"Matrix size Y:%i Matrix size X:%i\r\n",cMatrix.GetSizeY(),cMatrix.GetSizeX());
 for(size_t y=0;y<cMatrix.GetSizeY();y++)
 {
  for(size_t x=0;x<cMatrix.GetSizeX();x++)
  {
   fprintf(file,"%g\t",cMatrix.GetElement(y,x));
  }
  fprintf(file,"\r\n");
 }
 fclose(file);
}
//----------------------------------------------------------------------------------------------------
//записать набор матриц CUDA
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CCUDACNN<type_t>::SaveCUDAMatrixStorage(char *file_name,size_t first_index,CCUDAMatrixStorage<type_t> &matrix,bool save_image)
{
 //сохраняем результат свёртки
 for(size_t n=0;n<matrix.GetAmount();n++)
 {
  //printf("Save:%i\r\n",first_index+n);
  CMatrix<type_t> cMatrix(matrix.GetSizeY(),matrix.GetSizeX());
  matrix.Copy(n,cMatrix.GetColumnPtr(0));
  char output_file_name[STRING_BUFFER_SIZE];
  sprintf(output_file_name,"%s%i.txt",file_name,first_index+n);
  SaveMatrix(output_file_name,cMatrix);
  if (save_image==true)
  {
   sprintf(output_file_name,"%s%i.tga",file_name,first_index+n);
   cMatrix.SaveImage(output_file_name,1,1);
  }
 }
}



//----------------------------------------------------------------------------------------------------
//загрузка входных изображений
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CCUDACNN<type_t>::LoadImage(void)
{
 InputImage.clear();
 char string[STRING_BUFFER_SIZE];

 std::string path=GetCurrentPath();

 FindFile(std::string(path)+GetPathDivider()+"0",0,InputImage);
 FindFile(std::string(path)+GetPathDivider()+"1",1,InputImage);
 FindFile(std::string(path)+GetPathDivider()+"2",2,InputImage);
 FindFile(std::string(path)+GetPathDivider()+"3",3,InputImage);
 FindFile(std::string(path)+GetPathDivider()+"4",4,InputImage);
 FindFile(std::string(path)+GetPathDivider()+"5",5,InputImage);
 FindFile(std::string(path)+GetPathDivider()+"6",6,InputImage);
 FindFile(std::string(path)+GetPathDivider()+"7",7,InputImage);
 FindFile(std::string(path)+GetPathDivider()+"8",8,InputImage);
 FindFile(std::string(path)+GetPathDivider()+"9",9,InputImage);

 sprintf(string,"Найдено образов для обучения:%ld\r\n",InputImage.size());
 PutMessageToConsole(string);
}

//----------------------------------------------------------------------------------------------------
//обработка файлов
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CCUDACNN<type_t>::FindFile(const std::string &path,size_t output_neuron,std::vector<std::pair<CMatrix<type_t>,CMatrix<type_t> > > &training_image)
{
 std::vector<std::string> file_list;
 CreateFileList(path,file_list);
 //обрабатываем файлы
 size_t size=file_list.size();
 for(size_t n=0;n<size;n++)
 {
  std::string &file_name=file_list[n];
  //проверяем расширение
  size_t length=file_name.length();
  if (length<4) continue;
  if (file_name[length-4]!='.') continue;
  if (file_name[length-3]!='t' && file_name[length-3]!='T') continue;
  if (file_name[length-2]!='g' && file_name[length-2]!='G') continue;
  if (file_name[length-1]!='a' && file_name[length-1]!='A') continue;
  //отправляем файл на обработку
  ProcessingFile(path+GetPathDivider()+file_name,output_neuron,training_image);
 }
}
//----------------------------------------------------------------------------------------------------
//обработка файла
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CCUDACNN<type_t>::ProcessingFile(const std::string &file_name,size_t output_neuron,std::vector<std::pair<CMatrix<type_t>,CMatrix<type_t> > > &training_image)
{
 //создаём образ для обучения нейросети
 std::pair<CMatrix<type_t>,CMatrix<type_t> > tr=std::make_pair(CMatrix<type_t>(IMAGE_HEIGHT,IMAGE_WIDTH),CMatrix<type_t>(OUTPUT_LAYER_SIZE,1));
 CImage<type_t> cImage;
 //входной вектор
 if (cImage.Load(tr.first,file_name,IMAGE_WIDTH,IMAGE_HEIGHT)==false)
 {
  PutMessageToConsole("Файл: ");
  PutMessageToConsole(file_name);
  PutMessageToConsole("\tЗагрузка неудачна.\r\n");
  return;
 }
 tr.second.Zero();
 tr.second.SetElement(output_neuron,0,1);
 cImage.Normalize(tr.first);
 training_image.push_back(tr);
}
//----------------------------------------------------------------------------------------------------
//инициализация ядер
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CCUDACNN<type_t>::InitKernel(CMatrix<type_t> &cMatrixKernel,CMatrix<type_t> &cMatrixKernelBias,size_t kernel_width,size_t kernel_height,size_t kernel_depth,size_t image_width,size_t image_height)
{
 cMatrixKernel=CMatrix<type_t>(kernel_depth,kernel_width*kernel_height);
 cMatrixKernelBias=CMatrix<type_t>(1,1);

 //используем метод инициализации He (Ге)
 size_t size=kernel_width*kernel_height;
 size_t size_image=image_height*image_width*kernel_depth;
 type_t koeff=static_cast<type_t>(sqrt(2.0/size_image));
 type_t bias=static_cast<type_t>((GetRandValue(1.0))*koeff);
// cMatrixKernelBias.SetElement(0,0,bias);
 cMatrixKernelBias.SetElement(0,0,0);
 for(size_t k=0;k<kernel_depth;k++)
 {
  type_t *m_ptr=cMatrixKernel.GetColumnPtr(k);
  for(size_t n=0;n<size;n++,m_ptr++)
  {
   type_t rnd=static_cast<type_t>(GetRandValue(2.0)-1.0);
   type_t init=rnd*koeff;
   *m_ptr=init;
  }
 }
}
//----------------------------------------------------------------------------------------------------
//инициализация весов
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CCUDACNN<type_t>::InitWeight(CMatrix<type_t> &cMatrix)
{
 //инициализируем веса и сдвиги
 type_t size=cMatrix.GetSizeX();
 type_t koeff=sqrt(2.0/size);

 size_t width=cMatrix.GetSizeX();
 size_t height=cMatrix.GetSizeY();

 //веса
 for(size_t y=0;y<height;y++)
 {
  //используем метод инициализации He (Ге)
  for(size_t x=0;x<width;x++)
  {
   type_t rnd=static_cast<type_t>(GetRandValue(2.0)-1.0);
   type_t init=rnd*koeff;
   cMatrix.SetElement(y,x,init);
  }
 }
}

//----------------------------------------------------------------------------------------------------
//выполнить один проход обучения для набора с заданными номерами образов или только посчитать ошибку
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CCUDACNN<type_t>::NetProcessing(bool only_cost,double max_cost,const std::vector<size_t> &image_kit,double &cost)
{
 cost=0;
 size_t part_size=image_kit.size();
 if (part_size>CUDA_MAX_INPUT_IMAGE_AMOUNT) return(false);//нельзя превышать максимальный размер набора

 //копируем ядра
 for(size_t n=0;n<KERNEL_A_AMOUNT;n++)
 {
  CUDA_KernelA.Set(n,KernelA[n].GetColumnPtr(0));
  CUDA_KernelBiasA.Set(n,KernelBiasA[n].GetColumnPtr(0));
 }
 for(size_t n=0;n<KERNEL_B_AMOUNT;n++)
 {
  CUDA_KernelB.Set(n,KernelB[n].GetColumnPtr(0));
  CUDA_KernelBiasB.Set(n,KernelBiasB[n].GetColumnPtr(0));
 }
 //копируем полносвязную сеть
 for(size_t n=0;n<NN_LAYER_AMOUNT;n++)
 {
  CUDA_LayerWeigh[n].Set(0,LayerWeigh[n].GetColumnPtr(0));
  CUDA_LayerBias[n].Set(0,LayerBias[n].GetColumnPtr(0));
 }
 //выделяем память для входного набора изображений
 CCUDAMatrixStorage<type_t> CUDA_InputImage(CImage<type_t>::COLOR_CHANNEL,IMAGE_WIDTH*IMAGE_HEIGHT,part_size);
 CUDA_InputImage.Create();
 //копируем входные данные набора
 for(size_t n=0;n<part_size;n++)
 {
  size_t image_index=image_kit[n];
  CUDA_InputImage.Set(n,InputImage[image_index].first.GetColumnPtr(0));
 }

 //--------------------------------------------------
 //выполняем свёртку с первым слоем
 //--------------------------------------------------
 PauseInMs(PAUSE_IN_MS);
 CCUDAForwardConvolution<type_t> cCUDAForwardConvolution_A;
 //подключаемся к ядрам
 cCUDAForwardConvolution_A.cCUDAMatrixStorage_Kernel.Connect(CUDA_KernelA);
 //подключаемся к смещениям
 cCUDAForwardConvolution_A.cCUDAMatrixStorage_Bias.Connect(CUDA_KernelBiasA);
 //подключаемся к части исходных данных
 cCUDAForwardConvolution_A.cCUDAMatrixStorage_Image.Connect(CUDA_InputImage);
 //выполняем свёртку
 size_t forward_conv_a_width;
 size_t forward_conv_a_height;
 cCUDAForwardConvolution_A.ForwardConvolution(IMAGE_WIDTH,IMAGE_HEIGHT,KERNEL_A_WIDTH,KERNEL_A_HEIGHT,forward_conv_a_width,forward_conv_a_height);

 /*
 CUDA_InputImage.Reinterpret(IMAGE_WIDTH,IMAGE_HEIGHT,CUDA_InputImage.GetAmount()*CUDA_InputImage.GetSizeY());
 SaveCUDAMatrixStorage("image",0,CUDA_InputImage,true);
 throw "stop";
 */

 /*
 cCUDAForwardConvolution_A.cCUDAMatrixStorage_Output.Reinterpret(forward_conv_a_width,forward_conv_a_height,cCUDAForwardConvolution_A.cCUDAMatrixStorage_Output.GetAmount()*cCUDAForwardConvolution_A.cCUDAMatrixStorage_Output.GetSizeY());
 SaveCUDAMatrixStorage("conv_a",0,cCUDAForwardConvolution_A.cCUDAMatrixStorage_Output,true);
 throw "stop";
 */

 //--------------------------------------------------
 //применяем функцию нейрона
 //--------------------------------------------------
 PauseInMs(PAUSE_IN_MS);
 CCUDAFunction<type_t> cCUDAFunction_A(cCUDAForwardConvolution_A.cCUDAMatrixStorage_Output.GetAmount()*cCUDAForwardConvolution_A.cCUDAMatrixStorage_Output.GetSizeY());
 cCUDAFunction_A.cCUDAMatrixStorage_Input.Connect(cCUDAForwardConvolution_A.cCUDAMatrixStorage_Output);
 //перестраиваем входной вектор
 cCUDAFunction_A.cCUDAMatrixStorage_Input.Reinterpret(1,cCUDAFunction_A.cCUDAMatrixStorage_Input.GetSizeX(),cCUDAFunction_A.cCUDAMatrixStorage_Input.GetAmount()*cCUDAFunction_A.cCUDAMatrixStorage_Input.GetSizeY());
 cCUDAFunction_A.ApplyLeakyReLU();

/*
 cCUDAFunction_A.cCUDAMatrixStorage_Output.Reinterpret(forward_conv_a_width,forward_conv_a_height,cCUDAFunction_A.cCUDAMatrixStorage_Output.GetAmount()*cCUDAFunction_A.cCUDAMatrixStorage_Output.GetSizeY());
 SaveCUDAMatrixStorage("conv_f_a",0,cCUDAFunction_A.cCUDAMatrixStorage_Output,true);
 throw "stop";
 */

 //--------------------------------------------------
 //выполняем субдискретизацию после первого слоя
 //--------------------------------------------------
 PauseInMs(PAUSE_IN_MS);
 CCUDAMaxPooling<type_t> cCUDAMaxPooling_A(cCUDAFunction_A.cCUDAMatrixStorage_Output.GetAmount());
 cCUDAMaxPooling_A.cCUDAMatrixStorage_Input.Connect(cCUDAFunction_A.cCUDAMatrixStorage_Output);
 size_t forward_pooling_conv_a_width;
 size_t forward_pooling_conv_a_height;
 cCUDAMaxPooling_A.MaxPooling(forward_conv_a_width,forward_conv_a_height,POOLING_A_WIDTH,POOLING_A_HEIGHT,forward_pooling_conv_a_width,forward_pooling_conv_a_height);
 //перестраиваем результат
 cCUDAMaxPooling_A.cCUDAMatrixStorage_Output.Reinterpret(cCUDAMaxPooling_A.cCUDAMatrixStorage_Output.GetAmount()/part_size,cCUDAMaxPooling_A.cCUDAMatrixStorage_Output.GetSizeX(),part_size);
/*
 cCUDAMaxPooling_A.cCUDAMatrixStorage_Output.Reinterpret(forward_pooling_conv_a_width,forward_pooling_conv_a_height,cCUDAMaxPooling_A.cCUDAMatrixStorage_Output.GetAmount()*cCUDAMaxPooling_A.cCUDAMatrixStorage_Output.GetSizeY());
 SaveCUDAMatrixStorage("pooling_a",0,cCUDAMaxPooling_A.cCUDAMatrixStorage_Output,true);
 throw "stop";
 */

 //--------------------------------------------------
 //выполняем свёртку со вторым слоем
 //--------------------------------------------------
 PauseInMs(PAUSE_IN_MS);
 CCUDAForwardConvolution<type_t> cCUDAForwardConvolution_B;
 //подключаемся к ядрам,
 cCUDAForwardConvolution_B.cCUDAMatrixStorage_Kernel.Connect(CUDA_KernelB);
 //подключаемся к смещениям
 cCUDAForwardConvolution_B.cCUDAMatrixStorage_Bias.Connect(CUDA_KernelBiasB);
 //подключаемся к части исходных данных
 cCUDAForwardConvolution_B.cCUDAMatrixStorage_Image.Connect(cCUDAMaxPooling_A.cCUDAMatrixStorage_Output);
 //выполняем свёртку
 size_t forward_conv_b_width;
 size_t forward_conv_b_height;
 cCUDAForwardConvolution_B.ForwardConvolution(forward_pooling_conv_a_width,forward_pooling_conv_a_height,KERNEL_B_WIDTH,KERNEL_B_HEIGHT,forward_conv_b_width,forward_conv_b_height);
 /*
 cCUDAForwardConvolution_B.cCUDAMatrixStorage_Output.Reinterpret(forward_conv_b_width,forward_conv_b_height,cCUDAForwardConvolution_B.cCUDAMatrixStorage_Output.GetAmount()*cCUDAForwardConvolution_B.cCUDAMatrixStorage_Output.GetSizeY());
 SaveCUDAMatrixStorage("conv_b",0,cCUDAForwardConvolution_B.cCUDAMatrixStorage_Output,true);
 throw "stop";
 */

 //--------------------------------------------------
 //применяем функцию нейрона
 //--------------------------------------------------
 PauseInMs(PAUSE_IN_MS);
 CCUDAFunction<type_t> cCUDAFunction_B(cCUDAForwardConvolution_B.cCUDAMatrixStorage_Output.GetAmount()*cCUDAForwardConvolution_B.cCUDAMatrixStorage_Output.GetSizeY());
 cCUDAFunction_B.cCUDAMatrixStorage_Input.Connect(cCUDAForwardConvolution_B.cCUDAMatrixStorage_Output);
 //перестраиваем входной вектор
 cCUDAFunction_B.cCUDAMatrixStorage_Input.Reinterpret(1,cCUDAFunction_B.cCUDAMatrixStorage_Input.GetSizeX(),cCUDAFunction_B.cCUDAMatrixStorage_Input.GetAmount()*cCUDAFunction_B.cCUDAMatrixStorage_Input.GetSizeY());
 cCUDAFunction_B.ApplyLeakyReLU();

 /*
 cCUDAFunction_B.cCUDAMatrixStorage_Output.Reinterpret(forward_conv_b_width,forward_conv_b_height,cCUDAFunction_B.cCUDAMatrixStorage_Output.GetAmount()*cCUDAFunction_B.cCUDAMatrixStorage_Output.GetSizeY());
 SaveCUDAMatrixStorage("conv_f_b",0,cCUDAFunction_B.cCUDAMatrixStorage_Output,true);
 throw "stop";
 */

 //--------------------------------------------------
 //выполняем субдискретизацию после второго слоя
 //--------------------------------------------------
 PauseInMs(PAUSE_IN_MS);
 CCUDAMaxPooling<type_t> cCUDAMaxPooling_B(cCUDAFunction_B.cCUDAMatrixStorage_Output.GetAmount());
 cCUDAMaxPooling_B.cCUDAMatrixStorage_Input.Connect(cCUDAFunction_B.cCUDAMatrixStorage_Output);
 size_t forward_pooling_conv_b_width;
 size_t forward_pooling_conv_b_height;
 cCUDAMaxPooling_B.MaxPooling(forward_conv_b_width,forward_conv_b_height,POOLING_B_WIDTH,POOLING_B_HEIGHT,forward_pooling_conv_b_width,forward_pooling_conv_b_height);
 //перестраиваем результат
 cCUDAMaxPooling_B.cCUDAMatrixStorage_Output.Reinterpret(cCUDAMaxPooling_B.cCUDAMatrixStorage_Output.GetAmount()/part_size,cCUDAMaxPooling_B.cCUDAMatrixStorage_Output.GetSizeX(),part_size);

/*
 cCUDAMaxPooling_B.cCUDAMatrixStorage_Output.Reinterpret(forward_pooling_conv_b_width,forward_pooling_conv_b_height,cCUDAMaxPooling_B.cCUDAMatrixStorage_Output.GetAmount()*cCUDAMaxPooling_B.cCUDAMatrixStorage_Output.GetSizeY());
 SaveCUDAMatrixStorage("pooling_b",0,cCUDAMaxPooling_B.cCUDAMatrixStorage_Output,true);
 throw "stop";
 */


 /*
 SaveCUDAMatrixStorage("kernel_b",0,CUDA_KernelB,true);
 throw "stop";
 */

 //--------------------------------------------------
 //применяем полносвязную нейросеть
 //--------------------------------------------------
 PauseInMs(PAUSE_IN_MS);
 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_LayerOutputZ[NN_LAYER_AMOUNT+1];//выходы сети без функции нейрона
 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_LayerOutputH[NN_LAYER_AMOUNT+1];//выходы слоёв сети с функцией нейрона

 cCUDAMatrixStorage_LayerOutputZ[0].Connect(cCUDAMaxPooling_B.cCUDAMatrixStorage_Output);//подключаемся к выходу предыдущего слоя без функции нейрона
 cCUDAMatrixStorage_LayerOutputH[0].Connect(cCUDAMaxPooling_B.cCUDAMatrixStorage_Output);//подключаемся к выходу предыдущего слоя после функции нейрона

 size_t forward_pooling_b_amount=cCUDAMaxPooling_B.cCUDAMatrixStorage_Output.GetAmount();
 size_t forward_pooling_b_size_x=cCUDAMaxPooling_B.cCUDAMatrixStorage_Output.GetSizeX();
 size_t forward_pooling_b_size_y=cCUDAMaxPooling_B.cCUDAMatrixStorage_Output.GetSizeY();
 //входная матрица представляет собой 1 x c_output_size_x, представим её как вектор c_output_size_x x 1
 //это нужно, потому что умножать потребуется весовую матрицу на матрицу входа
 cCUDAMatrixStorage_LayerOutputZ[0].Reinterpret(forward_pooling_b_size_x*forward_pooling_b_size_y*(forward_pooling_b_amount/part_size),1,part_size);
 cCUDAMatrixStorage_LayerOutputH[0].Reinterpret(forward_pooling_b_size_x*forward_pooling_b_size_y*(forward_pooling_b_amount/part_size),1,part_size);

 //вычисляем сеть
 for(size_t n=0;n<NN_LAYER_AMOUNT;n++)
 {
  PauseInMs(PAUSE_IN_MS);
  CCUDAMatrixStorage<type_t> cCUDAMatrixStorage;
  CCUDAMatrixStorage<type_t>::MatrixMulMatrix(cCUDAMatrixStorage,CUDA_LayerWeigh[n],cCUDAMatrixStorage_LayerOutputH[n]);

  //прибавляем смещения
  PauseInMs(PAUSE_IN_MS);
  CCUDAMatrixStorage<type_t>::MatrixAddMatrix(cCUDAMatrixStorage_LayerOutputZ[n+1],cCUDAMatrixStorage,CUDA_LayerBias[n]);

  //применяем функцию нейрона
  PauseInMs(PAUSE_IN_MS);
  CCUDAFunction<type_t> cCUDAFunction(cCUDAMatrixStorage_LayerOutputZ[n+1].GetAmount());
  cCUDAFunction.cCUDAMatrixStorage_Input.Connect(cCUDAMatrixStorage_LayerOutputZ[n+1]);
  cCUDAFunction.ApplySigmoid();
  cCUDAMatrixStorage_LayerOutputH[n+1].Move(cCUDAFunction.cCUDAMatrixStorage_Output);
 }

/*
 SaveCUDAMatrixStorage("input_net",0,cCUDAMatrixStorage_LayerOutputH[1]);
 throw "stop";
 */

 //теперь у нас есть ответы сети для всех заданных обучающих образов
 //--------------------------------------------------
 //формируем эталонные ответы нейросети
 //--------------------------------------------------
 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Etalon(InputImage[0].second.GetSizeY(),InputImage[0].second.GetSizeX(),part_size);
 cCUDAMatrixStorage_Etalon.Create();
 for(size_t n=0;n<part_size;n++)
 {
  size_t image_index=image_kit[n];
  cCUDAMatrixStorage_Etalon.Set(n,InputImage[image_index].second.GetColumnPtr(0));
 }

 //--------------------------------------------------
 //вычисляем ошибку сети
 //--------------------------------------------------
 PauseInMs(PAUSE_IN_MS);
 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Error;//ошибка на выходах сети
 CCUDAMatrixStorage<type_t>::MatrixSubMatrix(cCUDAMatrixStorage_Error,cCUDAMatrixStorage_LayerOutputH[NN_LAYER_AMOUNT],cCUDAMatrixStorage_Etalon);
/*
 SaveCUDAMatrixStorage("output_net",0,cCUDAMatrixStorage_LayerOutputH[NN_LAYER_AMOUNT]);
 SaveCUDAMatrixStorage("etalon_net",0,cCUDAMatrixStorage_Etalon);
 SaveCUDAMatrixStorage("error_net",0,cCUDAMatrixStorage_Error);
 throw "stop";
 */

 cCUDAMatrixStorage_Etalon.Release();//эталон больше не нужен

 cost=0;
 for(size_t n=0;n<cCUDAMatrixStorage_Error.GetAmount();n++)
 {
  CMatrix<type_t> cMatrix_Error(cCUDAMatrixStorage_Error.GetSizeY(),cCUDAMatrixStorage_Error.GetSizeX());
  cCUDAMatrixStorage_Error.Copy(n,cMatrix_Error.GetColumnPtr(0));
  type_t local_cost=0;
  for(size_t y=0;y<cMatrix_Error.GetSizeY();y++)
  {
   for(size_t x=0;x<cMatrix_Error.GetSizeX();x++)
   {
    type_t value=cMatrix_Error.GetElement(y,x);
	local_cost+=value*value;
   }
  }
  if (local_cost>cost) cost=local_cost;
 }
 if (only_cost==true) return(true);//обучение сети не требуется
 if (cost<max_cost) return(true);//обучение сети не требуется

 //****************************************************************************************************
 //****************************************************************************************************
 //процесс обучения сети
 //****************************************************************************************************
 //****************************************************************************************************

 //**************************************************
 //делаем обратный проход по сети
 //**************************************************

 //--------------------------------------------------
 //применяем ко всем входным данным полносвязной сети производную - она потребуется для расчёта обратного распространения
 //--------------------------------------------------
 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_LayerDifferential[NN_LAYER_AMOUNT+1];//производные выходов
 for(size_t n=1;n<NN_LAYER_AMOUNT+1;n++)
 {
  //применяем производную функции нейрона
  PauseInMs(PAUSE_IN_MS);
  CCUDAFunction<type_t> cCUDAFunction(cCUDAMatrixStorage_LayerOutputZ[n].GetAmount());
  cCUDAFunction.cCUDAMatrixStorage_Input.Connect(cCUDAMatrixStorage_LayerOutputZ[n]);
  cCUDAFunction.ApplyDifferentialSigmoid();
  cCUDAMatrixStorage_LayerDifferential[n].Move(cCUDAFunction.cCUDAMatrixStorage_Output);
 }
 //на входном слое производная не применяется, поэтому её не считаем

 //--------------------------------------------------
 //вычисляем дельту на выходах сети
 //--------------------------------------------------
 PauseInMs(PAUSE_IN_MS);
 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_LayerDelta[NN_LAYER_AMOUNT+1];//дельта на выходах сети
 CCUDAMatrixStorage<type_t>::MatrixColumnScalarProduction(cCUDAMatrixStorage_LayerDelta[NN_LAYER_AMOUNT],cCUDAMatrixStorage_Error,cCUDAMatrixStorage_LayerDifferential[NN_LAYER_AMOUNT]);

 //представляем вектор дельты в виде столбца
 cCUDAMatrixStorage_LayerDelta[NN_LAYER_AMOUNT].Reinterpret(cCUDAMatrixStorage_LayerDelta[NN_LAYER_AMOUNT].GetSizeX()*cCUDAMatrixStorage_LayerDelta[NN_LAYER_AMOUNT].GetSizeY(),1,cCUDAMatrixStorage_LayerDelta[NN_LAYER_AMOUNT].GetAmount());
 //--------------------------------------------------
 //выполняем обратное распространение в полносвязной сети
 //--------------------------------------------------
 //распространяем дельту на все остальные слои
 for(size_t m=0,n=NN_LAYER_AMOUNT-1;m<NN_LAYER_AMOUNT;m++,n--)
 {
  PauseInMs(PAUSE_IN_MS);
  CCUDAMatrixStorage<type_t> cCUDAMatrixStorage;
  CCUDAMatrixStorage<type_t>::TransponseMatrixMulMatrix(cCUDAMatrixStorage,CUDA_LayerWeigh[n],cCUDAMatrixStorage_LayerDelta[n+1]);
  if (n!=0)//ко входному слою производная не применяется
  {
   PauseInMs(PAUSE_IN_MS);
   CCUDAMatrixStorage<type_t>::MatrixColumnScalarProduction(cCUDAMatrixStorage_LayerDelta[n],cCUDAMatrixStorage,cCUDAMatrixStorage_LayerDifferential[n]);
  }
  else cCUDAMatrixStorage_LayerDelta[n].Move(cCUDAMatrixStorage);
  cCUDAMatrixStorage_LayerDelta[n].Reinterpret(cCUDAMatrixStorage_LayerDelta[n].GetSizeX()*cCUDAMatrixStorage_LayerDelta[n].GetSizeY(),1,cCUDAMatrixStorage_LayerDelta[n].GetAmount());
 }

 //вычисляем поправки к весам
 //type_t speed=training_speed/static_cast<type_t>(part_size);
 for(size_t n=NN_LAYER_AMOUNT-1,k=0;k<NN_LAYER_AMOUNT;n--,k++)
 {
  PauseInMs(PAUSE_IN_MS);
  CCUDAMatrixStorage<type_t> cCUDAMatrixStorage;
  cCUDAMatrixStorage_LayerOutputH[n].Reinterpret(1,cCUDAMatrixStorage_LayerOutputH[n].GetSizeX()*cCUDAMatrixStorage_LayerOutputH[n].GetSizeY(),cCUDAMatrixStorage_LayerOutputH[n].GetAmount());
  CCUDAMatrixStorage<type_t>::MatrixMulMatrix(cCUDAMatrixStorage,cCUDAMatrixStorage_LayerDelta[n+1],cCUDAMatrixStorage_LayerOutputH[n]);

  //добавляем поправки к весам и смещениям
  CMatrix<type_t> cMatrix_dW(cCUDAMatrixStorage.GetSizeY(),cCUDAMatrixStorage.GetSizeX());
  CMatrix<type_t> cMatrix_dB(cCUDAMatrixStorage_LayerDelta[n+1].GetSizeY(),cCUDAMatrixStorage_LayerDelta[n+1].GetSizeX());

  for(size_t m=0;m<cCUDAMatrixStorage.GetAmount();m++)
  {
   //CMatrix<type_t> cMatrix_w(cMatrix_dW.GetSizeY(),cMatrix_dW.GetSizeX());
   cCUDAMatrixStorage.Copy(m,cMatrix_dW.GetColumnPtr(0));
   CMatrix<type_t>::Add(dLayerWeigh[n],dLayerWeigh[n],cMatrix_dW);
   //CMatrix<type_t>::Mul(cMatrix_w,cMatrix_dW,speed);
   //CMatrix<type_t>::Add(dLayerWeigh[n],dLayerWeigh[n],cMatrix_w);

   CMatrix<type_t> cMatrix_b(cMatrix_dB.GetSizeY(),cMatrix_dB.GetSizeX());
   cCUDAMatrixStorage_LayerDelta[n+1].Copy(m,cMatrix_dB.GetColumnPtr(0));
   CMatrix<type_t>::Add(dLayerBias[n],dLayerBias[n],cMatrix_dB);
   //CMatrix<type_t>::Mul(cMatrix_b,cMatrix_dB,speed);
   //CMatrix<type_t>::Add(dLayerBias[n],dLayerBias[n],cMatrix_b);
  }
 }

 //удаляем созданные матрицы
 cCUDAMatrixStorage_Error.Release();
 for(size_t n=0;n<NN_LAYER_AMOUNT+1;n++)
 {
  cCUDAMatrixStorage_LayerDifferential[n].Release();
  cCUDAMatrixStorage_LayerOutputZ[n].Release();
  cCUDAMatrixStorage_LayerOutputH[n].Release();
 }

 /*
 SaveCUDAMatrixStorage("input_delta",0,cCUDAMatrixStorage_LayerDelta[0]);
 throw "stop";
 */



 //--------------------------------------------------
 //выполняем обратное распространение в слое субдискретизации
 //--------------------------------------------------
 //приведём вертикальные матрицы к горизонтальным с размером результата субдискретизации с увеличением их количества
 size_t backward_delta_layer_b_amount=cCUDAMatrixStorage_LayerDelta[0].GetAmount();
 size_t backward_delta_layer_b_size_x=cCUDAMatrixStorage_LayerDelta[0].GetSizeX();
 size_t backward_delta_layer_b_size_y=cCUDAMatrixStorage_LayerDelta[0].GetSizeY();
 cCUDAMatrixStorage_LayerDelta[0].Reinterpret(1,forward_pooling_b_size_x,forward_pooling_b_size_y*(backward_delta_layer_b_amount*backward_delta_layer_b_size_x*backward_delta_layer_b_size_y)/(forward_pooling_b_size_x*forward_pooling_b_size_y));

 //делаем обратную субдискретизацию
 PauseInMs(PAUSE_IN_MS);
 CCUDAMaxDePooling<type_t> cCUDAMaxDePooling_B(cCUDAMatrixStorage_LayerDelta[0].GetAmount());
 cCUDAMaxDePooling_B.cCUDAMatrixStorage_Input.Connect(cCUDAMatrixStorage_LayerDelta[0]);
 cCUDAMaxDePooling_B.cCUDAMatrixStorage_InputIndex.Connect(cCUDAMaxPooling_B.cCUDAMatrixStorage_OutputIndex);
 cCUDAMaxDePooling_B.MaxDePooling(forward_pooling_conv_b_width,forward_pooling_conv_b_height,forward_conv_b_width,forward_conv_b_height);

 //удаляем ненужные данные
 cCUDAMaxPooling_B.Release();
 for(size_t n=0;n<NN_LAYER_AMOUNT+1;n++) cCUDAMatrixStorage_LayerDelta[n].Release();

 /*
 cCUDAMaxDePooling_B.cCUDAMatrixStorage_Output.Reinterpret(forward_conv_b_width,forward_conv_b_height,cCUDAMaxDePooling_B.cCUDAMatrixStorage_Output.GetAmount()*cCUDAMaxDePooling_B.cCUDAMatrixStorage_Output.GetSizeY());
 SaveCUDAMatrixStorage("de_pooling_b",0,cCUDAMaxDePooling_B.cCUDAMatrixStorage_Output,true);
 throw "stop";
 */

 //--------------------------------------------------
 //формируем производную по входу на функцию нейрона после слоя свёртки
 //--------------------------------------------------
 PauseInMs(PAUSE_IN_MS);
 CCUDAFunction<type_t> cCUDAFunction_dB(cCUDAForwardConvolution_B.cCUDAMatrixStorage_Output.GetAmount());
 cCUDAFunction_dB.cCUDAMatrixStorage_Input.Connect(cCUDAForwardConvolution_B.cCUDAMatrixStorage_Output);
 cCUDAFunction_dB.ApplyDifferentialLeakyReLU();

 //--------------------------------------------------
 //умножаем ошибку на выходе слоя на производную функции нейрона
 //--------------------------------------------------
 //представим строки столбцами
 size_t backward_db_amount=cCUDAFunction_dB.cCUDAMatrixStorage_Output.GetAmount();
 size_t backward_db_size_x=cCUDAFunction_dB.cCUDAMatrixStorage_Output.GetSizeX();
 size_t backward_db_size_y=cCUDAFunction_dB.cCUDAMatrixStorage_Output.GetSizeY();
 cCUDAFunction_dB.cCUDAMatrixStorage_Output.Reinterpret(backward_db_size_x*backward_db_size_y,1,backward_db_amount);

 //представим строки столбцами
 size_t backward_db_depooling_amount=cCUDAMaxDePooling_B.cCUDAMatrixStorage_Output.GetAmount();
 size_t backward_db_depooling_size_x=cCUDAMaxDePooling_B.cCUDAMatrixStorage_Output.GetSizeX();
 size_t backward_db_depooling_size_y=cCUDAMaxDePooling_B.cCUDAMatrixStorage_Output.GetSizeY();
 cCUDAMaxDePooling_B.cCUDAMatrixStorage_Output.Reinterpret(backward_db_depooling_size_x*backward_db_depooling_size_y*(backward_db_depooling_amount/backward_db_amount),1,backward_db_amount);

 //выполняем построчное скалярное произведение
 PauseInMs(PAUSE_IN_MS);
 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_DeltaB;
 CCUDAMatrixStorage<type_t>::MatrixColumnScalarProduction(cCUDAMatrixStorage_DeltaB,cCUDAFunction_dB.cCUDAMatrixStorage_Output,cCUDAMaxDePooling_B.cCUDAMatrixStorage_Output);

 //результат интерпретируем как строки матрицы
 cCUDAMatrixStorage_DeltaB.Reinterpret(cCUDAForwardConvolution_B.cCUDAMatrixStorage_Output.GetSizeY(),cCUDAForwardConvolution_B.cCUDAMatrixStorage_Output.GetSizeX(),cCUDAForwardConvolution_B.cCUDAMatrixStorage_Output.GetAmount());
 //удаляем ненужные матрицы
 cCUDAMaxDePooling_B.Release();
 cCUDAFunction_dB.Release();
 cCUDAFunction_B.Release();
 cCUDAForwardConvolution_B.Release();

 /*
 cCUDAMatrixStorage_DeltaB.Reinterpret(forward_conv_b_width,forward_conv_b_height,cCUDAMatrixStorage_DeltaB.GetAmount()*cCUDAMatrixStorage_DeltaB.GetSizeY());
 SaveCUDAMatrixStorage("de_pooling_f_b",0,cCUDAMatrixStorage_DeltaB,true);
 throw "stop";
 */


 //--------------------------------------------------
 //вычисляем поправки к ядрам B
 //--------------------------------------------------
 PauseInMs(PAUSE_IN_MS);
 CCUDABackConvolution<type_t> cCUDABackConvolution_B;
 cCUDABackConvolution_B.cCUDAMatrixStorage_Delta.Connect(cCUDAMatrixStorage_DeltaB);
 cCUDABackConvolution_B.cCUDAMatrixStorage_Image.Connect(cCUDAMaxPooling_A.cCUDAMatrixStorage_Output);
 size_t backward_conv_b_width;
 size_t backward_conv_b_height;
 cCUDABackConvolution_B.BackConvolution(forward_pooling_conv_a_width,forward_pooling_conv_a_height,forward_conv_b_width,forward_conv_b_height,backward_conv_b_width,backward_conv_b_height);
 /*
 cCUDABackConvolution_B.cCUDAMatrixStorage_Output.Reinterpret(KERNEL_B_HEIGHT,KERNEL_B_WIDTH,cCUDABackConvolution_B.cCUDAMatrixStorage_Output.GetAmount()*cCUDABackConvolution_B.cCUDAMatrixStorage_Output.GetSizeY());
 SaveCUDAMatrixStorage("back_conv_b",0,cCUDABackConvolution_B.cCUDAMatrixStorage_Output,true);
 throw "stop";
 */

 //обновляем ядра B
 //добавляем поправки к весам и смещениям
 for(size_t n=0;n<cCUDABackConvolution_B.cCUDAMatrixStorage_Output.GetAmount();n++)
 {
  CMatrix<type_t> cMatrix_dKBiasB(cCUDABackConvolution_B.cCUDAMatrixStorage_OutputBias.GetSizeY(),cCUDABackConvolution_B.cCUDAMatrixStorage_OutputBias.GetSizeX());
  cCUDABackConvolution_B.cCUDAMatrixStorage_OutputBias.Copy(n,cMatrix_dKBiasB.GetColumnPtr(0));
  //CMatrix<type_t>::Mul(cMatrix_dKBiasB,cMatrix_dKBiasB,speed_dkb);
  //корректируем смещения ядер B
  CMatrix<type_t>::Add(dKernelBiasB[n],dKernelBiasB[n],cMatrix_dKBiasB);

  CMatrix<type_t> cMatrix_dKB(cCUDABackConvolution_B.cCUDAMatrixStorage_Output.GetSizeY(),cCUDABackConvolution_B.cCUDAMatrixStorage_Output.GetSizeX());
  cCUDABackConvolution_B.cCUDAMatrixStorage_Output.Copy(n,cMatrix_dKB.GetColumnPtr(0));
  //CMatrix<type_t>::Mul(cMatrix_dKB,cMatrix_dKB,speed_dkb);
  //корректируем веса ядер B
  CMatrix<type_t>::Add(dKernelB[n],dKernelB[n],cMatrix_dKB);

  /*
   char str[255];

   sprintf(str,"dKernel_bias_b_%i.txt",n);
   SaveMatrix(str,cMatrix_dKBiasB);
   sprintf(str,"Kernel_bias_b_%i.txt",n);
   SaveMatrix(str,KernelBiasB[n]);

   sprintf(str,"dKernelb_%i.txt",n);
   SaveMatrix(str,cMatrix_dKB);
   sprintf(str,"Kernelb_%i.txt",n);
   SaveMatrix(str,KernelB[n]);
   */

 }

 /*
 cCUDABackConvolution_B.cCUDAMatrixStorage_Output.Reinterpret(KERNEL_B_HEIGHT,KERNEL_B_WIDTH,cCUDABackConvolution_B.cCUDAMatrixStorage_Output.GetAmount()*cCUDABackConvolution_B.cCUDAMatrixStorage_Output.GetSizeY());
 SaveCUDAMatrixStorage("p_b",0,cCUDABackConvolution_B.cCUDAMatrixStorage_Output,true);
 throw "stop";
 */

 /*
 for(size_t n=0;n<cCUDABackConvolution_B.cCUDAMatrixStorage_Output.GetAmount();n++)
 {
  for(size_t m=0;m<dKernelB[n].GetSizeY();m++)
  {
   CMatrix<type_t> cMatrix(KERNEL_B_HEIGHT,KERNEL_B_WIDTH);
   type_t *ptr_m=cMatrix.GetColumnPtr(0);
   type_t *ptr_k=dKernelB[n].GetColumnPtr(m);
   for(size_t p=0;p<dKernelB[n].GetSizeX();p++,ptr_m++,ptr_k++) *ptr_m=*ptr_k;
   char str[255];
   sprintf(str,"dkernel_b[%i][%i].tga",m,n);
   cMatrix.SaveImage(str,10,10);
  }
 }
 throw "stop";
 */


 //удаляем ненужные данные
 cCUDABackConvolution_B.Release();

 //--------------------------------------------------
 //переносим дельты на слой A
 //--------------------------------------------------
 PauseInMs(PAUSE_IN_MS);
 CCUDABackDeConvolution<type_t> cCUDABackDeConvolution_B;
 cCUDABackDeConvolution_B.cCUDAMatrixStorage_Delta.Connect(cCUDAMatrixStorage_DeltaB);
 cCUDABackDeConvolution_B.cCUDAMatrixStorage_Kernel.Connect(CUDA_KernelB);
 size_t backward_de_conv_b_width;
 size_t backward_de_conv_b_height;
 cCUDABackDeConvolution_B.BackDeConvolution(forward_conv_b_width,forward_conv_b_height,KERNEL_B_WIDTH,KERNEL_B_HEIGHT,backward_de_conv_b_width,backward_de_conv_b_height);

 //--------------------------------------------------
 //выполняем обратное распространение в слое субдискретизации
 //--------------------------------------------------
 //делаем обратную субдискретизацию
 PauseInMs(PAUSE_IN_MS);
 CCUDAMaxDePooling<type_t> cCUDAMaxDePooling_A(cCUDAMaxPooling_A.cCUDAMatrixStorage_OutputIndex.GetAmount());
 cCUDAMaxDePooling_A.cCUDAMatrixStorage_Input.Connect(cCUDABackDeConvolution_B.cCUDAMatrixStorage_Output);
 cCUDAMaxDePooling_A.cCUDAMatrixStorage_InputIndex.Connect(cCUDAMaxPooling_A.cCUDAMatrixStorage_OutputIndex);
 //реинтерпретируем входной вектор
 cCUDAMaxDePooling_A.cCUDAMatrixStorage_Input.Reinterpret(cCUDAMaxPooling_A.cCUDAMatrixStorage_OutputIndex.GetSizeY(),cCUDAMaxPooling_A.cCUDAMatrixStorage_OutputIndex.GetSizeX(),cCUDAMaxPooling_A.cCUDAMatrixStorage_OutputIndex.GetAmount());
 cCUDAMaxDePooling_A.MaxDePooling(forward_pooling_conv_a_width,forward_pooling_conv_a_height,forward_conv_a_width,forward_conv_a_height);

 //удаляем ненужные данные
 cCUDAMaxPooling_A.Release();
 cCUDABackDeConvolution_B.Release();
 //--------------------------------------------------
 //формируем производную по входу на функцию нейрона после слоя свёртки
 //--------------------------------------------------
 PauseInMs(PAUSE_IN_MS);
 CCUDAFunction<type_t> cCUDAFunction_dA(cCUDAForwardConvolution_A.cCUDAMatrixStorage_Output.GetAmount());
 cCUDAFunction_dA.cCUDAMatrixStorage_Input.Connect(cCUDAForwardConvolution_A.cCUDAMatrixStorage_Output);
 cCUDAFunction_dA.ApplyDifferentialLeakyReLU();

 //--------------------------------------------------
 //умножаем ошибку на выходе слоя на производную функции нейрона
 //--------------------------------------------------
 //представим строки столбцами
 size_t backward_da_amount=cCUDAFunction_dA.cCUDAMatrixStorage_Output.GetAmount();
 size_t backward_da_size_x=cCUDAFunction_dA.cCUDAMatrixStorage_Output.GetSizeX();
 size_t backward_da_size_y=cCUDAFunction_dA.cCUDAMatrixStorage_Output.GetSizeY();
 cCUDAFunction_dA.cCUDAMatrixStorage_Output.Reinterpret(backward_da_size_x*backward_da_size_y,1,backward_da_amount);
 //представим строки столбцами
 size_t backward_da_depooling_amount=cCUDAMaxDePooling_A.cCUDAMatrixStorage_Output.GetAmount();
 size_t backward_da_depooling_size_x=cCUDAMaxDePooling_A.cCUDAMatrixStorage_Output.GetSizeX();
 size_t backward_da_depooling_size_y=cCUDAMaxDePooling_A.cCUDAMatrixStorage_Output.GetSizeY();
 cCUDAMaxDePooling_A.cCUDAMatrixStorage_Output.Reinterpret(backward_da_depooling_size_x*backward_da_depooling_size_y*(backward_da_depooling_amount/backward_da_amount),1,backward_da_amount);

 //выполняем построчное скалярное произведение
 PauseInMs(PAUSE_IN_MS);
 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_DeltaA;
 CCUDAMatrixStorage<type_t>::MatrixColumnScalarProduction(cCUDAMatrixStorage_DeltaA,cCUDAFunction_dA.cCUDAMatrixStorage_Output,cCUDAMaxDePooling_A.cCUDAMatrixStorage_Output);

 //результат интерпретируем как строки матрицы
 cCUDAMatrixStorage_DeltaA.Reinterpret(cCUDAForwardConvolution_A.cCUDAMatrixStorage_Output.GetSizeY(),cCUDAForwardConvolution_A.cCUDAMatrixStorage_Output.GetSizeX(),cCUDAForwardConvolution_A.cCUDAMatrixStorage_Output.GetAmount());

 //удаляем ненужные матрицы
 cCUDAMaxDePooling_A.Release();
 cCUDAFunction_dA.Release();
 cCUDAFunction_A.Release();
 cCUDAForwardConvolution_A.Release();

 //--------------------------------------------------
 //вычисляем поправки к ядрам A
 //--------------------------------------------------
 PauseInMs(PAUSE_IN_MS);
 CCUDABackConvolution<type_t> cCUDABackConvolution_A;
 cCUDABackConvolution_A.cCUDAMatrixStorage_Delta.Connect(cCUDAMatrixStorage_DeltaA);
 cCUDABackConvolution_A.cCUDAMatrixStorage_Image.Connect(CUDA_InputImage);
 size_t backward_conv_a_width;
 size_t backward_conv_a_height;
 cCUDABackConvolution_A.BackConvolution(IMAGE_WIDTH,IMAGE_HEIGHT,forward_conv_a_width,forward_conv_a_height,backward_conv_a_width,backward_conv_a_height);

 //обновляем ядра A
 //добавляем поправки к весам и смещениям
 for(size_t n=0;n<cCUDABackConvolution_A.cCUDAMatrixStorage_Output.GetAmount();n++)
 {
  CMatrix<type_t> cMatrix_dKBiasA(cCUDABackConvolution_A.cCUDAMatrixStorage_OutputBias.GetSizeY(),cCUDABackConvolution_A.cCUDAMatrixStorage_OutputBias.GetSizeX());
  cCUDABackConvolution_A.cCUDAMatrixStorage_OutputBias.Copy(n,cMatrix_dKBiasA.GetColumnPtr(0));
  //CMatrix<type_t>::Mul(cMatrix_dKBiasA,cMatrix_dKBiasA,speed_dka);
  //корректируем смещения ядер A
  CMatrix<type_t>::Add(dKernelBiasA[n],dKernelBiasA[n],cMatrix_dKBiasA);

  CMatrix<type_t> cMatrix_dKA(cCUDABackConvolution_A.cCUDAMatrixStorage_Output.GetSizeY(),cCUDABackConvolution_A.cCUDAMatrixStorage_Output.GetSizeX());
  cCUDABackConvolution_A.cCUDAMatrixStorage_Output.Copy(n,cMatrix_dKA.GetColumnPtr(0));
  //CMatrix<type_t>::Mul(cMatrix_dKA,cMatrix_dKA,speed_dka);
  //корректируем веса ядер A
  CMatrix<type_t>::Add(dKernelA[n],dKernelA[n],cMatrix_dKA);

  /*
   char str[255];

   sprintf(str,"dKernel_bias_a_%i.txt",n);
   SaveMatrix(str,cMatrix_dKBiasA);
   sprintf(str,"Kernel_bias_a_%i.txt",n);
   SaveMatrix(str,KernelBiasA[n]);

   sprintf(str,"dKernela_%i.txt",n);
   SaveMatrix(str,cMatrix_dKA);
   sprintf(str,"Kernela_%i.txt",n);
   SaveMatrix(str,KernelA[n]);
   */
 }
 return(true);
}

//----------------------------------------------------------------------------------------------------
//перемешать обучающие образы
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CCUDACNN<type_t>::ExchangeTrainingImage(std::vector<size_t> &array_index_of_image)
{
 size_t training_amount=array_index_of_image.size();
 for(size_t n=0;n<training_amount;n++)
 {
  size_t index_1=n;
  size_t index_2=static_cast<size_t>((rand()*static_cast<double>(training_amount*10))/static_cast<double>(RAND_MAX));
  index_2%=training_amount;

  size_t tmp=array_index_of_image[index_1];
  array_index_of_image[index_1]=array_index_of_image[index_2];
  array_index_of_image[index_2]=tmp;
 }
}

//----------------------------------------------------------------------------------------------------
//сохранить изображения ядер
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CCUDACNN<type_t>::SaveKernelImage(void)
{
 for(size_t k=0;k<KERNEL_B_AMOUNT;k++)
 {
  for(size_t n=0;n<KernelB[k].GetSizeY();n++)
  {
   CMatrix<type_t> cMatrix(KERNEL_B_HEIGHT,KERNEL_B_WIDTH);
   type_t *ptr=cMatrix.GetColumnPtr(0);
   for(size_t m=0;m<KERNEL_B_HEIGHT*KERNEL_B_WIDTH;m++,ptr++) *ptr=KernelB[k].GetElement(n,m);
   char file_name[255];
   sprintf(file_name,"kernelb[%i][%i].tga",k,n);
   cMatrix.SaveImage(file_name,10,10);
  }
 }
 for(size_t k=0;k<KERNEL_A_AMOUNT;k++)
 {
  for(size_t n=0;n<KernelA[k].GetSizeY();n++)
  {
   CMatrix<type_t> cMatrix(KERNEL_A_HEIGHT,KERNEL_A_WIDTH);
   type_t *ptr=cMatrix.GetColumnPtr(0);
   for(size_t m=0;m<KERNEL_A_HEIGHT*KERNEL_A_WIDTH;m++,ptr++) *ptr=KernelA[k].GetElement(n,m);
   char file_name[255];
   sprintf(file_name,"kernela[%i][%i].tga",k,n);
   cMatrix.SaveImage(file_name,10,10);
  }
 }
}

//----------------------------------------------------------------------------------------------------
//загрузить нейросеть
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CCUDACNN<type_t>::LoadNet(const std::string &file_name)
{
 std::unique_ptr<IDataStream> iDataStream_Ptr(IDataStream::CreateNewDataStreamFile(file_name,false));
 if (iDataStream_Ptr->IsFail()==false)
 {
  for(size_t k=0;k<KERNEL_A_AMOUNT;k++)
  {
   KernelA[k].Load(iDataStream_Ptr.get());
   KernelBiasA[k].Load(iDataStream_Ptr.get());
  }
  for(size_t k=0;k<KERNEL_B_AMOUNT;k++)
  {
   KernelB[k].Load(iDataStream_Ptr.get());
   KernelBiasB[k].Load(iDataStream_Ptr.get());
  }
  for(size_t n=0;n<NN_LAYER_AMOUNT;n++)
  {
   LayerWeigh[n].Load(iDataStream_Ptr.get());
   LayerBias[n].Load(iDataStream_Ptr.get());
  }
  PutMessageToConsole("Нейросеть загружена.\r\n");
  return(true);
 }
 return(false);
}
//----------------------------------------------------------------------------------------------------
//сохранить нейросеть
//----------------------------------------------------------------------------------------------------
template<class type_t>
bool CCUDACNN<type_t>::SaveNet(const std::string &file_name)
{
 //сохраняем нейросеть
 std::unique_ptr<IDataStream> iDataStream_Ptr(IDataStream::CreateNewDataStreamFile(file_name,true));
 if (iDataStream_Ptr->IsFail()==false)
 {
  for(size_t k=0;k<KERNEL_A_AMOUNT;k++)
  {
   KernelA[k].Save(iDataStream_Ptr.get());
   KernelBiasA[k].Save(iDataStream_Ptr.get());
  }
  for(size_t k=0;k<KERNEL_B_AMOUNT;k++)
  {
   KernelB[k].Save(iDataStream_Ptr.get());
   KernelBiasB[k].Save(iDataStream_Ptr.get());
  }
  for(size_t n=0;n<NN_LAYER_AMOUNT;n++)
  {
   LayerWeigh[n].Save(iDataStream_Ptr.get());
   LayerBias[n].Save(iDataStream_Ptr.get());
  }
  PutMessageToConsole("Нейросеть записана.\r\n");
  return(true);
 }
 return(false);
}

//----------------------------------------------------------------------------------------------------
//инициализировать нейросеть
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CCUDACNN<type_t>::InitNet(void)
{
 char str[STRING_BUFFER_SIZE];

 //рассчитаем количество свёрток с одного входного изображения
 size_t conv_amount=KERNEL_B_AMOUNT;
 //рассчитаем выходной размер одного входного изображения
 size_t input_width=IMAGE_WIDTH;
 size_t input_height=IMAGE_HEIGHT;
 //свёртка A
 size_t conv_a_width=input_width-KERNEL_A_WIDTH+1;
 size_t conv_a_height=input_height-KERNEL_A_HEIGHT+1;
 //субдискретизация
 size_t pooling_a_width=conv_a_width/POOLING_A_WIDTH;
 size_t pooling_a_height=conv_a_height/POOLING_A_HEIGHT;
 //свёртка B
 size_t conv_b_width=pooling_a_width-KERNEL_B_WIDTH+1;
 size_t conv_b_height=pooling_a_height-KERNEL_B_HEIGHT+1;
 //субдискретизация
 size_t pooling_b_width=conv_b_width/POOLING_B_WIDTH;
 size_t pooling_b_height=conv_b_height/POOLING_B_HEIGHT;

 //инициализируем нейросеть
 for(size_t n=0;n<KERNEL_A_AMOUNT;n++)
 {
  InitKernel(KernelA[n],KernelBiasA[n],KERNEL_A_WIDTH,KERNEL_A_HEIGHT,CImage<type_t>::COLOR_CHANNEL,input_width,input_height);

  dKernelA[n]=CMatrix<type_t>(KernelA[n].GetSizeY(),KernelA[n].GetSizeX());
  dKernelBiasA[n]=CMatrix<type_t>(KernelBiasA[n].GetSizeY(),KernelBiasA[n].GetSizeX());

  LastdKernelA[n]=CMatrix<type_t>(KernelA[n].GetSizeY(),KernelA[n].GetSizeX());
  LastdKernelBiasA[n]=CMatrix<type_t>(KernelBiasA[n].GetSizeY(),KernelBiasA[n].GetSizeX());

  LastdKernelA[n].Zero();
  LastdKernelBiasA[n].Zero();

  KernelBiasA[n].Zero();
 }
 for(size_t n=0;n<KERNEL_B_AMOUNT;n++)
 {
  InitKernel(KernelB[n],KernelBiasB[n],KERNEL_B_WIDTH,KERNEL_B_HEIGHT,KERNEL_A_AMOUNT,pooling_a_width,pooling_a_height);

  dKernelB[n]=CMatrix<type_t>(KernelB[n].GetSizeY(),KernelB[n].GetSizeX());
  dKernelBiasB[n]=CMatrix<type_t>(KernelBiasB[n].GetSizeY(),KernelBiasB[n].GetSizeX());

  LastdKernelB[n]=CMatrix<type_t>(KernelB[n].GetSizeY(),KernelB[n].GetSizeX());
  LastdKernelBiasB[n]=CMatrix<type_t>(KernelBiasB[n].GetSizeY(),KernelBiasB[n].GetSizeX());

  LastdKernelB[n].Zero();
  LastdKernelBiasB[n].Zero();

  KernelBiasB[n].Zero();
 }
 sprintf(str,"Входов полносвязного слоя:%i (свёрток:%i ширина:%i высота:%i)\r\n",conv_amount*pooling_b_width*pooling_b_height,conv_amount,pooling_b_width,pooling_b_height);
 PutMessageToConsole(str);

 //создаём полносвязный слой и инициализируем его веса
 size_t neuron_in_layer[NN_LAYER_AMOUNT+1]={conv_amount*pooling_b_width*pooling_b_height,conv_amount*OUTPUT_LAYER_SIZE,OUTPUT_LAYER_SIZE};
 for(size_t n=1;n<NN_LAYER_AMOUNT+1;n++)
 {
  LayerWeigh[n-1]=CMatrix<type_t>(neuron_in_layer[n],neuron_in_layer[n-1]);
  InitWeight(LayerWeigh[n-1]);

  dLayerWeigh[n-1]=CMatrix<type_t>(neuron_in_layer[n],neuron_in_layer[n-1]);
  LastdLayerWeigh[n-1]=CMatrix<type_t>(neuron_in_layer[n],neuron_in_layer[n-1]);
  LastdLayerWeigh[n-1].Zero();

  LayerBias[n-1]=CMatrix<type_t>(neuron_in_layer[n],1);
  LayerBias[n-1].Zero();
  InitWeight(LayerBias[n-1]);

  dLayerBias[n-1]=CMatrix<type_t>(neuron_in_layer[n],1);
  LastdLayerBias[n-1]=CMatrix<type_t>(neuron_in_layer[n],1);
  LastdLayerBias[n-1].Zero();
 }

 //выделяем память для ядер свёртки и копируем ядра
 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_KernelA(KernelA[0].GetSizeY(),KernelA[0].GetSizeX(),KERNEL_A_AMOUNT);
 cCUDAMatrixStorage_KernelA.Create();
 CUDA_KernelA.Move(cCUDAMatrixStorage_KernelA);

 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_KernelBiasA(KernelBiasA[0].GetSizeY(),KernelBiasA[0].GetSizeX(),KERNEL_A_AMOUNT);
 cCUDAMatrixStorage_KernelBiasA.Create();
 CUDA_KernelBiasA.Move(cCUDAMatrixStorage_KernelBiasA);

 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_KernelB(KernelB[0].GetSizeY(),KernelB[0].GetSizeX(),KERNEL_B_AMOUNT);
 cCUDAMatrixStorage_KernelB.Create();
 CUDA_KernelB.Move(cCUDAMatrixStorage_KernelB);

 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_KernelBiasB(KernelBiasB[0].GetSizeY(),KernelBiasB[0].GetSizeX(),KERNEL_B_AMOUNT);
 cCUDAMatrixStorage_KernelBiasB.Create();
 CUDA_KernelBiasB.Move(cCUDAMatrixStorage_KernelBiasB);

 //выделяем память для полносвязного слоя
 for(size_t n=0;n<NN_LAYER_AMOUNT;n++)
 {
  CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_LayerWeigh(LayerWeigh[n].GetSizeY(),LayerWeigh[n].GetSizeX(),1);
  cCUDAMatrixStorage_LayerWeigh.Create();
  CUDA_LayerWeigh[n].Move(cCUDAMatrixStorage_LayerWeigh);

  CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_LayerBias(LayerBias[n].GetSizeY(),LayerBias[n].GetSizeX(),1);
  cCUDAMatrixStorage_LayerBias.Create();
  CUDA_LayerBias[n].Move(cCUDAMatrixStorage_LayerBias);
 }
}
//----------------------------------------------------------------------------------------------------
//обнулить поправки к весам и смещениям
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CCUDACNN<type_t>::ResetDeltaWeighAndBias(void)
{
 for(size_t n=0;n<NN_LAYER_AMOUNT;n++)
 {
  dLayerWeigh[n].Zero();
  dLayerBias[n].Zero();
 }
 for(size_t n=0;n<KERNEL_A_AMOUNT;n++)
 {
  dKernelA[n].Zero();
  dKernelBiasA[n].Zero();
 }
 for(size_t n=0;n<KERNEL_B_AMOUNT;n++)
 {
  dKernelB[n].Zero();
  dKernelBiasB[n].Zero();
 }
}

//----------------------------------------------------------------------------------------------------
//обновить веса и смещения
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CCUDACNN<type_t>::UpdateWeighAndBias(double speed)
{
/*
 for(size_t n=0;n<NN_LAYER_AMOUNT;n++)
 {
  CMatrix<type_t>::Mul(dLayerWeigh[n],dLayerWeigh[n],speed);
  CMatrix<type_t>::Mul(dLayerBias[n],dLayerBias[n],speed);
  CMatrix<type_t>::Sub(LayerWeigh[n],LayerWeigh[n],dLayerWeigh[n]);
  CMatrix<type_t>::Sub(LayerBias[n],LayerBias[n],dLayerBias[n]);
 }

 bool error=true;
 for(size_t n=0;n<KERNEL_A_AMOUNT;n++)
 {
  CMatrix<type_t>::Mul(dKernelA[n],dKernelA[n],speed);
  CMatrix<type_t>::Mul(dKernelBiasA[n],dKernelBiasA[n],speed);

  CMatrix<type_t>::Sub(KernelA[n],KernelA[n],dKernelA[n]);
  CMatrix<type_t>::Sub(KernelBiasA[n],KernelBiasA[n],dKernelBiasA[n]);

  for(size_t y=0;y<dKernelA[n].GetSizeY();y++)
  {
   for(size_t x=0;x<dKernelA[n].GetSizeX();x++)
   {
    type_t v=dKernelA[n].GetElement(y,x);
    if (fabs(v)>1e-9) error=false;
   }
  }
 }
 for(size_t n=0;n<KERNEL_B_AMOUNT;n++)
 {
  CMatrix<type_t>::Mul(dKernelB[n],dKernelB[n],speed);
  CMatrix<type_t>::Mul(dKernelBiasB[n],dKernelBiasB[n],speed);

  CMatrix<type_t>::Sub(KernelB[n],KernelB[n],dKernelB[n]);
  CMatrix<type_t>::Sub(KernelBiasB[n],KernelBiasB[n],dKernelBiasB[n]);

  for(size_t y=0;y<dKernelB[n].GetSizeY();y++)
  {
   for(size_t x=0;x<dKernelB[n].GetSizeX();x++)
   {
    type_t v=dKernelB[n].GetElement(y,x);
	if (fabs(v)>1e-9) error=false;
   }
  }
 }
 if (error==true) PutMessageToConsole("ЯДРА НЕ ОБУЧАЮТСЯ! ПОПРАВКИ БЛИЗКИ К НУЛЮ!\r\n");
*/


 type_t gamma=0;
 for(size_t n=0;n<NN_LAYER_AMOUNT;n++)
 {
  CMatrix<type_t>::Mul(dLayerWeigh[n],dLayerWeigh[n],speed);
  CMatrix<type_t>::Mul(dLayerBias[n],dLayerBias[n],speed);

  CMatrix<type_t>::Mul(LastdLayerWeigh[n],LastdLayerWeigh[n],gamma);
  CMatrix<type_t>::Mul(LastdLayerBias[n],LastdLayerBias[n],gamma);

  CMatrix<type_t>::Add(LastdLayerWeigh[n],LastdLayerWeigh[n],dLayerWeigh[n]);
  CMatrix<type_t>::Add(LastdLayerBias[n],LastdLayerBias[n],dLayerBias[n]);

  CMatrix<type_t>::Sub(LayerWeigh[n],LayerWeigh[n],LastdLayerWeigh[n]);
  CMatrix<type_t>::Sub(LayerBias[n],LayerBias[n],LastdLayerBias[n]);
 }
 bool error=true;
 for(size_t n=0;n<KERNEL_A_AMOUNT;n++)
 {
  CMatrix<type_t>::Mul(dKernelA[n],dKernelA[n],speed);
  CMatrix<type_t>::Mul(dKernelBiasA[n],dKernelBiasA[n],speed);


  CMatrix<type_t>::Mul(LastdKernelA[n],LastdKernelA[n],gamma);
  CMatrix<type_t>::Mul(LastdKernelBiasA[n],LastdKernelBiasA[n],gamma);

  CMatrix<type_t>::Add(LastdKernelA[n],LastdKernelA[n],dKernelA[n]);
  CMatrix<type_t>::Add(LastdKernelBiasA[n],LastdKernelBiasA[n],dKernelBiasA[n]);

  CMatrix<type_t>::Sub(KernelA[n],KernelA[n],LastdKernelA[n]);
  CMatrix<type_t>::Sub(KernelBiasA[n],KernelBiasA[n],LastdKernelBiasA[n]);

  for(size_t y=0;y<dKernelA[n].GetSizeY();y++)
  {
   for(size_t x=0;x<dKernelA[n].GetSizeX();x++)
   {
    type_t v=dKernelA[n].GetElement(y,x);
    if (fabs(v)>1e-9) error=false;
   }
  }
 }
 for(size_t n=0;n<KERNEL_B_AMOUNT;n++)
 {
  CMatrix<type_t>::Mul(dKernelB[n],dKernelB[n],speed);
  CMatrix<type_t>::Mul(dKernelBiasB[n],dKernelBiasB[n],speed);


  CMatrix<type_t>::Mul(LastdKernelB[n],LastdKernelB[n],gamma);
  CMatrix<type_t>::Mul(LastdKernelBiasB[n],LastdKernelBiasB[n],gamma);

  CMatrix<type_t>::Add(LastdKernelB[n],LastdKernelB[n],dKernelB[n]);
  CMatrix<type_t>::Add(LastdKernelBiasB[n],LastdKernelBiasB[n],dKernelBiasB[n]);

  CMatrix<type_t>::Sub(KernelB[n],KernelB[n],LastdKernelB[n]);
  CMatrix<type_t>::Sub(KernelBiasB[n],KernelBiasB[n],LastdKernelBiasB[n]);

  for(size_t y=0;y<dKernelB[n].GetSizeY();y++)
  {
   for(size_t x=0;x<dKernelB[n].GetSizeX();x++)
   {
    type_t v=dKernelB[n].GetElement(y,x);
	if (fabs(v)>1e-9) error=false;
   }
  }
 }
 if (error==true) PutMessageToConsole("ЯДРА НЕ ОБУЧАЮТСЯ! ПОПРАВКИ БЛИЗКИ К НУЛЮ!\r\n");

}

//----------------------------------------------------------------------------------------------------
//обучить нейросеть
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CCUDACNN<type_t>::TrainingNet(void)
{
 char str[STRING_BUFFER_SIZE];

 //начинаем обучение блоками
 size_t image_amount=InputImage.size();
 static const double max_cost=0.1;//максимальная ошибка
 static const double speed=0.25;//скорость обучения

 std::vector<size_t> array_index_of_image(image_amount);//индексы обучающих образов
 for(size_t n=0;n<image_amount;n++) array_index_of_image[n]=n;

 size_t iteration=0;
 while(1)
 {
  //выполняем обучение наборов
  ExchangeTrainingImage(array_index_of_image);//перемешиваем индексы обучающих образов

  size_t image_index=0;
  size_t image_counter=image_amount;
  double current_max_cost=0;
  bool training_done=true;
  double begin_time=GetSecondCounter();
  size_t part_index=0;

  while(image_index<image_amount)
  {
   size_t part_size=CUDA_MAX_INPUT_IMAGE_AMOUNT;
   if (image_counter<part_size) part_size=image_counter;
   std::vector<size_t> image_kit(part_size);
   for(size_t n=0;n<part_size;n++) image_kit[n]=array_index_of_image[image_index+n];

   //очищаем добавки к весам и смещениям
   ResetDeltaWeighAndBias();

   double cost=0;
   bool only_cost=false;
   double net_begin_time=GetSecondCounter();
   if (NetProcessing(only_cost,max_cost,image_kit,cost)==false) throw "CCUDACNN<type_t>::NetProcessing ошибка выполнения";

   if (cost>=max_cost) UpdateWeighAndBias(speed/static_cast<double>(part_size));//обновляем веса и смещения

   if (cost>current_max_cost) current_max_cost=cost;
   if (cost>=max_cost) training_done=false;
   double delta_t=GetSecondCounter()-net_begin_time;
   sprintf(str,"\tIteration:%i Part:%i cost:%f time:%f second\r\n",iteration+1,part_index,cost,delta_t);
   PutMessageToConsole(str);

   //char net_name[255];
   //sprintf(net_name,"cnn-neuronet-i%i-p%i-c%f.net",iteration,part_index,cost);
   //SaveNet(net_name);

   SaveNet("cnn-neuronet.net");

   image_counter-=part_size;
   image_index+=part_size;
   part_index++;
  }

  iteration++;
  sprintf(str,"Cost:%f Iteration:%i\r\n",current_max_cost,iteration);
  PutMessageToConsole(str);
  double delta_t=GetSecondCounter()-begin_time;
  sprintf(str,"time: %f second\r\n\r\n",delta_t);
  PutMessageToConsole(str);
  PutMessageToConsole("--------------------------------------------------\r\n");

  //набор не обучается, если его ошибка меньше требуемой: если все наборы не обучаются, значит, обучение завершено
  if (training_done==true) break;
 }
}

//****************************************************************************************************
//открытые функции класса
//****************************************************************************************************

//----------------------------------------------------------------------------------------------------
//запустить на выполнение
//----------------------------------------------------------------------------------------------------
template<class type_t>
void CCUDACNN<type_t>::Execute(void)
{
 //производим тестирование классов
 CCUDAForwardConvolution<float>::Test();
 CCUDABackConvolution<float>::Test();
 CCUDABackDeConvolution<float>::Test();
 //CCUDAMatrixStorage<float>::Test();
 CMatrix<float>::Test();

 //инициализируем нейросеть
 InitNet();
 //загружаем нейросеть
 LoadNet("cnn-neuronet.net");

 //SaveKernelImage();
 //throw "stop";

 //загружаем образы для обучения
 LoadImage();

 //обучаем нейросеть
 TrainingNet();
}


 /*
 CCUDAMatrixStorage<type_t> cCUDAMatrixStorage_Input(2,4*4,2);
 cCUDAMatrixStorage_Input.Create();
 type_t input_1[]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16, 16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1};
 type_t input_2[]={-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16, -16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1};
 cCUDAMatrixStorage_Input.Set(0,input_1);
 cCUDAMatrixStorage_Input.Set(1,input_2);
 cCUDAMatrixStorage_Input.Reinterpret(1,4*4,2*2);

 PauseInMs(PAUSE_IN_MS);
 CCUDAMaxPooling<type_t> cCUDAMaxPooling(2*2);
 cCUDAMaxPooling.cCUDAMatrixStorage_Input.Connect(cCUDAMatrixStorage_Input);
 size_t pooling_width;
 size_t pooling_height;
 cCUDAMaxPooling.MaxPooling(4,4,2,2,pooling_width,pooling_height);


 PauseInMs(PAUSE_IN_MS);
 CCUDAMaxDePooling<type_t> cCUDAMaxDePooling(2*2);
 cCUDAMaxDePooling.cCUDAMatrixStorage_Input.Connect(cCUDAMaxPooling.cCUDAMatrixStorage_Output);
 cCUDAMaxDePooling.cCUDAMatrixStorage_InputIndex.Connect(cCUDAMaxPooling.cCUDAMatrixStorage_OutputIndex);
 cCUDAMaxDePooling.MaxDePooling(pooling_width,pooling_height,4,4);

 SaveCUDAMatrixStorage("depooling",0,cCUDAMaxDePooling.cCUDAMatrixStorage_Output);
 return;
 */

#endif

#include <windows.h>
#include <math.h>
#include <conio.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

//------------------------------------------------------------------------------
#include "cuda/cuda.cu.h"
//------------------------------------------------------------------------------

const size_t image_width=5;
const size_t image_height=5;

const size_t delta_width=3;
const size_t delta_height=3;

const size_t image_amount=5;
const size_t image_depth=3;

const size_t delta_depth=7;

const size_t output_width=image_width-delta_width+1;
const size_t output_height=image_height-delta_height+1;

static size_t TestArray[image_depth*output_height*output_width*image_amount*delta_depth];

void Test(size_t image_index,size_t delta_index,size_t i)
{	
 size_t output_index=delta_index;
 size_t delta_ptr=image_index*delta_width*delta_height*delta_depth+delta_index*delta_width*delta_height;
 size_t image_ptr=image_index*image_width*image_height*image_depth;
 size_t output_ptr=output_index*image_amount*output_width*output_height*image_depth+image_index*output_width*output_height;

 size_t padding=0;
 size_t step=1;

 //расчитываем градиенты весов фильтров и смещений
 
 FILE *file=fopen("out.txt","ab");
 fprintf(file,"ImageIndex:%i DeltaIndex:%i\r\n",image_index,delta_index);

 //for(size_t i=0;i<output_height;i++)
 {
  for(size_t j=0;j<output_width;j++)
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

     size_t d_ptr=delta_ptr+y*delta_width+x;

     //наращиваем градиент фильтра
     for(size_t c=0;c<image_depth;c++)
     {
      //size_t i_ptr=image_ptr+c*image_width*image_height+i0*image_width+j0;
	  size_t o_ptr=output_ptr+c*output_width*output_height*image_amount+i*output_width+j;
	  TestArray[o_ptr]++;
	  fprintf(file,"%i ",o_ptr);
	  //(*o_ptr)+=delta*(*i_ptr);	  
     }
	 fprintf(file,"\r\n");
    }
	fprintf(file,"\r\n");
   }
  }
 }
 fprintf(file,"\r\n----------\r\n");
 fclose(file);
}

int WINAPI WinMain(HINSTANCE hInstance,HINSTANCE hPrevstance,LPSTR lpstrCmdLine,int nCmdShow)
{
	/*
 for(size_t image=0;image<5;image++)
 {
  for(size_t depth=0;depth<7;depth++)
  {
   for(size_t i=0;i<output_height;i++)
   {
    Test(image,depth,i);
   }
  }
 }
 
 FILE *file=fopen("counter.txt","wb");
 for(size_t n=0;n<image_depth*output_height*output_width*image_amount*delta_depth;n++)
 {
  fprintf(file,"%i\r\n",TestArray[n]);
 }
 fclose(file);
 */
	
 try
 {  
  CUDA_Start();
 }
 catch (const char *text)
 {
  MessageBox(NULL,text,"Ошибка исключения",MB_OK);
 } 
 
 MessageBox(NULL,"Завершено","Сообщение", MB_OK);
 return(0);
}

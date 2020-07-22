#include <windows.h>
#include <math.h>
#include <conio.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

//------------------------------------------------------------------------------
#include "cuda/cuda.cu.h"
//------------------------------------------------------------------------------

int WINAPI WinMain(HINSTANCE hInstance,HINSTANCE hPrevstance,LPSTR lpstrCmdLine,int nCmdShow)
{
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

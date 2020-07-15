/* $Id: handle_error.h,v 1.1 2015/01/27 10:31:11 mechanoid Exp $ */
#ifndef _HANDLE_ERROR_H_
#define _HANDLE_ERROR_H_

#include "../system/system.h"

static void HandleError( cudaError_t err, const char *file, int line )
{
 if (err!=cudaSuccess)
 {
  char str[1024];
  sprintf(str,"%s in %s at line %d\n", cudaGetErrorString(err),file,line);
  PutMessage(str);
  exit(EXIT_FAILURE);
 }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))




#endif

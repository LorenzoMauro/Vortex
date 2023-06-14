#pragma once
#ifndef nvccUTILS_H
#define nvccUTILS_H

#ifdef __CUDACC__
#define cuda_SYNCTHREADS() __syncthreads()
#define cuAtomicAdd(a,b) atomicAdd(a,b)
#define cuda_ATOMICCAS(a,b,c) ::atomicCAS(a,b,c)
#define cuda_FLOAT_AS_INT(a) __float_as_int(a)
#define cuda_INT_AS_FLOAT(a) __int_as_float(a)
#else
#define cuda_SYNCTHREADS()
#define cuAtomicAdd(a,b) 1
#define cuda_ATOMICCAS(a,b,c) 1
#define cuda_FLOAT_AS_INT(a) 1
#define cuda_INT_AS_FLOAT(a) 1.0f
#endif


#endif

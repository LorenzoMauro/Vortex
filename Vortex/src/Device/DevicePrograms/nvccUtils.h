#pragma once
#ifndef nvccUTILS_H
#define nvccUTILS_H

#ifdef __CUDACC__
#define cuSynchThread() __syncthreads()
#define cuAtomicAdd(a,b) atomicAdd(a,b)
#define cuAtomicSub(a,b) atomicSub(a,b)
#define cuAtomicCas(a,b,c) ::atomicCAS(a,b,c)
#define cuda_FLOAT_AS_INT(a) __float_as_int(a)
#define cuda_FLOAT_AS_UINT(a) __float_as_uint(a)
#define cuda_INT_AS_FLOAT(a) __int_as_float(a)
#else
#define cuSynchThread()
#define cuAtomicAdd(a,b) 1
#define cuAtomicSub(a,b) 1
#define cuAtomicCas(a,b,c) 1
#define cuda_FLOAT_AS_INT(a) 1
#define cuda_INT_AS_FLOAT(a) 1.0f
#define cuda_FLOAT_AS_UINT(a) 1
#endif

#endif

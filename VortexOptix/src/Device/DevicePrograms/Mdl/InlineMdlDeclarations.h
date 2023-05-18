#pragma once
#ifndef MDL_DECLARATIONS_H
#define MDL_DECLARATIONS_H


#include "MdlStructs.h"
//
// Functions needed by texture runtime when this file is compiled with Clang
//


extern "C" __device__ __inline__ void __itex2D_float(
    float* retVal, cudaTextureObject_t texObject, float x, float y)
{
    float4 tmp;
    asm volatile ("tex.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
                  : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
                  : "l"(texObject), "f"(x), "f"(y));
    *retVal = (float)(tmp.x);
}

extern "C" __device__ __inline__ void __itex2D_float4(
    float4 * retVal, cudaTextureObject_t texObject, float x, float y)
{
    float4 tmp;
    asm volatile ("tex.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
                  : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
                  : "l"(texObject), "f"(x), "f"(y));
    *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

extern "C" __device__ __inline__ void __itex2DGrad_float4(
    float4 * retVal, cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
    float4 tmp;
    asm volatile ("tex.grad.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}], {%7, %8}, {%9, %10};"
                  : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
                  : "l"(texObject), "f"(x), "f"(y), "f"(dPdx.x), "f"(dPdx.y), "f"(dPdy.x), "f"(dPdy.y));
    *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

extern "C" __device__ __inline__ void __itex2DLod_float4(
    float4 * retVal, cudaTextureObject_t texObject, float x, float y, float level)
{
    float4 tmp;
    asm volatile ("tex.level.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}], %7;"
                  : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
                  : "l"(texObject), "f"(x), "f"(y), "f"(level));
    *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

extern "C" __device__ __inline__ void __itex3D_float(
    float* retVal, cudaTextureObject_t texObject, float x, float y, float z)
{
    float4 tmp;
    asm volatile ("tex.3d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
                  : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
                  : "l"(texObject), "f"(x), "f"(y), "f"(z));
    *retVal = (float)(tmp.x);
}

extern "C" __device__ __inline__ void __itex3D_float4(
    float4 * retVal, cudaTextureObject_t texObject, float x, float y, float z)
{
    float4 tmp;
    asm volatile ("tex.3d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
                  : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
                  : "l"(texObject), "f"(x), "f"(y), "f"(z));
    *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

extern "C" __device__ __inline__ void __itexCubemap_float4(
    float4 * retVal, cudaTextureObject_t texObject, float x, float y, float z)
{
    float4 tmp;
    asm volatile ("tex.cube.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %7}];"
                  : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
                  : "l"(texObject), "f"(x), "f"(y), "f"(z));
    *retVal = make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}


typedef mi::neuraylib::Material_expr_function               MatExprFunc;
typedef mi::neuraylib::Bsdf_init_function                   BsdfInitFunc;
typedef mi::neuraylib::Bsdf_sample_function                 BsdfSampleFunc;
typedef mi::neuraylib::Bsdf_evaluate_function               BsdfEvaluateFunc;
typedef mi::neuraylib::Bsdf_auxiliary_function              BsdfAuxiliaryFunc;
typedef mi::neuraylib::Edf_sample_function                  EdfSampleFunc;
typedef mi::neuraylib::Edf_evaluate_function                EdfEvaluateFunc;


//
// Declarations of generated MDL functions
//

extern "C" __device__ BsdfInitFunc init;
extern "C" __device__ MatExprFunc thinWalled;
extern "C" __device__ MatExprFunc iorEvaluation;

extern "C" __device__ BsdfSampleFunc    frontBsdf_sample;
extern "C" __device__ BsdfEvaluateFunc  frontBsdf_evaluate;
extern "C" __device__ BsdfAuxiliaryFunc frontBsdf_auxiliary;

extern "C" __device__ BsdfSampleFunc    backBsdf_sample;
extern "C" __device__ BsdfEvaluateFunc  backBsdf_evaluate;
extern "C" __device__ BsdfAuxiliaryFunc backBsdf_auxiliary;

extern "C" __device__ EdfEvaluateFunc frontEdf_evaluate;
extern "C" __device__ MatExprFunc     frontEdfIntensity;
extern "C" __device__ MatExprFunc     frontEdfMode;

extern "C" __device__ EdfEvaluateFunc backEdf_evaluate;
extern "C" __device__ MatExprFunc     backEdfIntensity;
extern "C" __device__ MatExprFunc     backEdfMode;


//extern "C" __device__ MatExprFunc volumeAbsorptionCoefficient;
//extern "C" __device__ MatExprFunc volumeScatteringCoefficient;
//extern "C" __device__ MatExprFunc volumeDirectionalBias;

extern "C" __device__ MatExprFunc opacityEvaluation;

//extern "C" __device__ BsdfSampleFunc   HairSample;
//extern "C" __device__ BsdfEvaluateFunc HairEval;

#endif // !MDL_DECLARATIONS_H
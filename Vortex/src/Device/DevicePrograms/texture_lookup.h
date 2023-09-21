/*
 * Copyright (c) 2013-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 * * Neither the name of NVIDIA CORPORATION nor the names of its
 * contributors may be used to endorse or promote products derived
 * from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once
#ifndef TEXTURE_LOOKUP_H
#define TEXTURE_LOOKUP_H

#include <optix.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "DataFetcher.h"
#include <mi/neuraylib/target_code_types.h>
#include <limits>

 // PERF Disabled to not slow down the texure lookup functions.
 //#define USE_SMOOTHERSTEP_FILTER

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define M_ONE_OVER_PI 0.318309886183790671538


typedef mi::neuraylib::tct_deriv_float                    tct_deriv_float;
typedef mi::neuraylib::tct_deriv_float2                   tct_deriv_float2;
typedef mi::neuraylib::tct_deriv_arr_float_2              tct_deriv_arr_float_2;
typedef mi::neuraylib::tct_deriv_arr_float_3              tct_deriv_arr_float_3;
typedef mi::neuraylib::tct_deriv_arr_float_4              tct_deriv_arr_float_4;
typedef mi::neuraylib::Shading_state_material_with_derivs Shading_state_material_with_derivs;
typedef mi::neuraylib::Shading_state_material             Shading_state_material;
typedef mi::neuraylib::Texture_handler_base               Texture_handler_base;
typedef mi::neuraylib::Tex_wrap_mode                      Tex_wrap_mode;
typedef mi::neuraylib::Mbsdf_part                         Mbsdf_part;
//using namespace mi::neuraylib;
using namespace vtx;

__device__ const float INVALID_FLOAT = std::numeric_limits<float>::quiet_NaN();

// Stores a float4 in a float[4] array.
__forceinline__ __device__ void storeResult4(float res[4], const float4& v)
{
    res[0] = v.x;
    res[1] = v.y;
    res[2] = v.z;
    res[3] = v.w;
}

// Stores a float in all elements of a float[4] array.
__forceinline__ __device__ void storeResult4(float res[4], const float s)
{
    res[0] = res[1] = res[2] = res[3] = s;
}

// Stores the given float values in a float[4] array.
__forceinline__ __device__ void storeResult4(float res[4], const float v0, const float v1, const float v2, const float v3)
{
    res[0] = v0;
    res[1] = v1;
    res[2] = v2;
    res[3] = v3;
}

// Stores a float3 in a float[3] array.
__forceinline__ __device__ void storeResult3(float res[3], const float3& v)
{
    res[0] = v.x;
    res[1] = v.y;
    res[2] = v.z;
}

// Stores a float4 in a float[3] array, ignoring v.w.
__forceinline__ __device__ void storeResult3(float res[3], const float4& v)
{
    res[0] = v.x;
    res[1] = v.y;
    res[2] = v.z;
}

// Stores a float in all elements of a float[3] array.
__forceinline__ __device__ void storeResult3(float res[3], const float s)
{
    res[0] = res[1] = res[2] = s;
}

// Stores the given float values in a float[3] array.
__forceinline__ __device__ void storeResult3(float res[3], const float v0, const float v1, const float v2)
{
    res[0] = v0;
    res[1] = v1;
    res[2] = v2;
}

// Stores the luminance of a given float3 in a float.
__forceinline__ __device__ void storeResult1(float* res, const float3& v)
{
    // store luminance
    *res = 0.212671f * v.x + 0.71516f * v.y + 0.072169f * v.z;
}

// Stores the luminance of 3 float scalars in a float.
__forceinline__ __device__ void storeResult1(float* res, const float v0, const float v1, const float v2)
{
    // store luminance
    *res = 0.212671f * v0 + 0.715160f * v1 + 0.072169f * v2;
}

// Stores a given float in a float
__forceinline__ __device__ void storeResult1(float* res, const float s)
{
    *res = s;
}


// ------------------------------------------------------------------------------------------------
// Textures
// ------------------------------------------------------------------------------------------------

__forceinline__ __device__ const TextureData* getTextureData(Texture_handler_base const* selfBase, const uint32_t mdlIndex)
{
    auto const* textureHandler = reinterpret_cast<TextureHandler const*>(selfBase);

    // Note that self->num_textures == texture_idx is a valid case because 1 (the invalid index 0) is subtracted to get the final zerop based index.
    if (mdlIndex == 0 || textureHandler->numTextures < mdlIndex)
    {
        // invalid texture returns zero
        return nullptr;
    }
    return textureHandler->textures[mdlIndex - 1];
}


__forceinline__ __device__ const LightProfileData* getLightProfile(Texture_handler_base const* selfBase, const uint32_t mdlIndex)
{
    auto const* textureHandler = reinterpret_cast<TextureHandler const*>(selfBase);

    // Note that self->num_textures == texture_idx is a valid case because 1 (the invalid index 0) is subtracted to get the final zerop based index.
    if (mdlIndex == 0 || textureHandler->numLightProfiles < mdlIndex)
    {
        // invalid texture returns zero
        return nullptr;
    }
    return textureHandler->lightProfiles[mdlIndex];
}

__forceinline__ __device__ const BsdfData* getBsdf(Texture_handler_base const* selfBase, const uint32_t mdlIndex)
{
    auto const* textureHandler = reinterpret_cast<TextureHandler const*>(selfBase);

    // Note that self->num_textures == texture_idx is a valid case because 1 (the invalid index 0) is subtracted to get the final zerop based index.
    if (mdlIndex == 0 || textureHandler->numBsdfs < mdlIndex)
    {
        // invalid texture returns zero
        return nullptr;
    }
    return textureHandler->bsdfs[mdlIndex - 1];
}

__forceinline__ __device__ const BsdfSamplingPartData* getBsdfPart(Texture_handler_base const* selfBase, const uint32_t mdlIndex, const int part)
{
    auto const* textureHandler = reinterpret_cast<TextureHandler const*>(selfBase);
    // Note that self->num_textures == texture_idx is a valid case because 1 (the invalid index 0) is subtracted to get the final zerop based index.
    if (mdlIndex == 0 || textureHandler->numBsdfs < mdlIndex)
    {
        // invalid texture returns zero
        return nullptr;
    }
    return getBsdfPart(textureHandler->bsdfs[mdlIndex - 1], part);
    
}


__forceinline__ __device__ float getActualUv(
    float const coordinateValue,
    Tex_wrap_mode const wrapMode,
    float const cropRange[2],
    const float invImageSize
)
{
    float finalCoordinate = coordinateValue;
    if (wrapMode != mi::neuraylib::TEX_WRAP_REPEAT || cropRange[0] != 0.0f || cropRange[1] != 1.0f)
    {
        if (wrapMode == mi::neuraylib::TEX_WRAP_REPEAT)
            finalCoordinate = finalCoordinate - floorf(finalCoordinate);
        else
        {
            if (wrapMode == mi::neuraylib::TEX_WRAP_CLIP && (finalCoordinate < 0.0f || 1.0f <= finalCoordinate))
            {
                return INVALID_FLOAT;
            }
            if (wrapMode == mi::neuraylib::TEX_WRAP_MIRRORED_REPEAT)
            {
	            if (const float flooredCoordinateValue = floorf(finalCoordinate); (static_cast<int>(flooredCoordinateValue) & 1) != 0)
		            finalCoordinate = 1.0f - (finalCoordinate - flooredCoordinateValue);
	            else
		            finalCoordinate = finalCoordinate - flooredCoordinateValue;
            }
            const float invHdim = 0.5f * invImageSize;
            finalCoordinate = fminf(fmaxf(finalCoordinate, invHdim), 1.f - invHdim);
        }
        finalCoordinate = finalCoordinate * (cropRange[1] - cropRange[0]) + cropRange[0];
    }
    return finalCoordinate;
}

#ifdef USE_SMOOTHERSTEP_FILTER
// Modify texture coordinates to get better texture filtering,
// see http://www.iquilezles.org/www/articles/texture/texture.htm
#define APPLY_SMOOTHERSTEP_FILTER()                            \
  do                                                           \
  {                                                            \
    u = u * tex.size.x + 0.5f;                                 \
    v = v * tex.size.y + 0.5f;                                 \
    float u_i = floorf(u), v_i = floorf(v);                    \
    float u_f = u - u_i;                                       \
    float v_f = v - v_i;                                       \
    u_f = u_f * u_f * u_f * (u_f * (u_f * 6.f - 15.f) + 10.f); \
    v_f = v_f * v_f * v_f * (v_f * (v_f * 6.f - 15.f) + 10.f); \
    u = u_i + u_f;                                             \
    v = v_i + v_f;                                             \
    u = (u - 0.5f) * tex.inv_size.x;                           \
    v = (v - 0.5f) * tex.inv_size.y;                           \
  } while (0)
#else
#define APPLY_SMOOTHERSTEP_FILTER()
#endif


// Implementation of tex::lookup_float4() for a texture_2d texture.
extern "C" __device__ void tex_lookup_float4_2d(
    float result[4],
    Texture_handler_base const* self_base,
    const unsigned int texture_idx,
    float const coord[2], // uv coordinate
    Tex_wrap_mode const wrap_u,
    Tex_wrap_mode const wrap_v,
    float const crop_u[2], // subregion of the uv space used 
    float const crop_v[2],
    float /*frame*/)
{
	const TextureData* textureData = getTextureData(self_base, texture_idx);
    if (textureData == nullptr)
    {
        // invalid texture returns zero
        storeResult4(result, 0.0f);
        return;
    }

	const float u = getActualUv(coord[0], wrap_u, crop_u, textureData->invSize.x);
	const float v = getActualUv(coord[1], wrap_v, crop_v, textureData->invSize.y);

    if (isnan(u) || isnan(v))
    {
        storeResult4(result, 0.0f);
    }

    APPLY_SMOOTHERSTEP_FILTER();

    storeResult4(result, tex2D<float4>(textureData->texObj, u, v));
}

// Implementation of tex::lookup_float4() for a texture_2d texture.
extern "C" __device__ void tex_lookup_deriv_float4_2d(
    float result[4],
    Texture_handler_base const* self_base,
    const unsigned int texture_idx,
    tct_deriv_float2 const* coord,
    Tex_wrap_mode const wrap_u,
    Tex_wrap_mode const wrap_v,
    float const crop_u[2],
    float const crop_v[2],
    float /*frame*/)
{
    const TextureData* textureData = getTextureData(self_base, texture_idx);
    if (textureData == nullptr)
    {
        // invalid texture returns zero
        storeResult4(result, 0.0f);
        return;
    }

    const float u = getActualUv(coord->val.x, wrap_u, crop_u, textureData->invSize.x);
    const float v = getActualUv(coord->val.y, wrap_v, crop_v, textureData->invSize.y);

    if (isnan(u) || isnan(v))
    {
        storeResult4(result, 0.0f);
    }

    APPLY_SMOOTHERSTEP_FILTER();

    storeResult4(result, tex2DGrad<float4>(textureData->texObj, u, v, coord->dx, coord->dy));
}

// Implementation of tex::lookup_float3() for a texture_2d texture.
extern "C" __device__ void tex_lookup_float3_2d(
    float result[3],
    Texture_handler_base const* self_base,
    const unsigned int texture_idx,
    float const coord[2],
    Tex_wrap_mode const wrap_u,
    Tex_wrap_mode const wrap_v,
    float const crop_u[2],
    float const crop_v[2],
    float /*frame*/)
{
    const TextureData* textureData = getTextureData(self_base, texture_idx);
    if (textureData == nullptr)
    {
        // invalid texture returns zero
        storeResult3(result, 0.0f);
        return;
    }

    const float u = getActualUv(coord[0], wrap_u, crop_u, textureData->invSize.x);
    const float v = getActualUv(coord[1], wrap_v, crop_v, textureData->invSize.y);

    if (isnan(u) || isnan(v))
    {
        storeResult3(result, 0.0f);
    }

    APPLY_SMOOTHERSTEP_FILTER();

    storeResult3(result, tex2D<float4>(textureData->texObj, u, v));
}

// Implementation of tex::lookup_float3() for a texture_2d texture.
extern "C" __device__ void tex_lookup_deriv_float3_2d(
    float result[3],
    Texture_handler_base const* self_base,
    const unsigned int texture_idx,
    tct_deriv_float2 const* coord,
    Tex_wrap_mode const wrap_u,
    Tex_wrap_mode const wrap_v,
    float const crop_u[2],
    float const crop_v[2],
    float /*frame*/)
{
    const TextureData* textureData = getTextureData(self_base, texture_idx);
    if (textureData == nullptr)
    {
        // invalid texture returns zero
        storeResult3(result, 0.0f);
        return;
    }

    const float u = getActualUv(coord->val.x, wrap_u, crop_u, textureData->invSize.x);
    const float v = getActualUv(coord->val.y, wrap_v, crop_v, textureData->invSize.y);

    if (isnan(u) || isnan(v))
    {
        storeResult3(result, 0.0f);
    }

    APPLY_SMOOTHERSTEP_FILTER();

    storeResult3(result, tex2DGrad<float4>(textureData->texObj, u, v, coord->dx, coord->dy));
}

// Implementation of tex::texel_float4() for a texture_2d texture.
// Note: uvtile and/or animated textures are not supported
extern "C" __device__ void tex_texel_float4_2d(
    float result[4],
    Texture_handler_base const* self_base,
    const unsigned int texture_idx,
    int const coord[2],
    int const /*uv_tile*/ [2],
    float /*frame*/)
{

    const TextureData* textureData = getTextureData(self_base, texture_idx);
    if (textureData == nullptr)
    {
        // invalid texture returns zero
        storeResult4(result, 0.0f);
        return;
    }

    storeResult4(result, tex2D<float4>(textureData->texObjUnfiltered,
                                        static_cast<float>(coord[0]) * textureData->invSize.x,
                                        static_cast<float>(coord[1]) * textureData->invSize.y));
}

// Implementation of tex::lookup_float4() for a texture_3d texture.
extern "C" __device__ void tex_lookup_float4_3d(
    float result[4],
    Texture_handler_base const* self_base,
    const unsigned int texture_idx,
    float const coord[3],
    const Tex_wrap_mode wrap_u,
    const Tex_wrap_mode wrap_v,
    const Tex_wrap_mode wrap_w,
    float const crop_u[2],
    float const crop_v[2],
    float const crop_w[2],
    float /*frame*/)
{
    const TextureData* textureData = getTextureData(self_base, texture_idx);
    if (textureData == nullptr)
    {
        // invalid texture returns zero
        storeResult4(result, 0.0f);
        return;
    }

    const float u = getActualUv(coord[0], wrap_u, crop_u, textureData->invSize.x);
    const float v = getActualUv(coord[1], wrap_v, crop_v, textureData->invSize.y);
    const float w = getActualUv(coord[2], wrap_w, crop_w, textureData->invSize.z);

    if (isnan(u) || isnan(v)|| isnan(w))
    {
        storeResult4(result, 0.0f);
    }

    //storeResult4(result, tex3D<float4>(textureData->texObj, u, v, w));
}

// Implementation of tex::lookup_float3() for a texture_3d texture.
extern "C" __device__ void tex_lookup_float3_3d(
    float result[3],
    Texture_handler_base const* self_base,
    const unsigned int texture_idx,
    float const coord[3],
    const Tex_wrap_mode wrap_u,
    const Tex_wrap_mode wrap_v,
    const Tex_wrap_mode wrap_w,
    float const crop_u[2],
    float const crop_v[2],
    float const crop_w[2],
    float /*frame*/)
{
    const TextureData* textureData = getTextureData(self_base, texture_idx);
    if (textureData == nullptr)
    {
        // invalid texture returns zero
        storeResult3(result, 0.0f);
        return;
    }

    const float u = getActualUv(coord[0], wrap_u, crop_u, textureData->invSize.x);
    const float v = getActualUv(coord[1], wrap_v, crop_v, textureData->invSize.y);
    const float w = getActualUv(coord[2], wrap_w, crop_w, textureData->invSize.z);

    if (isnan(u) || isnan(v) || isnan(w))
    {
        storeResult3(result, 0.0f);
    }

    storeResult3(result, tex3D<float4>(textureData->texObj, u, v, w));
}

// Implementation of tex::texel_float4() for a texture_3d texture.
extern "C" __device__ void tex_texel_float4_3d(
    float result[4],
    Texture_handler_base const* self_base,
    const unsigned int texture_idx,
    const int coord[3],
    float /*frame*/)
{
    const TextureData* textureData = getTextureData(self_base, texture_idx);
    if (textureData == nullptr)
    {
        // invalid texture returns zero
        storeResult4(result, 0.0f);
        return;
    }

    storeResult4(result, tex3D<float4>(textureData->texObjUnfiltered,
                                        static_cast<float>(coord[0]) * textureData->invSize.x,
                                        static_cast<float>(coord[1]) * textureData->invSize.y,
										static_cast<float>(coord[2])* textureData->invSize.z));
}

// Implementation of tex::lookup_float4() for a texture_cube texture.
extern "C" __device__ void tex_lookup_float4_cube(
    float result[4],
    Texture_handler_base const* self_base,
    const unsigned int texture_idx,
    float const coord[3])
{
    const TextureData* textureData = getTextureData(self_base, texture_idx);
    if (textureData == nullptr)
    {
        // invalid texture returns zero
        storeResult4(result, 0.0f);
        return;
    }

    storeResult4(result, texCubemap<float4>(textureData->texObj, coord[0], coord[1], coord[2]));
}

// Implementation of tex::lookup_float3() for a texture_cube texture.
extern "C" __device__ void tex_lookup_float3_cube(
    float result[3],
    Texture_handler_base const* self_base,
    const unsigned int texture_idx,
    float const coord[3])
{
    const TextureData* textureData = getTextureData(self_base, texture_idx);
    if (textureData == nullptr)
    {
        // invalid texture returns zero
        storeResult3(result, 0.0f);
        return;
    }

    storeResult3(result, texCubemap<float4>(textureData->texObj, coord[0], coord[1], coord[2]));
}

// Implementation of resolution_2d function needed by generated code.
// Note: uvtile and/or animated textures are not supported
extern "C" __device__ void tex_resolution_2d(
    int result[2],
    Texture_handler_base const* self_base,
    const unsigned int texture_idx,
    int const /*uv_tile*/ [2],
    float /*frame*/)
{

    const TextureData* textureData = getTextureData(self_base, texture_idx);

    if (textureData == nullptr)
    {
        // invalid texture returns zero
        result[0] = 0;
        result[1] = 0;
        return;
    }

    result[0] = textureData->dimension.x;
    result[1] = textureData->dimension.y;
}

// Implementation of resolution_3d function needed by generated code.
extern "C" __device__ void tex_resolution_3d(
    int result[3],
    Texture_handler_base const* self_base,
    const unsigned int texture_idx,
    float /*frame*/)
{
    const TextureData* textureData = getTextureData(self_base, texture_idx);

    if (textureData == nullptr)
    {
        // invalid texture returns zero
        result[0] = 0;
        result[1] = 0;
        result[2] = 0;
        return;
    }

    result[0] = textureData->dimension.x;
    result[1] = textureData->dimension.y;
    result[2] = textureData->dimension.z;
}

// Implementation of texture_isvalid().
extern "C" __device__ bool tex_texture_isvalid(
    Texture_handler_base const* self_base,
    const unsigned int texture_idx)
{
	if (const TextureData* textureData = getTextureData(self_base, texture_idx); textureData == nullptr)
    {
        return false;
    }
    return true;
}

// Implementation of frame function needed by generated code.
extern "C" __device__ void tex_frame(
    int result[2],
    Texture_handler_base const* self_base,
    unsigned int texture_idx)
{
    //Texture_handler const* self = static_cast<Texture_handler const*>(self_base);
    //if (texture_idx == 0 || self->num_textures < texture_idx)
    //{
    //    // invalid texture returns zero
    //    result[0] = 0;
    //    result[1] = 0;
    //    return;
    //}
    // TextureMDL const& tex = self->textures[texture_idx - 1];

    result[0] = 0;
    result[1] = 0;
}


// ------------------------------------------------------------------------------------------------
// Light Profiles
// ------------------------------------------------------------------------------------------------


// Implementation of light_profile_power() for a light profile.
extern "C" __device__ float df_light_profile_power(
    Texture_handler_base const* self_base,
    const unsigned int light_profile_idx)
{
	const LightProfileData* lp = getLightProfile(self_base, light_profile_idx);

    if (lp==nullptr)
    {
        return 0.0f; // invalid light profile returns zero
    }

    return lp->totalPower;
}

// Implementation of light_profile_maximum() for a light profile.
extern "C" __device__ float df_light_profile_maximum(
    Texture_handler_base const* self_base,
    const unsigned int light_profile_idx)
{
	const LightProfileData* lp = getLightProfile(self_base, light_profile_idx);

    if (lp == nullptr)
    {
        return 0.0f; // invalid light profile returns zero
    }

    return lp->candelaMultiplier;
}

// Implementation of light_profile_isvalid() for a light profile.
extern "C" __device__ bool df_light_profile_isvalid(
    Texture_handler_base const* self_base,
    const unsigned int light_profile_idx)
{
    const LightProfileData* lp = getLightProfile(self_base, light_profile_idx);

    if (lp == nullptr)
    {
        return false; // invalid light profile returns zero
    }
    return true;
}

// binary search through CDF
__forceinline__ __device__ unsigned int sample_cdf(
    const float* cdf,
    const unsigned int cdf_size,
    const float xi)
{
    unsigned int li = 0;
    unsigned int ri = cdf_size - 1; // This fails for cdf_size == 0.
    unsigned int m = (li + ri) / 2;

    while (ri > li)
    {
        if (xi < cdf[m])
        {
            ri = m;
        }
        else
        {
            li = m + 1;
        }

        m = (li + ri) / 2;
    }

    return m;
}


// Implementation of df::light_profile_evaluate() for a light profile.
extern "C" __device__ float df_light_profile_evaluate(
    Texture_handler_base const* self_base,
    const unsigned int light_profile_idx,
    float const theta_phi[2])
{
    const LightProfileData* lp = getLightProfile(self_base, light_profile_idx);

    if (lp == nullptr)
    {
        return 0.0f; // invalid light profile returns zero
    }

    // map theta to 0..1 range
    float u = (theta_phi[0] - lp->thetaPhiStart.x) * lp->thetaPhiInvDelta.x * lp->invAngularResolution.x;

    // converting input phi from -pi..pi to 0..2pi
    float phi = (theta_phi[1] > 0.0f) ? theta_phi[1] : (static_cast<float>(2.0 * M_PI) + theta_phi[1]);

    // floorf wraps phi range into 0..2pi
    phi = phi - lp->thetaPhiStart.y - floorf((phi - lp->thetaPhiStart.y) * static_cast<float>(0.5 / M_PI)) * static_cast<float>(2.0 * M_PI);

    // (phi < 0.0f) is no problem, this is handle by the (black) border
    // since it implies lp->theta_phi_start.y > 0 (and we really have "no data" below that)
    float v = phi * lp->thetaPhiInvDelta.y * lp->invAngularResolution.y;

    // half pixel offset
    // see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#linear-filtering
    u += 0.5f * lp->invAngularResolution.x;
    v += 0.5f * lp->invAngularResolution.y;

    // wrap_mode: border black would be an alternative (but it produces artifacts at low res)
    if (u < 0.0f || 1.0f < u || v < 0.0f || 1.0f < v)
    {
        return 0.0f;
    }

    return tex2D<float>(lp->evalData, u, v) * lp->candelaMultiplier;
}

// Implementation of df::light_profile_sample() for a light profile.
extern "C" __device__ void df_light_profile_sample(
    float result[3], // output: theta, phi, pdf
    Texture_handler_base const* self_base,
    const unsigned int light_profile_idx,
    float const xi[3]) // uniform random values
{
    result[0] = -1.0f; // negative theta means no emission
    result[1] = -1.0f;
    result[2] = 0.0f;

    const LightProfileData* lp = getLightProfile(self_base, light_profile_idx);

    if (lp == nullptr)
    {
        return; // invalid light profile returns zero
    }

    const uint2 res = lp->angularResolution;

    // sample theta_out
    //-------------------------------------------
    float xi0 = xi[0];
    const float* cdfDataTheta = lp->cdfData; // CDF theta
    const unsigned int idxTheta = sample_cdf(cdfDataTheta, res.x - 1, xi0); // binary search

    float probTheta = cdfDataTheta[idxTheta];
    if (idxTheta > 0)
    {
        const float tmp = cdfDataTheta[idxTheta - 1];
        probTheta -= tmp;
        xi0 -= tmp;
    }

    xi0 /= probTheta; // rescale for re-usage

    // sample phi_out
    //-------------------------------------------
    float xi1 = xi[1];
    const float* cdfDataPhi = cdfDataTheta
        + (res.x - 1) // CDF theta block
        + (idxTheta * (res.y - 1)); // selected CDF for phi

    const unsigned int idxPhi = sample_cdf(cdfDataPhi, res.y - 1, xi1); // binary search

    float probPhi = cdfDataPhi[idxPhi];
    if (idxPhi > 0)
    {
        const float tmp = cdfDataPhi[idxPhi - 1];

        probPhi -= tmp;
        xi1 -= tmp;
    }

    xi1 /= probPhi; // rescale for re-usage

    // compute theta and phi
    //-------------------------------------------
    // sample uniformly within the patch (grid cell)
    const float2 start = lp->thetaPhiStart;
    const float2 delta = lp->thetaPhiDelta;

    const float cosTheta0 = cosf(start.x + static_cast<float>(idxTheta) * delta.x);
    const float cosTheta1 = cosf(start.x + static_cast<float>(idxTheta + 1u) * delta.x);

    // n = \int_{\theta_0}^{\theta_1} \sin{\theta} \delta \theta
    //   = 1 / (\cos{\theta_0} - \cos{\theta_1})
    //
    // \xi = n * \int_{\theta_0}^{\theta_1} \sin{\theta} \delta \theta
    //     => \cos{\theta} = (1 - \xi) \cos{\theta_0} + \xi \cos{\theta_1}

    const float cosTheta = (1.0f - xi1) * cosTheta0 + xi1 * cosTheta1;

    result[0] = acosf(cosTheta);
    result[1] = start.y + (static_cast<float>(idxPhi) + xi0) * delta.y;

    // align phi
    if (result[1] > static_cast<float>(2.0 * M_PI))
    {
        result[1] -= static_cast<float>(2.0 * M_PI); // wrap
    }
    if (result[1] > static_cast<float>(1.0 * M_PI))
    {
        result[1] = static_cast<float>(-2.0 * M_PI) + result[1]; // to [-pi, pi]
    }

    // compute pdf
    //-------------------------------------------
    result[2] = probTheta * probPhi / (delta.y * (cosTheta0 - cosTheta1));
}


// Implementation of df::light_profile_pdf() for a light profile.
extern "C" __device__ float df_light_profile_pdf(
    Texture_handler_base const* self_base,
    const unsigned int light_profile_idx,
    float const theta_phi[2])
{
    const LightProfileData* lp = getLightProfile(self_base, light_profile_idx);

    if (lp == nullptr)
    {
        return 0.0f; // invalid light profile returns zero
    }

    // CDF data
    const uint2 res = lp->angularResolution;
    const float* cdfDataTheta = lp->cdfData;

    // map theta to 0..1 range
    const float theta = theta_phi[0] - lp->thetaPhiStart.x;
    const int idxTheta = static_cast<int>(theta * lp->thetaPhiInvDelta.x);

    // converting input phi from -pi..pi to 0..2pi
    float phi = (theta_phi[1] > 0.0f) ? theta_phi[1] : (static_cast<float>(2.0 * M_PI) + theta_phi[1]);

    // floorf wraps phi range into 0..2pi
    phi = phi - lp->thetaPhiStart.y - floorf((phi - lp->thetaPhiStart.y) * static_cast<float>(0.5 / M_PI)) * static_cast<float>(2.0 * M_PI);

    // (phi < 0.0f) is no problem, this is handle by the (black) border
    // since it implies lp->thetaPhiStart.y > 0 (and we really have "no data" below that)
    const int idxPhi = static_cast<int>(phi * lp->thetaPhiInvDelta.y);

    // wrap_mode: border black would be an alternative (but it produces artifacts at low res)
    if (idxTheta < 0 || (res.x - 2) < idxTheta || idxPhi < 0 || (res.y - 2) < idxPhi) // DAR BUG Was: (res.x - 2) < idx_phi
    {
        return 0.0f;
    }

    // get probability for theta
    //-------------------------------------------

    float probTheta = cdfDataTheta[idxTheta];
    if (idxTheta > 0)
    {
        const float tmp = cdfDataTheta[idxTheta - 1];
        probTheta -= tmp;
    }

    // get probability for phi
    //-------------------------------------------
    const float* cdfDataPhi = cdfDataTheta
        + (res.x - 1) // CDF theta block
        + (idxTheta * (res.y - 1)); // selected CDF for phi


    float probPhi = cdfDataPhi[idxPhi];
    if (idxPhi > 0)
    {
        const float tmp = cdfDataPhi[idxPhi - 1];
        probPhi -= tmp;
    }

    // compute probability to select a position in the sphere patch
    const float2 start = lp->thetaPhiStart;
    const float2 delta = lp->thetaPhiDelta;

    const float cosTheta0 = cos(start.x + static_cast<float>(idxTheta) * delta.x);
    const float cosTheta1 = cos(start.x + static_cast<float>(idxTheta + 1u) * delta.x);

    return probTheta * probPhi / (delta.y * (cosTheta0 - cosTheta1));
}


// ------------------------------------------------------------------------------------------------
// BSDF Measurements
// ------------------------------------------------------------------------------------------------

// Implementation of bsdf_measurement_isvalid() for an MBSDF.
extern "C" __device__ bool df_bsdf_measurement_isvalid(
    Texture_handler_base const* self_base,
	const unsigned int          bsdf_measurement_index)
{

    const BsdfData* bm = getBsdf(self_base, bsdf_measurement_index);

    if (bm == nullptr)
    {
        return false;
    }
    return true;
}

// Implementation of df::bsdf_measurement_resolution() function needed by generated code,
// which retrieves the angular and chromatic resolution of the given MBSDF.
// The returned triple consists of: number of equi-spaced steps of theta_i and theta_o,
// number of equi-spaced steps of phi, and number of color channels (1 or 3).
extern "C" __device__ void df_bsdf_measurement_resolution(
    unsigned int                    result[3],
    Texture_handler_base const*     self_base,
	const unsigned int              bsdf_measurement_index,
    const mi::neuraylib::Mbsdf_part part)
{
	const BsdfSamplingPartData* bmp = getBsdfPart(self_base, bsdf_measurement_index,part);

    if (bmp == nullptr)
    {
        result[0] = 0;
        result[1] = 0;
        result[2] = 0;
    }

    // pass out the information
    result[0] = bmp->angularResolution.x;
    result[1] = bmp->angularResolution.y;
    result[2] = bmp->numChannels;
}

__forceinline__ __device__ math::vec3f bsdfComputeUvw(const float thetaPhiIn[2],
                                                        const float thetaPhiOut[2])
{
    // assuming each phi is between -pi and pi
    float u = thetaPhiOut[1] - thetaPhiIn[1];
    if (u < 0.0f)
    {
        u += static_cast<float>(2.0 * M_PI);
    }
    if (u > static_cast<float>(1.0 * M_PI))
    {
        u = static_cast<float>(2.0 * M_PI) - u;
    }
    u *= M_ONE_OVER_PI;

    const float v = thetaPhiOut[0] * static_cast<float>(2.0 / M_PI);
    const float w = thetaPhiIn[0] * static_cast<float>(2.0 / M_PI);

    return { u, v, w };
}

template<typename T>
__forceinline__ __device__ T bsdfMeasurementLookup(
    const cudaTextureObject_t& evalVolume,
    const float thetaPhiIn[2],
    const float thetaPhiOut[2])
{
    // 3D volume on the GPU (phi_delta x theta_out x theta_in)
    const math::vec3f uvw = bsdfComputeUvw(thetaPhiIn, thetaPhiOut);

    return tex3D<T>(evalVolume, uvw.x, uvw.y, uvw.z);
}

// Implementation of df::bsdf_measurement_evaluate() for an MBSDF.
extern "C" __device__ void df_bsdf_measurement_evaluate(
    float                       result[3],
    Texture_handler_base const* self_base,
	const unsigned int          bsdf_measurement_index,
    float const                 theta_phi_in[2],
    float const                 theta_phi_out[2],
    const Mbsdf_part            part)
{
    const BsdfSamplingPartData* bmp = getBsdfPart(self_base, bsdf_measurement_index, part);

    if (bmp == nullptr)
    {
        storeResult3(result, 0.0f);
        return;
    }

    // handle channels
    if (bmp->numChannels == 3)
    {
        const auto sample = bsdfMeasurementLookup<float4>(bmp->evalData, theta_phi_in, theta_phi_out);
        storeResult3(result, sample.x, sample.y, sample.z);
    }
    else
    {
        const auto sample = bsdfMeasurementLookup<float>(bmp->evalData, theta_phi_in, theta_phi_out);
        storeResult3(result, sample);
    }
}

// Implementation of df::bsdf_measurement_sample() for an MBSDF.
extern "C" __device__ void df_bsdf_measurement_sample(
    float                       result[3], // output: theta, phi, pdf
    Texture_handler_base const* self_base,
	const unsigned int          bsdf_measurement_index,
    float const                 theta_phi_out[2],
    float const                 xi[3], // uniform random values
    const Mbsdf_part            part)
{
    result[0] = -1.0f; // negative theta means absorption
    result[1] = -1.0f;
    result[2] = 0.0f;

    const BsdfSamplingPartData* bmp = getBsdfPart(self_base, bsdf_measurement_index, part);

    if (bmp == nullptr)
    {
        return;
    }

    // CDF data
    const math::vec2ui res = bmp->angularResolution;
    const float* sampleData = bmp->sampleData;

    // compute the theta_in index (flipping input and output, BSDFs are symmetric)
    auto idxThetaIn = static_cast<unsigned int>(theta_phi_out[0] * M_ONE_OVER_PI * 2.0f * static_cast<float>(res.x));
    idxThetaIn = min(idxThetaIn, res.x - 1);

    // sample theta_out
    //-------------------------------------------
    float xi0 = xi[0];
    const float* cdfTheta = sampleData + idxThetaIn * res.x;
    const unsigned int idxThetaOut = sample_cdf(cdfTheta, res.x, xi0); // binary search

    float prob_theta = cdfTheta[idxThetaOut];
    if (idxThetaOut > 0)
    {
        const float tmp = cdfTheta[idxThetaOut - 1];
        prob_theta -= tmp;
        xi0 -= tmp;
    }
    xi0 /= prob_theta; // rescale for re-usage

    // sample phi_out
    //-------------------------------------------
    float xi1 = xi[1];
    const float* cdfPhi = sampleData +
        (res.x * res.x) + // CDF theta block
        (idxThetaIn * res.x + idxThetaOut) * res.y; // selected CDF phi

    // select which half-circle to choose with probability 0.5
    const bool flip = (xi1 > 0.5f);
    if (flip)
    {
        xi1 = 1.0f - xi1;
    }
    xi1 *= 2.0f;

    const unsigned int idxPhiOut = sample_cdf(cdfPhi, res.y, xi1); // binary search
    float probPhi = cdfPhi[idxPhiOut];
    if (idxPhiOut > 0)
    {
        const float tmp = cdfPhi[idxPhiOut - 1];
        probPhi -= tmp;
        xi1 -= tmp;
    }
    xi1 /= probPhi; // rescale for re-usage

    // compute theta and phi out
    //-------------------------------------------
    const math::vec2f invRes = bmp->invAngularResolution;

    const float sTheta = static_cast<float>(0.5 * M_PI) * invRes.x;
    const float sPhi = static_cast<float>(1.0 * M_PI) * invRes.y;

    const float cosTheta0 = cosf(static_cast<float>(idxThetaOut) * sTheta);
    const float cosTheta1 = cosf(static_cast<float>(idxThetaOut + 1u) * sTheta);

    const float cosTheta = cosTheta0 * (1.0f - xi1) + cosTheta1 * xi1;
    result[0] = acosf(cosTheta);
    result[1] = (static_cast<float>(idxPhiOut) + xi0) * sPhi;

    if (flip)
    {
        result[1] = static_cast<float>(2.0 * M_PI) - result[1]; // phi \in [0, 2pi]
    }

    // align phi
    result[1] += (theta_phi_out[1] > 0) ? theta_phi_out[1] : (static_cast<float>(2.0 * M_PI) + theta_phi_out[1]);
    if (result[1] > static_cast<float>(2.0 * M_PI))
    {
        result[1] -= static_cast<float>(2.0 * M_PI);
    }
    if (result[1] > static_cast<float>(1.0 * M_PI))
    {
        result[1] = static_cast<float>(-2.0 * M_PI) + result[1]; // to [-pi, pi]
    }

    // compute pdf
    //-------------------------------------------
    result[2] = prob_theta * probPhi * 0.5f / (sPhi * (cosTheta0 - cosTheta1));
}

// Implementation of df::bsdf_measurement_pdf() for an MBSDF.
extern "C" __device__ float df_bsdf_measurement_pdf(
    Texture_handler_base const* self_base,
	const unsigned int          bsdf_measurement_index,
    float const                 theta_phi_in[2],
    float const                 theta_phi_out[2],
	const Mbsdf_part            part)
{
    const BsdfSamplingPartData* bmp = getBsdfPart(self_base, bsdf_measurement_index, part);

    if (bmp == nullptr)
    {
        return 0.0f;
    }

    // CDF data and resolution
    const float* sample_data = bmp->sampleData;
    uint2 res = bmp->angularResolution;

    // compute indices in the CDF data
    float3 uvw = bsdfComputeUvw(theta_phi_in, theta_phi_out); // phi_delta, theta_out, theta_in
    auto idxThetaIn = static_cast<unsigned int>(theta_phi_in[0] * M_ONE_OVER_PI * 2.0f * static_cast<float>(res.x));
    auto idxThetaOut = static_cast<unsigned int>(theta_phi_out[0] * M_ONE_OVER_PI * 2.0f * static_cast<float>(res.x));
    auto idxPhiOut = static_cast<unsigned int>(uvw.x * static_cast<float>(res.y));

    idxThetaIn = min(idxThetaIn, res.x - 1);
    idxThetaOut = min(idxThetaOut, res.x - 1);
    idxPhiOut = min(idxPhiOut, res.y - 1);

    // get probability to select theta_out
    const float* cdfTheta = sample_data + idxThetaIn * res.x;
    float probTheta = cdfTheta[idxThetaOut];
    if (idxThetaOut > 0)
    {
        const float tmp = cdfTheta[idxThetaOut - 1];
        probTheta -= tmp;
    }

    // get probability to select phi_out
    const float* cdfPhi = sample_data +
        (res.x * res.x) + // CDF theta block
        (idxThetaIn * res.x + idxThetaOut) * res.y; // selected CDF phi

    float probPhi = cdfPhi[idxPhiOut];
    if (idxPhiOut > 0)
    {
        const float tmp = cdfPhi[idxPhiOut - 1];
        probPhi -= tmp;
    }

    // compute probability to select a position in the sphere patch
    float2 inv_res = bmp->invAngularResolution;

    const float sTheta = static_cast<float>(0.5 * M_PI) * inv_res.x;
    const float sPhi = static_cast<float>(1.0 * M_PI) * inv_res.y;

    const float cosTheta0 = cosf(static_cast<float>(idxThetaOut) * sTheta);
    const float cosTheta1 = cosf(static_cast<float>(idxThetaOut + 1u) * sTheta);

    return probTheta * probPhi * 0.5f / (sPhi * (cosTheta0 - cosTheta1));
}

// Implementation of df::bsdf_measurement_albedos() for an MBSDF.
extern "C" __device__ void df_bsdf_measurement_albedos(
    float result[4], // output: [0] albedoAccumulator refl. for theta_phi
    // [1] max albedoAccumulator refl. global
    // [2] albedoAccumulator trans. for theta_phi
    // [3] max albedoAccumulator trans. global
    Texture_handler_base const* self_base,
	const unsigned int          bsdf_measurement_index,
    float const                 theta_phi[2])
{
    result[0] = 0.0f;
    result[1] = 0.0f;
    result[2] = 0.0f;
    result[3] = 0.0f;

    const BsdfSamplingPartData* bmpReflection = getBsdfPart(self_base, bsdf_measurement_index, mi::neuraylib::MBSDF_DATA_REFLECTION);
    const BsdfSamplingPartData* bmpTransmission = getBsdfPart(self_base, bsdf_measurement_index, mi::neuraylib::MBSDF_DATA_TRANSMISSION);

    if (bmpReflection != nullptr)
    {
        const math::vec2ui res = bmpReflection->angularResolution;
        auto idxTheta = static_cast<unsigned int>(theta_phi[0] * static_cast<float>(2.0 / M_PI) * static_cast<float>(res.x));

        idxTheta = min(idxTheta, res.x - 1u);
        result[0] = bmpReflection->albedoData[idxTheta];
        result[1] = bmpReflection->maxAlbedo;
    }
    if (bmpTransmission != nullptr)
    {
        const math::vec2ui res = bmpTransmission->angularResolution;
        auto idxTheta = static_cast<unsigned int>(theta_phi[0] * static_cast<float>(2.0 / M_PI) * static_cast<float>(res.x));

        idxTheta = min(idxTheta, res.x - 1u);
        result[2] = bmpTransmission->albedoData[idxTheta];
        result[3] = bmpTransmission->maxAlbedo;
    }
}


// ------------------------------------------------------------------------------------------------
// Normal adaption (dummy functions)
//
// Can be enabled via backend option "use_renderer_adapt_normal".
// ------------------------------------------------------------------------------------------------

#ifndef TEX_SUPPORT_NO_DUMMY_ADAPTNORMAL

// Implementation of adapt_normal().
extern "C" __device__ void adapt_normal(
    float result[3],
    Texture_handler_base const* self_base,
    Shading_state_material * state,
    float const normal[3])
{
    // just return original normal
    result[0] = normal[0];
    result[1] = normal[1];
    result[2] = normal[2];
}

#endif // TEX_SUPPORT_NO_DUMMY_ADAPTNORMAL


// ------------------------------------------------------------------------------------------------
// Scene data (dummy functions)
// ------------------------------------------------------------------------------------------------

#ifndef TEX_SUPPORT_NO_DUMMY_SCENEDATA

// Implementation of scene_data_isvalid().
extern "C" __device__ bool scene_data_isvalid(
    Texture_handler_base const* self_base,
    Shading_state_material * state,
    unsigned int scene_data_id)
{
    return false;
}

// Implementation of scene_data_lookup_float4().
extern "C" __device__ void scene_data_lookup_float4(
    float result[4],
    Texture_handler_base const* self_base,
    Shading_state_material * state,
    unsigned int scene_data_id,
    float const default_value[4],
    bool uniform_lookup)
{
    // just return default value
    result[0] = default_value[0];
    result[1] = default_value[1];
    result[2] = default_value[2];
    result[3] = default_value[3];
}

// Implementation of scene_data_lookup_float3().
extern "C" __device__ void scene_data_lookup_float3(
    float result[3],
    Texture_handler_base const* self_base,
    Shading_state_material * state,
    unsigned int scene_data_id,
    float const default_value[3],
    bool uniform_lookup)
{
    // just return default value
    result[0] = default_value[0];
    result[1] = default_value[1];
    result[2] = default_value[2];
}

// Implementation of scene_data_lookup_color().
extern "C" __device__ void scene_data_lookup_color(
    float result[3],
    Texture_handler_base const* self_base,
    Shading_state_material * state,
    unsigned int scene_data_id,
    float const default_value[3],
    bool uniform_lookup)
{
    // just return default value
    result[0] = default_value[0];
    result[1] = default_value[1];
    result[2] = default_value[2];
}

// Implementation of scene_data_lookup_float2().
extern "C" __device__ void scene_data_lookup_float2(
    float result[2],
    Texture_handler_base const* self_base,
    Shading_state_material * state,
    unsigned int scene_data_id,
    float const default_value[2],
    bool uniform_lookup)
{
    // just return default value
    result[0] = default_value[0];
    result[1] = default_value[1];
}

// Implementation of scene_data_lookup_float().
extern "C" __device__ float scene_data_lookup_float(
    Texture_handler_base const* self_base,
    Shading_state_material * state,
    unsigned int scene_data_id,
    float const default_value,
    bool uniform_lookup)
{
    // just return default value
    return default_value;
}

// Implementation of scene_data_lookup_int4().
extern "C" __device__ void scene_data_lookup_int4(
    int result[4],
    Texture_handler_base const* self_base,
    Shading_state_material * state,
    unsigned int scene_data_id,
    int const default_value[4],
    bool uniform_lookup)
{
    // just return default value
    result[0] = default_value[0];
    result[1] = default_value[1];
    result[2] = default_value[2];
    result[3] = default_value[3];
}

// Implementation of scene_data_lookup_int3().
extern "C" __device__ void scene_data_lookup_int3(
    int result[3],
    Texture_handler_base const* self_base,
    Shading_state_material * state,
    unsigned int scene_data_id,
    int const default_value[3],
    bool uniform_lookup)
{
    // just return default value
    result[0] = default_value[0];
    result[1] = default_value[1];
    result[2] = default_value[2];
}

// Implementation of scene_data_lookup_int2().
extern "C" __device__ void scene_data_lookup_int2(
    int result[2],
    Texture_handler_base const* self_base,
    Shading_state_material * state,
    unsigned int scene_data_id,
    int const default_value[2],
    bool uniform_lookup)
{
    // just return default value
    result[0] = default_value[0];
    result[1] = default_value[1];
}

// Implementation of scene_data_lookup_int().
extern "C" __device__ int scene_data_lookup_int(
    Texture_handler_base const* self_base,
    Shading_state_material * state,
    unsigned int scene_data_id,
    const int default_value,
    bool uniform_lookup)
{
    // just return default value
    return default_value;
}

// Implementation of scene_data_lookup_float4() with derivatives.
extern "C" __device__ void scene_data_lookup_deriv_float4(
    tct_deriv_arr_float_4 * result,
    Texture_handler_base const* self_base,
    Shading_state_material_with_derivs * state,
    unsigned int scene_data_id,
    tct_deriv_arr_float_4 const* default_value,
    bool uniform_lookup)
{
    // just return default value
    *result = *default_value;
}

// Implementation of scene_data_lookup_float3() with derivatives.
extern "C" __device__ void scene_data_lookup_deriv_float3(
    tct_deriv_arr_float_3 * result,
    Texture_handler_base const* self_base,
    Shading_state_material_with_derivs * state,
    unsigned int scene_data_id,
    tct_deriv_arr_float_3 const* default_value,
    bool uniform_lookup)
{
    // just return default value
    *result = *default_value;
}

// Implementation of scene_data_lookup_color() with derivatives.
extern "C" __device__ void scene_data_lookup_deriv_color(
    tct_deriv_arr_float_3 * result,
    Texture_handler_base const* self_base,
    Shading_state_material_with_derivs * state,
    unsigned int scene_data_id,
    tct_deriv_arr_float_3 const* default_value,
    bool uniform_lookup)
{
    // just return default value
    *result = *default_value;
}

// Implementation of scene_data_lookup_float2() with derivatives.
extern "C" __device__ void scene_data_lookup_deriv_float2(
    tct_deriv_arr_float_2 * result,
    Texture_handler_base const* self_base,
    Shading_state_material_with_derivs * state,
    unsigned int scene_data_id,
    tct_deriv_arr_float_2 const* default_value,
    bool uniform_lookup)
{
    // just return default value
    *result = *default_value;
}

// Implementation of scene_data_lookup_float() with derivatives.
extern "C" __device__ void scene_data_lookup_deriv_float(
    tct_deriv_float * result,
    Texture_handler_base const* self_base,
    Shading_state_material_with_derivs * state,
    unsigned int scene_data_id,
    tct_deriv_float const* default_value,
    bool uniform_lookup)
{
    // just return default value
    *result = *default_value;
}

#endif // TEX_SUPPORT_NO_DUMMY_SCENEDATA


//// ------------------------------------------------------------------------------------------------
//// Vtables
//// ------------------------------------------------------------------------------------------------
//
//#define TEX_SUPPORT_NO_VTABLES
//#ifndef TEX_SUPPORT_NO_VTABLES
//// The vtable containing all texture access handlers required by the generated code
//// in "vtable" mode.
//__device__ mi::neuraylib::Texture_handler_vtable tex_vtable = {
//  tex_lookup_float4_2d,
//  tex_lookup_float3_2d,
//  tex_texel_float4_2d,
//  tex_lookup_float4_3d,
//  tex_lookup_float3_3d,
//  tex_texel_float4_3d,
//  tex_lookup_float4_cube,
//  tex_lookup_float3_cube,
//  tex_resolution_2d,
//  tex_resolution_3d,
//  tex_texture_isvalid,
//  tex_frame,
//  df_light_profile_power,
//  df_light_profile_maximum,
//  df_light_profile_isvalid,
//  df_light_profile_evaluate,
//  df_light_profile_sample,
//  df_light_profile_pdf,
//  df_bsdf_measurement_isvalid,
//  df_bsdf_measurement_resolution,
//  df_bsdf_measurement_evaluate,
//  df_bsdf_measurement_sample,
//  df_bsdf_measurement_pdf,
//  df_bsdf_measurement_albedos,
//  adapt_normal,
//  scene_data_isvalid,
//  scene_data_lookup_float,
//  scene_data_lookup_float2,
//  scene_data_lookup_float3,
//  scene_data_lookup_float4,
//  scene_data_lookup_int,
//  scene_data_lookup_int2,
//  scene_data_lookup_int3,
//  scene_data_lookup_int4,
//  scene_data_lookup_color,
//};
//
//// The vtable containing all texture access handlers required by the generated code
//// in "vtable" mode with derivatives.
//__device__ mi::neuraylib::Texture_handler_deriv_vtable tex_deriv_vtable = {
//  tex_lookup_deriv_float4_2d,
//  tex_lookup_deriv_float3_2d,
//  tex_texel_float4_2d,
//  tex_lookup_float4_3d,
//  tex_lookup_float3_3d,
//  tex_texel_float4_3d,
//  tex_lookup_float4_cube,
//  tex_lookup_float3_cube,
//  tex_resolution_2d,
//  tex_resolution_3d,
//  tex_texture_isvalid,
//  tex_frame,
//  df_light_profile_power,
//  df_light_profile_maximum,
//  df_light_profile_isvalid,
//  df_light_profile_evaluate,
//  df_light_profile_sample,
//  df_light_profile_pdf,
//  df_bsdf_measurement_isvalid,
//  df_bsdf_measurement_resolution,
//  df_bsdf_measurement_evaluate,
//  df_bsdf_measurement_sample,
//  df_bsdf_measurement_pdf,
//  df_bsdf_measurement_albedos,
//  adapt_normal,
//  scene_data_isvalid,
//  scene_data_lookup_float,
//  scene_data_lookup_float2,
//  scene_data_lookup_float3,
//  scene_data_lookup_float4,
//  scene_data_lookup_int,
//  scene_data_lookup_int2,
//  scene_data_lookup_int3,
//  scene_data_lookup_int4,
//  scene_data_lookup_color,
//  scene_data_lookup_deriv_float,
//  scene_data_lookup_deriv_float2,
//  scene_data_lookup_deriv_float3,
//  scene_data_lookup_deriv_float4,
//  scene_data_lookup_deriv_color,
//};
//#endif // TEX_SUPPORT_NO_VTABLES

#endif // TEXTURE_LOOKUP_H

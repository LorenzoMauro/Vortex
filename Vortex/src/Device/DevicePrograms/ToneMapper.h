#pragma once
#ifndef TONEMAPPER_H
#define TONEMAPPER_H

#include <cuda_runtime.h>
#include "Core/Math.h"
#include "Utils.h"
namespace vtx
{
    __forceinline__ __device__ math::vec3f floatToScientificRGB(float value) {
        assert(value >= 0.0f && value <= 1.0f);

        float r = 0.0f;
        float g = 0.0f;
        float b = 0.0f;

        if (value < 0.5f) {
            // Value is between 0 and 0.5, so we interpolate between blue and green
            b = 1.0f - 2.0f * value;
            g = 2.0f * value;
        }
        else {
            // Value is between 0.5 and 1.0, so we interpolate between green and red
            g = 2.0f * (1.0f - value);
            r = 2.0f * (value - 0.5f);
        }

        return math::vec3f(r, g, b);
    }

    __forceinline__ __device__ math::vec3f toneMap(ToneMapperSettings* tm, const math::vec3f& radianceSample)
    {
        //printf("ToneMapperSettings pointer %p\n", tm);
        //printf("ToneMapperSettings: %f %f %f %f\n", tm->invGamma, tm->saturation, tm->burnHighlights, tm->crushBlacks);
        //return math::vec4f(radianceSample, 1.0f);
        math::vec3f ldrColor = tm->invWhitePoint * tm->colorBalance * radianceSample;

        ldrColor *= (ldrColor * tm->burnHighlights + 1.0f) / (ldrColor + 1.0f);
        float luminance = dot(ldrColor, math::vec3f(0.3f, 0.59f, 0.11f));
        ldrColor        = utl::max<math::vec3f>(utl::lerp(math::vec3f(luminance), ldrColor, tm->saturation), 0.0f);
        luminance       = dot(ldrColor, math::vec3f(0.3f, 0.59f, 0.11f));

        if (luminance < 1.0f)
        {
            ldrColor = utl::max<math::vec3f>(utl::lerp(utl::pow(ldrColor, math::vec3f(tm->crushBlacks)), ldrColor, sqrtf(luminance)), 0.0f);
        }

        ldrColor = utl::pow(ldrColor, math::vec3f(tm->invGamma));
        return { ldrColor};
    }
    // The code in this file was originally written by Stephen Hill (@self_shadow), who deserves all
	// credit for coming up with this fit and implementing it. Buy him a beer next time you see him. :)

	// Vector-Matrix multiplication
    __forceinline__ __device__ math::vec3f mul(const float ACESMat[3][3], const math::vec3f& v)
    {
        return math::vec3f(
            ACESMat[0][0] * v.x + ACESMat[0][1] * v.y + ACESMat[0][2] * v.z,
            ACESMat[1][0] * v.x + ACESMat[1][1] * v.y + ACESMat[1][2] * v.z,
            ACESMat[2][0] * v.x + ACESMat[2][1] * v.y + ACESMat[2][2] * v.z
        );
    }

    // RRTAndODTFit function
    __forceinline__ __device__ math::vec3f RRTAndODTFit(const math::vec3f& v)
    {
        math::vec3f a = v * (v + math::vec3f(0.0245786f)) - math::vec3f(0.000090537f);
        math::vec3f b = v * (math::vec3f(0.983729f) * v + math::vec3f(0.4329510f)) + math::vec3f(0.238081f);
        return a / b;
    }

    // ACESFitted function
    __forceinline__ __device__ math::vec3f ACESFitted(math::vec3f color)
    {
        // sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
        const float ACESInputMat[3][3] =
        {
            {0.59719f, 0.35458f, 0.04823f},
            {0.07600f, 0.90834f, 0.01566f},
            {0.02840f, 0.13383f, 0.83777f}
        };

        // ODT_SAT => XYZ => D60_2_D65 => sRGB
        const float ACESOutputMat[3][3] =
        {
            { 1.60475f, -0.53108f, -0.07367f},
            {-0.10208f,  1.10813f, -0.00605f},
            {-0.00327f, -0.07276f,  1.07602f}
        };

        color = mul(ACESInputMat, color);

        // Apply RRT and ODT
        color = RRTAndODTFit(color);

        color = mul(ACESOutputMat, color);

        // Clamp to [0, 1]
        color = math::saturate(color);

        return color;
    }
}




#endif

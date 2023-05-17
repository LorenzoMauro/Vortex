#pragma once
#ifndef TONEMAPPER_H
#define TONEMAPPER_H
#include <cuda_runtime.h>
#include "Core/Math.h"
#include "DataFetcher.h"
#include "Utils.h"

namespace vtx
{
    __device__ math::vec3f floatToScientificRGB(float value) {
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

    __forceinline__ __device__ math::vec3f toneMap(const math::vec3f& radianceSample)
    {
		const ToneMapperSettings* tm = optixLaunchParams.toneMapperSettings;

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
}



#endif

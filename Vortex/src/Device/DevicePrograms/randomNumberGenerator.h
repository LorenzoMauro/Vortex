﻿/*
 * Copyright (c) 2013-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#ifndef RANDOM_NUMBER_GENERATORS_H
#define RANDOM_NUMBER_GENERATORS_H

#include "cuda_runtime.h"

 // Tiny Encryption Algorithm (TEA) to calculate a the seed per launch index and iteration.
 // This results in a ton of integer instructions! Use the smallest N necessary.
template<unsigned int N>
__forceinline__ __both__ unsigned int tea(const unsigned int val0, const unsigned int val1)
{
    unsigned int v0 = val0;
    unsigned int v1 = val1;
    unsigned int s0 = 0;

    for (unsigned int n = 0; n < N; ++n)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xA341316C) ^ (v1 + s0) ^ ((v1 >> 5) + 0xC8013EA4);
        v1 += ((v0 << 4) + 0xAD90777D) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7E95761E);
    }
    return v0;
}

// Just do one LCG step. The new random unsigned int number is in the referenced argument.
__forceinline__ __both__ void lcg(unsigned int& previous)
{
    previous = previous * 1664525u + 1013904223u;
}


// Return a random sample in the range [0, 1) with a simple Linear Congruential Generator.
__forceinline__ __device__ float rng(unsigned int& previous)
{
    previous = previous * 1664525u + 1013904223u;

    //return float(previous & 0x00FFFFFF) / float(0x01000000u); // Use the lower 24 bits.
    return static_cast<float>(previous >> 8) / static_cast<float>(0x01000000u);      // Use the upper 24 bits
}

// Convenience function
__forceinline__ __device__ float2 rng2(unsigned int& previous)
{
    float2 s;

    previous = previous * 1664525u + 1013904223u;
    //s.x = float(previous & 0x00FFFFFF) / float(0x01000000u); // Use the lower 24 bits.
    s.x = static_cast<float>(previous >> 8) / static_cast<float>(0x01000000u);      // Use the upper 24 bits

    previous = previous * 1664525u + 1013904223u;
    //s.y = float(previous & 0x00FFFFFF) / float(0x01000000u); // Use the lower 24 bits.
    s.y = static_cast<float>(previous >> 8) / static_cast<float>(0x01000000u);      // Use the upper 24 bits

    return s;
}

// Convenience function
__forceinline__ __device__ float3 rng3(unsigned int& previous)
{
    float3 s;

    previous = previous * 1664525u + 1013904223u;
    //s.x = float(previous & 0x00FFFFFF) / float(0x01000000u); // Use the lower 24 bits.
    s.x = static_cast<float>(previous >> 8) / static_cast<float>(0x01000000u);      // Use the upper 24 bits

    previous = previous * 1664525u + 1013904223u;
    //s.y = float(previous & 0x00FFFFFF) / float(0x01000000u); // Use the lower 24 bits.
    s.y = static_cast<float>(previous >> 8) / static_cast<float>(0x01000000u);      // Use the upper 24 bits

    previous = previous * 1664525u + 1013904223u;
    //s.z = float(previous & 0x00FFFFFF) / float(0x01000000u); // Use the lower 24 bits.
    s.z = static_cast<float>(previous >> 8) / static_cast<float>(0x01000000u);      // Use the upper 24 bits

    return s;
}


// Convenience function
__forceinline__ __device__ float4 rng4(unsigned int& previous)
{
    float4 s;

    previous = previous * 1664525u + 1013904223u;
    //s.x = float(previous & 0x00FFFFFF) / float(0x01000000u); // Use the lower 24 bits.
    s.x = static_cast<float>(previous >> 8) / static_cast<float>(0x01000000u);      // Use the upper 24 bits

    previous = previous * 1664525u + 1013904223u;
    //s.y = float(previous & 0x00FFFFFF) / float(0x01000000u); // Use the lower 24 bits.
    s.y = static_cast<float>(previous >> 8) / static_cast<float>(0x01000000u);      // Use the upper 24 bits

    previous = previous * 1664525u + 1013904223u;
    //s.z = float(previous & 0x00FFFFFF) / float(0x01000000u); // Use the lower 24 bits.
    s.z = static_cast<float>(previous >> 8) / static_cast<float>(0x01000000u);      // Use the upper 24 bits

    previous = previous * 1664525u + 1013904223u;
    //s.w = float(previous & 0x00FFFFFF) / float(0x01000000u); // Use the lower 24 bits.
    s.w = static_cast<float>(previous >> 8) / static_cast<float>(0x01000000u);      // Use the upper 24 bits

    return s;
}

__forceinline__ __device__ float nRng(unsigned int& previous)
{
    const float epsilon = 1e-6f;
    float u1 = rng(previous);
    u1 = (u1 < epsilon) ? epsilon : (u1 > 1.0f - epsilon) ? 1.0f - epsilon : u1;
    const float u2 = rng(previous);
    const float r = sqrtf(-2.0f * logf(u1));
    const float phi = 2.0f * 3.1415926535897932384626433832795f * u2;
    const float x = r * cosf(phi);

    return x;
}

__forceinline__ __device__ float2 nRng2(unsigned int& previous)
{
    float2 z;
    z.x = nRng(previous);
    z.y = nRng(previous);
    return z;
}

#endif // RANDOM_NUMBER_GENERATORS_H

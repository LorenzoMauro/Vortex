#pragma once

#ifndef PER_RAY_DATA_H
#define PER_RAY_DATA_H

#include <vector_types.h>
#include "Core/Math.h"
#include <mi/neuraylib/target_code_types.h>
#include "LaunchParams.h"

namespace vtx {

#define NUM_TEXTURE_SPACES 2
#define NUM_TEXTURE_RESULTS 16
#define MATERIAL_STACK_LAST 3
#define MATERIAL_STACK_SIZE 4
#define FLAG_HIT           0x00000001
#define FLAG_SHADOW        0x00000002
// Prevent that division by very small floating point values results in huge values, for example dividing by pdf.
#define DENOMINATOR_EPSILON 1.0e-6f
#define REDCOLOR  math::vec3f(1.0f, 0.0f, 0.0f)
#define GREENCOLOR  math::vec3f(0.0f, 1.0f, 0.0f)
#define BLUECOLOR  math::vec3f(0.0f, 0.0f, 1.0f)

	// Alias the PerRayData pointer and an math::vec2f for the payload split and merge operations. This generates only move instructions.
	union Payload {
		__host__ __device__ Payload(): dat(math::vec2ui(0)) {};
		void* ptr;
		math::vec2ui dat; // Initialization removed here
	};

	__forceinline__ __device__ math::vec2ui splitPointer(void* ptr)
	{
		Payload payload;

		payload.ptr = ptr;

		return payload.dat;
	}

	__forceinline__ __device__ void* mergePointer(const unsigned int p0, const unsigned int p1)
	{
		Payload payload;

		payload.dat.x = p0;
		payload.dat.y = p1;

		return payload.ptr;
	}


}

#endif
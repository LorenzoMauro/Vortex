#pragma once

#ifndef PER_RAY_DATA_H
#define PER_RAY_DATA_H

#include <vector_types.h>
#include "Core/Math.h"

namespace vtx {

	struct MaterialStack
	{
		math::vec3f ior;
		math::vec3f absorption;
		math::vec3f scattering;
		float bias;
	};

	struct PerRayData {
		math::vec3f position;
		math::vec3f wi;
		math::vec3f wo;
		math::vec3f	distance;
		math::vec3f	radiance;
		math::vec3f debugColor;
		math::vec3f pdf;
		math::vec3f throughput;
		MaterialStack stack;
	};

	// Alias the PerRayData pointer and an math::vec2f for the payload split and merge operations. This generates only move instructions.
	typedef union
	{
		PerRayData* ptr;
		math::vec2ui dat{0,0};
	} Payload;

	__forceinline__ __device__ math::vec2ui splitPointer(PerRayData* ptr)
	{
		Payload payload;

		payload.ptr = ptr;

		return payload.dat;
	}

	__forceinline__ __device__ PerRayData* mergePointer(const unsigned int p0, const unsigned int p1)
	{
		Payload payload;

		payload.dat.x = p0;
		payload.dat.y = p1;

		return payload.ptr;
	}


}

#endif
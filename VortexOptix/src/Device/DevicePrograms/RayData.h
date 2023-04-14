#pragma once
#include "Core/Math.h"

namespace vtx {


	struct PerRayData {
		math::vec3f position;
		math::vec3f wi;
		math::vec3f	distance;
		math::vec3f	color;
	};

	// Alias the PerRayData pointer and an uint2 for the payload split and merge operations. This generates only move instructions.
	typedef union
	{
		PerRayData* ptr;
		uint2       dat;
	} Payload;

	__forceinline__ __device__ uint2 splitPointer(PerRayData* ptr)
	{
		Payload payload;

		payload.ptr = ptr;

		return payload.dat;
	}

	__forceinline__ __device__ PerRayData* mergePointer(unsigned int p0, unsigned int p1)
	{
		Payload payload;

		payload.dat.x = p0;
		payload.dat.y = p1;

		return payload.ptr;
	}


}
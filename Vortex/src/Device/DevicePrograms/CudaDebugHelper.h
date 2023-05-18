#pragma once
#ifndef CUDADEBUGHELPER_H
#define CUDADEBUGHELPER_H

#if defined(__CUDACC__) || defined(__CUDA__)
	#ifdef PRINT_ALL
		#define CUDA_DEBUG_PRINT(message, ...) printf(message, __VA_ARGS__)
	#else
		#define CUDA_DEBUG_PRINT(message, ...)
	#endif

	#define CUDA_ERROR_PRINT(message, ...) \
		printf("Error in line %d, file %s, function %s.\n", __LINE__, __FILE__, __FUNCTION__);\
		printf(message, __VA_ARGS__);\
		//assert(0);


#else
	#define CUDA_DEBUG_PRINT(message, ...)
	#define CUDA_ERROR_PRINT(message, ...)
#endif

#include "Core/Math.h"

namespace vtx
{

	__forceinline__ __device__ void printMath(const char* message, const float4* matrix)
	{
		printf("%s :\n"
					"\t %.1f %.1f %.1f %.1f\n"
					"\t %.1f %.1f %.1f %.1f\n"
					"\t %.1f %.1f %.1f %.1f\n",
				message,
					matrix[0].x, matrix[0].y, matrix[0].z, matrix[0].w,
					matrix[1].x, matrix[1].y, matrix[1].z, matrix[1].w,
					matrix[2].x, matrix[2].y, matrix[2].z, matrix[2].w);
	}

	__forceinline__ __device__ void printMath(const char* message, const math::vec3f& vector)
	{
		printf("%s :\n"
					"\t %.1f %.1f %.1f\n",
			   message, 
					vector.x, vector.y, vector.z);		
	}


	__forceinline__ __device__ void printMath(const char* message, const math::affine3f& affine)
	{
		printf("%s :\n"
					"\t %.1f %.1f %.1f %.1f\n"
					"\t %.1f %.1f %.1f %.1f\n"
					"\t %.1f %.1f %.1f %.1f\n",
			   message,
				 affine.l.vx.x, affine.l.vy.x, affine.l.vz.x, affine.p.x,
				 affine.l.vx.y, affine.l.vy.y, affine.l.vz.y, affine.p.y,
				 affine.l.vx.z, affine.l.vy.z, affine.l.vz.z, affine.p.z);
	}


}



#endif
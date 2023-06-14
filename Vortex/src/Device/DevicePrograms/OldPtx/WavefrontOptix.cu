#include <optix_device.h>
#include <cuda_runtime_api.h>
#include "Core/Math.h"
#include "Device/DevicePrograms/RayData.h"

namespace vtx
{
	extern "C" __constant__ LaunchParams optixLaunchParams;

	struct TracePayload
	{
	};

	__forceinline__ __device__ int getLaunchIndex()
	{

		const int      ix = optixGetLaunchIndex().x;
		const int      iy = optixGetLaunchIndex().y;
		const uint32_t fbIndex = ix + iy * frameSize.x;
	}
	__forceinline__ __device__ void trace(math::vec3f& origin, math::vec3f& direction, TraceType traceType)
	{
		TracePayload tracePrd;
		math::vec2ui payload = splitPointer(&tracePrd);

		optixTrace(optixLaunchParams.topObject,
			origin,
			direction, // origin, direction
			optixLaunchParams.settings->minClamp,
			prd.distance,
			0.0f, // tmin, tmax, time
			static_cast<OptixVisibilityMask>(0xFF),
			OPTIX_RAY_FLAG_DISABLE_ANYHIT,    //OPTIX_RAY_FLAG_NONE,
			optixLaunchParams.programs->hit,  //SBT Offset
			0,                                // SBT stride
			optixLaunchParams.programs->miss, // missSBTIndex
			payload.x,
			payload.y);

	}

	extern "C" __global__ void __raygen__closestTrace()
	{
		const int index = optixGetLaunchIndex().x;

	}

	extern "C" __global__ void __raygen__anyTrace()
	{
		const int index = optixGetLaunchIndex().x;

	}


}

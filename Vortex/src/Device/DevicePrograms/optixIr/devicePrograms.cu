#ifndef OPTIXCODE
#define OPTIXCODE
#endif

#include "../RayData.h"
#include "../Utils.h"
#include "../ToneMapper.h"
#define ARCHITECTURE_OPTIX
#include "Device/DevicePrograms/rendererFunctions.h"

namespace vtx
{
	extern "C" __constant__ LaunchParams optixLaunchParams;

	//extern "C" __global__ void __exception__all()
	//{
	//	//const uint3 theLaunchDim     = optixGetLaunchDimensions(); 
	//	const uint3 theLaunchIndex = optixGetLaunchIndex();
	//	const int   theExceptionCode = optixGetExceptionCode();
	//	const char* exceptionLineInfo = optixGetExceptionLineInfo();

	//	printf("Optix Exception: \n"
	//		"    Code: %d\n"
	//		"    LineInfo: %s\n"
	//		"    at launch Index (pixel): x = %u y = %u\n",
	//		theExceptionCode, exceptionLineInfo, theLaunchIndex.x, theLaunchIndex.y);

	//	// FIXME This only works for render strategies where the launch dimension matches the outputBuffer resolution.
	//	//float4* buffer = reinterpret_cast<float4*>(sysData.outputBuffer);
	//	//const unsigned int index = theLaunchIndex.y * theLaunchDim.x + theLaunchIndex.x;

	//	//buffer[index] = make_float4(1000000.0f, 0.0f, 1000000.0f, 1.0f); // super magenta
	//}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////// SHADOW TRACE //////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	extern "C" __global__ void __miss__radianceAndShadow()
	{
		optixSetPayload_0(0);
	}

	extern "C" __global__ void __anyhit__shadow()
	{
		auto* prd = reinterpret_cast<ShadowWorkItem*>(mergePointer(optixGetPayload_0(), optixGetPayload_1()));
		bool hasOpacity = optixLaunchParams.instances[optixGetInstanceId()]->hasOpacity;
		
		if(hasOpacity)
		{
			prd->distance = optixGetRayTmax();
			HitProperties hitProperties;
			hitProperties.position = prd->origin;
			const float2 baricenter2D = optixGetTriangleBarycentrics();
			const auto baricenter = math::vec3f(1.0f - baricenter2D.x - baricenter2D.y, baricenter2D.x, baricenter2D.y);
			const math::vec3f position = hitProperties.position + prd->direction * prd->distance;
			hitProperties.init(optixGetInstanceId(), optixGetPrimitiveIndex(), baricenter, position);

			transparentAnyHit(
				hitProperties,
				prd->direction,
				prd->mediumIor,
				prd->seed,
				&optixLaunchParams
			);
		}
	}

	extern "C" __global__ void __anyhit__radianceHit()
	{
		auto* prd = reinterpret_cast<RayWorkItem*>(mergePointer(optixGetPayload_0(), optixGetPayload_1()));
		RayWorkItem prdCopy = *prd;
		bool hasOpacity = optixLaunchParams.instances[optixGetInstanceId()]->hasOpacity;
		if (hasOpacity)
		{
			optixHitProperties(&prdCopy);
			transparentAnyHit(&prdCopy, &optixLaunchParams);
		}
	}

	extern "C" __global__ void __closesthit__radiance()
	{
		auto* prd = reinterpret_cast<RayWorkItem*>(mergePointer(optixGetPayload_0(), optixGetPayload_1()));
		optixHitProperties(prd);
	}


	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////// FULL OPTIX RAYGEN /////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	extern "C" __global__ void __raygen__renderFrame()
	{
		const int      ix = optixGetLaunchIndex().x;
		const int      iy = optixGetLaunchIndex().y;
		const int	   fbIndex = ix + iy * optixLaunchParams.frameBuffer.frameSize.x;

		cleanFrameBuffer(fbIndex, &optixLaunchParams);

		const int samplesPerLaunch = getAdaptiveSampleCount(fbIndex, &optixLaunchParams);
		
		TraceWorkItem twi;
		
		for (int i = 0; i < samplesPerLaunch; i++)
		{
			twi.seed = tea<4>(fbIndex + i, optixLaunchParams.settings.renderer.iteration + *optixLaunchParams.frameID);
			generateCameraRay(fbIndex, &optixLaunchParams, twi);
			for (int i = 0; i < optixLaunchParams.settings.renderer.maxBounces; i++)
			{
				elaborateRadianceTrace(twi, optixLaunchParams, A_FULL_OPTIX);
				if(!twi.extendRay)
				{
					break;
				}
			}
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////// WAVEFRONT RADIANCE TRACE //////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	extern "C" __global__ void __raygen__trace()
	{
		const int queueWorkId = optixGetLaunchIndex().x;
		wfTraceRadianceEntry(queueWorkId, optixLaunchParams);
	}

	extern "C" __global__ void __raygen__shadow()
	{
		const int queueWorkId = optixGetLaunchIndex().x;
		wfTraceShadowEntry(queueWorkId, optixLaunchParams);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////// WAVEFRONT SHADE  //////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	extern "C" __global__ void __raygen__shade()
	{
		const int queueWorkId = optixGetLaunchIndex().x;
		handleShading(queueWorkId, optixLaunchParams);
	}

}
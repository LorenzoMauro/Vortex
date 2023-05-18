#ifndef OPTIXCODE
#define OPTIXCODE
#endif

#include <optix_device.h>
#include "../RayData.h"
#include "../DataFetcher.h"
#include "../Utils.h"
#include "../ToneMapper.h"
#include "Device/DevicePrograms/randomNumberGenerator.h"
//#include "../LightSampler.h"

namespace vtx
{

	//------------------------------------------------------------------------------
	// closest hit and anyhit programs for radiance-type rays.
	//------------------------------------------------------------------------------
	//extern "C" __constant__ LaunchParams optixLaunchParams;

	extern "C" __global__ void __exception__all()
	{
		//const uint3 theLaunchDim     = optixGetLaunchDimensions(); 
		const uint3 theLaunchIndex    = optixGetLaunchIndex();
		const int   theExceptionCode  = optixGetExceptionCode();
		const char* exceptionLineInfo = optixGetExceptionLineInfo();

		printf("Optix Exception: \n"
			   "    Code: %d\n"
			   "    LineInfo: %s\n"
			   "    at launch Index (pixel): x = %u y = %u\n",
			   theExceptionCode, exceptionLineInfo, theLaunchIndex.x, theLaunchIndex.y);

		// FIXME This only works for render strategies where the launch dimension matches the outputBuffer resolution.
		//float4* buffer = reinterpret_cast<float4*>(sysData.outputBuffer);
		//const unsigned int index = theLaunchIndex.y * theLaunchDim.x + theLaunchIndex.x;

		//buffer[index] = make_float4(1000000.0f, 0.0f, 1000000.0f, 1.0f); // super magenta
	}




	//------------------------------------------------------------------------------
	// miss program that gets called for any ray that did not have a
	// valid intersection
	//
	// as with the anyhit/closest hit programs, in this example we only
	// need to have _some_ dummy function to set up a valid SBT
	// ------------------------------------------------------------------------------

	extern "C" __global__ void __miss__radiance()
	{
		/*! for this simple example, this will remain empty */
		PerRayData* prd         = mergePointer(optixGetPayload_0(), optixGetPayload_1());
		/*printf("INSIDE			- prd pointer	%p\n"
			   "FIRST			-				%d\n"
			   "SECOND			-				%d\n",
			   prd, optixGetPayload_0(), optixGetPayload_1());*/
		prd->traceResult         = TR_MISS;
		
		if(prd->traceOperation == TR_HIT)
		{
			if(optixLaunchParams.envLight != nullptr)
			{
				const LightData* envLight              = optixLaunchParams.envLight;
				EnvLightAttributesData attrib = *reinterpret_cast<EnvLightAttributesData*>(envLight->attributes);
				auto texture = attrib.texture;

				math::vec3f dir = math::transformNormal3F(attrib.invTransformation, prd->wi);
				

				{
					bool computeOriginalUV = false;
					float u, v;
					if(computeOriginalUV)
					{
						u = fmodf(atan2f(dir.y, dir.x) * (float)(0.5 / M_PI) + 0.5f, 1.0f);
						v = acosf(fmax(fminf(-dir.z, 1.0f), -1.0f)) * (float)(1.0 / M_PI);
					}
					else{
						float theta = acosf(-dir.z);
						v = theta / (float)M_PI;
						float phi = atan2f(dir.y, dir.x);// + M_PI / 2.0f; // azimuth angle (theta)
						u = (phi + (float)M_PI) / (float)(2.0f * M_PI);
					}
					

					const auto x = math::min<unsigned>((unsigned int)(u * (float)texture->dimension.x), texture->dimension.x - 1);
					const auto y = math::min<unsigned>((unsigned int)(v * (float)texture->dimension.y), texture->dimension.y - 1);
					math::vec3f emission = tex2D<float4>(texture->texObj, u, v);
					//emission = emission; *attrib.envIntensity;

					// to incorporate the point light selection probability
					float misWeight = 1.0f;
					// If the last surface intersection was a diffuse event which was directly lit with multiple importance sampling,
					// then calculate light emission with multiple importance sampling for this implicit light hit as well.
					if (prd->pdf > 0.0f && optixLaunchParams.settings->samplingTechnique == RendererDeviceSettings::S_MIS)
					{
						float envSamplePdf = attrib.aliasMap[y * texture->dimension.x + x].pdf;
						misWeight = utl::heuristic(prd->pdf, envSamplePdf);
						//misWeight = prd->pdf / (prd->pdf + envSamplePdf);
					}
					prd->radiance += prd->throughput * emission * misWeight;
					prd->eventType = mi::neuraylib::BSDF_EVENT_ABSORB;

					if (prd->depth == 0)
					{
						prd->colors.diffuse = emission;
						prd->colors.trueNormal = prd->wi;
						prd->colors.shadingNormal = 0.0f;
					}
				}
			}
		}
	}

	__forceinline__ __device__ math::vec3f integrator(PerRayData& prd)
	{
		// The integrator starts with black radiance and full path throughput.
		prd.radiance = math::vec3f(0.0f);
		prd.pdf = 0.0f;
		prd.throughput = math::vec3f(1.0f);
		prd.sigmaT = math::vec3f(0.0f); // Extinction coefficient: sigma_a + sigma_s.
		prd.walk = 0; // Number of random walk steps taken through volume scattering. 
		prd.eventType = mi::neuraylib::BSDF_EVENT_ABSORB; // Initialize for exit. (Otherwise miss programs do not work.)

		prd.idxStack = 0; // Nested material handling. 
		// Small stack of four entries of which the first is vacuum.
		prd.stack[0].ior        = math::vec3f(1.0f); // No effective IOR.
		prd.stack[0].absorption = math::vec3f(0.0f); // No volume absorption.
		prd.stack[0].scattering = math::vec3f(0.0f); // No volume scattering.
		prd.stack[0].bias       = 0.0f;              // Isotropic volume scattering.
		prd.depth               = 0;

		prd.colors.diffuse = 0.0f;
		prd.colors.trueNormal = 0.0f;
		prd.colors.shadingNormal = 0.0f;
		prd.colors.tangent = 0.0f;
		prd.colors.orientation = 0.0f;
		prd.colors.debugColor1 = 0.0f;
		prd.colors.uv = 0.0f;


		int maxDepth = (optixLaunchParams.settings->samplingTechnique== RendererDeviceSettings::SamplingTechnique::S_DIRECT_LIGHT) ? 0 :optixLaunchParams.settings->maxBounces;

		uint2 payload = splitPointer(&prd);

		//uint32_t ptr1;
		//uint32_t ptr2;
		//pack_pointer(&prd, ptr1, ptr2);
		while (prd.depth <= maxDepth)
		{
			prd.wo         = -prd.wi;
			prd.distance   = optixLaunchParams.settings->maxClamp;
			prd.traceResult = TR_UNKNOWN;
			prd.traceOperation = TR_HIT;

			//printf("OUT - PRD POINTER %p\n", &prd);
			/*printf(	"OUTSIDE		- prd pointer	%p\n"
					"FIRST			-				%d\n"
					"SECOND			-				%d\n",
					(void*) & prd, payload.x, payload.y);*/

			//printf("OUTSIDE PTR : %p\n", &prd);

			optixTrace(optixLaunchParams.topObject,
					   prd.position,
					   prd.wi, // origin, direction
					   optixLaunchParams.settings->minClamp,
					   prd.distance,
					   0.0f, // tmin, tmax, time
					   static_cast<OptixVisibilityMask>(0xFF),
					   OPTIX_RAY_FLAG_NONE,//OPTIX_RAY_FLAG_DISABLE_ANYHIT,    //OPTIX_RAY_FLAG_NONE,
					   0, //optixLaunchParams.programs->hit,  //SBT Offset
					   0,                                // SBT stride
					   optixLaunchParams.programs->miss, //, // missSBTIndex
					   payload.x,
					   payload.y);


			// Path termination by miss shader or sample() routines.
			if (prd.eventType == mi::neuraylib::BSDF_EVENT_ABSORB || prd.throughput == math::vec3f(0.0f) || prd.traceResult == TR_MISS)
			{
				break;
			}

			// Unbiased Russian Roulette path termination.
			if (2 <= prd.depth) // Start termination after a minimum number of bounces.
			{
				const float probability = fmaxf(fmaxf(prd.throughput.x, prd.throughput.y), prd.throughput.z);

				if (probability < rng(prd.seed)) // Paths with lower probability to continue are terminated earlier.
				{
					break;
				}

				prd.throughput /= probability; // Path isn't terminated. Adjust the throughput so that the average is right again.
			}


			++prd.depth; // Next path segment.
		}

		return prd.radiance;
	}


	//------------------------------------------------------------------------------
	// ray gen program - the actual rendering happens in here
	//------------------------------------------------------------------------------
	extern "C" __global__ void __raygen__renderFrame()
	{
		const RendererDeviceSettings* settings    = getData<RendererDeviceSettings>();
		const FrameBufferData*        frameBuffer = getData<FrameBufferData>();
		const math::vec2ui&           frameSize   = frameBuffer->frameSize;
		const int      ix      = optixGetLaunchIndex().x;
		const int      iy      = optixGetLaunchIndex().y;
		const uint32_t fbIndex = ix + iy * frameSize.x;
		const math::vec2f pixel = math::vec2f(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
		float& noise = frameBuffer->noiseBuffer[fbIndex].noise;
		auto outputBuffer = reinterpret_cast<math::vec4f*>(frameBuffer->outputBuffer);
		bool runningAverage = false;

		int& iteration = optixLaunchParams.settings->iteration;
		int& samples = frameBuffer->noiseBuffer[fbIndex].samples;
		if(iteration <= 0)
		{
			samples = 0;
		}

		PerRayData prd;
		prd.seed = tea<4>(fbIndex, *optixLaunchParams.frameID);

		int* actualSamples;
		int samplesPerLaunch = 1;
		if (settings->adaptiveSampling)
		{
			actualSamples = &samples;
			if (settings->minAdaptiveSamples <= iteration)
			{

				if (noise < settings->noiseCutOff)
				{
					samplesPerLaunch = 0;
				}
				else
				{
					samplesPerLaunch = utl::lerp((float)settings->minPixelSamples, (float)settings->maxPixelSamples, noise*noise);
				}
			}
		}
		else
		{
			actualSamples = &iteration;
		}

		for (int i = 0; i < samplesPerLaunch; i++)
		{
			const math::vec2f sample = rng2(prd.seed);
			const math::vec2f screen{ static_cast<float>(frameSize.x), static_cast<float>(frameSize.y) };

			LensRay cameraRay;
			//cameraRay = optixDirectCall<LensRay, const math::vec2f, const math::vec2f, const math::vec2f>(optixLaunchParams.programs->pinhole, screen, pixel, sample);
			const math::vec2f fragment = pixel + sample;                    // Jitter the sub-pixel location
			const math::vec2f ndc = (fragment / screen) * 2.0f - 1.0f;      // Normalized device coordinates in range [-1, 1].

			const CameraData camera = optixLaunchParams.cameraData;

			cameraRay.org = camera.position;
			cameraRay.dir = math::normalize<float>(camera.horizontal * ndc.x + camera.vertical * ndc.y + camera.direction);

			prd.position = cameraRay.org;
			prd.wi = cameraRay.dir;

			math::vec3f radiance=0.0f;
			radiance = integrator(prd);

			// DEBUG Highlight numerical errors.
			//if (isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z))
			//{
			//	radiance = make_float3(0.0f, 1000000.0f, 0.0f); // super green
			//}
			//else if (isinf(radiance.x) || isinf(radiance.y) || isinf(radiance.z))
			//{
			//	radiance = make_float3(1000000.0f, 0.0f, 0.0f); // super red
			//}
			//else if (radiance.x < 0.0f || radiance.y < 0.0f || radiance.z < 0.0f)
			//{
			//	radiance = make_float3(0.0f, 0.0f, 1000000.0f); // super blue
			//}
			
			if ((!(isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z))))
			{
				if ((!settings->accumulate || *actualSamples <= 0))
				{
					frameBuffer->radianceBuffer[fbIndex] = radiance;
				}
				else
				{
					if (runningAverage)
					{
						const math::vec3f dst = frameBuffer->radianceBuffer[fbIndex];
						frameBuffer->radianceBuffer[fbIndex] = utl::lerp(dst, radiance, (float)(*actualSamples+1));
					}
					else
					{
						frameBuffer->radianceBuffer[fbIndex] += radiance;
					}
				}
			}
			samples++;
		}

		if ((!settings->accumulate || *actualSamples == 0))
		{
			frameBuffer->toneMappedRadiance[fbIndex] = toneMap(frameBuffer->radianceBuffer[fbIndex]);
		}
		else
		{
			if (runningAverage)
			{
				frameBuffer->toneMappedRadiance[fbIndex] = toneMap(frameBuffer->radianceBuffer[fbIndex]);
			}
			else
			{
				int divideSamples = (settings->adaptiveSampling) ? *actualSamples : *actualSamples + 1;
				frameBuffer->toneMappedRadiance[fbIndex] = toneMap(frameBuffer->radianceBuffer[fbIndex] / ((float)divideSamples));
			}
		}
		frameBuffer->albedo[fbIndex] = toneMap(prd.colors.diffuse);
		frameBuffer->normal[fbIndex] = toneMap(prd.colors.shadingNormal);

		switch (settings->displayBuffer)
		{
			case(RendererDeviceSettings::DisplayBuffer::FB_NOISY):
			{
				outputBuffer[fbIndex] = math::vec4f(frameBuffer->toneMappedRadiance[fbIndex], 1.0f);
			}
			break;
			case(RendererDeviceSettings::DisplayBuffer::FB_DIFFUSE):
			{
				outputBuffer[fbIndex] = math::vec4f(prd.colors.diffuse, 1.0f);
			}
			break;
			case(RendererDeviceSettings::DisplayBuffer::FB_ORIENTATION):
			{
				outputBuffer[fbIndex] = math::vec4f(prd.colors.orientation, 1.0f);
			}
			break;
			case(RendererDeviceSettings::DisplayBuffer::FB_TRUE_NORMAL):
			{
				outputBuffer[fbIndex] = math::vec4f(prd.colors.trueNormal, 1.0f);
			}
			break;
			case(RendererDeviceSettings::DisplayBuffer::FB_SHADING_NORMAL):
			{
				outputBuffer[fbIndex] = math::vec4f(prd.colors.shadingNormal, 1.0f);
			}
			break;
			case(RendererDeviceSettings::DisplayBuffer::FB_TANGENT):
			{
				outputBuffer[fbIndex] = math::vec4f(prd.colors.tangent, 1.0f);
			}
			break;
			case(RendererDeviceSettings::DisplayBuffer::FB_UV):
			{
				outputBuffer[fbIndex] = math::vec4f(prd.colors.uv, 1.0f);
			}
			break;
			case(RendererDeviceSettings::DisplayBuffer::FB_NOISE):
			{
				outputBuffer[fbIndex] = math::vec4f(floatToScientificRGB(noise), 1.0f);
			}
			break;
			case(RendererDeviceSettings::DisplayBuffer::FB_SAMPLES):
			{
				float samplesMetric;
				if(iteration> settings->minAdaptiveSamples)
				{
					samplesMetric = (float)(samples - settings->minAdaptiveSamples) / (float)(iteration - settings->minAdaptiveSamples);
				}
				else
				{
					samplesMetric = 0.5f;
				}
				outputBuffer[fbIndex] = math::vec4f(floatToScientificRGB(toneMap(samplesMetric).x), 1.0f);
			}
			break;
			case(RendererDeviceSettings::DisplayBuffer::FB_DEBUG_1):
			{
				outputBuffer[fbIndex] = math::vec4f(prd.colors.debugColor1, 1.0f);
			}
			break;
		}
	}
}

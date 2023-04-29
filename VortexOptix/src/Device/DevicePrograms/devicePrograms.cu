#ifndef OPTIXCODE
#define OPTIXCODE
#endif

#include <optix_device.h>
#include "RayData.h"
#include "DataFetcher.h"
#include "mdlDeviceWrapper.h"
#include "Utils.h"
#include "texture_lookup.h"

namespace vtx
{

	//------------------------------------------------------------------------------
	// closest hit and anyhit programs for radiance-type rays.
	//------------------------------------------------------------------------------

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


	extern "C" __global__ void __closesthit__radiance()
	{
		PerRayData*        prd         = mergePointer(optixGetPayload_0(), optixGetPayload_1());
		const unsigned int instanceId  = optixGetInstanceId();
		const unsigned int triangleIdx = optixGetPrimitiveIndex();
		prd->distance                  = optixGetRayTmax();
		prd->position                  = prd->position + prd->wi * prd->distance;

		HitProperties hitP;
		hitP.position     = prd->position;
		hitP.direction    = prd->wo;
		hitP.baricenter   = math::vec3f(0.0f, optixGetTriangleBarycentrics().x, optixGetTriangleBarycentrics().y);
		hitP.baricenter.x = 1.0f - hitP.baricenter.y - hitP.baricenter.z;

		utl::getInstanceAndGeometry(&hitP, instanceId);
		utl::getVertices(&hitP, triangleIdx);
		utl::fetchTransformsFromHandle(&hitP);
		utl::computeGeometricHitProperties(&hitP);
		utl::determineMaterialHitProperties(&hitP, triangleIdx);

		//Auxiliary Data
		if (prd->depth == 0)
		{
			prd->colors.trueNormal  = 0.5f * (hitP.ngW + 1.0f);
			prd->colors.orientation = hitP.isFrontFace ? math::vec3f(0.0f, 0.0f, 1.0f) : math::vec3f(1.0f, 0.0f, 0.0f);
		}

		if (hitP.material != nullptr)
		{
			mdl::MdlData    mdlData;
			mdl::InitConfig mdlConfig;
			mdlConfig.evaluateOpacity    = true;
			mdlConfig.evaluateEmission   = true;
			mdlConfig.evaluateScattering = true;
			mdl::initMdl(hitP, &mdlData, mdlConfig);

			math::vec3f ior = mdl::getIor(mdlData);

			// Auxiliary Data
			if (prd->depth == 0)
			{
				mdl::BsdfAuxiliaryData auxiliaryData = mdl::getAuxiliaryData(
					mdlData, ior, prd->idxStack, prd->stack, hitP.ngW);

				if (auxiliaryData.isValid)
				{
					prd->colors.diffuse       = auxiliaryData.data.albedo;
					prd->colors.shadingNormal = auxiliaryData.data.normal;
				}
				else
				{
					prd->colors.diffuse       = math::vec3f(1.0f, 0.0f, 1.0f);
					prd->colors.shadingNormal = hitP.ngW;
				}
			}

			RendererDeviceSettings::SamplingTechnique& samplingTechnique = optixLaunchParams.settings->samplingTechnique;

			//Evauluate Hit Point Emission
			if (hitP.meshLightAttributes != nullptr && mdlData.emissionFunctions.hasEmission)
			{
				mdl::EmissionEvaluateData evalData = mdl::evaluateEmission(mdlData, prd->wo);
				if (evalData.isValid)
				{
					const float area  = hitP.meshLightAttributes->totalArea;
					evalData.data.pdf = prd->distance * prd->distance / (area * evalData.data.cos);
					// Solid angle measure.

					float misWeight = 1.0f;

					// If the last event was diffuse or glossy, calculate the opposite MIS weight for this implicit light hit.
					if(samplingTechnique == RendererDeviceSettings::S_MIS && prd->eventType & (mi::neuraylib::BSDF_EVENT_DIFFUSE | mi::neuraylib::BSDF_EVENT_GLOSSY))
					{
						misWeight = utl::balanceHeuristic(prd->pdf, evalData.data.pdf);
					}
					// Power (flux) [W] divided by light area gives radiant exitance [W/m^2].
					const float factor = (mdlData.emissionFunctions.mode == 0) ? 1.0f : 1.0f / area;

					prd->radiance += prd->throughput * mdlData.emissionFunctions.intensity * evalData.data.edf * (factor* misWeight);
				}
			}

			const math::vec3f throughput = prd->throughput;
			prd->pdf                       = 0.0f;

			// BSDF Sampling
			if(samplingTechnique == RendererDeviceSettings::S_MIS | samplingTechnique==RendererDeviceSettings::S_BSDF)
			{
				mdl::BsdfSampleData sampleData = mdl::sampleBsdf(mdlData, ior, prd->idxStack, prd->stack, prd->wo, prd->seed);
				//Importance Sampling the Bsdf
				if (sampleData.isValid)
				{
					prd->wi = sampleData.data.k2; // Continuation direction.
					prd->throughput *= sampleData.data.bsdf_over_pdf;
					// Adjust the path throughput for all following incident lighting.
					prd->pdf = sampleData.data.pdf;
					// Note that specular events return pdf == 0.0f! (=> Not a path termination condition.)
					prd->eventType = sampleData.data.event_type;
					// This replaces the PRD flags used inside the other examples.
				}
				else
				{
					prd->traceResult = TR_HIT;
					return;
					// None of the following code will have any effect in that case.
				}
			}

			//Direct Light Sampling
			if (int numLights = optixLaunchParams.lightMap->size; numLights > 0 && (samplingTechnique == RendererDeviceSettings::S_MIS | samplingTechnique == RendererDeviceSettings::S_DIRECT_LIGHT))
			{
				//Randomly Selecting a Light
				//TODO, I think here we can do some better selection by giving more importance to lights with greater power
				const int indexLight = (1 < numLights) ? gdt::clamp(static_cast<int>(floorf(rng(prd->seed) * numLights)), 0,numLights - 1): 0;

				LightData light = optixLaunchParams.lightMap->values[indexLight];

				LightType typeLight = light.type;

				int lightSampleProgramIdx = -1;
				switch (typeLight)
				{
					case L_MESH:
					{
						lightSampleProgramIdx = optixLaunchParams.programs->meshLightSample;
					}
					break;
					default: break;
				}

				if (lightSampleProgramIdx != -1)
				{
					LightSample lightSample = optixDirectCall<LightSample, const LightData&, PerRayData*>(lightSampleProgramIdx, light, prd);
					if (lightSample.isValid)
					{
						mdl::BsdfEvaluateData evalData = mdl::evaluateBsdf(mdlData, ior, prd->idxStack, prd->stack, lightSample.direction, prd->wo);

						auto bxdf = math::vec3f(0.0f, 0.0f, 0.0f);
						bxdf += evalData.data.bsdf_diffuse;
						bxdf += evalData.data.bsdf_glossy;



						if (0.0f < evalData.data.pdf && bxdf != math::vec3f(0.0f, 0.0f, 0.0f))
						{
							// Pass the current payload registers through to the shadow ray.
							unsigned int p0 = optixGetPayload_0();
							unsigned int p1 = optixGetPayload_1();

							// Note that the sysData.sceneEpsilon is applied on both sides of the shadow ray [t_min, t_max] interval 
							// to prevent self-intersections with the actual light geometry in the scene.
							prd->traceOperation = TR_SHADOW;
							optixTrace(optixLaunchParams.topObject,
									   prd->position,
									   lightSample.direction, // origin, direction
									   optixLaunchParams.settings->minClamp,
									   lightSample.distance - optixLaunchParams.settings->minClamp,
									   0.0f, // tmin, tmax, time
									   OptixVisibilityMask(0xFF),
									   OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, // The shadow ray type only uses anyhit programs.
									   optixLaunchParams.programs->hit,
									   0,
									   optixLaunchParams.programs->miss,
									   p0, p1); // Pass through thePrd to the shadow ray.

							if (prd->traceResult != TR_SHADOW)
							{
								float weightMis = 1.0f;
								if (typeLight == L_MESH)
								{
									weightMis = utl::balanceHeuristic(lightSample.pdf, evalData.data.pdf);
								}

								// The sampled emission needs to be scaled by the inverse probability to have selected this light,
								// Selecting one of many lights means the inverse of 1.0f / numLights.
								// This is using the path throughput before the sampling modulated it above.

								prd->radiance += throughput * bxdf * lightSample.radianceOverPdf * (float(numLights) * weightMis);
								
							}
						}
					}
				}
			}
		}
		prd->traceResult = TR_HIT;
	}


	extern "C" __global__ void __anyhit__radiance()
	{
		PerRayData* prd = mergePointer(optixGetPayload_0(), optixGetPayload_1());
		const unsigned int instanceId = optixGetInstanceId();
		const unsigned int triangleIdx = optixGetPrimitiveIndex();

		HitProperties hitP;

		utl::getInstanceAndGeometry(&hitP, instanceId);

		if(hitP.instance->hasOpacity)
		{
			utl::determineMaterialHitProperties(&hitP, triangleIdx);
			hitP.position = prd->wo;
			hitP.baricenter = math::vec3f(0.0f, optixGetTriangleBarycentrics().x, optixGetTriangleBarycentrics().y);
			hitP.baricenter.x = 1.0f - hitP.baricenter.x - hitP.baricenter.y;
			utl::getVertices(&hitP, triangleIdx);
			utl::fetchTransformsFromHandle(&hitP);
			utl::computeGeometricHitProperties(&hitP);
			mdl::MdlData    mdlData;
			mdl::InitConfig mdlConfig;
			mdlConfig.evaluateOpacity = true;
			mdl::initMdl(hitP, &mdlData, mdlConfig);
			// Stochastic alpha test to get an alpha blend effect.
			// No need to calculate an expensive random number if the test is going to fail anyway.
			if (mdlData.opacity < 1.0f && mdlData.opacity <= rng(prd->seed))
			{
				optixIgnoreIntersection();
				return;
			}
		}
		prd->traceResult = TR_SHADOW;
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
		prd->traceResult         = TR_MISS;
		
		if(prd->traceOperation == TR_HIT)
		{
			prd->colors.debugColor1 = math::vec3f(1.0f, 1.0f, 0.0f);
			if (prd->depth == 0)
			{
				prd->colors.diffuse = prd->colors.debugColor1;
				prd->colors.orientation = prd->colors.debugColor1;
				prd->colors.shadingNormal = prd->colors.debugColor1;
				prd->colors.trueNormal = prd->colors.debugColor1;
				prd->colors.debugColor2 = prd->colors.debugColor1;
				prd->colors.debugColor3 = prd->colors.debugColor1;
			}
			prd->radiance += prd->throughput * math::vec3f(0.2f, 0.2f, 0.2f);
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

		int maxDepth = (optixLaunchParams.settings->samplingTechnique== RendererDeviceSettings::SamplingTechnique::S_DIRECT_LIGHT) ? 1 :optixLaunchParams.settings->maxBounces;

		math::vec2ui payload = splitPointer(&prd);

		while (prd.depth < maxDepth)
		{
			prd.wo         = -prd.wi;
			prd.distance   = optixLaunchParams.settings->maxClamp;
			prd.traceResult = TR_UNKNOWN;
			prd.traceOperation = TR_HIT;

			optixTrace(optixLaunchParams.topObject,
					   prd.position,
					   prd.wi, // origin, direction
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


			// Path termination by miss shader or sample() routines.
			if (prd.eventType == mi::neuraylib::BSDF_EVENT_ABSORB || prd.throughput == math::vec3f(0.0f) || prd.traceResult == TR_MISS)
			{
				break;
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

		PerRayData prd;


		const math::vec2f pixel = math::vec2f(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
		//const math::vec2f sample = rng2(prd.seed);
		const math::vec2f sample = math::vec2f{0.5f, 0.5f};
		const math::vec2f screen{static_cast<float>(frameSize.x), static_cast<float>(frameSize.y)};

		const LensRay cameraRay = optixDirectCall<LensRay, const math::vec2f, const math::vec2f, const math::vec2f>(
			optixLaunchParams.programs->pinhole, screen, pixel, sample);

		prd.position = cameraRay.org;
		prd.wi       = cameraRay.dir;
		#define RANDOM_SAMPLING
		#ifdef RANDOM_SAMPLING
		prd.seed = tea<4>(fbIndex, *optixLaunchParams.frameID);
		// PERF This template really generates a lot of instructions.
		#else
        prd.seed = 0.5f; // Initialize the random number generator.
		#endif

		math::vec3f radiance = integrator(prd);

		math::vec4f* outputBuffer = reinterpret_cast<math::vec4f*>(frameBuffer->outputBuffer);
		// This is a per device launch sized buffer in this renderer strategy.
		if (!settings->accumulate || settings->iteration == 0)
		{
			frameBuffer->radianceBuffer[fbIndex] = radiance;
		}
		else
		{
			frameBuffer->radianceBuffer[fbIndex] += radiance;
		}
		switch (settings->displayBuffer)
		{
		case(RendererDeviceSettings::DisplayBuffer::FB_NOISY):
			{
				if (optixLaunchParams.settings->accumulate)
				{
					outputBuffer[fbIndex] = math::vec4f(frameBuffer->radianceBuffer[fbIndex] / static_cast<float>(optixLaunchParams.settings->iteration+ 1), 1.0f);
				}
				else
				{
					outputBuffer[fbIndex] = math::vec4f(frameBuffer->radianceBuffer[fbIndex], 1.0f);
				}
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
		case(RendererDeviceSettings::DisplayBuffer::FB_DEBUG_1):
			{
				outputBuffer[fbIndex] = math::vec4f(prd.colors.debugColor1, 1.0f);
			}
			break;
		case(RendererDeviceSettings::DisplayBuffer::FB_DEBUG_2):
			{
				outputBuffer[fbIndex] = math::vec4f(prd.colors.debugColor2, 1.0f);
			}
			break;
		case(RendererDeviceSettings::DisplayBuffer::FB_DEBUG_3):
			{
				outputBuffer[fbIndex] = math::vec4f(prd.colors.debugColor3, 1.0f);
			}
			break;
		}
	}
}

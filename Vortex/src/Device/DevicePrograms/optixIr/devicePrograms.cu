#ifndef OPTIXCODE
#define OPTIXCODE
#endif

#include <optix_device.h>
#include "../RayData.h"
#include "../DataFetcher.h"
#include "../Utils.h"
#include "../ToneMapper.h"
#include "Device/WorkQueues.h"
#include "Device/DevicePrograms/HitPropertiesComputation.h"
#include "Device/DevicePrograms/ToneMapper.h"
#include "Device/DevicePrograms/Mdl/directMdlWrapper.h"

namespace vtx
{

#define materialEvaluation(sbtIndex, request)  optixDirectCall<mdl::MaterialEvaluation>(sbtIndex, request)

	extern "C" __device__ LensRay __direct_callable__pinhole(const math::vec2f screen, const math::vec2f pixel, const math::vec2f sample) {
		const math::vec2f fragment = pixel + sample;                    // Jitter the sub-pixel location
		const math::vec2f ndc = (fragment / screen) * 2.0f - 1.0f;      // Normalized device coordinates in range [-1, 1].

		const CameraData camera = optixLaunchParams.cameraData;

		LensRay ray;

		ray.org = camera.position;
		ray.dir = math::normalize<float>(camera.horizontal * ndc.x + camera.vertical * ndc.y + camera.direction);
		return ray;
	}

	extern "C" __device__ LightSample __direct_callable__meshLightSample(const LightData & light, PerRayData * prd)
	{
		MeshLightAttributesData meshLightAttributes = *reinterpret_cast<MeshLightAttributesData*>(light.attributes);
		LightSample lightSample;

		lightSample.pdf = 0.0f;

		const float* cdfArea = meshLightAttributes.cdfArea;
		const float3 sample3D = rng3(prd->seed);

		// Uniformly sample the triangles over their surface area.
		// Note that zero-area triangles (e.g. at the poles of spheres) are automatically never sampled with this method!
		// The cdfU is one bigger than res.y.

		unsigned int idxTriangle = utl::binarySearchCdf(cdfArea, meshLightAttributes.size, sample3D.z);
		idxTriangle = meshLightAttributes.actualTriangleIndices[idxTriangle];

		// Unit square to triangle via barycentric coordinates.
		const float sqrtSampleX = sqrtf(sample3D.x);
		// Barycentric coordinates.
		const float alpha = 1.0f - sqrtSampleX;
		const float beta = sample3D.y * sqrtSampleX;
		const float gamma = 1.0f - alpha - beta;

		HitProperties hitP;
		hitP.baricenter = math::vec3f(alpha, beta, gamma);
		utl::getInstanceAndGeometry(&hitP, meshLightAttributes.instanceId, optixLaunchParams);
		utl::getVertices(&hitP, idxTriangle);
		utl::fetchTransformsFromInstance(&hitP);
		utl::computeHit(&hitP, prd->position);

		lightSample.position = hitP.position;
		lightSample.direction = -hitP.direction;
		lightSample.distance = hitP.distance;
		if (lightSample.distance < DENOMINATOR_EPSILON)
		{
			return lightSample;
		}

		utl::computeGeometricHitProperties(&hitP, true);
		utl::determineMaterialHitProperties(&hitP, idxTriangle);

		hitP.seed = prd->seed;
		mdl::MdlRequest request;
		request.edf = true;
		request.opacity = true;
		request.lastRayDirection = -lightSample.direction;
		request.hitP = &hitP;
		request.surroundingIor = 1.0f;

		mdl::MaterialEvaluation matEval;
		if (!hitP.materialConfiguration->directCallable)
		{
			const int sbtIndex = hitP.materialConfiguration->idxCallEvaluateMaterialStandard;
			optixDirectCall<void, mdl::MdlRequest*, mdl::MaterialEvaluation*>(sbtIndex, &request, &matEval);
		}
		else
		{
			matEval = mdl::evaluateMdlMaterial(&request);
		}

		if (matEval.opacity <= 0.0f)
		{
			return lightSample;
		}

		if (matEval.edf.isValid)
		{
			const float totArea = meshLightAttributes.totalArea;

			// Modulate the emission with the cutout opacity value to get the correct value.
			// The opacity value must not be greater than one here, which could happen for HDR textures.
			float opacity = math::min(matEval.opacity, 1.0f);

			// Power (flux) [W] divided by light area gives radiant exitance [W/m^2].
			const float factor = (matEval.edf.mode == 0) ? matEval.opacity : matEval.opacity / totArea;

			lightSample.pdf = lightSample.distance * lightSample.distance / (totArea * matEval.edf.cos); // Solid angle measure.
			lightSample.radianceOverPdf = matEval.edf.intensity * matEval.edf.edf * (factor / lightSample.pdf);
			lightSample.isValid = true;
		}

		return lightSample;
	}

	extern "C" __device__ LightSample __direct_callable__envLightSample(const LightData & light, PerRayData * prd)
	{
		EnvLightAttributesData attrib = *reinterpret_cast<EnvLightAttributesData*>(light.attributes);
		auto texture = attrib.texture;

		unsigned width = texture->dimension.x;
		unsigned height = texture->dimension.y;

		LightSample            lightSample;

		// importance sample an envmap pixel using an alias map
		const float3       sample = rng3(prd->seed);
		const unsigned int size = width * height;
		const auto         idx = math::min<unsigned>((unsigned int)(sample.x * (float)size), size - 1);
		unsigned int       envIdx;
		float              sampleY = sample.y;
		if (sampleY < attrib.aliasMap[idx].q) {
			envIdx = idx;
			sampleY /= attrib.aliasMap[idx].q;
		}
		else {
			envIdx = attrib.aliasMap[idx].alias;
			sampleY = (sampleY - attrib.aliasMap[idx].q) / (1.0f - attrib.aliasMap[idx].q);
		}

		const unsigned int py = envIdx / width;
		const unsigned int px = envIdx % width;
		lightSample.pdf = attrib.aliasMap[envIdx].pdf;

		const float u = (float)(px + sampleY) / (float)width;
		//const float phi = (M_PI_2)*(1.0f-u);

		//const float phi = (float)M_PI  -u * (float)(2.0 * M_PI);
		const float phi = u * (float)(2.0 * M_PI) - (float)M_PI;
		float sinPhi, cosPhi;
		sincosf(phi > float(-M_PI) ? phi : (phi + (float)(2.0 * M_PI)), &sinPhi, &cosPhi);
		const float stepTheta = (float)M_PI / (float)height;
		const float theta0 = (float)(py)*stepTheta;
		const float cosTheta = cosf(theta0) * (1.0f - sample.z) + cosf(theta0 + stepTheta) * sample.z;
		const float theta = acosf(cosTheta);
		const float sinTheta = sinf(theta);
		const float v = theta * (float)(1.0 / M_PI);

		float x = cosPhi * sinTheta;
		float y = sinPhi * sinTheta;
		float z = -cosTheta;

		math::vec3f dir{ x, y, z };
		// Now rotate that normalized object space direction into world space. 
		lightSample.direction = math::transformNormal3F(attrib.transformation, dir);

		lightSample.distance = optixLaunchParams.settings->maxClamp; // Environment light.

		// Get the emission from the spherical environment texture.
		math::vec3f emission = tex2D<float4>(texture->texObj, u, v);
		// For simplicity we pretend that we perfectly importance-sampled the actual texture-filtered environment map
		// and not the Gaussian-smoothed one used to actually generate the CDFs and uniform sampling in the texel.
		// (Note that this does not contain the light.emission which just modulates the texture.)
		if (DENOMINATOR_EPSILON < lightSample.pdf)
		{
			//lightSample.radianceOverPdf = light.emission * emission / lightSample.pdf;
			lightSample.radianceOverPdf = emission / lightSample.pdf;
		}

		lightSample.isValid = true;
		return lightSample;

	}

	//------------------------------------------------------------------------------
	// closest hit and anyhit programs for radiance-type rays.
	//------------------------------------------------------------------------------

	extern "C" __global__ void __exception__all()
	{
		//const uint3 theLaunchDim     = optixGetLaunchDimensions(); 
		const uint3 theLaunchIndex = optixGetLaunchIndex();
		const int   theExceptionCode = optixGetExceptionCode();
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

	__forceinline__ __device__ LightSample sampleLight(PerRayData* prd, const HitProperties* hitP)
	{
		LightSample lightSample;
		if (const int numLights = optixLaunchParams.numberOfLights; numLights > 0)
		{
			//Randomly Selecting a Light
			//TODO, I think here we can do some better selection by giving more importance to lights with greater power
			const int indexLight = (1 < numLights) ? gdt::clamp(static_cast<int>(floorf(rng(prd->seed) * numLights)), 0, numLights - 1) : 0;

			const LightData light = *(optixLaunchParams.lights[indexLight]);

			const LightType typeLight = light.type;

			int lightSampleProgramIdx = -1;
			switch (typeLight)
			{
			case L_MESH:
			{
				lightSampleProgramIdx = optixLaunchParams.programs->meshLightSample;
			}
			break;
			case L_ENV:
			{
				lightSampleProgramIdx = optixLaunchParams.programs->envLightSample;
			}
			break;
			default: {
				lightSample.isValid = false;
				return lightSample;
			}break;
			}

			if (lightSampleProgramIdx != -1)
			{
				lightSample = optixDirectCall<LightSample, const LightData&, PerRayData*>(lightSampleProgramIdx, light, prd);
				lightSample.typeLight = typeLight;

				if (lightSample.isValid) // && dot(lightSample.direction, hitP->ngW) >= -0.05f)
				{
					lightSample.isValid = true;
					return lightSample;
				}
			}
		}
		lightSample.isValid = false;
		lightSample.pdf = 0.0f;
		return lightSample;
	}

	extern "C" __global__ void __closesthit__radiance()
	{

		PerRayData* prd = reinterpret_cast<PerRayData*>(mergePointer(optixGetPayload_0(), optixGetPayload_1()));
		const unsigned int instanceId = optixGetInstanceId();
		const unsigned int triangleIdx = optixGetPrimitiveIndex();
		prd->distance = optixGetRayTmax();
		prd->position = prd->position + prd->wi * prd->distance;

		HitProperties hitP;
		hitP.position = prd->position;
		hitP.direction = prd->wo;
		hitP.baricenter = math::vec3f(0.0f, optixGetTriangleBarycentrics().x, optixGetTriangleBarycentrics().y);
		hitP.baricenter.x = 1.0f - hitP.baricenter.y - hitP.baricenter.z;
		hitP.seed = prd->seed;

		utl::getInstanceAndGeometry(&hitP, instanceId, optixLaunchParams);
		utl::getVertices(&hitP, triangleIdx);
		utl::fetchTransformsFromHandle(&hitP);
		utl::computeGeometricHitProperties(&hitP, triangleIdx);
		utl::determineMaterialHitProperties(&hitP, triangleIdx);

		//Auxiliary Data
		if (prd->depth == 0)
		{
			prd->colors.trueNormal = 0.5f * (hitP.ngW + 1.0f);
			prd->colors.uv = hitP.textureCoordinates[0];
			//prd->colors.trueNormal  = hitP.ngW;
			prd->colors.orientation = hitP.isFrontFace ? math::vec3f(0.0f, 0.0f, 1.0f) : math::vec3f(1.0f, 0.0f, 0.0f);
			prd->colors.tangent = 0.5f * (hitP.tgW + 1.0f);
			//prd->colors.tangent     = hitP.tgW;
		}

		if (hitP.material != nullptr)
		{
			RendererDeviceSettings::SamplingTechnique& samplingTechnique = optixLaunchParams.settings->samplingTechnique;

			mdl::MdlRequest request;
			request.auxiliary = true;
			request.ior = true;
			request.edf = true;
			request.lastRayDirection = prd->wo;
			request.hitP = &hitP;

			request.surroundingIor = prd->mediumIor;

			if (hitP.meshLightAttributes != nullptr)
			{
				request.edf = true;
			}
			LightSample lightSample;
			if (samplingTechnique == RendererDeviceSettings::S_MIS) {
				request.bsdfSample = true;
				lightSample = sampleLight(prd, &hitP);
				if (lightSample.isValid)
				{
					request.bsdfEvaluation = true;
					request.toSampledLight = lightSample.direction;
				}
			}
			else if (samplingTechnique == RendererDeviceSettings::S_BSDF)
			{
				request.bsdfSample = true;
			}
			else if (samplingTechnique == RendererDeviceSettings::S_DIRECT_LIGHT)
			{
				lightSample = sampleLight(prd, &hitP);
				if (lightSample.isValid)
				{
					request.bsdfEvaluation = true;
					request.toSampledLight = lightSample.direction;
				}
			}


			mdl::MaterialEvaluation matEval;
			if (!hitP.materialConfiguration->directCallable)
			{
				const int sbtIndex = hitP.materialConfiguration->idxCallEvaluateMaterialStandard;
				optixDirectCall<void, mdl::MdlRequest*, mdl::MaterialEvaluation*>(sbtIndex, &request, &matEval);
			}
			else
			{
				matEval = mdl::evaluateMdlMaterial(&request);
			}

			// Auxiliary Data

			if (matEval.aux.isValid)
			{
				prd->colors.bounceDiffuse = matEval.aux.albedo;
				if (prd->depth == 0)
				{
					prd->colors.shadingNormal = 0.5f * (matEval.aux.normal + 1.0f);
				}
			}
			else
			{
				prd->colors.bounceDiffuse = math::vec3f(1.0f, 0.0f, 1.0f);
				if (prd->depth == 0)
				{
					prd->colors.shadingNormal = 0.5f * (hitP.nsW + 1.0f);
				}
			}


			//Evauluate Hit Point Emission
			if (matEval.edf.isValid)
			{
				const float area = hitP.meshLightAttributes->totalArea;
				matEval.edf.pdf = prd->distance * prd->distance / (area * matEval.edf.cos);
				// Solid angle measure.

				float misWeight = 1.0f;

				// If the last event was diffuse or glossy, calculate the opposite MIS weight for this implicit light hit.
				if (samplingTechnique == RendererDeviceSettings::S_MIS && prd->eventType & (mi::neuraylib::BSDF_EVENT_DIFFUSE | mi::neuraylib::BSDF_EVENT_GLOSSY))
				{
					misWeight = utl::heuristic(prd->pdf, matEval.edf.pdf);
				}
				// Power (flux) [W] divided by light area gives radiant exitance [W/m^2].
				const float factor = (matEval.edf.mode == 0) ? 1.0f : 1.0f / area;
				prd->radiance += prd->throughput * matEval.edf.intensity * matEval.edf.edf * (factor * misWeight);
			}

			math::vec3f currentThroughput = prd->throughput;
			prd->pdf = 0.0f;

			// BSDF Sampling
			if (samplingTechnique == RendererDeviceSettings::S_MIS || samplingTechnique == RendererDeviceSettings::S_BSDF)
			{
				//matEval.bsdfSample.print("Outside:\n");

				//Importance Sampling the Bsdf
				if (matEval.bsdfSample.eventType != mi::neuraylib::BSDF_EVENT_ABSORB)
				{
					prd->wi = matEval.bsdfSample.nextDirection; // Continuation direction.
					prd->throughput *= matEval.bsdfSample.bsdfOverPdf;
					// Adjust the path throughput for all following incident lighting.
					prd->pdf = matEval.bsdfSample.pdf;
					// Note that specular events return pdf == 0.0f! (=> Not a path termination condition.)
					prd->eventType = matEval.bsdfSample.eventType;
				}
				else
				{
					prd->traceResult = TR_HIT;
					return;
				}
			}

			const bool isDiffuse = (prd->eventType & mi::neuraylib::BSDF_EVENT_DIFFUSE) != 0;
			const bool isGlossy = (prd->eventType & mi::neuraylib::BSDF_EVENT_GLOSSY) != 0;
			const bool isTransmission = (prd->eventType & mi::neuraylib::BSDF_EVENT_TRANSMISSION) != 0;
			const bool isSpecular = (prd->eventType & mi::neuraylib::BSDF_EVENT_SPECULAR) != 0;

			if (prd->depth == 0)
			{
				if (isDiffuse)
				{
					prd->colors.debugColor1 = REDCOLOR;
				}
				if (isSpecular)
				{
					prd->colors.debugColor1 = GREENCOLOR;
				}
				if (isGlossy)
				{
					prd->colors.debugColor1 = BLUECOLOR;
				}
			}



			//Direct Light Sampling
			if ((samplingTechnique == RendererDeviceSettings::S_MIS || samplingTechnique == RendererDeviceSettings::S_DIRECT_LIGHT) && lightSample.isValid)
			{
				//printf("number of tries: %d\n", numberOfTries);
				auto bxdf = math::vec3f(0.0f, 0.0f, 0.0f);
				bxdf += matEval.bsdfEvaluation.diffuse;
				bxdf += matEval.bsdfEvaluation.glossy;

				if (0.0f < matEval.bsdfEvaluation.pdf && bxdf != math::vec3f(0.0f, 0.0f, 0.0f))
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
						if ((lightSample.typeLight == L_MESH || lightSample.typeLight == L_ENV) && samplingTechnique == RendererDeviceSettings::S_MIS)
						{
							weightMis = utl::heuristic(lightSample.pdf, matEval.bsdfEvaluation.pdf);
							//weightMis = utl::heuristic(matEval.bsdfEvaluation.pdf, lightSample.pdf);
							//weightMis = matEval.bsdfEvaluation.pdf*lightSample.pdf;
						}

						// The sampled emission needs to be scaled by the inverse probability to have selected this light,
						// Selecting one of many lights means the inverse of 1.0f / numLights.
						// This is using the path throughput before the sampling modulated it above.

						prd->radiance += currentThroughput * bxdf * lightSample.radianceOverPdf * weightMis * (float)optixLaunchParams.numberOfLights; // *float(numLights);

					}
				}
			}

			if (!matEval.isThinWalled && (prd->eventType & mi::neuraylib::BSDF_EVENT_TRANSMISSION) != 0)
			{
				if (hitP.isFrontFace) // Entered a volume. 
				{
					prd->mediumIor = matEval.ior;
					//printf("Entering Volume Ior = %f %f %f\n", matEval.ior.x, matEval.ior.y, matEval.ior.z);
				}
				else // if !isFrontFace. Left a volume.
				{
					prd->mediumIor = 1.0f;
				}
			}

		}
		prd->traceResult = TR_HIT;
	}


	extern "C" __global__ void __anyhit__radiance()
	{
		PerRayData* prd = reinterpret_cast<PerRayData*>(mergePointer(optixGetPayload_0(), optixGetPayload_1()));
		const unsigned int instanceId = optixGetInstanceId();
		const unsigned int triangleIdx = optixGetPrimitiveIndex();

		HitProperties hitP;

		utl::getInstanceAndGeometry(&hitP, instanceId, optixLaunchParams);

		if (hitP.instance->hasOpacity)
		{
			printf("Evaluating Opacity");
			float opacity = 1.0f;
			utl::determineMaterialHitProperties(&hitP, triangleIdx);
			hitP.position = prd->wo;
			hitP.baricenter = math::vec3f(0.0f, optixGetTriangleBarycentrics().x, optixGetTriangleBarycentrics().y);
			hitP.baricenter.x = 1.0f - hitP.baricenter.x - hitP.baricenter.y;

			hitP.seed = prd->seed;
			utl::getVertices(&hitP, triangleIdx);
			utl::fetchTransformsFromHandle(&hitP);
			utl::computeGeometricHitProperties(&hitP, triangleIdx);

			mdl::MdlRequest request;
			request.hitP = &hitP;
			request.opacity = true;
			mdl::MaterialEvaluation eval;
			if (!hitP.materialConfiguration->directCallable)
			{
				const int sbtIndex = hitP.materialConfiguration->idxCallEvaluateMaterialStandard;
				optixDirectCall<void, mdl::MdlRequest*, mdl::MaterialEvaluation*>(sbtIndex, &request, &eval);
				opacity = eval.opacity;
			}
			else
			{
				eval = mdl::evaluateMdlMaterial(&request);
			}

			// Stochastic alpha test to get an alpha blend effect.
			// No need to calculate an expensive random number if the test is going to fail anyway.
			if (eval.opacity < 1.0f && eval.opacity <= rng(prd->seed))
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
		PerRayData* prd = reinterpret_cast<PerRayData*>(mergePointer(optixGetPayload_0(), optixGetPayload_1()));
		prd->traceResult = TR_MISS;

		if (prd->traceOperation == TR_HIT)
		{
			if (optixLaunchParams.envLight != nullptr)
			{
				const LightData* envLight = optixLaunchParams.envLight;
				EnvLightAttributesData attrib = *reinterpret_cast<EnvLightAttributesData*>(envLight->attributes);
				auto texture = attrib.texture;

				math::vec3f dir = math::transformNormal3F(attrib.invTransformation, prd->wi);


				{
					bool computeOriginalUV = false;
					float u, v;
					if (computeOriginalUV)
					{
						u = fmodf(atan2f(dir.y, dir.x) * (float)(0.5 / M_PI) + 0.5f, 1.0f);
						v = acosf(fmax(fminf(-dir.z, 1.0f), -1.0f)) * (float)(1.0 / M_PI);
					}
					else {
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
					bool MiSCondition = (optixLaunchParams.settings->samplingTechnique == RendererDeviceSettings::S_MIS) && (prd->eventType & (mi::neuraylib::BSDF_EVENT_DIFFUSE | mi::neuraylib::BSDF_EVENT_GLOSSY));
					if (prd->pdf > 0.0f && MiSCondition)
					{
						float envSamplePdf = attrib.aliasMap[y * texture->dimension.x + x].pdf;
						misWeight = utl::heuristic(prd->pdf, envSamplePdf);
						//misWeight = prd->pdf / (prd->pdf + envSamplePdf);
					}
					prd->radiance += prd->throughput * emission * misWeight;
					prd->eventType = mi::neuraylib::BSDF_EVENT_ABSORB;

					prd->colors.bounceDiffuse = math::normalize(emission);
					if (prd->depth == 0)
					{
						prd->colors.trueNormal = prd->wi;
						prd->colors.shadingNormal = 0.0f;
					}
				}
			}
		}
	}

	struct Radiances
	{
		math::vec3f radiance;
		math::vec3f directLight;
		math::vec3f diffuseIndirect;
		math::vec3f transmissionIndirect;
		math::vec3f glossyIndirect;
	};

	__forceinline__ __device__ Radiances integrator(PerRayData& prd)
	{
		Radiances radiances;

		radiances.radiance = math::vec3f(0.0f);
		radiances.directLight = math::vec3f(0.0f);
		radiances.diffuseIndirect = math::vec3f(0.0f);
		radiances.transmissionIndirect = math::vec3f(0.0f);
		radiances.glossyIndirect = math::vec3f(0.0f);
		// The integrator starts with black radiance and full path throughput.
		prd.radiance = math::vec3f(0.0f);
		prd.pdf = 0.0f;
		prd.throughput = math::vec3f(1.0f);
		prd.sigmaT = math::vec3f(0.0f); // Extinction coefficient: sigma_a + sigma_s.
		prd.walk = 0; // Number of random walk steps taken through volume scattering. 
		prd.eventType = mi::neuraylib::BSDF_EVENT_ABSORB; // Initialize for exit. (Otherwise miss programs do not work.)

		prd.idxStack = 0; // Nested material handling. 
		// Small stack of four entries of which the first is vacuum.
		prd.stack[0].ior = math::vec3f(1.0f); // No effective IOR.
		prd.stack[0].absorption = math::vec3f(0.0f); // No volume absorption.
		prd.stack[0].scattering = math::vec3f(0.0f); // No volume scattering.
		prd.stack[0].bias = 0.0f;              // Isotropic volume scattering.
		prd.depth = 0;

		prd.mediumIor = math::vec3f(1.0f);

		prd.colors.finalDiffuse = 0.0f;
		prd.colors.bounceDiffuse = 0.0f;
		prd.colors.trueNormal = 0.0f;
		prd.colors.shadingNormal = 0.0f;
		prd.colors.tangent = 0.0f;
		prd.colors.orientation = 0.0f;
		prd.colors.debugColor1 = 0.0f;
		prd.colors.uv = 0.0f;


		int maxDepth = (optixLaunchParams.settings->samplingTechnique == RendererDeviceSettings::SamplingTechnique::S_DIRECT_LIGHT) ? 0 : optixLaunchParams.settings->maxBounces;

		math::vec2ui payload = splitPointer(&prd);

		bool isDiffuse = false;
		bool isGlossy = false;
		bool isTransmission = false;
		bool isSpecular = false;
		bool isFirstSpecular = false;

		while (prd.depth <= maxDepth)
		{
			prd.wo = -prd.wi;
			prd.distance = optixLaunchParams.settings->maxClamp;
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



			if (prd.depth == 0) {
				isDiffuse = (prd.eventType & mi::neuraylib::BSDF_EVENT_DIFFUSE) != 0;
				isGlossy = (prd.eventType & mi::neuraylib::BSDF_EVENT_GLOSSY) != 0;
				isTransmission = (prd.eventType & mi::neuraylib::BSDF_EVENT_TRANSMISSION) != 0;
				isSpecular = (prd.eventType & mi::neuraylib::BSDF_EVENT_SPECULAR) != 0;
				if (isSpecular == true)
					isFirstSpecular = true;
				prd.colors.finalDiffuse = prd.colors.bounceDiffuse;
			}

			//if (isFirstSpecular && prd.depth > 0)
			//{
			//	if(!((prd.eventType & mi::neuraylib::BSDF_EVENT_SPECULAR) != 0))
			//	{
			//		prd.colors.finalDiffuse = prd.colors.bounceDiffuse;
			//		isFirstSpecular = false;
			//		// update prd. colors albedo
			//	}
			//}

			if (prd.depth == 0 && prd.traceResult == TR_MISS)
			{
				radiances.directLight = prd.radiance;
				break;
			}
			if (prd.depth == 1 && prd.traceResult == TR_MISS)
			{
				radiances.directLight = prd.radiance;
			}

			// Path termination by miss shader or sample() routines.
			if (prd.eventType == mi::neuraylib::BSDF_EVENT_ABSORB || prd.throughput == math::vec3f(0.0f) || prd.traceResult == TR_MISS)
			{
				break;
			}

			// Unbiased Russian Roulette path termination.
			if (optixLaunchParams.settings->useRussianRoulette && 2 <= prd.depth) // Start termination after a minimum number of bounces.
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

		math::vec3f deltaRadiance = prd.radiance - radiances.directLight;

		if (isTransmission)
		{
			radiances.transmissionIndirect = deltaRadiance;
		}
		else if (isDiffuse) {
			radiances.diffuseIndirect = deltaRadiance;
		}
		else if (isGlossy || isSpecular) {
			radiances.glossyIndirect = deltaRadiance;
		}

		radiances.radiance = prd.radiance;
		return radiances;
	}

	extern "C" __global__ void __raygen__renderFrame()
	{
		const RendererDeviceSettings* settings = getData<RendererDeviceSettings>();
		const FrameBufferData* frameBuffer = getData<FrameBufferData>();
		const math::vec2ui& frameSize = frameBuffer->frameSize;
		const int      ix = optixGetLaunchIndex().x;
		const int      iy = optixGetLaunchIndex().y;
		const uint32_t fbIndex = ix + iy * frameSize.x;
		const math::vec2f pixel = math::vec2f(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
		float& noise = frameBuffer->noiseBuffer[fbIndex].noise;
		bool runningAverage = false;

		int& iteration = optixLaunchParams.settings->iteration;
		int samples = iteration >= 0 ? iteration : 0;
		int samplesPerLaunch = 1;

		math::vec3f& directLightBuffer = frameBuffer->directLight[fbIndex];
		math::vec3f& diffuseIndirectBuffer = frameBuffer->diffuseIndirect[fbIndex];
		math::vec3f& glossyIndirectBuffer = frameBuffer->glossyIndirect[fbIndex];
		math::vec3f& transmissionIndirect = frameBuffer->transmissionIndirect[fbIndex];

		if (samples == 0)
		{
			directLightBuffer = math::vec3f(0.0f);
			diffuseIndirectBuffer = math::vec3f(0.0f);
			glossyIndirectBuffer = math::vec3f(0.0f);
			transmissionIndirect = math::vec3f(0.0f);
		}

		if (settings->adaptiveSampling)
		{
			if (samples == 0)
			{
				frameBuffer->noiseBuffer[fbIndex].samples = 0;
			}
			else
			{
				samples = frameBuffer->noiseBuffer[fbIndex].samples;
			}
			if (settings->minAdaptiveSamples <= iteration)
			{
				if (noise < settings->noiseCutOff)
				{
					samplesPerLaunch = 0;
				}
				else
				{
					samplesPerLaunch = utl::lerp((float)settings->minPixelSamples, (float)settings->maxPixelSamples, noise * noise);
				}
			}
		}

		PerRayData prd;
		prd.seed = tea<4>(fbIndex, *optixLaunchParams.frameID);

		for (int i = 0; i < samplesPerLaunch; i++)
		{
			const math::vec2f sample = rng2(prd.seed);
			const math::vec2f screen{ static_cast<float>(frameSize.x), static_cast<float>(frameSize.y) };

			const LensRay cameraRay = optixDirectCall<LensRay, const math::vec2f, const math::vec2f, const math::vec2f>(
				optixLaunchParams.programs->pinhole, screen, pixel, sample);

			prd.position = cameraRay.org;
			prd.wi = cameraRay.dir;

			Radiances radiance = integrator(prd);

			++samples;

			if (!settings->accumulate || samples <= 1)
			{
				utl::replaceNanCheck(radiance.radiance, frameBuffer->rawRadiance[fbIndex]);
				utl::replaceNanCheck(radiance.directLight, directLightBuffer);
				utl::replaceNanCheck(radiance.diffuseIndirect, diffuseIndirectBuffer);
				utl::replaceNanCheck(radiance.glossyIndirect, glossyIndirectBuffer);
				utl::replaceNanCheck(radiance.transmissionIndirect, transmissionIndirect);
				utl::replaceNanCheck(prd.colors.finalDiffuse, frameBuffer->albedo[fbIndex]);
				utl::replaceNanCheck(prd.colors.shadingNormal, frameBuffer->normal[fbIndex]);
				utl::replaceNanCheck(prd.colors.trueNormal, frameBuffer->trueNormal[fbIndex]);
				utl::replaceNanCheck(prd.colors.tangent, frameBuffer->tangent[fbIndex]);
				utl::replaceNanCheck(prd.colors.orientation, frameBuffer->orientation[fbIndex]);
				utl::replaceNanCheck(prd.colors.uv, frameBuffer->uv[fbIndex]);
				utl::replaceNanCheck(prd.colors.debugColor1, frameBuffer->debugColor1[fbIndex]);
			}
			else
			{
				if (runningAverage)
				{
					const float fSamples = (float)samples;
					utl::lerpAccumulate(radiance.radiance, frameBuffer->rawRadiance[fbIndex], fSamples);
					utl::lerpAccumulate(radiance.directLight, directLightBuffer, fSamples);
					utl::lerpAccumulate(radiance.diffuseIndirect, diffuseIndirectBuffer, fSamples);
					utl::lerpAccumulate(radiance.glossyIndirect, glossyIndirectBuffer, fSamples);
					utl::lerpAccumulate(radiance.transmissionIndirect, transmissionIndirect, fSamples);
					utl::lerpAccumulate(prd.colors.finalDiffuse, frameBuffer->albedo[fbIndex], fSamples);
					utl::lerpAccumulate(prd.colors.shadingNormal, frameBuffer->normal[fbIndex], fSamples);
					utl::lerpAccumulate(prd.colors.trueNormal, frameBuffer->trueNormal[fbIndex], fSamples);
					utl::lerpAccumulate(prd.colors.tangent, frameBuffer->tangent[fbIndex], fSamples);
					utl::lerpAccumulate(prd.colors.orientation, frameBuffer->orientation[fbIndex], fSamples);
					utl::lerpAccumulate(prd.colors.uv, frameBuffer->uv[fbIndex], fSamples);
					utl::lerpAccumulate(prd.colors.debugColor1, frameBuffer->debugColor1[fbIndex], fSamples);
				}
				else
				{
					float kFactor = 1.0f / (float)samples;
					float jFactor = ((float)samples - 1.0f) * kFactor;
					utl::accumulate(radiance.radiance, frameBuffer->rawRadiance[fbIndex], kFactor, jFactor);
					utl::accumulate(radiance.directLight, directLightBuffer, kFactor, jFactor);
					utl::accumulate(radiance.diffuseIndirect, diffuseIndirectBuffer, kFactor, jFactor);
					utl::accumulate(radiance.glossyIndirect, glossyIndirectBuffer, kFactor, jFactor);
					utl::accumulate(radiance.transmissionIndirect, transmissionIndirect, kFactor, jFactor);
					utl::accumulate(prd.colors.finalDiffuse, frameBuffer->albedo[fbIndex], kFactor, jFactor);
					utl::accumulate(prd.colors.shadingNormal, frameBuffer->normal[fbIndex], kFactor, jFactor);
					utl::accumulate(prd.colors.trueNormal, frameBuffer->trueNormal[fbIndex], kFactor, jFactor);
					utl::accumulate(prd.colors.tangent, frameBuffer->tangent[fbIndex], kFactor, jFactor);
					utl::accumulate(prd.colors.orientation, frameBuffer->orientation[fbIndex], kFactor, jFactor);
					utl::accumulate(prd.colors.uv, frameBuffer->uv[fbIndex], kFactor, jFactor);
					utl::accumulate(prd.colors.debugColor1, frameBuffer->debugColor1[fbIndex], kFactor, jFactor);
				}
			}

		}

		if (settings->adaptiveSampling)
		{
			frameBuffer->noiseBuffer[fbIndex].samples = samples;
		}

		frameBuffer->tmRadiance[fbIndex] = toneMap(optixLaunchParams.toneMapperSettings, frameBuffer->rawRadiance[fbIndex]);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////// WAVEFRONT RADIANCE TRACE //////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//Dummy AnyHit
	extern "C" __global__ void __anyhit__dummyAnyHit()
	{

	}


	extern "C" __global__ void __closesthit__rtClosest()
	{
		RayWorkItem* prd = reinterpret_cast<RayWorkItem*>(mergePointer(optixGetPayload_0(), optixGetPayload_1()));


		prd->hitInstanceId = optixGetInstanceId();
		prd->hitTriangleId = optixGetPrimitiveIndex();
		prd->hitDistance = optixGetRayTmax();
		prd->hitPosition = prd->origin + prd->direction * prd->hitDistance;

		float2 bari = optixGetTriangleBarycentrics();
		prd->hitBaricenter = math::vec3f(1.0f - bari.x - bari.y, bari.x, bari.y);

		const OptixTraversableHandle handle = optixGetTransformListHandle(0);
		// UNSURE IF THIS IS CORRECT! WE ALWAYS HAVE THE TRANSFORM FROM THE INSTANCE DATA IN CASE
		const float4* wTo = optixGetInstanceInverseTransformFromHandle(handle);
		const float4* oTw = optixGetInstanceTransformFromHandle(handle);

		prd->hitOTW = math::affine3f(oTw);
		prd->hitWTO = math::affine3f(wTo);

		optixLaunchParams.shadeQueue->Push(*prd);
		//enqueueRayData(optixLaunchParams.shadeQueue, prd, optixLaunchParams.shadeQueueSize, optixLaunchParams.maxQueueSize);

		//optixLaunchParams.shadeQueue->addJob(prd, "Shade From Trace Closest");
	}

	extern "C" __global__ void __miss__rtMiss()
	{
		RayWorkItem* prd = reinterpret_cast<RayWorkItem*>(mergePointer(optixGetPayload_0(), optixGetPayload_1()));

		optixLaunchParams.escapedQueue->Push(*prd);
		//enqueueRayData(optixLaunchParams.escapedQueue, prd, optixLaunchParams.escapedQueueSize, optixLaunchParams.maxQueueSize);
		//optixLaunchParams.escapedQueue->addJob(prd, "Escaped from Trace Miss");
	}

	__forceinline__ __device__ void trace(RayWorkItem* rd, float distance)
	{
		math::vec2ui payload = splitPointer(rd);

		optixTrace(optixLaunchParams.topObject,
			rd->origin,
			rd->direction, // origin, direction
			optixLaunchParams.settings->minClamp,
			distance,
			0.0f, // tmin, tmax, time
			static_cast<OptixVisibilityMask>(0xFF),
			OPTIX_RAY_FLAG_DISABLE_ANYHIT,    //OPTIX_RAY_FLAG_NONE,
			0,  //SBT Offset
			0,                                // SBT stride
			0, // missSBTIndex
			payload.x,
			payload.y);
	}

	extern "C" __global__ void __raygen__rtRaygen()
	{
		const int queueWorkId = optixGetLaunchIndex().x;
		/*if (queueWorkId == 0)
		{
			if (optixLaunchParams.radianceTraceQueue->Size() == 0)
			{
				printf("Trace Queue Size 0!!\n");
			}
			else
			{
				printf("Trace Queue Size: %d\n", optixLaunchParams.radianceTraceQueue->Size());
			}
		}*/
		
		if (optixLaunchParams.radianceTraceQueue->Size() <= queueWorkId)
			return;

		

		float distance = optixLaunchParams.settings->maxClamp;

		RayWorkItem rayData = (*optixLaunchParams.radianceTraceQueue)[queueWorkId];

		trace(&rayData, distance);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////// WAVEFRONT SHADE  //////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	extern "C" __global__ void __closesthit__shadeDummy()
	{
	}

	extern "C" __global__ void __anyhit__shadeShadowHit()
	{
		optixSetPayload_0(0);
	}

	extern "C" __global__ void __miss__shadeShadowMiss()
	{
		optixSetPayload_0(1);
	}

	__forceinline__ __device__ LightSample sampleMeshLight(const LightData& light, RayWorkItem* prd)
	{
		MeshLightAttributesData meshLightAttributes = *reinterpret_cast<MeshLightAttributesData*>(light.attributes);
		LightSample lightSample;

		lightSample.pdf = 0.0f;

		const float* cdfArea = meshLightAttributes.cdfArea;
		const float3 sample3D = rng3(prd->seed);

		// Uniformly sample the triangles over their surface area.
		// Note that zero-area triangles (e.g. at the poles of spheres) are automatically never sampled with this method!
		// The cdfU is one bigger than res.y.

		unsigned int idxTriangle = utl::binarySearchCdf(cdfArea, meshLightAttributes.size, sample3D.z);
		idxTriangle = meshLightAttributes.actualTriangleIndices[idxTriangle];

		// Unit square to triangle via barycentric coordinates.
		const float sqrtSampleX = sqrtf(sample3D.x);
		// Barycentric coordinates.
		const float alpha = 1.0f - sqrtSampleX;
		const float beta = sample3D.y * sqrtSampleX;
		const float gamma = 1.0f - alpha - beta;

		HitProperties hitP;
		hitP.baricenter = math::vec3f(alpha, beta, gamma);
		utl::getInstanceAndGeometry(&hitP, meshLightAttributes.instanceId, optixLaunchParams);
		utl::getVertices(&hitP, idxTriangle);
		utl::fetchTransformsFromInstance(&hitP);
		utl::computeHit(&hitP, prd->hitPosition);

		lightSample.position = hitP.position;
		lightSample.direction = -hitP.direction;
		lightSample.distance = hitP.distance;
		if (lightSample.distance < DENOMINATOR_EPSILON)
		{
			return lightSample;
		}

		utl::computeGeometricHitProperties(&hitP, true);
		utl::determineMaterialHitProperties(&hitP, idxTriangle);

		hitP.seed = prd->seed;
		mdl::MdlRequest request;
		request.edf = true;
		request.opacity = true;
		request.lastRayDirection = -lightSample.direction;
		request.hitP = &hitP;
		request.surroundingIor = 1.0f;

		mdl::MaterialEvaluation matEval;
		if (!hitP.materialConfiguration->directCallable)
		{
			const int sbtIndex = hitP.materialConfiguration->idxCallEvaluateMaterialWavefront;
			optixDirectCall<void, mdl::MdlRequest*, mdl::MaterialEvaluation*>(sbtIndex, &request, &matEval);
		}
		else
		{
			matEval = mdl::evaluateMdlMaterial(&request);
		}

		if (matEval.opacity <= 0.0f)
		{
			return lightSample;
		}

		if (matEval.edf.isValid)
		{
			const float totArea = meshLightAttributes.totalArea;

			// Modulate the emission with the cutout opacity value to get the correct value.
			// The opacity value must not be greater than one here, which could happen for HDR textures.
			float opacity = math::min(matEval.opacity, 1.0f);

			// Power (flux) [W] divided by light area gives radiant exitance [W/m^2].
			const float factor = (matEval.edf.mode == 0) ? matEval.opacity : matEval.opacity / totArea;

			lightSample.pdf = lightSample.distance * lightSample.distance / (totArea * matEval.edf.cos); // Solid angle measure.
			lightSample.radianceOverPdf = matEval.edf.intensity * matEval.edf.edf * (factor / lightSample.pdf);
			lightSample.isValid = true;
		}

		return lightSample;
	}

	__forceinline__ __device__ LightSample sampleEnvironment(const LightData & light, RayWorkItem* prd)
	{
		EnvLightAttributesData attrib = *reinterpret_cast<EnvLightAttributesData*>(light.attributes);
		auto texture = attrib.texture;

		unsigned width = texture->dimension.x;
		unsigned height = texture->dimension.y;

		LightSample            lightSample;

		// importance sample an envmap pixel using an alias map
		const float3       sample = rng3(prd->seed);
		const unsigned int size = width * height;
		const auto         idx = math::min<unsigned>((unsigned int)(sample.x * (float)size), size - 1);
		unsigned int       envIdx;
		float              sampleY = sample.y;
		if (sampleY < attrib.aliasMap[idx].q) {
			envIdx = idx;
			sampleY /= attrib.aliasMap[idx].q;
		}
		else {
			envIdx = attrib.aliasMap[idx].alias;
			sampleY = (sampleY - attrib.aliasMap[idx].q) / (1.0f - attrib.aliasMap[idx].q);
		}

		const unsigned int py = envIdx / width;
		const unsigned int px = envIdx % width;
		lightSample.pdf = attrib.aliasMap[envIdx].pdf;

		const float u = (float)(px + sampleY) / (float)width;
		//const float phi = (M_PI_2)*(1.0f-u);

		//const float phi = (float)M_PI  -u * (float)(2.0 * M_PI);
		const float phi = u * (float)(2.0 * M_PI) - (float)M_PI;
		float sinPhi, cosPhi;
		sincosf(phi > float(-M_PI) ? phi : (phi + (float)(2.0 * M_PI)), &sinPhi, &cosPhi);
		const float stepTheta = (float)M_PI / (float)height;
		const float theta0 = (float)(py)*stepTheta;
		const float cosTheta = cosf(theta0) * (1.0f - sample.z) + cosf(theta0 + stepTheta) * sample.z;
		const float theta = acosf(cosTheta);
		const float sinTheta = sinf(theta);
		const float v = theta * (float)(1.0 / M_PI);

		float x = cosPhi * sinTheta;
		float y = sinPhi * sinTheta;
		float z = -cosTheta;

		math::vec3f dir{ x, y, z };
		// Now rotate that normalized object space direction into world space. 
		lightSample.direction = math::transformNormal3F(attrib.transformation, dir);

		lightSample.distance = optixLaunchParams.settings->maxClamp; // Environment light.

		// Get the emission from the spherical environment texture.
		math::vec3f emission = tex2D<float4>(texture->texObj, u, v);
		// For simplicity we pretend that we perfectly importance-sampled the actual texture-filtered environment map
		// and not the Gaussian-smoothed one used to actually generate the CDFs and uniform sampling in the texel.
		// (Note that this does not contain the light.emission which just modulates the texture.)
		if (DENOMINATOR_EPSILON < lightSample.pdf)
		{
			//lightSample.radianceOverPdf = light.emission * emission / lightSample.pdf;
			lightSample.radianceOverPdf = emission / lightSample.pdf;
		}

		lightSample.isValid = true;
		return lightSample;
	}

	__forceinline__ __device__ LightSample sampleLight(RayWorkItem* prd, const HitProperties* hitP)
	{
		LightSample lightSample;
		if (const int& numLights = optixLaunchParams.numberOfLights; numLights > 0)
		{
			//Randomly Selecting a Light
			//TODO, I think here we can do some better selection by giving more importance to lights with greater power
			const int indexLight = (1 < numLights) ? gdt::clamp(static_cast<int>(floorf(rng(prd->seed) * numLights)), 0, numLights - 1) : 0;

			const LightData& light = *(optixLaunchParams.lights[indexLight]);

			const LightType& typeLight = light.type;

			switch (typeLight)
			{
			case L_MESH:
			{
				lightSample = sampleMeshLight(light, prd);
			}
			break;
			case L_ENV:
			{
				lightSample = sampleEnvironment(light, prd);
			}
			break;
			default: {
				lightSample.isValid = false;
				return lightSample;
			}
			}

			lightSample.typeLight = typeLight;

			if (lightSample.isValid) // && dot(lightSample.direction, hitP->ngW) >= -0.05f)
			{
				lightSample.isValid = true;
				return lightSample;
			}
		}
		lightSample.isValid = false;
		lightSample.pdf = 0.0f;
		return lightSample;
	}


	struct Profiler
	{
		float hitProperties = 0;
		float auxiliaryGeneral = 0;
		float materialEvaluation = 0;
		float auxiliaryMaterial = 0;
		float lightSampling = 0;
		float hitEmission = 0;
		float russianRoulette = 0;
		float bsdfSampling = 0;
		float nextWork = 0;
		float tot = 0;

		__forceinline__ __device__ void increment(float* counter, const float delta){
			*counter = *counter + delta;
		}

		__device__ __forceinline__ float percentage(const float delta)
		{
			return (float)delta / (float)tot * 100.0f;
		}

		__device__ __forceinline__ void print()
		{
			tot = hitProperties + auxiliaryGeneral + materialEvaluation + auxiliaryMaterial + lightSampling + hitEmission + russianRoulette + bsdfSampling + nextWork;
			


			float hitValue = hitProperties;
			float auxGenValue = auxiliaryGeneral;
			float matEvalValue = materialEvaluation;
			float auxMatvalue = auxiliaryMaterial;
			float lightSampleValue = lightSampling;
			float hitEmissionValue = hitEmission;
			float russianValue = russianRoulette;
			float bsdfValue = bsdfSampling;
			float nextWorkValue = nextWork;



			printf("Profile:\n"
				"\t Tot:			%.3f\n"
				"\t Hit Properties:	%.3f\n"
				"\t Aux Data 1:		%.3f\n"
				"\t Mat Eval:		%.3f\n"
				"\t Aux Data 2:		%.3f\n"
				"\t Direct Light:		%.3f\n"
				"\t Eval Emission:		%.3f\n"
				"\t Russian Roulette:	%.3f\n"
				"\t enqueue:		%.3f\n"
				"\t Bsdf and enqueue:	%.3f\n\n",
				tot,
				hitValue,
				auxGenValue,
				matEvalValue,
				auxMatvalue,
				lightSampleValue,
				hitEmissionValue,
				russianValue,
				nextWorkValue,
				bsdfValue);

			/*float hitValue = percentage(hitProperties);
			float auxGenValue = percentage(auxiliaryGeneral);
			float matEvalValue = percentage(materialEvaluation);
			float auxMatvalue = percentage(auxiliaryMaterial);
			float lightSampleValue = percentage(lightSampling);
			float hitEmissionValue = percentage(hitEmission);
			float russianValue = percentage(russianRoulette);
			float bsdfValue = percentage(bsdfSampling);
			float nextWorkValue = percentage(nextWork);
			printf("Profile:\n"
				"\t Tot:			%.lu\n"
				"\t Hit Properties:	%.3f\n"
				"\t Aux Data 1:		%.3f\n"
				"\t Mat Eval:		%.3f\n"
				"\t Aux Data 2:		%.3f\n"
				"\t Direct Light:		%.3f\n"
				"\t Eval Emission:		%.3f\n"
				"\t Russian Roulette:	%.3f\n"
				"\t enqueue:			%.3f\n"
				"\t Bsdf and enqueue:	%.3f\n\n",
				tot,
				hitValue,
				auxGenValue,
				matEvalValue,
				auxMatvalue,
				lightSampleValue,
				hitEmissionValue,
				russianValue,
				nextWorkValue,
				bsdfValue);*/
		}

	};



	template<typename Func>
	__forceinline__ __device__ float profile(Func func)
	{
		float start = (float)clock()/(10000.0f);
		func();
		float end = (float)clock()/(10000.0f);

		float delta = end - start;
		return delta;
	}

	__forceinline__ __device__ void prepareHitProperties(HitProperties* hitP, RayWorkItem& prd)
	{
		hitP->position = prd.hitPosition;
		hitP->direction = -prd.direction;
		hitP->baricenter = prd.hitBaricenter;
		hitP->seed = prd.seed;

		utl::getInstanceAndGeometry(hitP, prd.hitInstanceId, optixLaunchParams);
		utl::getVertices(hitP, prd.hitTriangleId);
		utl::setTransform(hitP, &prd);
		utl::computeGeometricHitProperties(hitP, prd.hitTriangleId);
		utl::determineMaterialHitProperties(hitP, prd.hitTriangleId);
	}

	__forceinline__ __device__ void setGeometricAuxiliaryData(RayWorkItem& prd, const HitProperties& hitP)
	{
		//Auxiliary Data
		if (prd.depth == 0)
		{
			prd.colorsTrueNormal = 0.5f * (hitP.ngW + 1.0f);
			prd.colorsUv = hitP.textureCoordinates[0];
			prd.colorsOrientation = hitP.isFrontFace ? math::vec3f(0.0f, 0.0f, 1.0f) : math::vec3f(1.0f, 0.0f, 0.0f);
			prd.colorsTangent = 0.5f * (hitP.tgW + 1.0f);
			prd.colorsBounceDiffuse = math::vec3f(1.0f, 0.0f, 1.0f);
			prd.colorsShadingNormal = 0.5f * (hitP.nsW + 1.0f);
		}
	}

	__forceinline__ __device__ void evaluateMaterialAndSampleLight(mdl::MaterialEvaluation* matEval, LightSample* lightSample, HitProperties& hitP, RayWorkItem& prd)
	{
		mdl::MdlRequest request;

		RendererDeviceSettings::SamplingTechnique& samplingTechnique = optixLaunchParams.settings->samplingTechnique;

		request.auxiliary = true;
		request.ior = true;
		request.edf = true;
		request.lastRayDirection = -prd.direction;
		request.hitP = &hitP;

		request.surroundingIor = prd.mediumIor;

		const bool doSampleLight = samplingTechnique == RendererDeviceSettings::S_DIRECT_LIGHT || samplingTechnique == RendererDeviceSettings::S_MIS;
		const bool doSampleBsdf  = samplingTechnique == RendererDeviceSettings::S_MIS || samplingTechnique == RendererDeviceSettings::S_BSDF;

		if(doSampleLight)
		{
			*lightSample = sampleLight(&prd, &hitP);
			if (lightSample->isValid)
			{
				request.bsdfEvaluation = true;
				request.toSampledLight = lightSample->direction;
			}
		}

		if(doSampleBsdf)
		{
			request.bsdfSample = true;
		}
		if (hitP.meshLightAttributes != nullptr)
		{
			request.edf = true;
		}
		

		if (hitP.materialConfiguration->directCallable)
		{
			*matEval = mdl::evaluateMdlMaterial(&request);
		}
		else
		{
			const int sbtIndex = hitP.materialConfiguration->idxCallEvaluateMaterialWavefront;
			optixDirectCall<void, mdl::MdlRequest*, mdl::MaterialEvaluation*>(sbtIndex, &request, matEval);
		}
	}

	__forceinline__ __device__ void setAuxiliaryMaterial(const mdl::MaterialEvaluation& matEval, RayWorkItem& prd)
	{
		// Auxiliary Data
		if(prd.depth == 0 && matEval.aux.isValid)
		{
			prd.colorsBounceDiffuse = matEval.aux.albedo;
			prd.colorsShadingNormal = 0.5f * (matEval.aux.normal + 1.0f);
		}
	}

	__forceinline__ __device__ void nextEventEstimation(const mdl::MaterialEvaluation& matEval, LightSample& lightSample, RayWorkItem& prd)
	{
		RendererDeviceSettings::SamplingTechnique& samplingTechnique = optixLaunchParams.settings->samplingTechnique;

		//Direct Light Sampling
		if ((samplingTechnique == RendererDeviceSettings::S_MIS || samplingTechnique == RendererDeviceSettings::S_DIRECT_LIGHT) && lightSample.isValid)
		{
			//printf("number of tries: %d\n", numberOfTries);
			auto bxdf = math::vec3f(0.0f, 0.0f, 0.0f);
			bxdf += matEval.bsdfEvaluation.diffuse;
			bxdf += matEval.bsdfEvaluation.glossy;

			if (0.0f < matEval.bsdfEvaluation.pdf && bxdf != math::vec3f(0.0f, 0.0f, 0.0f))
			{
				// Pass the current payload registers through to the shadow ray.

				unsigned escaped = 1;
				// Note that the sysData.sceneEpsilon is applied on both sides of the shadow ray [t_min, t_max] interval 
				// to prevent self-intersections with the actual light geometry in the scene.
				optixTrace(optixLaunchParams.topObject,
					prd.hitPosition,
					lightSample.direction, // origin, direction
					optixLaunchParams.settings->minClamp,
					lightSample.distance - optixLaunchParams.settings->minClamp,
					0.0f, // tmin, tmax, time
					OptixVisibilityMask(0xFF),
					OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, // The shadow ray type only uses anyhit programs.
					0,
					0,
					0,
					escaped);

				if (escaped == 1)
				{
					float weightMis = 1.0f;
					if ((lightSample.typeLight == L_MESH || lightSample.typeLight == L_ENV) && samplingTechnique == RendererDeviceSettings::S_MIS)
					{
						weightMis = utl::heuristic(lightSample.pdf, matEval.bsdfEvaluation.pdf);
					}

					// The sampled emission needs to be scaled by the inverse probability to have selected this light,
					// Selecting one of many lights means the inverse of 1.0f / numLights.
					// This is using the path throughput before the sampling modulated it above.

					prd.radiance += prd.throughput * bxdf * lightSample.radianceOverPdf * weightMis * (float)optixLaunchParams.numberOfLights; // *float(numLights);
				}
			}
		}		//Evauluate Hit Point Emission
	}

	__forceinline__ __device__ void evaluateEmission(mdl::MaterialEvaluation& matEval, RayWorkItem& prd, const HitProperties& hitP)
	{
		RendererDeviceSettings::SamplingTechnique& samplingTechnique = optixLaunchParams.settings->samplingTechnique;
		if (matEval.edf.isValid)
		{
			const float area = hitP.meshLightAttributes->totalArea;
			matEval.edf.pdf = prd.hitDistance * prd.hitDistance / (area * matEval.edf.cos);
			// Solid angle measure.

			float misWeight = 1.0f;

			// If the last event was diffuse or glossy, calculate the opposite MIS weight for this implicit light hit.
			if (samplingTechnique == RendererDeviceSettings::S_MIS && prd.eventType & (mi::neuraylib::BSDF_EVENT_DIFFUSE | mi::neuraylib::BSDF_EVENT_GLOSSY))
			{
				misWeight = utl::heuristic(prd.pdf, matEval.edf.pdf);
			}
			// Power (flux) [W] divided by light area gives radiant exitance [W/m^2].
			const float factor = (matEval.edf.mode == 0) ? 1.0f : 1.0f / area;
			prd.radiance += prd.throughput * matEval.edf.intensity * matEval.edf.edf * (factor * misWeight);
		}
	}

	__forceinline__ __device__ void russianRoulette(RayWorkItem& prd, bool* terminate)
	{
		if (optixLaunchParams.settings->useRussianRoulette && 2 <= prd.depth) // Start termination after a minimum number of bounces.
		{
			const float probability = fmaxf(fmaxf(prd.throughput.x, prd.throughput.y), prd.throughput.z);

			if (probability < rng(prd.seed)) // Paths with lower probability to continue are terminated earlier.
			{
				*terminate = true;
			}
			prd.throughput /= probability; // Path isn't terminated. Adjust the throughput so that the average is right again.
		}
	}

	__forceinline__ __device__ void bsdfSample(RayWorkItem& prd, const mdl::MaterialEvaluation& matEval, const HitProperties& hitP, bool* terminate)
	{
		if(*terminate)
		{
			return;
		}
		RendererDeviceSettings::SamplingTechnique& samplingTechnique = optixLaunchParams.settings->samplingTechnique;
		if ((samplingTechnique == RendererDeviceSettings::S_MIS || samplingTechnique == RendererDeviceSettings::S_BSDF)
			&& prd.depth + 1 <= optixLaunchParams.settings->maxBounces
			&& matEval.bsdfSample.eventType != mi::neuraylib::BSDF_EVENT_ABSORB)
		{
			prd.origin = prd.hitPosition;
			prd.direction = matEval.bsdfSample.nextDirection; // Continuation direction.
			prd.throughput *= matEval.bsdfSample.bsdfOverPdf;
			// Adjust the path throughput for all following incident lighting.
			prd.pdf = matEval.bsdfSample.pdf;
			// Note that specular events return pdf == 0.0f! (=> Not a path termination condition.)
			prd.eventType = matEval.bsdfSample.eventType;

			if (!matEval.isThinWalled && (prd.eventType & mi::neuraylib::BSDF_EVENT_TRANSMISSION) != 0)
			{
				if (hitP.isFrontFace) // Entered a volume. 
				{
					prd.mediumIor = matEval.ior;
					//printf("Entering Volume Ior = %f %f %f\n", matEval.ior.x, matEval.ior.y, matEval.ior.z);
				}
				else // if !isFrontFace. Left a volume.
				{
					prd.mediumIor = 1.0f;
				}
			}
			if (prd.depth == 0)
			{
				prd.firstHitType = prd.eventType;
			}
			prd.depth++;
			// Unbiased Russian Roulette path termination.

			*terminate = false;
		}
	}

	__forceinline__ __device__ void nextWork(const RayWorkItem& prd, const bool& terminate)
	{
		if(!terminate)
		{
			optixLaunchParams.radianceTraceQueue->Push(prd);
		}
		else
		{
			optixLaunchParams.accumulationQueue->Push(prd);
		}
	}

	extern "C" __global__ void __raygen__shade()
	{
		const int queueWorkId = optixGetLaunchIndex().x;

		if (optixLaunchParams.shadeQueue->Size() <= queueWorkId)
			return;


		RayWorkItem prd = (*optixLaunchParams.shadeQueue)[queueWorkId];

		HitProperties hitP;

		prepareHitProperties(&hitP, prd);

		setGeometricAuxiliaryData(prd, hitP);

		if (hitP.material != nullptr)
		{
			mdl::MaterialEvaluation matEval{};
			LightSample lightSample{};
			evaluateMaterialAndSampleLight(&matEval, &lightSample, hitP, prd);

			//setAuxiliaryMaterial(matEval, prd);

			//nextEventEstimation(matEval, lightSample, prd);

			//evaluateEmission(matEval, prd, hitP);

			prd.pdf = 0.0f;

			bool terminate = false;

			//russianRoulette(prd, &terminate);

			//bsdfSample(prd, matEval, hitP, &terminate);

			nextWork(prd, terminate);

		}
	}

}
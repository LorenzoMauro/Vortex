#ifndef OPTIXCODE
#define OPTIXCODE
#endif

#include <optix_device.h>
#include "../RayData.h"
#include "../DataFetcher.h"
#include "../Utils.h"
#include "../ToneMapper.h"
#include "Device/DevicePrograms/Mdl/directMdlWrapper.h"

namespace vtx
{

#define materialEvaluation(sbtIndex, request)  optixDirectCall<mdl::MaterialEvaluation>(sbtIndex, request)

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
		utl::getInstanceAndGeometry(&hitP, meshLightAttributes.instanceId);
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
		if (!hitP.shaderConfiguration->directCallable)
		{
			const int sbtIndex = hitP.shaderConfiguration->idxCallEvaluateMaterial;
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

		math::vec3f dir{ x,y,z };
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
		lightSample.pdf     = 0.0f;
		return lightSample;
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
		hitP.seed = prd->seed;

		utl::getInstanceAndGeometry(&hitP, instanceId);
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
			request.ior       = true;
			request.edf = true;
			request.lastRayDirection = prd->wo;
			request.hitP = &hitP;

			//printf("OUTSIDE - request pointer : %p\n"
			//	   "OUTSIDE - hitP pointer    : %p\n", (void*)&request, (void*)request.hitP);

			/*printf("OUTISDE - HiP tex coords: %f %f\n\n",
				   request.hitP->textureCoordinates[0].x, request.hitP->textureCoordinates[0].y);*/
			request.surroundingIor = math::vec3f{1.0f};

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
			else if(samplingTechnique == RendererDeviceSettings::S_BSDF)
			{
				request.bsdfSample = true;
			}
			else if (samplingTechnique ==  RendererDeviceSettings::S_DIRECT_LIGHT)
			{
				lightSample = sampleLight(prd, &hitP);
				if(lightSample.isValid)
				{
					request.bsdfEvaluation = true;
					request.toSampledLight = lightSample.direction;
				}
			}


			mdl::MaterialEvaluation matEval;
			if (!hitP.shaderConfiguration->directCallable)
			{
				const int sbtIndex = hitP.shaderConfiguration->idxCallEvaluateMaterial;
				optixDirectCall<void, mdl::MdlRequest*, mdl::MaterialEvaluation*>(sbtIndex, &request, &matEval);
			}
			else
			{
				matEval = mdl::evaluateMdlMaterial(&request);
			}

			/*printf("Aux Valid %i\n"
				   "Bsdf Sample Valid %i\n"
				   "Bsdf Eval Valid %i\n"
				   "Light Sample Valid %i\n\n",
				   matEval.aux.isValid,
				   matEval.bsdfEvaluation.isValid,
				   matEval.bsdfEvaluation.isValid,
				   lightSample.isValid);*/

			// Auxiliary Data
			if (prd->depth == 0)
			{
				if (matEval.aux.isValid)
				{
					prd->colors.diffuse       = matEval.aux.albedo;
					prd->colors.shadingNormal = matEval.aux.normal;
					prd->colors.shadingNormal = 0.5f * (prd->colors.shadingNormal + 1.0f);
				}
				else
				{
					prd->colors.diffuse       = math::vec3f(1.0f, 0.0f, 1.0f);
					prd->colors.shadingNormal = hitP.nsW;
					prd->colors.shadingNormal = 0.5f * (prd->colors.shadingNormal + 1.0f);
				}
			}


			//Evauluate Hit Point Emission
			if (matEval.edf.isValid)
			{
				const float area  = hitP.meshLightAttributes->totalArea;
				matEval.edf.pdf = prd->distance * prd->distance / (area * matEval.edf.cos);
				// Solid angle measure.

				float misWeight = 1.0f;

				// If the last event was diffuse or glossy, calculate the opposite MIS weight for this implicit light hit.
				if(samplingTechnique == RendererDeviceSettings::S_MIS && prd->eventType & (mi::neuraylib::BSDF_EVENT_DIFFUSE | mi::neuraylib::BSDF_EVENT_GLOSSY))
				{
					misWeight = utl::heuristic(prd->pdf, matEval.edf.pdf);
				}
				// Power (flux) [W] divided by light area gives radiant exitance [W/m^2].
				const float factor = (matEval.edf.mode == 0) ? 1.0f : 1.0f / area;

				prd->radiance += prd->throughput * matEval.edf.intensity * matEval.edf.edf * (factor* misWeight);
			}

			math::vec3f currentThroughput = prd->throughput;
			prd->pdf                       = 0.0f;

			// BSDF Sampling
			if(samplingTechnique == RendererDeviceSettings::S_MIS || samplingTechnique==RendererDeviceSettings::S_BSDF)
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
					// This replaces the PRD flags used inside the other examples.
					if ((matEval.bsdfSample.eventType & mi::neuraylib::BSDF_EVENT_TRANSMISSION) != 0)
					{
						// continue on the opposite side
						utl::offsetRay(prd->position, -hitP.ngW);
					}
					else
					{
						// continue on the current side
						utl::offsetRay(prd->position, hitP.ngW);
					}
				}
				else
				{
					prd->traceResult = TR_HIT;
					return;
					// None of the following code will have any effect in that case.
				}
			}

			//Direct Light Sampling

			if ((samplingTechnique == RendererDeviceSettings::S_MIS || samplingTechnique == RendererDeviceSettings::S_DIRECT_LIGHT) && lightSample.isValid)
			{
				//printf("number of tries: %d\n", numberOfTries);
				auto bxdf = math::vec3f(0.0f, 0.0f, 0.0f);
				bxdf += matEval.bsdfEvaluation.diffuse;
				bxdf += matEval.bsdfEvaluation.glossy;
				if(matEval.bsdfEvaluation.pdf>0.0f)
				{
					prd->colors.debugColor1 = 1.0f;
				}

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
						if (lightSample.typeLight == L_MESH || lightSample.typeLight == L_ENV)
						{
							weightMis = utl::heuristic(lightSample.pdf, matEval.bsdfEvaluation.pdf);
						}

						// The sampled emission needs to be scaled by the inverse probability to have selected this light,
						// Selecting one of many lights means the inverse of 1.0f / numLights.
						// This is using the path throughput before the sampling modulated it above.

						prd->radiance += currentThroughput * bxdf * lightSample.radianceOverPdf * weightMis * (float)optixLaunchParams.numberOfLights; // *float(numLights);

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

		if (hitP.instance->hasOpacity)
		{
			printf("Evaluating Opacity");
			float opacity = 1.0f;
			utl::determineMaterialHitProperties(&hitP, triangleIdx);
			hitP.position     = prd->wo;
			hitP.baricenter   = math::vec3f(0.0f, optixGetTriangleBarycentrics().x, optixGetTriangleBarycentrics().y);
			hitP.baricenter.x = 1.0f - hitP.baricenter.x - hitP.baricenter.y;

			hitP.seed = prd->seed;
			utl::getVertices(&hitP, triangleIdx);
			utl::fetchTransformsFromHandle(&hitP);
			utl::computeGeometricHitProperties(&hitP, triangleIdx);

			mdl::MdlRequest request;
			request.hitP    = &hitP;
			request.opacity = true;
			mdl::MaterialEvaluation eval;
			if (!hitP.shaderConfiguration->directCallable)
			{
				const int sbtIndex = hitP.shaderConfiguration->idxCallEvaluateMaterial;
				optixDirectCall<void, mdl::MdlRequest*, mdl::MaterialEvaluation*>(sbtIndex, &request, &eval);
				opacity            = eval.opacity;
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
		PerRayData* prd         = mergePointer(optixGetPayload_0(), optixGetPayload_1());
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

		math::vec2ui payload = splitPointer(&prd);

		while (prd.depth <= maxDepth)
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

			const LensRay cameraRay = optixDirectCall<LensRay, const math::vec2f, const math::vec2f, const math::vec2f>(
				optixLaunchParams.programs->pinhole, screen, pixel, sample);

			prd.position = cameraRay.org;
			prd.wi = cameraRay.dir;

			math::vec3f radiance = integrator(prd);

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

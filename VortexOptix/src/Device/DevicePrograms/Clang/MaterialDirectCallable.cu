#include <optix.h>
#include <optix_device.h>
#include "../Utils.h"
#include "Device/DevicePrograms/LightSampler.h"
#include "Device/DevicePrograms/Mdl/InlineMdlWrapper.h"
#include "vector_types.h"
namespace vtx
{
	//extern "C" __device__ LightSample sampleLight(PerRayData * prd, const HitProperties * hitP);
}

namespace vtx
{

    extern "C" __device__ void __direct_callable__EvaluateMaterial(mdl::MdlRequest* request, mdl::MaterialEvaluation* result)
    {
		mdl::MdlData mdlData = mdl::mdlInit(request->hitP);

        if (request->ior)
        {
            iorEvaluation(&result->ior, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
        }
        if (request->opacity)
        {
            opacityEvaluation(&result->opacity, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
        }
        if (request->edf || request->bsdfEvaluation || request->bsdfSample)
        {
            thinWalled(&mdlData.isThinWalled, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
        }
        if (request->edf)
        {
            result->edf = evaluateEmission(&mdlData, request->lastRayDirection);
        }
        if(request->bsdfEvaluation)
        {
            result->bsdfEvaluation = evaluateBsdf(&mdlData, request->surroundingIor, request->lastRayDirection, request->toSampledLight);
        }
        if (request->bsdfSample)
        {
            result->bsdfSample = sampleBsdf(&mdlData, request->surroundingIor, request->lastRayDirection, request->hitP->seed);
        }
        if (request->auxiliary)
        {
            result->aux = auxiliaryBsdf(&mdlData, request->surroundingIor, request->lastRayDirection);
        }
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
			hitP.position = prd->wo;
			hitP.baricenter = math::vec3f(0.0f, optixGetTriangleBarycentrics().x, optixGetTriangleBarycentrics().y);
			hitP.baricenter.x = 1.0f - hitP.baricenter.x - hitP.baricenter.y;

			hitP.seed = prd->seed;
			utl::getVertices(&hitP, triangleIdx);
			utl::fetchTransformsFromHandle(&hitP);
			utl::computeGeometricHitProperties(&hitP, triangleIdx);

			mdl::MdlData mdlData = mdl::mdlInit(&hitP);
			opacityEvaluation(&opacity, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
			// Stochastic alpha test to get an alpha blend effect.
			// No need to calculate an expensive random number if the test is going to fail anyway.
			if (opacity < 1.0f && opacity <= rng(prd->seed))
			{
				optixIgnoreIntersection();
				return;
			}
		}
		prd->traceResult = TR_SHADOW;
	}

	extern "C" __global__ void __closesthit__radiance()
	{
		//unsigned long long address = ((unsigned long long)optixGetPayload_1() * (1ULL << 32)) + optixGetPayload_0();
		//void* pointer = (void*)address;
		
		PerRayData* prd = mergePointer(optixGetPayload_0(), optixGetPayload_1());
		/*printf("INSIDE CH		- prd pointer	%p\n"
			   "FIRST			-				%d\n"
			   "SECOND			-				%d\n",
			   prd, optixGetPayload_0(), optixGetPayload_1());*/
		//printf("PRD POINTER %p\n DISTANCE POINTER %f\n", prd , prd->distance);
		const unsigned int instanceId = optixGetInstanceId();
		const unsigned int triangleIdx = optixGetPrimitiveIndex();
		float distance = optixGetRayTmax();
		prd->distance = distance;
		prd->position = prd->position + prd->wi * prd->distance;

		HitProperties hitP;
		hitP.position = prd->position;
    	hitP.direction = prd->wo;
		hitP.baricenter = math::vec3f(0.0f, optixGetTriangleBarycentrics().x, optixGetTriangleBarycentrics().y);
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

			mdl::MdlData mdlData = mdl::mdlInit(&hitP);

	    	float ior;
	    	math::vec3f surroundingIor = 1.0f;

	    	iorEvaluation(&ior, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
	    	thinWalled(&mdlData.isThinWalled, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);

			if (prd->depth == 0)
	    	{
				mdl::BsdfAuxResult auxEval = auxiliaryBsdf(&mdlData, surroundingIor, prd->wo);
	    		if (auxEval.isValid)
	    		{
	    			prd->colors.diffuse = auxEval.albedo;
	    			prd->colors.shadingNormal = auxEval.normal;
	    			prd->colors.shadingNormal = 0.5f * (prd->colors.shadingNormal + 1.0f);
	    		}
	    		else
	    		{
	    			prd->colors.diffuse = math::vec3f(1.0f, 0.0f, 1.0f);
	    			prd->colors.shadingNormal = hitP.nsW;
	    			prd->colors.shadingNormal = 0.5f * (prd->colors.shadingNormal + 1.0f);
	    		}
	    	}

			//Evauluate Hit Point Emission
	    	if (hitP.meshLightAttributes != nullptr)
	    	{
				mdl::EdfResult edfEval = evaluateEmission(&mdlData, prd->wo);
	    		if (edfEval.isValid)
	    		{
	    			const float area = hitP.meshLightAttributes->totalArea;
	    			edfEval.pdf = prd->distance * prd->distance / (area * edfEval.cos);
	    			// Solid angle measure.

	    			float misWeight = 1.0f;

	    			// If the last event was diffuse or glossy, calculate the opposite MIS weight for this implicit light hit.
	    			if (samplingTechnique == RendererDeviceSettings::S_MIS && prd->eventType & (mi::neuraylib::BSDF_EVENT_DIFFUSE | mi::neuraylib::BSDF_EVENT_GLOSSY))
	    			{
	    				misWeight = utl::heuristic(prd->pdf, edfEval.pdf);
	    			}
	    			// Power (flux) [W] divided by light area gives radiant exitance [W/m^2].
	    			const float factor = (edfEval.mode == 0) ? 1.0f : 1.0f / area;

	    			prd->radiance += prd->throughput * edfEval.intensity * edfEval.edf * (factor * misWeight);
	    		}
	    	}


	    	math::vec3f currentThroughput = prd->throughput;
	    	prd->pdf = 0.0f;

	    	// BSDF Sampling
	    	if (samplingTechnique == RendererDeviceSettings::S_MIS || samplingTechnique == RendererDeviceSettings::S_BSDF)
	    	{
	    		//matEval.bsdfSample.print("Outside:\n");
				mdl::BsdfSampleResult bsdfSampleEval = sampleBsdf(&mdlData, surroundingIor, prd->wo, prd->seed);

	    		//Importance Sampling the Bsdf
	    		if (bsdfSampleEval.eventType != mi::neuraylib::BSDF_EVENT_ABSORB)
	    		{
	    			prd->wi = bsdfSampleEval.nextDirection; // Continuation direction.
	    			prd->throughput *= bsdfSampleEval.bsdfOverPdf;
	    			// Adjust the path throughput for all following incident lighting.
	    			prd->pdf = bsdfSampleEval.pdf;
	    			// Note that specular events return pdf == 0.0f! (=> Not a path termination condition.)
	    			prd->eventType = bsdfSampleEval.eventType;
	    			// This replaces the PRD flags used inside the other examples.
	    			if ((bsdfSampleEval.eventType & mi::neuraylib::BSDF_EVENT_TRANSMISSION) != 0)
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
	    	if ((samplingTechnique == RendererDeviceSettings::S_MIS || samplingTechnique == RendererDeviceSettings::S_DIRECT_LIGHT)) {
				LightSample lightSample;
	    		lightSample = sampleLight(prd, &hitP);
	    		if (lightSample.isValid)
	    		{
	    			//printf("number of tries: %d\n", numberOfTries);
					mdl::BsdfEvalResult bsdfEval = evaluateBsdf(&mdlData, surroundingIor, prd->wo, lightSample.direction);
	    			auto           bxdf = math::vec3f(0.0f, 0.0f, 0.0f);
	    			bxdf += bsdfEval.diffuse;
	    			bxdf += bsdfEval.glossy;
	    			if (bsdfEval.pdf > 0.0f)
	    			{
	    				prd->colors.debugColor1 = 1.0f;
	    			}

	    			if (0.0f < bsdfEval.pdf && bxdf != math::vec3f(0.0f, 0.0f, 0.0f))
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
								   0,
								   0,
								   0,
								   p0, p1); // Pass through thePrd to the shadow ray.

	    				if (prd->traceResult != TR_SHADOW)
	    				{
	    					float weightMis = 1.0f;
	    					if (lightSample.typeLight == L_MESH || lightSample.typeLight == L_ENV)
	    					{
	    						weightMis = utl::heuristic(lightSample.pdf, bsdfEval.pdf);
	    					}

	    					// The sampled emission needs to be scaled by the inverse probability to have selected this light,
	    					// Selecting one of many lights means the inverse of 1.0f / numLights.
	    					// This is using the path throughput before the sampling modulated it above.

	    					prd->radiance += currentThroughput * bxdf * lightSample.radianceOverPdf * weightMis * (float)optixLaunchParams.numberOfLights; // *float(numLights);

	    				}
	    			}
	    		}
	    	}
	    }
		prd->traceResult = TR_HIT;
	}
}


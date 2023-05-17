#pragma once
#ifndef MDL_DEVICE_WRAPPER_H
#define MDL_DEVICE_WRAPPER_H
#include <optix_device.h>
#include "../randomNumberGenerator.h"
#include "../RayData.h"
#include "MdlStructs.h"

namespace vtx::mdl
{

    __forceinline__ __device__ float determineOpacity(MdlData& mdlData, const DeviceShaderConfiguration* shaders)
    {
        // Arbitrary mesh lights can have cutout opacity!
        float opacity = shaders->cutoutOpacity;
        if (0 <= shaders->idxCallGeometryCutoutOpacity)
        {
            optixDirectCall<void>(shaders->idxCallGeometryCutoutOpacity, &opacity, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
        }

        // If the current light sample is inside a fully cutout region, reject that sample.
        if (opacity <= 0.0f)
        {
        }

        return opacity;
    }

    __forceinline__ __device__ math::vec3f getIor(MdlData& mdlData, const DeviceShaderConfiguration* shaders)
    {
        // IOR value in case the material ior expression is constant.
        math::vec3f ior = shaders->ior;

        if (0 <= shaders->idxCallIor)
        {
            optixDirectCall<void>(shaders->idxCallIor, &ior, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
        }

        return ior;
    }

    __forceinline__ __device__ bool isThinWalled(MdlData& mdlData, const DeviceShaderConfiguration* shaders)
    {
        // IOR value in case the material ior expression is constant.
        bool isThinWalled = shaders->isThinWalled;

        if (0 <= shaders->idxCallThinWalled)
        {
            optixDirectCall<void>(shaders->idxCallThinWalled, &isThinWalled, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
        }

        return isThinWalled;
    }

    __forceinline__ __device__ BsdfSampleResult sampleBsdf(const MdlData& mdlData,const DeviceShaderConfiguration* shaders, math::vec3f& surroundingIor, math::vec3f outgoingDirection, unsigned seed)
    {
        BsdfSampleData data;
        BsdfSampleResult result;
        // If the hit is either on the surface or a thin-walled material,
        // the ray is inside the surrounding material and the material ior is on the other side.
        // When hitting the backface of a non-thin-walled material, 
        // the ray is inside the current material and the surrounding material is on the other side.
        // The material's IOR is the current top-of-stack. We need the one further down!
        data.k1 = outgoingDirection; // == -optixGetWorldRayDirection()
        data.xi = rng4(seed);
        if (mdlData.isFrontFace || mdlData.isThinWalled)
        {
            data.ior1 = surroundingIor;
            data.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
            if (shaders->idxCallSurfaceScatteringSample < 0) {
                result.isValid = false;
                result.eventType = mi::neuraylib::BSDF_EVENT_ABSORB;
                return result;
            }
            optixDirectCall<void>(shaders->idxCallSurfaceScatteringSample, &data, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
        }
        else
        {
            data.ior1.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
            data.ior2 = surroundingIor;
            if (shaders->idxCallBackfaceScatteringSample < 0) {
                result.isValid = false;
                result.eventType = mi::neuraylib::BSDF_EVENT_ABSORB;
                return result;
            }
            optixDirectCall<void>(shaders->idxCallBackfaceScatteringSample, &data, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
        }

        result.isValid = true;
        result.nextDirection = data.k2;
        result.bsdfOverPdf = data.bsdf_over_pdf;
        result.eventType = data.event_type;
        result.pdf = data.pdf;
        return result;

    }

    __forceinline__ __device__ BsdfEvalResult evaluateBsdf(const MdlData& mdlData,const DeviceShaderConfiguration* shaders, math::vec3f& surroundingIor, math::vec3f outgoingDirection, math::vec3f incomingDirection)
    {
        BsdfEvaluateData evalData;
        BsdfEvalResult result;

        evalData.k1 = outgoingDirection;
        evalData.k2 = incomingDirection;

        // If the hit is either on the surface or a thin-walled material,
        // the ray is inside the surrounding material and the material ior is on the other side.
        if (mdlData.isFrontFace || mdlData.isThinWalled)
        {
            evalData.ior1 = surroundingIor; // From surrounding medium ior
            evalData.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;                    // to material ior.
            if (shaders->idxCallSurfaceScatteringEval < 0) {
                result.isValid = false;
                return result;
            }
            optixDirectCall<void>(shaders->idxCallSurfaceScatteringEval, &evalData, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
        }
        else
        {
            // When hitting the backface of a non-thin-walled material, 
            // the ray is inside the current material and the surrounding material is on the other side.
            // The material's IOR is the current top-of-stack. We need the one further down!
            evalData.ior1.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;                    // From material ior 
            evalData.ior2 = surroundingIor; // From surrounding medium ior

            if (shaders->idxCallBackfaceScatteringEval < 0) {
                result.isValid = false;
                return result;
            }
            optixDirectCall<void>(shaders->idxCallBackfaceScatteringEval, &evalData, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
        }

        result.isValid = true;
        result.pdf = evalData.pdf;
        result.diffuse = evalData.bsdf_diffuse;
        result.glossy = evalData.bsdf_glossy;

        return result;

    }

	__forceinline__ __device__ BsdfAuxResult getAuxiliaryData(const MdlData& mdlData,const DeviceShaderConfiguration* shaders, math::vec3f& surroundingIor, math::vec3f outgoingDirection)
    {

        BsdfAuxiliaryData auxData;
        BsdfAuxResult result;
        auxData.k1 = outgoingDirection;
        // If the hit is either on the surface or a thin-walled material,
        // the ray is inside the surrounding material and the material ior is on the other side.
        if (mdlData.isFrontFace || mdlData.isThinWalled)
        {
            auxData.ior1 = surroundingIor; // From surrounding medium ior
            auxData.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;                    // to material ior.
            if (shaders->idxCallSurfaceScatteringAuxiliary < 0) {
                result.isValid = false;
                return result;
            }
        	optixDirectCall<void>(shaders->idxCallSurfaceScatteringAuxiliary, &auxData, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
        }
        else
        {
            // When hitting the backface of a non-thin-walled material, 
            // the ray is inside the current material and the surrounding material is on the other side.
            // The material's IOR is the current top-of-stack. We need the one further down!
            auxData.ior1.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;                    // From material ior 
        	auxData.ior2 = surroundingIor; // From surrounding medium ior
        	if (shaders->idxCallBackfaceScatteringAuxiliary <0) {
        		result.isValid = false;
                return result;
        	}
            optixDirectCall<void>(shaders->idxCallBackfaceScatteringAuxiliary, &auxData, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
        }
        result.isValid = true;
        result.albedo = auxData.albedo;
        result.normal = auxData.normal;
        return result;
    }

    __forceinline__ __device__ EdfResult evaluateEmission(const MdlData& mdlData,const DeviceShaderConfiguration* shaders, math::vec3f& outgoingDirection)
    {
        EdfEvaluateData evalData;
        EdfResult result;

        struct EdfFunctions
        {
	        const int* evalF;
			const int* intensityF;
			const int* intensityModeF;
			const math::vec3f* intensity;
			const int* intensityMode;
        };

        EdfFunctions edfFunctions;
        if (mdlData.isFrontFace)
        {
            edfFunctions.evalF = &shaders->idxCallSurfaceEmissionEval;
            if (*edfFunctions.evalF < 0)
            {
                result.isValid = false;
                return result;
            }
            edfFunctions.intensityF = &shaders->idxCallSurfaceEmissionIntensity;
            edfFunctions.intensityModeF = &shaders->idxCallSurfaceEmissionIntensityMode;
            edfFunctions.intensity = &shaders->surfaceIntensity;
            edfFunctions.intensityMode = &shaders->surfaceIntensityMode;
        }
		else if (mdlData.isThinWalled)
		{
            // MDL Specs: There is no emission on the back-side unless an EDF is specified with the backface field and thin_walled is set to true.
            edfFunctions.evalF = &shaders->idxCallBackfaceEmissionEval;
            if (*edfFunctions.evalF < 0)
            {
            	result.isValid = false;
				return result;
			}
			edfFunctions.intensityF = &shaders->idxCallBackfaceEmissionIntensity;
			edfFunctions.intensityModeF = &shaders->idxCallBackfaceEmissionIntensityMode;
			edfFunctions.intensity = &shaders->backfaceIntensity;
			edfFunctions.intensityMode = &shaders->backfaceIntensityMode;
		}
        else
        {
            result.isValid = false;
            return result;
        }
        

        evalData.k1 = outgoingDirection; // input: outgoing direction (-ray.direction)

        result.intensity = *edfFunctions.intensity;
        result.mode = *edfFunctions.intensityMode;

        if (0 <= *edfFunctions.intensityF) // Emission intensity is not a constant.
        {
            optixDirectCall<void>(*edfFunctions.intensityF, &result.intensity, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
        }

        if (result.intensity == math::vec3f(0.0f)) {
            result.isValid = false;
            return result;
        }

        optixDirectCall<void>(*edfFunctions.evalF, &evalData, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);

        if (0 <= *edfFunctions.intensityModeF) // Emission intensity mode is not a constant.
        {
            optixDirectCall<void>(*edfFunctions.intensityModeF, &result.mode, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
        }
        result.isValid = true;
        result.cos = evalData.cos;
        result.edf = evalData.edf;
        result.pdf = evalData.pdf;
        
        return result;
    }

    __forceinline__ __device__ MdlData directMdlInit(HitProperties& hitP)
    {
        MdlData mdlData;
        mdlData.state.normal = hitP.nsW;
        mdlData.state.geom_normal = hitP.ngW;
        mdlData.state.position = hitP.position;
        mdlData.state.animation_time = 0.0f; // This renderer implements no support for animations.
        mdlData.state.text_coords = hitP.textureCoordinates;
        mdlData.state.tangent_u = hitP.textureBitangents;
        mdlData.state.tangent_v = hitP.textureTangents;
        float4 texture_results[16]; //TODO add macro
        mdlData.state.text_results = texture_results;
        mdlData.state.ro_data_segment = nullptr;
        mdlData.state.world_to_object = hitP.wToF4;
        mdlData.state.object_to_world = hitP.oTwF4;
        mdlData.state.object_id = hitP.instance->instanceId; // idObject, this is the sg::Instance node ID.
        mdlData.state.meters_per_scene_unit = 1.0f;

        mdlData.resourceData.shared_data = nullptr;
        mdlData.resourceData.texture_handler = hitP.shader->textureHandler;
        //mdlData.shaderConfiguration = hitP.shader->shaderConfiguration;
        mdlData.isFrontFace = hitP.isFrontFace;
        mdlData.argBlock = hitP.material->argBlock;

        // Using a single material init function instead of per distribution init functions.
        // This is always present, even if it just returns.
        optixDirectCall<void>(hitP.shaderConfiguration->idxCallInit, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);

        return mdlData;
    }

    __forceinline__ __device__ MaterialEvaluation evaluateMdlMaterial(MdlRequest* request)
    {
        MaterialEvaluation result;
        MdlData mdlData = directMdlInit(*request->hitP);

        HitProperties& hitP = *request->hitP;
        if (request->ior)
        {
            result.ior = getIor(mdlData, hitP.shaderConfiguration);
        }
        if (request->opacity)
        {
            result.opacity = determineOpacity(mdlData, hitP.shaderConfiguration);
        }
        if (request->edf || request->bsdfEvaluation || request->bsdfSample)
        {
            mdlData.isThinWalled = isThinWalled(mdlData, hitP.shaderConfiguration);
        }
        if (request->edf)
        {
            result.edf = evaluateEmission(mdlData, hitP.shaderConfiguration, request->lastRayDirection);
        }
        if (request->bsdfEvaluation)
        {
            result.bsdfEvaluation = evaluateBsdf(mdlData, hitP.shaderConfiguration, request->surroundingIor, request->lastRayDirection, request->toSampledLight);
        }
        if (request->bsdfSample)
        {
            result.bsdfSample = sampleBsdf(mdlData, hitP.shaderConfiguration, request->surroundingIor, request->lastRayDirection, request->hitP->seed);
        }
        if (request->auxiliary)
        {
            result.aux = getAuxiliaryData(mdlData, hitP.shaderConfiguration, request->surroundingIor, request->lastRayDirection);
        }

        return result;
    }
}

#endif
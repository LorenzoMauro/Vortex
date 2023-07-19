#pragma once
#ifndef MDL_WRAPPER_DEVICE_H
#define MDL_WRAPPER_DEVICE_H
#include "Core/Math.h"
#include "Device/DevicePrograms/RayData.h"
#include "InlineMdlDeclarations.h"
#include "Device/DevicePrograms/randomNumberGenerator.h"

namespace vtx::mdl {
    // Importance sample the BSDF. 
    __forceinline__ __device__ BsdfSampleResult sampleBsdf(const MdlData* mdlData, const math::vec3f& surroundingIor, const math::vec3f& outgoingDirection, unsigned& seed)
    {
        BsdfSampleData data;
        // If the hit is either on the surface or a thin-walled material,
        // the ray is inside the surrounding material and the material ior is on the other side.
        // When hitting the backface of a non-thin-walled material, 
        // the ray is inside the current material and the surrounding material is on the other side.
        // The material's IOR is the current top-of-stack. We need the one further down!
        data.k1 = outgoingDirection; // == -optixGetWorldRayDirection()
        data.xi = rng4(seed);
        data.pdf = 1.0f;

        if (mdlData->isFrontFace || mdlData->isThinWalled)
        {
            //printf("Evaluating Frontface\n");
            data.ior1 = surroundingIor;
            data.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
        }
        else
        {
            //printf("Evaluating Backface\n");
            data.ior1.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
            data.ior2 = surroundingIor;
        }

        if (mdlData->isFrontFace || !mdlData->isThinWalled)
        {
            frontBsdf_sample(&data, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
        }
        else
        {
            backBsdf_sample(&data, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
        }

        BsdfSampleResult result;
        result.isValid = true;
        result.isComputed = true;
        result.nextDirection = data.k2;
        result.nextDirection = math::normalize(result.nextDirection);
        result.bsdfOverPdf = data.bsdf_over_pdf;
        result.eventType = data.event_type;
        result.pdf = data.pdf;

        return result;
    }

    __forceinline__ __device__ BsdfEvalResult evaluateBsdf(const MdlData* mdlData, const math::vec3f& surroundingIor, const math::vec3f& outgoingDirection, const math::vec3f& incomingDirection)
    {
        BsdfEvaluateData evalData;

        evalData.k1 = outgoingDirection;
        evalData.k2 = incomingDirection;
        evalData.bsdf_diffuse = math::vec3f(0.0f);
        evalData.bsdf_glossy = math::vec3f(0.0f);

        // If the hit is either on the surface or a thin-walled material,
        // the ray is inside the surrounding material and the material ior is on the other side.
        if (mdlData->isFrontFace || mdlData->isThinWalled)
        {
            evalData.ior1 = surroundingIor; // From surrounding medium ior
            evalData.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;                    // to material ior.
        }
        else
        {
            // When hitting the backface of a non-thin-walled material, 
            // the ray is inside the current material and the surrounding material is on the other side.
            // The material's IOR is the current top-of-stack. We need the one further down!
            evalData.ior1.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;                    // From material ior 
            evalData.ior2 = surroundingIor; // From surrounding medium ior
        }

        if(mdlData->isFrontFace || !mdlData->isThinWalled)
        {
            frontBsdf_evaluate(&evalData, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
        }
        else
        {
            backBsdf_evaluate(&evalData, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
        }

        BsdfEvalResult result;
        result.isValid = true;
        result.pdf = evalData.pdf;
        result.diffuse = evalData.bsdf_diffuse;
        result.glossy = evalData.bsdf_glossy;

        return result;

    }

    __forceinline__ __device__ BsdfAuxResult auxiliaryBsdf(const MdlData* mdlData, const math::vec3f& surroundingIor, const math::vec3f& outgoingDirection)
    {
        BsdfAuxiliaryData auxData;

        auxData.k1 = outgoingDirection;

        // If the hit is either on the surface or a thin-walled material,
        // the ray is inside the surrounding material and the material ior is on the other side.
        if (mdlData->isFrontFace || mdlData->isThinWalled)
        {
            auxData.ior1 = surroundingIor; // From surrounding medium ior
            auxData.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;                    // to material ior.
        }
        else
        {
            // When hitting the backface of a non-thin-walled material, 
            // the ray is inside the current material and the surrounding material is on the other side.
            // The material's IOR is the current top-of-stack. We need the one further down!
            auxData.ior1.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;                    // From material ior 
            auxData.ior2 = surroundingIor; // From surrounding medium ior
        }


        if (mdlData->isFrontFace || !mdlData->isThinWalled)
        {
            frontBsdf_auxiliary(&auxData, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
        }
        else
        {
            backBsdf_auxiliary(&auxData, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
        }

        BsdfAuxResult result;
        result.isValid = true;
        result.albedo = auxData.albedo;
        result.normal = auxData.normal;
        return result;
    }

    __forceinline__ __device__ EdfResult evaluateEmission(const MdlData* mdlData, const math::vec3f& outgoingDirection)
    {
        EdfEvaluateData evalData;
        EdfResult result;

        float3 emissionIntensity;
        emissionIntensity.x = 0.0f;
        emissionIntensity.y = 0.0f;
        emissionIntensity.z = 0.0f;

        // Emission 
        if (mdlData->isFrontFace || !mdlData->isThinWalled)
        {
            frontEdfIntensity(&emissionIntensity, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);

            result.intensity = emissionIntensity;
            //printMath("Front Intensity", emissionIntensity);

            if (result.intensity == math::vec3f(0.0f)) {
                result.isValid = false;
                return result;
            }

            evalData.k1 = outgoingDirection; // input: outgoing direction (-ray.direction)
            frontEdfMode(&result.mode, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
            frontEdf_evaluate(&evalData, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
        }
        else
        {
            backEdfIntensity(&emissionIntensity, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
            result.intensity = emissionIntensity;


            if (result.intensity == math::vec3f(0.0f)) {
                result.isValid = false;
                return result;
            }
            evalData.k1 = outgoingDirection; // input: outgoing direction (-ray.direction)
            backEdfMode(&result.mode, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
            backEdf_evaluate(&evalData, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
        }

        // Check if the hit geometry contains any emission.
        result.isValid = true;
        result.cos = evalData.cos;
        result.edf = evalData.edf;
        result.pdf = evalData.pdf;
        return result;
    }

    __forceinline__ __device__ MdlData mdlInit(const MdlRequest* request)
    {
        float4 oTwF4[3];
        float4 wToF4[3];
        float3 textureCoordinates[2];
        float3 textureBitangents[2];
        float3 textureTangents[2];
        float4 textureResults[16]; //TODO add macro
        MdlData mdlData;

        const math::affine3f& objectToWorld = *request->hitProperties->oTw;
        const math::affine3f& worldToObject = math::affine3f(objectToWorld.l.inverse(), objectToWorld.p);
        objectToWorld.toFloat4(oTwF4);
        worldToObject.toFloat4(wToF4);

        textureCoordinates[0] = request->hitProperties->uv;
        textureBitangents[0] = request->hitProperties->bitangent;
        textureTangents[0] = request->hitProperties->tangent;

        textureCoordinates[1] = textureCoordinates[0];
        textureBitangents[1] = textureBitangents[0];
        textureTangents[1] = textureTangents[0];

        mdlData.state.normal = request->hitProperties->shadingNormal;
        mdlData.state.geom_normal = request->hitProperties->trueNormal;
        mdlData.state.position = request->hitProperties->position;
        mdlData.state.animation_time = 0.0f;
        mdlData.state.text_coords = textureCoordinates;
        mdlData.state.tangent_u = textureBitangents;
        mdlData.state.tangent_v = textureTangents;
        mdlData.state.text_results = textureResults;
        mdlData.state.ro_data_segment = nullptr;
        mdlData.state.world_to_object = wToF4;
        mdlData.state.object_to_world = oTwF4;
        mdlData.state.object_id = request->hitProperties->instanceId;
        mdlData.state.meters_per_scene_unit = 1.0f;

        mdlData.resourceData.shared_data = nullptr;
        mdlData.resourceData.texture_handler = request->hitProperties->textureHandler;
        mdlData.argBlock = request->hitProperties->argBlock;
        mdlData.isFrontFace = request->hitProperties->isFrontFace;

        init(&mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
        
        return mdlData;
    }
}

#endif
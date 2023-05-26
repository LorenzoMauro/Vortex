#pragma once
#ifndef MDL_WRAPPER_DEVICE_H
#define MDL_WRAPPER_DEVICE_H
#include "Core/Math.h"
#include "Device/DevicePrograms/RayData.h"
#include "InlineMdlDeclarations.h"
#include "Device/DevicePrograms/randomNumberGenerator.h"

namespace vtx::mdl {
    // Importance sample the BSDF. 
    __forceinline__ __device__ BsdfSampleResult sampleBsdf(MdlData* mdlData, math::vec3f& surroundingIor, math::vec3f& outgoingDirection, unsigned seed)
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

        // I've noticed that if the mdl doesn't contain df_specular but roughness is not zero the eventType will always be glossy even for
        // specular paths (pdf becomes a large floating number)
        if(data.pdf > 1.0f || data.pdf < 0.0f && (data.event_type & mi::neuraylib::BSDF_EVENT_ABSORB)==0)
        {
			data.pdf = 1.0f;
            if((data.event_type & mi::neuraylib::BSDF_EVENT_REFLECTION)!=0)
            {
                data.event_type = mi::neuraylib::BSDF_EVENT_SPECULAR_REFLECTION;
            }
            else if ((data.event_type & mi::neuraylib::BSDF_EVENT_TRANSMISSION) != 0)
            {
                data.event_type = mi::neuraylib::BSDF_EVENT_SPECULAR_TRANSMISSION;
            }
		}

        BsdfSampleResult result;
        result.isValid = true;
        result.isComputed = true;
        result.nextDirection = data.k2;
        result.bsdfOverPdf = data.bsdf_over_pdf;
        result.eventType = data.event_type;
        result.pdf = data.pdf;

        //switch(result.eventType)
        //{
		//case mi::neuraylib::BSDF_EVENT_ABSORB:
        //    printf("Event Type : BSDF_EVENT_ABSORB\n");
		//break;
        //case mi::neuraylib::BSDF_EVENT_DIFFUSE:
        //    printf("Event Type : BSDF_EVENT_DIFFUSE\n"); 
        //           break;
        //case mi::neuraylib::BSDF_EVENT_GLOSSY:
        //    printf("Event Type : BSDF_EVENT_GLOSSY\n"); 
        //           break;
        //case mi::neuraylib::BSDF_EVENT_SPECULAR:
        //    printf("Event Type : BSDF_EVENT_SPECULAR\n");
        //           break;
        //case mi::neuraylib::BSDF_EVENT_REFLECTION:
        //    printf("Event Type : BSDF_EVENT_REFLECTION\n"); 
        //           break;
        //case mi::neuraylib::BSDF_EVENT_TRANSMISSION:
        //    printf("Event Type : BSDF_EVENT_TRANSMISSION\n"); 
        //           break;
        //case mi::neuraylib::BSDF_EVENT_DIFFUSE_REFLECTION:
        //    printf("Event Type : BSDF_EVENT_DIFFUSE_REFLECTION\n");
        //           break;
        //case mi::neuraylib::BSDF_EVENT_DIFFUSE_TRANSMISSION:
        //    printf("Event Type : BSDF_EVENT_DIFFUSE_TRANSMISSION\n"); 
        //           break;
        //case mi::neuraylib::BSDF_EVENT_GLOSSY_REFLECTION:
        //    printf("Event Type : BSDF_EVENT_GLOSSY_REFLECTION\n"); 
        //           break;
        //case mi::neuraylib::BSDF_EVENT_GLOSSY_TRANSMISSION:
        //    printf("Event Type : BSDF_EVENT_GLOSSY_TRANSMISSION\n"); 
        //           break;
        //case mi::neuraylib::BSDF_EVENT_SPECULAR_REFLECTION:
        //    printf("Event Type : BSDF_EVENT_SPECULAR_REFLECTION\n"); 
        //           break;
        //case mi::neuraylib::BSDF_EVENT_SPECULAR_TRANSMISSION:
        //    printf("Event Type : BSDF_EVENT_SPECULAR_TRANSMISSION\n"); 
        //           break;
        //case mi::neuraylib::BSDF_EVENT_FORCE_32_BIT:
        //    printf("Event Type : BSDF_EVENT_FORCE_32_BIT\n"); 
        //    break;
		//}
        /*printf("Sampled BSDF: %f %f %f\n"
               "BSDF over PDF:  %f %f %f\n"
               "PDF: %f\n",
               result.bsdfOverPdf.x, result.bsdfOverPdf.y, result.bsdfOverPdf.z,
               result.nextDirection.x, result.nextDirection.y, result.nextDirection.z,
               result.pdf);*/

        //result.print("Inside:\n");
        return result;
    }

    __forceinline__ __device__ BsdfEvalResult evaluateBsdf(MdlData* mdlData, math::vec3f& surroundingIor, math::vec3f& outgoingDirection, math::vec3f& incomingDirection)
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

    __forceinline__ __device__ BsdfAuxResult auxiliaryBsdf(MdlData* mdlData, math::vec3f& surroundingIor, math::vec3f& outgoingDirection)
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

    __forceinline__ __device__ EdfResult evaluateEmission(MdlData* mdlData, math::vec3f& outgoingDirection)
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

    __forceinline__ __device__ MdlData mdlInit(HitProperties* hitP)
    {
        MdlData mdlData;

        mdlData.state.normal = hitP->nsW;
        mdlData.state.geom_normal = hitP->ngW;
        mdlData.state.position = hitP->position;
        mdlData.state.animation_time = 0.0f;
        mdlData.state.text_coords = hitP->textureCoordinates;
        mdlData.state.tangent_u = hitP->textureBitangents;
        mdlData.state.tangent_v = hitP->textureTangents;
        float4 texture_results[16]; //TODO add macro
        mdlData.state.text_results = texture_results;
        mdlData.state.ro_data_segment = nullptr;
        mdlData.state.world_to_object = hitP->wToF4;
        mdlData.state.object_to_world = hitP->oTwF4;
        mdlData.state.object_id = hitP->instance->instanceId;
        mdlData.state.meters_per_scene_unit = 1.0f;

        mdlData.resourceData.shared_data = nullptr;
        mdlData.resourceData.texture_handler = hitP->material->textureHandler;
        mdlData.argBlock = hitP->material->argBlock;
        mdlData.isFrontFace = hitP->isFrontFace;

        init(&mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
        return mdlData;
    }
}

#endif
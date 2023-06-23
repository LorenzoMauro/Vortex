#pragma once
#ifndef MDL_WRAPPER_DEVICE_H
#define MDL_WRAPPER_DEVICE_H
#include "Core/Math.h"
#include "Device/DevicePrograms/RayData.h"
#include "InlineMdlDeclarations.h"
#include "Device/DevicePrograms/randomNumberGenerator.h"

namespace vtx::mdl {
    // Importance sample the BSDF. 
    __forceinline__ __device__ BsdfSampleResult sampleBsdf(MdlData* mdlData, math::vec3f& surroundingIor, math::vec3f& outgoingDirection, unsigned& seed)
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

    __forceinline__ __device__ MdlData mdlInit(MdlRequest* request, MaterialEvaluation* matEval)
    {
        float4 oTwF4[3];
        float4 wToF4[3];
        float3 textureCoordinates[2];
        float3 textureBitangents[2];
        float3 textureTangents[2];
        float4 textureResults[16]; //TODO add macro
        MdlData mdlData;
        const math::vec3ui  triVerticesIndices = reinterpret_cast<math::vec3ui*>(request->geometry->indicesData)[request->triangleId];
        graph::VertexAttributes* vertices[3]{ nullptr, nullptr, nullptr };
        vertices[0] = &(request->geometry->vertexAttributeData[triVerticesIndices.x]);
        vertices[1] = &(request->geometry->vertexAttributeData[triVerticesIndices.y]);
        vertices[2] = &(request->geometry->vertexAttributeData[triVerticesIndices.z]);
        math::vec3f ngO = math::normalize(cross(vertices[1]->position - vertices[0]->position, vertices[2]->position - vertices[0]->position));

        math::vec3f nsO = math::normalize(vertices[0]->normal * request->baricenter.x + vertices[1]->normal * request->baricenter.y + vertices[2]->normal * request->baricenter.z);
        math::vec3f tgO = math::normalize(vertices[0]->tangent * request->baricenter.x + vertices[1]->tangent * request->baricenter.y + vertices[2]->tangent * request->baricenter.z);
        math::vec3f btO = math::normalize(vertices[0]->bitangent * request->baricenter.x + vertices[1]->bitangent * request->baricenter.y + vertices[2]->bitangent * request->baricenter.z);

        if (dot(ngO, nsO) < 0.0f) // make sure that shading and geometry normal agree on sideness
        {
            ngO = -ngO;
        }

        const math::affine3f& objectToWorld = request->instance->transform;
        const math::affine3f& worldToObject = math::affine3f(objectToWorld.l.inverse(), objectToWorld.p);
        
        objectToWorld.toFloat4(oTwF4);
        worldToObject.toFloat4(wToF4);
        // TODO we already have the inverse so there can be some OPTIMIZATION here
        math::vec3f nsW = math::normalize(math::transformNormal3F(objectToWorld, nsO));
        math::vec3f ngW = math::normalize(math::transformNormal3F(objectToWorld, ngO));
        math::vec3f tgW = math::normalize(math::transformVector3F(objectToWorld, tgO));
        math::vec3f btW = math::normalize(math::transformVector3F(objectToWorld, btO));

        // Calculate an ortho-normal system respective to the shading normal.
        // Expanding the TBN tbn(tg, ns) constructor because TBN members can't be used as pointers for the Mdl_state with NUM_TEXTURE_SPACES > 1.
        btW = math::normalize(cross(nsW, tgW));
        tgW = cross(btW, nsW); // Now the tangent is orthogonal to the shading normal.

        textureCoordinates[0] = vertices[0]->texCoord * request->baricenter.x + vertices[1]->texCoord * request->baricenter.y + vertices[2]->texCoord * request->baricenter.z;
        textureBitangents[0] = btW;
        textureTangents[0] = tgW;

        textureCoordinates[1] = textureCoordinates[0];
        textureBitangents[1] = btW;
        textureTangents[1] = tgW;

        // Explicitly include edge-on cases as frontface condition!
        bool isFrontFace = 0.0f <= dot(request->outgoingDirection, ngW);
        matEval->isFrontFace = isFrontFace;
        matEval->trueNormal = ngW;
        matEval->tangent = tgW;
        matEval->uv = textureCoordinates[0];

        mdlData.state.normal = nsW;
        mdlData.state.geom_normal = ngW;
        mdlData.state.position = request->position;
        mdlData.state.animation_time = 0.0f;
        mdlData.state.text_coords = textureCoordinates;
        mdlData.state.tangent_u = textureBitangents;
        mdlData.state.tangent_v = textureTangents;
        mdlData.state.text_results = textureResults;
        mdlData.state.ro_data_segment = nullptr;
        mdlData.state.world_to_object = wToF4;
        mdlData.state.object_to_world = oTwF4;
        mdlData.state.object_id = request->instance->instanceId;
        mdlData.state.meters_per_scene_unit = 1.0f;

        mdlData.resourceData.shared_data = nullptr;
        mdlData.resourceData.texture_handler = request->textureHandler;
        mdlData.argBlock = request->argBlock;
        mdlData.isFrontFace = isFrontFace;

        init(&mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
        
        return mdlData;
    }
}

#endif
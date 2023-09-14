#ifndef HIT_PROPERTIES_H
#define HIT_PROPERTIES_H
#pragma once
#include "LaunchParams.h"
#include "Utils.h"
#include "Core/Math.h"

namespace vtx
{
	struct HitProperties
    {
        bool isInit = false;
        unsigned instanceId;
        unsigned triangleId;
        math::vec3f position;
        math::vec3f baricenter;
        math::vec3f trueNormal;
        math::vec3f shadingNormal;
        math::vec3f uv;
        math::vec3f bitangent;
        math::vec3f tangent;
        math::affine3f* oTw;
        bool isFrontFace;

        // Material Related data
        bool hasMaterial = false;
        int programCall;
        bool hasEdf = false;
        bool hasOpacity = false;
        char* argBlock = nullptr;
        TextureHandler* textureHandler = nullptr;
        LightData* lightData = nullptr;

        __forceinline__ __device__ void init(const unsigned& _instanceId, const unsigned& _triangleId, const math::vec3f& _baricenter, const math::vec3f& _position)
        {
            instanceId = _instanceId;
            triangleId = _triangleId;
            baricenter = _baricenter;
            position = _position;
            isInit = true;
        }

        __forceinline__ __device__ void calculateForMeshLightSampling(const LaunchParams* params, const math::vec3f& rayOrigin, math::vec3f* outgoingDirection, float* distance)
        {
            InstanceData* instance = params->instances[instanceId];
            const GeometryData* geometry = instance->geometryData;
            const math::vec3ui       triVerticesIndices = reinterpret_cast<math::vec3ui*>(geometry->indicesData)[triangleId];
            graph::VertexAttributes* vertices[3]{ nullptr, nullptr, nullptr };
            vertices[0] = &(geometry->vertexAttributeData[triVerticesIndices.x]);
            vertices[1] = &(geometry->vertexAttributeData[triVerticesIndices.y]);
            vertices[2] = &(geometry->vertexAttributeData[triVerticesIndices.z]);

            math::vec3f ngO = math::normalize(cross(vertices[1]->position - vertices[0]->position, vertices[2]->position - vertices[0]->position));
            const math::vec3f nsO = math::normalize(vertices[0]->normal * baricenter.x + vertices[1]->normal * baricenter.y + vertices[2]->normal * baricenter.z);
            const math::vec3f tgO = math::normalize(vertices[0]->tangent * baricenter.x + vertices[1]->tangent * baricenter.y + vertices[2]->tangent * baricenter.z);
            const math::vec3f btO = math::normalize(vertices[0]->bitangent * baricenter.x + vertices[1]->bitangent * baricenter.y + vertices[2]->bitangent * baricenter.z);
            uv = vertices[0]->texCoord * baricenter.x + vertices[1]->texCoord * baricenter.y + vertices[2]->texCoord * baricenter.z;

            if (dot(ngO, nsO) < 0.0f) // make sure that shading and geometry normal agree on sideness
            {
                ngO = -ngO;
            }

            oTw = &(instance->transform);
            shadingNormal = math::normalize(math::transformNormal3F(*oTw, nsO));
            trueNormal = math::normalize(math::transformNormal3F(*oTw, ngO));
            tangent = math::normalize(math::transformVector3F(*oTw, tgO));
            bitangent = math::normalize(math::transformVector3F(*oTw, btO));

            position = vertices[0]->position * baricenter.x + vertices[1]->position * baricenter.y + vertices[2]->position * baricenter.z;
            position = math::transformPoint3F(*oTw, position);
            *outgoingDirection = position - rayOrigin;
            *distance = math::length(*outgoingDirection);
            *outgoingDirection = *outgoingDirection / (*distance + EPS);

            // Calculate an ortho-normal system respective to the shading normal.
            // Expanding the TBN tbn(tg, ns) constructor because TBN members can't be used as pointers for the Mdl_state with NUM_TEXTURE_SPACES > 1.
            bitangent = math::normalize(cross(shadingNormal, tangent));
            if(utl::isNan(bitangent))
            {
                printf(
                    "Nan Bitangent\n"
                    "Shading Normal : %f %f %f\n"
                    "Tangent : %f %f %f\n\n",
                    shadingNormal.x, shadingNormal.y, shadingNormal.z,
                    tangent.x, tangent.y, tangent.z
                );
            }
            tangent = cross(bitangent, shadingNormal); // Now the tangent is orthogonal to the shading normal.

            // Explicitly include edge-on cases as frontface condition!
            isFrontFace = 0.0f <= dot(*outgoingDirection, trueNormal);
        }
        __forceinline__ __device__ void calculate(const LaunchParams* params, const math::vec3f& outgoingDirection)
        {
            InstanceData* instance = params->instances[instanceId];
            const GeometryData* geometry = instance->geometryData;
            const math::vec3ui       triVerticesIndices = reinterpret_cast<math::vec3ui*>(geometry->indicesData)[triangleId];
            graph::VertexAttributes* vertices[3]{ nullptr, nullptr, nullptr };
            vertices[0] = &(geometry->vertexAttributeData[triVerticesIndices.x]);
            vertices[1] = &(geometry->vertexAttributeData[triVerticesIndices.y]);
            vertices[2] = &(geometry->vertexAttributeData[triVerticesIndices.z]);



            math::vec3f ngO = math::normalize(cross(vertices[1]->position - vertices[0]->position, vertices[2]->position - vertices[0]->position));
            const math::vec3f nsO = math::normalize(vertices[0]->normal * baricenter.x + vertices[1]->normal * baricenter.y + vertices[2]->normal * baricenter.z);
            const math::vec3f tgO = math::normalize(vertices[0]->tangent * baricenter.x + vertices[1]->tangent * baricenter.y + vertices[2]->tangent * baricenter.z);
            const math::vec3f btO = math::normalize(vertices[0]->bitangent * baricenter.x + vertices[1]->bitangent * baricenter.y + vertices[2]->bitangent * baricenter.z);
            uv = vertices[0]->texCoord * baricenter.x + vertices[1]->texCoord * baricenter.y + vertices[2]->texCoord * baricenter.z;

            if (dot(ngO, nsO) < 0.0f) // make sure that shading and geometry normal agree on sideness
            {
                ngO = -ngO;
            }

            oTw = &(instance->transform);
            shadingNormal = math::normalize(math::transformNormal3F(*oTw, nsO));
            trueNormal = math::normalize(math::transformNormal3F(*oTw, ngO));
            tangent = math::normalize(math::transformVector3F(*oTw, tgO));
            bitangent = math::normalize(math::transformVector3F(*oTw, btO));

            // Calculate an ortho-normal system respective to the shading normal.
            // Expanding the TBN tbn(tg, ns) constructor because TBN members can't be used as pointers for the Mdl_state with NUM_TEXTURE_SPACES > 1.
            bitangent = math::normalize(cross(shadingNormal, tangent));
            if (utl::isNan(bitangent))
            {
                printf(
                    "Nan Bitangent\n"
                    "Shading Normal : %f %f %f\n"
                    "Tangent : %f %f %f\n\n"
                    "tgO: %f %f %f\n"
                    "Tangent Vertex 0: %f %f %f\n"
                    "Tanget Vertex 1: %f %f %f\n"
                    "Tanget Vertex 2: %f %f %f\n"
                    ,
                    shadingNormal.x, shadingNormal.y, shadingNormal.z,
                    tangent.x, tangent.y, tangent.z,
                    tgO.x, tgO.y, tgO.z,
                    vertices[0]->tangent.x, vertices[0]->tangent.y, vertices[0]->tangent.z,
                    vertices[1]->tangent.x, vertices[1]->tangent.y, vertices[1]->tangent.z,
                    vertices[2]->tangent.x, vertices[2]->tangent.y, vertices[2]->tangent.z
                );
            }

            tangent = cross(bitangent, shadingNormal); // Now the tangent is orthogonal to the shading normal.

            // Explicitly include edge-on cases as frontface condition!
            isFrontFace = 0.0f <= dot(-outgoingDirection, trueNormal);

            determineMaterialInfo(params);
        }
        __forceinline__ __device__ void calculateForInferenceQuery(const LaunchParams* params)
        {
            const InstanceData* instance = params->instances[instanceId];
            const GeometryData* geometry = instance->geometryData;
            const math::affine3f& objectToWorld = instance->transform;
            const math::vec3ui       triVerticesIndices = reinterpret_cast<math::vec3ui*>(geometry->indicesData)[triangleId];
            graph::VertexAttributes* vertices[3]{ nullptr, nullptr, nullptr };
            vertices[0] = &(geometry->vertexAttributeData[triVerticesIndices.x]);
            vertices[1] = &(geometry->vertexAttributeData[triVerticesIndices.y]);
            vertices[2] = &(geometry->vertexAttributeData[triVerticesIndices.z]);
            const math::vec3f nsO = math::normalize(vertices[0]->normal * baricenter.x + vertices[1]->normal * baricenter.y + vertices[2]->normal * baricenter.z);
            shadingNormal = math::normalize(math::transformNormal3F(objectToWorld, nsO));

        }
        __forceinline__ __device__ void determineMaterialInfo(const LaunchParams* params)
        {
            const InstanceData* instance = params->instances[instanceId];
            const GeometryData* geometry = (instance->geometryData);
            const unsigned& materialSlotIndex = geometry->faceAttributeData[triangleId].materialSlotId;
            const InstanceData::SlotIds slotIds = instance->materialSlots[materialSlotIndex];
            if (slotIds.material != nullptr)
            {
                textureHandler = slotIds.material->textureHandler;
                argBlock = slotIds.material->argBlock;

#ifdef ARCHITECTURE_OPTIX
                programCall = slotIds.material->materialConfiguration->idxCallEvaluateMaterialWavefront;
#else
                programCall = slotIds.material->materialConfiguration->idxCallEvaluateMaterialWavefrontCuda;
#endif

                if (slotIds.meshLight != nullptr)
                {
                    hasEdf = true;
                    lightData = instance->materialSlots[materialSlotIndex].meshLight;
                }
                if (instance->hasOpacity)
                {
                    hasOpacity = true;
                }
                hasMaterial = true;
            }
            else
            {
                hasMaterial = false;
            }
        }
    };
}

#endif
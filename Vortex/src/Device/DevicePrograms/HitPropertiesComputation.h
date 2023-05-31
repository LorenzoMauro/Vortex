#pragma once

#include "Utils.h"
#include <optix_device.h>

#include "DataFetcher.h"
#include "LaunchParams.h"
#include "RayData.h"
#include <optix_device.h>

namespace vtx::utl
{

	__forceinline__ __device__ void getInstanceAndGeometry(HitProperties* hitP, const vtxID& instanceId)
	{
		hitP->instance = optixLaunchParams.instances[instanceId];
		hitP->geometry = (hitP->instance->geometryData);
	}

	__forceinline__ __device__ void getVertices(HitProperties* hitP, const unsigned int triangleId)
	{
		if (hitP->geometry == nullptr)
		{
			CUDA_ERROR_PRINT("Trying to access geometry in hit properties getVertices Function but geometry is null! You need to call getInstanceAndGeometry First")
		}
		const math::vec3ui  triVerticesIndices = reinterpret_cast<math::vec3ui*>(hitP->geometry->indicesData)[triangleId];

		hitP->vertices[0] = &(hitP->geometry->vertexAttributeData[triVerticesIndices.x]);
		hitP->vertices[1] = &(hitP->geometry->vertexAttributeData[triVerticesIndices.y]);
		hitP->vertices[2] = &(hitP->geometry->vertexAttributeData[triVerticesIndices.z]);
	}

	__forceinline__ __device__ void computeHit(HitProperties* hitP, const math::vec3f& originPosition)
	{
		if (hitP->vertices[0] == nullptr)
		{
			CUDA_ERROR_PRINT("Trying to access vertices in hit properties computeHit Function but vertices is null! You need to call getVertices First")
		}
		// Object space vertex attributes at the hit point.
		hitP->position = hitP->vertices[0]->position * hitP->baricenter.x + hitP->vertices[1]->position * hitP->baricenter.y + hitP->vertices[2]->position * hitP->baricenter.z;
		hitP->position = math::transformPoint3F(hitP->objectToWorld, hitP->position);
		hitP->direction = originPosition - hitP->position;
		hitP->distance = math::length<float>(hitP->direction);
		if (hitP->distance < DENOMINATOR_EPSILON)
		{
			return;
		}
		hitP->direction /= hitP->distance;
	}

	__forceinline__ __device__ void fetchTransformsFromHandle(HitProperties* hitP)
	{
		const OptixTraversableHandle handle = optixGetTransformListHandle(0);
		// UNSURE IF THIS IS CORRECT! WE ALWAYS HAVE THE TRANSFORM FROM THE INSTANCE DATA IN CASE
		const float4* wTo = optixGetInstanceInverseTransformFromHandle(handle);
		const float4* oTw = optixGetInstanceTransformFromHandle(handle);

		hitP->objectToWorld = math::affine3f(oTw);
		hitP->worldToObject = math::affine3f(wTo);

		hitP->objectToWorld.toFloat4(hitP->oTwF4);
		hitP->worldToObject.toFloat4(hitP->wToF4);
	}

	__forceinline__ __device__ void fetchTransformsFromInstance(HitProperties* hitP)
	{
		hitP->objectToWorld = hitP->instance->transform;
		hitP->worldToObject = math::affine3f(hitP->objectToWorld.l.inverse(), hitP->objectToWorld.p);
		hitP->objectToWorld.toFloat4(hitP->oTwF4);
		hitP->worldToObject.toFloat4(hitP->wToF4);
	}

	__forceinline__ __device__ void computeGeometricHitProperties(HitProperties* hitP, const unsigned int triangleId, const bool useInstanceData = false)
	{
		if (hitP->geometry == nullptr)
		{
			CUDA_ERROR_PRINT("Trying to access geometry in hit properties getVertices Function but geometry is null! You need to call getInstanceAndGeometry First")
		}
		if (hitP->vertices[0] == nullptr)
		{
			CUDA_ERROR_PRINT("Trying to access vertices in hit properties computeHit Function but vertices is null! You need to call getVertices First")
		}
		//hitP->ngO = hitP->geometry->faceAttributeData[triangleId].normal;
		hitP->ngO = math::normalize(cross(hitP->vertices[1]->position - hitP->vertices[0]->position, hitP->vertices[2]->position - hitP->vertices[0]->position));

		hitP->nsO = math::normalize(hitP->vertices[0]->normal * hitP->baricenter.x + hitP->vertices[1]->normal * hitP->baricenter.y + hitP->vertices[2]->normal * hitP->baricenter.z);
		hitP->tgO = math::normalize(hitP->vertices[0]->tangent * hitP->baricenter.x + hitP->vertices[1]->tangent * hitP->baricenter.y + hitP->vertices[2]->tangent * hitP->baricenter.z);
		hitP->btO = math::normalize(hitP->vertices[0]->bitangent * hitP->baricenter.x + hitP->vertices[1]->bitangent * hitP->baricenter.y + hitP->vertices[2]->bitangent * hitP->baricenter.z);

		if (dot(hitP->ngO, hitP->nsO) < 0.0f) // make sure that shading and geometry normal agree on sideness
		{
			hitP->ngO = -hitP->ngO;
		}

		// TODO we already have the inverse so there can be some OPTIMIZATION here
		hitP->nsW = math::normalize(transformNormal3F(hitP->objectToWorld, hitP->nsO));
		hitP->ngW = math::normalize(transformNormal3F(hitP->objectToWorld, hitP->ngO));
		hitP->tgW = math::normalize(transformVector3F(hitP->objectToWorld, hitP->tgO));
		hitP->btW = math::normalize(transformVector3F(hitP->objectToWorld, hitP->btO));
		//hitP->tgW = hitP->tgO;
		//hitP->btW = hitP->btO;


		// Calculate an ortho-normal system respective to the shading normal.
		// Expanding the TBN tbn(tg, ns) constructor because TBN members can't be used as pointers for the Mdl_state with NUM_TEXTURE_SPACES > 1.
		hitP->btW = math::normalize(cross(hitP->nsW, hitP->tgW));
		hitP->tgW = cross(hitP->btW, hitP->nsW); // Now the tangent is orthogonal to the shading normal.



		//math::vec3f bt = math::normalize(cross(hitP->nsW, hitP->tgW));
		//hitP->tgW      = cross(bt, hitP->nsW);

		hitP->textureCoordinates[0] = hitP->vertices[0]->texCoord * hitP->baricenter.x + hitP->vertices[1]->texCoord * hitP->baricenter.y + hitP->vertices[2]->texCoord * hitP->baricenter.z;
		hitP->textureBitangents[0] = hitP->btW;
		hitP->textureTangents[0] = hitP->tgW;

		hitP->textureCoordinates[1] = hitP->textureCoordinates[0];
		hitP->textureBitangents[1] = hitP->btW;
		hitP->textureTangents[1] = hitP->tgW;

		// Explicitly include edge-on cases as frontface condition!
		hitP->isFrontFace = 0.0f <= dot(hitP->direction, hitP->ngW);
	}

	__forceinline__ __device__ void determineMaterialHitProperties(HitProperties* hitP, const unsigned int triangleId)
	{
		if (hitP->geometry == nullptr)
		{
			CUDA_ERROR_PRINT("Trying to access geometry in hit properties getVertices Function but geometry is null! You need to call getInstanceAndGeometry First")
		}
		if (hitP->instance->numberOfSlots > 0)
		{
			const unsigned& materialSlotIndex = hitP->geometry->faceAttributeData[triangleId].materialSlotId;
			InstanceData::SlotIds slotIds = hitP->instance->materialSlots[materialSlotIndex];

			if (slotIds.material != nullptr)
			{
				hitP->material = slotIds.material;
				hitP->materialConfiguration = hitP->material->materialConfiguration;
			}
			if (slotIds.meshLight != nullptr)
			{
				const LightData* lightData = slotIds.meshLight;
				const MeshLightAttributesData* attributes = reinterpret_cast<MeshLightAttributesData*>(lightData->attributes);
				hitP->meshLight = lightData;
				hitP->meshLightAttributes = attributes;
			}
		}
	}

	__forceinline__ __device__ void computeHitProperties(HitProperties* hitP,
		const vtxID instanceId,
		const unsigned int& triangleId,
		const bool doComputeHit = false,
		const math::vec3f& originPosition = math::vec3f(0.0f, 0.0f, 0.0f))
	{
		getInstanceAndGeometry(hitP, instanceId);
		getVertices(hitP, triangleId);

		if (doComputeHit)
		{
			computeHit(hitP, originPosition);
		}

		computeGeometricHitProperties(hitP, triangleId);

		determineMaterialHitProperties(hitP, triangleId);
	}
}

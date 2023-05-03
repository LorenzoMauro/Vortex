#pragma once
#ifndef UTILS_H
#define UTILS_H
#include <optix_device.h>

namespace vtx::utl
{

	__forceinline__ __host__ __device__ float luminance(const math::vec3f& rgb)
	{
		const math::vec3f ntscLuminance{ 0.30f, 0.59f, 0.11f };
		return dot(rgb, ntscLuminance);
	}

	__forceinline__ __host__ __device__ float intensity(const math::vec3f& rgb)
	{
		return (rgb.x + rgb.y + rgb.z) * 0.3333333333f;
	}


	__forceinline__ __device__ uint32_t makeColor(const math::vec3f& radiance)
	{
		const auto     r = static_cast<uint8_t>(radiance.x * 255.0f);
		const auto     g = static_cast<uint8_t>(radiance.y * 255.0f);
		const auto     b = static_cast<uint8_t>(radiance.z * 255.0f);
		const auto     a = static_cast<uint8_t>(1.0f * 255.0f);
		const uint32_t returnColor = (a << 24) | (b << 16) | (g << 8) | r;
		return returnColor;
	}

	template<typename T>
	__forceinline__ __host__ __device__ bool isNull(const T& v)
	{
		return T==0;
	}

	template<typename T>
	__forceinline__ __host__ __device__ bool isNull(const gdt::vec_t<T,3>& v)
	{
		return (v.x == 0.0f && v.y == 0.0f && v.z == 0.0f);
	}

	__forceinline__ __host__ __device__ float balanceHeuristic(const float a, const float b)
	{
		//return __fdiv_rn(a,__fadd_rn(a,b));
		return a / (a + b);
	}

	// Binary-search and return the highest cell index with CDF value <= sample.
	// Arguments are the CDF values array pointer, the index of the last element and the random sample in the range [0.0f, 1.0f).
	__forceinline__ __device__ unsigned int binarySearchCdf(const float* cdf, const unsigned int last, const float sample)
	{
		unsigned int lowerBound = 0;
		unsigned int upperBound = last; // Index on the last entry containing 1.0f. Can never be reached with the sample in the range [0.0f, 1.0f).

		while (lowerBound + 1 != upperBound) // When a pair of limits have been found, the lower index indicates the cell to use.
		{
			const unsigned int index = (lowerBound + upperBound) >> 1;

			if (sample < cdf[index]) // If the CDF value is greater than the sample, use that as new higher limit.
			{
				upperBound = index;
			}
			else // If the sample is greater than or equal to the CDF value, use that as new lower limit.
			{
				lowerBound = index;
			}
		}

		return lowerBound;
	}

	__forceinline__ __device__ void getInstanceAndGeometry(HitProperties* hitP, const vtxID& instanceId)
	{
		hitP->instance = optixLaunchParams.instances[instanceId];
		hitP->geometry = (hitP->instance->geometryData);
	}

	__forceinline__ __device__ void getVertices(HitProperties* hitP, const unsigned int triangleId)
	{
		if(hitP->geometry == nullptr)
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

	}

	__forceinline__ __device__ void fetchTransformsFromInstance(HitProperties* hitP)
	{
		hitP->objectToWorld = hitP->instance->transform;
		hitP->worldToObject = math::affine3f(hitP->objectToWorld.l.inverse(), hitP->objectToWorld.p);
	}

	__forceinline__ __device__ void computeGeometricHitProperties(HitProperties* hitP, const bool useInstanceData =false)
	{
		if (hitP->geometry == nullptr)
		{
			CUDA_ERROR_PRINT("Trying to access geometry in hit properties getVertices Function but geometry is null! You need to call getInstanceAndGeometry First")
		}
		if (hitP->vertices[0] == nullptr)
		{
			CUDA_ERROR_PRINT("Trying to access vertices in hit properties computeHit Function but vertices is null! You need to call getVertices First")
		}
		hitP->nsO = hitP->vertices[0]->normal * hitP->baricenter.x + hitP->vertices[1]->normal * hitP->baricenter.y + hitP->vertices[2]->normal * hitP->baricenter.z;
		hitP->ngO = cross(hitP->vertices[1]->position - hitP->vertices[0]->position, hitP->vertices[2]->position - hitP->vertices[0]->position);
		hitP->tgO = hitP->vertices[0]->tangent * hitP->baricenter.x + hitP->vertices[1]->tangent * hitP->baricenter.y + hitP->vertices[2]->tangent * hitP->baricenter.z;
		
		// TODO we already have the inverse so there can be some OPTIMIZATION here
		hitP->nsW = math::normalize(transformNormal3F(hitP->objectToWorld, hitP->nsO));
		hitP->ngW = math::normalize(transformNormal3F(hitP->objectToWorld, hitP->ngO));
		hitP->tgW = math::normalize(transformVector3F(hitP->objectToWorld, hitP->tgO));

		math::vec3f bt = math::normalize(cross(hitP->nsW, hitP->tgW));
		hitP->tgW      = cross(bt, hitP->nsW);

		hitP->textureCoordinates[0] = hitP->vertices[0]->texCoord * hitP->baricenter.x + hitP->vertices[1]->texCoord * hitP->baricenter.y + hitP->vertices[2]->texCoord * hitP->baricenter.z;
		hitP->textureBitangents[0] = bt;
		hitP->textureTangents[0] = hitP->tgW;

		hitP->textureCoordinates[1] = hitP->textureCoordinates[0];
		hitP->textureBitangents[1]  = bt;
		hitP->textureTangents[1]    = hitP->tgW;

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
			const unsigned&       materialSlotIndex = hitP->geometry->faceAttributeData[triangleId].materialSlotId;
			InstanceData::SlotIds slotIds           = hitP->instance->materialSlots[materialSlotIndex];

			if (slotIds.material != nullptr)
			{
				hitP->material            = slotIds.material;
				hitP->shader              = hitP->material->shader;
				hitP->shaderConfiguration = hitP->shader->shaderConfiguration;
			}
			if (slotIds.meshLight != nullptr)
			{
				const LightData*               lightData  = slotIds.meshLight;
				const MeshLightAttributesData* attributes = reinterpret_cast<MeshLightAttributesData*>(lightData->
					attributes);
				hitP->meshLight           = lightData;
				hitP->meshLightAttributes = attributes;
			}
		}
	}

	__forceinline__ __device__ void computeHitProperties(HitProperties* hitP,
														 const vtxID instanceId,
														 const unsigned int& triangleId,
														 const bool doComputeHit =false,
														 const math::vec3f& originPosition = math::vec3f(0.0f, 0.0f, 0.0f))
	{
		getInstanceAndGeometry(hitP, instanceId);
		getVertices(hitP, triangleId);

		if (doComputeHit)
		{
			computeHit(hitP, originPosition);
		}

		computeGeometricHitProperties(hitP);
        
		determineMaterialHitProperties(hitP, triangleId);
	}

}

#endif

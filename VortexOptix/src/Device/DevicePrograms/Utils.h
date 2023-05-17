#pragma once
#ifndef UTILS_H
#define UTILS_H
#include <optix_device.h>

namespace vtx::utl
{
	

	template<typename T>
	__device__ T lerp(T a, T b, float t)
	{
		return a * (1.0f - t) + b * t;
	}

	__forceinline__ __host__ __device__ float luminance(const math::vec3f& rgb)
	{
		const math::vec3f ntscLuminance{ 0.30f, 0.59f, 0.11f };
		return dot(rgb, ntscLuminance);
	}

	__forceinline__ __host__ __device__ float intensity(const math::vec3f& rgb)
	{
		return (rgb.x + rgb.y + rgb.z) * 0.3333333333f;
	}

	template<typename T>
	__device__ T pow(const T& a, const T& b);


	template<>
	__forceinline__ __device__ float pow(const float& a, const float& b) {
		return powf(a, b);
	}


	template<>
	__forceinline__ __device__ math::vec3f pow(const math::vec3f& a, const math::vec3f& b)
	{
		return { pow(a.x, b.x), pow(a.y, b.y), pow(a.z, b.z) };
	}


	template<typename T>
	__device__ T max(const T& a, const T& b);


	template<>
	__forceinline__ __device__ float max(const float& a, const float& b) {
		return a>b? a: b;
	}

	template<>
	__forceinline__ __device__ math::vec3f max(const math::vec3f& a, const math::vec3f& b)
	{
		return { max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) };
	}

	//-------------------------------------------------------------------------------------------------
		// Avoiding self intersections (see Ray Tracing Gems, Ch. 6)
		//-------------------------------------------------------------------------------------------------
	__forceinline__ __device__ float intAsFloat(const uint32_t v)
	{
		union
		{
			uint32_t bit;
			float    value;
		} temp;

		temp.bit = v;
		return temp.value;
	}

	__forceinline__ __device__ uint32_t floatAsInt(const float v)
	{
		union
		{
			uint32_t bit;
			float    value;
		} temp;

		temp.value = v;
		return temp.bit;
	}

	__forceinline__ __device__ void offsetRay(math::vec3f hitPosition, const math::vec3f& normal)
	{
		constexpr float origin = 1.0f / 32.0f;
		constexpr float floatScale = 1.0f / 65536.0f;
		constexpr float intScale = 256.0f;

		const math::vec3ui ofI(
			static_cast<int>(intScale * normal.x),
			static_cast<int>(intScale * normal.y),
			static_cast<int>(intScale * normal.z));

		math::vec3f pI(
			intAsFloat(floatAsInt(hitPosition.x) + ((hitPosition.x < 0.0f) ? -ofI.x : ofI.x)),
			intAsFloat(floatAsInt(hitPosition.y) + ((hitPosition.y < 0.0f) ? -ofI.y : ofI.y)),
			intAsFloat(floatAsInt(hitPosition.z) + ((hitPosition.z < 0.0f) ? -ofI.z : ofI.z)));

		hitPosition.x = abs(hitPosition.x) < origin ? hitPosition.x + floatScale * normal.x : pI.x;
		hitPosition.y = abs(hitPosition.y) < origin ? hitPosition.y + floatScale * normal.y : pI.y;
		hitPosition.z = abs(hitPosition.z) < origin ? hitPosition.z + floatScale * normal.z : pI.z;
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

	__forceinline__ __host__ __device__ float powerHeuristic(const float a, const float b)
	{
		//return __fdiv_rn(a,__fadd_rn(a,b));
		return a*a / (a*a + b*b);
	}

	__forceinline__ __host__ __device__ float heuristic(const float a, const float b)
	{
		//return __fdiv_rn(a,__fadd_rn(a,b));
		return powerHeuristic(a, b);
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

		hitP->objectToWorld.toFloat4(hitP->oTwF4);
		hitP->worldToObject.toFloat4(hitP->wToF4);
		//printMath("wTo", wTo);
		//printMath("oTw", oTw);
		//printMath("objectToWorld", hitP->objectToWorld);
		//printMath("worldToObject", hitP->worldToObject);
		//printMath("oTwF4", hitP->oTwF4);
		//printMath("oTwF4", hitP->wToF4);
	}

	__forceinline__ __device__ void fetchTransformsFromInstance(HitProperties* hitP)
	{
		hitP->objectToWorld = hitP->instance->transform;
		hitP->worldToObject = math::affine3f(hitP->objectToWorld.l.inverse(), hitP->objectToWorld.p);
		hitP->objectToWorld.toFloat4(hitP->oTwF4);
		hitP->worldToObject.toFloat4(hitP->wToF4);
	}

	__forceinline__ __device__ void computeGeometricHitProperties(HitProperties* hitP, const unsigned int triangleId, const bool useInstanceData =false)
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

		computeGeometricHitProperties(hitP, triangleId);
        
		determineMaterialHitProperties(hitP, triangleId);
	}

}

#endif

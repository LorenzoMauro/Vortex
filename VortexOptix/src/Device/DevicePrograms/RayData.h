#pragma once

#ifndef PER_RAY_DATA_H
#define PER_RAY_DATA_H

#include <vector_types.h>
#include "Core/Math.h"
#include <mi/neuraylib/target_code_types.h>
#include "LaunchParams.h"

namespace vtx {

#define NUM_TEXTURE_SPACES 2
#define NUM_TEXTURE_RESULTS 16
#define MATERIAL_STACK_LAST 3
#define MATERIAL_STACK_SIZE 4
#define FLAG_HIT           0x00000001
#define FLAG_SHADOW        0x00000002


	struct HitProperties
	{
		const InstanceData* instance;
		const GeometryData* geometry;
		const MaterialData* material;
		const ShaderData* shader;
		const DeviceShaderConfiguration* shaderConfiguration;

		const LightData* meshLight;
		const MeshLightAttributes* meshLightAttributes;

		// Ray properties
		math::vec3f             position;
		float                   distance;
		// World normals and Tangents
		math::vec3f           ngW;
		math::vec3f           nsW;
		math::vec3f           tgW;

		// Object normals and Tangents
		math::vec3f           ngO;
		math::vec3f           nsO;
		math::vec3f           tgO;

		float3 textureCoordinates[2];
		float3 textureBitangents[2];
		float3 textureTangents[2];

		//Transformations
		math::affine3f        objectToWorld;
		math::affine3f        worldToObject;

		bool isFrontFace;
	};

	struct MaterialStack
	{
		float3 ior;
		float3 absorption;
		float3 scattering;
		float bias;
	};

	struct ShadingColors
	{
		math::vec3f diffuse;
		math::vec3f shadingNormal;
		math::vec3f trueNormal;
		math::vec3f orientation;
		math::vec3f debugColor1;
		math::vec3f debugColor2;
		math::vec3f debugColor3;
	};


	struct PerRayData {

		math::vec3f position; //Current Hit Position
		float		distance; //Distance of hit Position to Ray origin
		int			depth;

		math::vec3f wo; //Outgoing direction, to observer in world space
		math::vec3f wi; //Incoming direction, to light, in world space

		math::vec3f	radiance; //Radiance along the current path segment
		float pdf; //last Bdsf smaple, tracked for multiple importance sampling
		math::vec3f throughput; //Throughput of the current path segment, starts white and gets modulated with bsdf_over_pdf with each sample.
		unsigned int flags; // Bitfield with flags. See FLAG_* defines above for its contents.
		mi::neuraylib::Bsdf_event_type eventType; // The type of events created by BSDF importance sampling.


		math::vec3f sigmaT;     // Extinction coefficient in a homogeneous medium.
		int			walk;        // Number of random walk steps done through scattering volume.
		math::vec3f pdfVolume;   // Volume extinction sample pdf. Used to adjust the throughput along the random walk.

		unsigned int seed;  // Random number generator input.

		// Small material stack tracking IOR, absorption ansd scattering coefficients of the entered materials. Entry 0 is vacuum.
		int           idxStack;
		MaterialStack stack[MATERIAL_STACK_SIZE];

		ShadingColors colors;
	};

	// Alias the PerRayData pointer and an math::vec2f for the payload split and merge operations. This generates only move instructions.
	typedef union
	{
		PerRayData* ptr;
		math::vec2ui dat{0,0};
	} Payload;

	__forceinline__ __device__ math::vec2ui splitPointer(PerRayData* ptr)
	{
		Payload payload;

		payload.ptr = ptr;

		return payload.dat;
	}

	__forceinline__ __device__ PerRayData* mergePointer(const unsigned int p0, const unsigned int p1)
	{
		Payload payload;

		payload.dat.x = p0;
		payload.dat.y = p1;

		return payload.ptr;
	}


}

#endif
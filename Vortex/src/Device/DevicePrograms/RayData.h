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
// Prevent that division by very small floating point values results in huge values, for example dividing by pdf.
#define DENOMINATOR_EPSILON 1.0e-6f
#define REDCOLOR  math::vec3f(1.0f, 0.0f, 0.0f)
#define GREENCOLOR  math::vec3f(0.0f, 1.0f, 0.0f)
#define BLUECOLOR  math::vec3f(0.0f, 0.0f, 1.0f)

	enum TraceEvent
	{
		TR_MISS,
		TR_HIT,
		TR_SHADOW,

		TR_UNKNOWN
	};

	struct HitProperties
	{
		vtxID				instanceId;
		vtxID				geometryId;

		const InstanceData* instance = nullptr;
		const GeometryData* geometry = nullptr;

		graph::VertexAttributes*   vertices[3]{nullptr, nullptr, nullptr};

		// Material Properties
		const MaterialData* material = nullptr;
		const DeviceShaderConfiguration* materialConfiguration = nullptr;
		const LightData* meshLight = nullptr;
		const MeshLightAttributesData* meshLightAttributes = nullptr;

		// Hit Point Properties
		math::vec3f				baricenter;
		math::vec3f             position;
		math::vec3f             direction;
		float                   distance;

		//Transformations
		math::affine3f        objectToWorld;
		math::affine3f        worldToObject;

		// World normals and Tangents
		math::vec3f           ngW;
		math::vec3f           nsW;
		math::vec3f           tgW;
		math::vec3f           btW;

		// Object normals and Tangents
		math::vec3f           ngO;
		math::vec3f           nsO;
		math::vec3f           tgO;
		math::vec3f           btO;

		float3 textureCoordinates[2];
		float3 textureBitangents[2];
		float3 textureTangents[2];

		bool isFrontFace;
		unsigned  seed;

		float4					oTwF4[3];
		float4					wToF4[3];
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
		math::vec3f finalDiffuse;
		math::vec3f bounceDiffuse;
		math::vec3f shadingNormal;
		math::vec3f trueNormal;
		math::vec3f orientation;
		math::vec3f tangent;
		math::vec3f uv;
		math::vec3f debugColor1;
	};


	struct PerRayData {
		math::vec3f						position;					//Current Hit Position
		float							distance;					//Distance of hit Position to Ray origin
		int								depth;
		TraceEvent						traceResult;				// Bitfield with flags. See FLAG_* defines above for its contents.
		TraceEvent						traceOperation = TR_HIT;	// Bitfield with flags. See FLAG_* defines above for its contents.

		math::vec3f						wo;							//Outgoing direction, to observer in world space
		math::vec3f						wi;							//Incoming direction, to light, in world space

		math::vec3f						radiance;					//Radiance along the current path segment

		float							pdf;						//last Bdsf smaple, tracked for multiple importance sampling
		math::vec3f						throughput;					//Throughput of the current path segment, starts white and gets modulated with bsdf_over_pdf with each sample.
		mi::neuraylib::Bsdf_event_type	eventType;					// The type of events created by BSDF importance sampling.


		math::vec3f sigmaT;											// Extinction coefficient in a homogeneous medium.
		int         walk;											// Number of random walk steps done through scattering volume.
		math::vec3f pdfVolume;										// Volume extinction sample pdf. Used to adjust the throughput along the random walk.

		unsigned int seed;											// Random number generator input.

		// Small material stack tracking IOR, absorption and scattering coefficients of the entered materials. Entry 0 is vacuum.
		int           idxStack;
		MaterialStack stack[MATERIAL_STACK_SIZE];
		math::vec3f           mediumIor;

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
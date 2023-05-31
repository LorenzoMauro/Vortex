#pragma once

#ifndef LAUNCH_PARAMS_H
#define LAUNCH_PARAMS_H

#include "Core/Math.h"
#include "Core/VortexID.h"
#include "cuda.h"
#include <mi/neuraylib/target_code_types.h>
#include "NoiseData.h"
#include "Scene/DataStructs/VertexAttribute.h"
#include "Scene/Nodes/LightTypes.h"
#include <optix.h>

#include "Device/UploadCode/UploadBuffers.h"

namespace vtx {

    struct SbtProgramIdx
    {
        int raygen          = -1;
        int exception       = -1;
        int miss            = -1;
        int hit             = -1;
        int pinhole         = -1;
        int meshLightSample = -1;
		int envLightSample = -1;
	};


    enum DataType
    {
	    DV_INSTANCE,
        DV_GEOMETRY,
        DV_MATERIAL,
        DV_SHADER,
        DV_TEXTURE,
        DV_BSDF,
        DV_LIGHTPROFILE
    };
    enum PrimitiveType {
        PT_TRIANGLES,

        NUM_PT
    };
    // Camera
    struct CameraData {
        math::vec3f  position;
        math::vec3f  vertical;
        math::vec3f  horizontal;
        math::vec3f  direction;
    };

    struct GeometryData {
        PrimitiveType				type;
        OptixTraversableHandle		traversable;
        graph::VertexAttributes*	vertexAttributeData;
        graph::FaceAttributes*      faceAttributeData;

        vtxID*					    indicesData;
        size_t						numVertices;
        size_t						numIndices;
        size_t                      numFaces;
    };


    struct DeviceShaderConfiguration
    {
        bool isEmissive = false;
        bool isThinWalled = true;
        bool hasOpacity = false;
        bool directCallable;

        int idxCallEvaluateMaterial = -1;

        int idxCallInit = -1; // The material global init function.

        int idxCallThinWalled = -1;

        int idxCallSurfaceScatteringSample = -1;
        int idxCallSurfaceScatteringEval = -1;
        int idxCallSurfaceScatteringAuxiliary = -1;

        int idxCallBackfaceScatteringSample = -1;
        int idxCallBackfaceScatteringEval = -1;
        int idxCallBackfaceScatteringAuxiliary = -1;

        int idxCallSurfaceEmissionEval = -1;
        int idxCallSurfaceEmissionIntensity = -1;
        int idxCallSurfaceEmissionIntensityMode = -1;

        int idxCallBackfaceEmissionEval = -1;
        int idxCallBackfaceEmissionIntensity = -1;
        int idxCallBackfaceEmissionIntensityMode = -1;

        int idxCallIor = -1;

        int idxCallVolumeAbsorptionCoefficient = -1;
        int idxCallVolumeScatteringCoefficient = -1;
        int idxCallVolumeDirectionalBias = -1;

        int idxCallGeometryCutoutOpacity = -1;

        int idxCallHairSample = -1;
        int idxCallHairEval = -1;

        // The constant expression values:
        //bool thin_walled; // Stored inside flags.
        math::vec3f surfaceIntensity;
        int         surfaceIntensityMode;
        math::vec3f backfaceIntensity;
        int         backfaceIntensityMode;
        math::vec3f ior;
        math::vec3f absorptionCoefficient;
        math::vec3f scatteringCoefficient;
        float       directionalBias;
        float       cutoutOpacity;
	};

    struct TextureData
    {
        cudaTextureObject_t     texObj;
        cudaTextureObject_t     texObjUnfiltered;
        math::vec4ui            dimension;
        math::vec3f             invSize;

    };

    struct BsdfSamplingPartData
    {
        math::vec2ui    angularResolution;
        math::vec2f     invAngularResolution;
        int             numChannels;
        float*          sampleData;
        float*          albedoData;
        CUtexObject     evalData;
        CUarray         deviceMbsdfData;
        float           maxAlbedo;
    };

    struct BsdfData
    {
        BsdfSamplingPartData* reflectionBsdf = nullptr;
        BsdfSamplingPartData* transmissionBsdf = nullptr;
        bool hasReflectionBsdf = false;
        bool hasTransmissionBsdf = false;
    };

    struct LightProfileData
    {
        CUarray             lightProfileArray;
        CUtexObject         evalData;
        float*              cdfData;
        math::vec2ui        angularResolution;     // angular resolution of the grid
        math::vec2f         invAngularResolution; // inverse angular resolution of the grid
        math::vec2f         thetaPhiStart;        // start of the grid
        math::vec2f         thetaPhiDelta;        // angular step size
        math::vec2f         thetaPhiInvDelta;    // inverse step size
        // 4 byte aligned
        float               candelaMultiplier;     // factor to rescale the normalized data
        float               totalPower;
    };

    struct TextureHandler : mi::neuraylib::Texture_handler_base
    {
        // All indexed by the mdl indices
        TextureData** textures;
        uint32_t numTextures;
        BsdfData** bsdfs;
        uint32_t numBsdfs;
        LightProfileData** lightProfiles;
        uint32_t numLightProfiles;
    };


    struct MaterialData
    {
        char*                       argBlock;
        DeviceShaderConfiguration*  materialConfiguration;
        TextureHandler*             textureHandler;
    };

    struct AliasData {
        unsigned int alias;
        float q;
        float pdf;
    };

    struct EnvLightAttributesData
    {
        TextureData* texture;
        //float  invIntegral;
        //float* cdfU;
        //float* cdfV;
        AliasData* aliasMap;
        math::affine3f    transformation;
        math::affine3f    invTransformation;
    };

    struct MeshLightAttributesData
    {
        vtxID           instanceId; //To retrieve the transform
        GeometryData*   geometryData;
        MaterialData*   materialId;

        float* cdfArea;
        uint32_t* actualTriangleIndices;
        float           totalArea;
        int             size;
    };

    struct LightData
    {
        LightType   type;
        CUdeviceptr attributes;

    };

    struct InstanceData
    {
        struct SlotIds
        {
            MaterialData*   material;
            LightData*      meshLight;
        };
        vtxID						instanceId;
        GeometryData*               geometryData;
        SlotIds*                    materialSlots;
        int                         numberOfSlots;
        math::affine3f              transform;
        bool                        hasEmission;
        bool                        hasOpacity;
    };


    struct FrameBufferData
    {
        //math::vec3f* toneMappedRadiance;

        ///////////////////////////////////////////
        /////////////// Passes ////////////////////
        ///////////////////////////////////////////
        math::vec3f* rawRadiance;
        math::vec3f* directLight;
        math::vec3f* diffuseIndirect;
        math::vec3f* glossyIndirect;
        math::vec3f* transmissionIndirect;

        math::vec3f* tmRadiance;
        math::vec3f* tmDirectLight;
        math::vec3f* tmDiffuseIndirect;
        math::vec3f* tmGlossyIndirect;
        math::vec3f* tmTransmissionIndirect;

        math::vec3f* albedo;
        math::vec3f* normal;
        math::vec3f* trueNormal;
        math::vec3f* tangent;
        math::vec3f* orientation;
        math::vec3f* uv;
        math::vec3f* debugColor1;

        math::vec3f* fireflyPass;
        NoiseData*   noiseBuffer;
        CUdeviceptr  outputBuffer{};
        math::vec2ui frameSize;
	};

    struct RendererDeviceSettings
    {
        enum SamplingTechnique
        {
	        S_BSDF,
            S_DIRECT_LIGHT,
            S_MIS,

            S_COUNT
        };

        inline static const char* samplingTechniqueNames[] = {
                "Bsdf Sampling",
                "Light Sampling",
                "Multiple Importance Sampling",
        };

        enum DisplayBuffer
        {
            FB_BEAUTY,
            FB_NOISY,
            FB_DIRECT_LIGHT,
            FB_DIFFUSE_INDIRECT,
            FB_GLOSSY_INDIRECT,
            FB_TRANSMISSION_INDIRECT,

            FB_DIFFUSE,
            FB_ORIENTATION,
            FB_TRUE_NORMAL,
            FB_SHADING_NORMAL,
            FB_TANGENT,
            FB_UV,
            FB_NOISE,
            FB_SAMPLES,
            FB_DEBUG_1,

            FB_COUNT,
        };

        inline static const char* displayBufferNames[] = {
				"Beauty",
                "Noisy",
                "Direct Light",
                "diffuse Indirect",
                "Glossy Indirect",
                "Transmission Indirect",
                "Diffuse",
                "Orientation",
                "True Normal",
                "Shading Normal",
				"Tangent",
                "Uv",
				"Noise",
				"Samples",
                "Debug1"
        };

        int                 iteration;
        int                 maxBounces;
        bool                accumulate;
        SamplingTechnique   samplingTechnique;
		DisplayBuffer       displayBuffer;
        float               minClamp;
        float               maxClamp;

        bool                adaptiveSampling;
        int                 minAdaptiveSamples;
        int                 minPixelSamples;
        int                 maxPixelSamples;
        float               noiseCutOff;

        bool                enableDenoiser;
        bool                removeFirefly;
	};

    struct ToneMapperSettings
    {
	    math::vec3f invWhitePoint;
        math::vec3f colorBalance;
        float 	 burnHighlights;
        float 	 crushBlacks;
        float 	 saturation;
        float 	 invGamma;
    };

	struct LaunchParams
    {
		int*                    frameID;
		FrameBufferData         frameBuffer;
		CameraData              cameraData;
		RendererDeviceSettings* settings;
        ToneMapperSettings*     toneMapperSettings;
		SbtProgramIdx*          programs;
		OptixTraversableHandle  topObject;
		InstanceData**          instances;
		LightData*              envLight = nullptr;
        LightData**             lights;
        int                     numberOfLights;
	};

    enum TypeRay
    {
        TYPE_RAY_RADIANCE = 0,
        NUM_RAY_TYPES
    };

    struct LensRay
    {
        math::vec3f org;
        math::vec3f dir;
    };

} // ::osc

#endif

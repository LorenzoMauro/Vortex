#pragma once

#ifndef LAUNCH_PARAMS_H
#define LAUNCH_PARAMS_H

#include "Core/math.h"
#include "Core/VortexID.h"
#include "cuda.h"
#include "Device/UploadCode/CUDAmap.h"
#include <mi/neuraylib/target_code_types.h>
#include "Scene/DataStructs/VertexAttribute.h"
#include "Scene/Nodes/LightTypes.h"
//#include "Scene/Nodes/Renderer.h"

namespace vtx {

    struct SbtProgramIdx
    {
        int raygen = -1;
        int exception = -1;
        int miss = -1;
        int hit = -1;
        int pinhole = -1;
        int meshLightSample = -1;
    };

    struct TextureHandler : mi::neuraylib::Texture_handler_base
    {
        vtxID* textureMdlIndexMap;
        uint32_t numTextures;
        vtxID* bsdfMdlIndexMap;
        uint32_t numBsdfs;
        vtxID* lightProfileMdlIndexMap;
        uint32_t numLightProfiles;
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

    struct InstanceData
    {
        struct SlotIds
        {
            vtxID materialId;
            vtxID meshLightId;
        };
	    vtxID						instanceId;
        vtxID                       geometryDataId;
        SlotIds*                    materialSlotsId;
        int                         numberOfSlots;
        math::affine3f              transform;
        bool                        hasEmission;
        bool                        hasOpacity;
    };

    struct DeviceShaderConfiguration
    {
        bool isEmissive = false;
        bool isThinWalled = true;
        bool hasOpacity = false;

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


    struct MeshLightAttributesData
    {
        vtxID           meshId;
        vtxID           materialId;
        vtxID           instanceId; //To retrieve the transform

        float*          cdfArea;
        uint32_t*       actualTriangleIndices;
        float           totalArea;
        int             size;
    };

    struct LightData
    {
        LightType   type;
        CUdeviceptr attributes;
        
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


    struct ShaderData
    {
        DeviceShaderConfiguration*  shaderConfiguration;
        TextureHandler*             textureHandler;
        //vtxID*                      lightProfilesId;
        //vtxID*                      bsdfsId;
        //vtxID*                      texturesId;
    };

    struct MaterialData
    {
        CUdeviceptr                 argBlock;
        vtxID                       shaderId;
    };

    struct FrameBufferData
    {
        math::vec3f*                                radianceBuffer;
        CUdeviceptr                                 outputBuffer{};
        math::vec2ui                                frameSize;
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
            FB_NOISY,
            FB_DIFFUSE,
            FB_ORIENTATION,
            FB_TRUE_NORMAL,
            FB_SHADING_NORMAL,
            FB_DEBUG_1,
            FB_DEBUG_2,
            FB_DEBUG_3,

            FB_COUNT
        };

        inline static const char* displayBufferNames[] = {
                "Noisy",
                "Diffuse",
                "Orientation",
                "True Normal",
                "Shading Normal",
                "Debug1",
                "Debug2",
                "Debug3"
        };

        int               iteration;
        int               maxBounces;
        bool              accumulate;
        SamplingTechnique samplingTechnique;
		DisplayBuffer     displayBuffer;
        float             minClamp;
        float             maxClamp;
	};

	struct LaunchParams
    {
        int*                                    frameID;
        FrameBufferData                         frameBuffer;
    	CameraData                              cameraData;
        RendererDeviceSettings*                 settings;

        SbtProgramIdx*                           programs;

        OptixTraversableHandle                  topObject;
    	CudaMap<vtxID, InstanceData>*			instanceMap;

        CudaMap<vtxID, GeometryData>*			geometryMap;

        CudaMap<vtxID, MaterialData>*			materialMap;
        CudaMap<vtxID, ShaderData>*				shaderMap;
		CudaMap<vtxID, TextureData>*			textureMap;
        CudaMap<vtxID, BsdfData>*				bsdfMap;
        CudaMap<vtxID, LightProfileData>*		lightProfileMap;
        CudaMap<vtxID, LightData>*              lightMap;
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

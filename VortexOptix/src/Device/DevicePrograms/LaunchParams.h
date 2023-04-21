#pragma once

#ifndef LAUNCH_PARAMS_H
#define LAUNCH_PARAMS_H

#include "Core/math.h"
#include "Core/VortexID.h"
#include "cuda.h"
#include "Device/CUDAmap.h"
#include <mi/neuraylib/target_code_types.h>
#include "Scene/DataStructs/VertexAttribute.h"

namespace vtx {


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
        vtxID*					    indicesData;
        size_t						numVertices;
        size_t						numIndices;
    };

    struct InstanceData
    {
	    vtxID						instanceId;
        vtxID                       geometryDataId;
        vtxID*                      materialsDataId;
        int                         numberOfMaterials;
	};

    struct DeviceShaderConfiguration
    {
        unsigned int flags; // See defines above.

        int idxCallInit; // The material global init function.

        int idxCallThinWalled;

        int idxCallSurfaceScatteringSample;
        int idxCallSurfaceScatteringEval;

        int idxCallBackfaceScatteringSample;
        int idxCallBackfaceScatteringEval;

        int idxCallSurfaceEmissionEval;
        int idxCallSurfaceEmissionIntensity;
        int idxCallSurfaceEmissionIntensityMode;

        int idxCallBackfaceEmissionEval;
        int idxCallBackfaceEmissionIntensity;
        int idxCallBackfaceEmissionIntensityMode;

        int idxCallIor;

        int idxCallVolumeAbsorptionCoefficient;
        int idxCallVolumeScatteringCoefficient;
        int idxCallVolumeDirectionalBias;

        int idxCallGeometryCutoutOpacity;

        int idxCallHairSample;
        int idxCallHairEval;

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
        CUtexObject     texObj; 
        CUtexObject     texObjUnfiltered;
        math::vec4ui    dimension;
        math::vec3f     invSize;

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
        CUdeviceptr                                 colorBuffer{};
        math::vec2ui                                frameSize;
    };

	struct LaunchParams
    {
        int*                                    frameID;
        FrameBufferData                         frameBuffer;
    	CameraData                              cameraData;

        OptixTraversableHandle                  topObject;
    	CudaMap<vtxID, InstanceData>*			instanceMap;

        CudaMap<vtxID, GeometryData>*			geometryMap;

        CudaMap<vtxID, MaterialData>*			materialMap;
        CudaMap<vtxID, ShaderData>*				shaderMap;
		CudaMap<vtxID, TextureData>*			textureMap;
        CudaMap<vtxID, BsdfData>*				bsdfMap;
        CudaMap<vtxID, LightProfileData>*		lightProfileMap;
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

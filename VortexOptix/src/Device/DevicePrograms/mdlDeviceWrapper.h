#pragma once
#ifndef MDL_DEVICE_WRAPPER_H
#define MDL_DEVICE_WRAPPER_H

#define TEX_SUPPORT_NO_VTABLES
#define TEX_SUPPORT_NO_DUMMY_SCENEDATA
#include <optix_device.h>
#include "texture_lookup.h"
#include "randomNumberGenerator.h"
#include "RayData.h"

namespace vtx::mdl
{

    struct BsdfEvaluateData
    {
        mi::neuraylib::Bsdf_evaluate_data<mi::neuraylib::DF_HSM_NONE>   data;
        bool                                                            isValid;

    };

    struct BsdfSampleData
    {
    	mi::neuraylib::Bsdf_sample_data     data;
		bool                                isValid;
	};

    struct EmissionEvaluateData
    {
        mi::neuraylib::Edf_evaluate_data<mi::neuraylib::DF_HSM_NONE>    data;
		bool                                                            isValid;
	};

    struct BsdfAuxiliaryData
    {
        mi::neuraylib::Bsdf_auxiliary_data<mi::neuraylib::DF_HSM_NONE>  data;
        bool                                                            isValid;
    };

    struct MdlData
    {

        struct ScatteringFunctions
        {
            int         sampleF = -1;
            int         evalF = -1;
            int         auxF = -1;
            bool        hasEvaluation;
            bool        hasSample;
            bool        hasAux;
        };

        struct EmissionFunctions
        {
            int             evalF = -1;
            int             intensityF = -1;
            int             modeF = -1;
            math::vec3f     intensity = math::vec3f(0.0f);
            int             mode = 0;
            bool            hasEmission;
        };

        mi::neuraylib::Shading_state_material   state;
        mi::neuraylib::Resource_data            resourceData;
        ScatteringFunctions                     scatteringFunctions;
        EmissionFunctions                       emissionFunctions;
        CUdeviceptr                             argBlock;
        DeviceShaderConfiguration*              shaderConfiguration;    //This is not actually an mdl exclusive info but it is placed here for convenience
        bool                                    isFrontFace;            //This is not actually an mdl exclusive info but it is placed here for convenience
        bool                                    isThinWalled;           //might be the same as in shaderConfiguration if it is constant, else, it needs to be evaluated
    };

    __forceinline__ __device__ void determineScatteringFunctions(MdlData* mdlData)
    {
        // Determine which BSDF to use when the material is thin-walled.
        mdlData->scatteringFunctions.sampleF = mdlData->shaderConfiguration->idxCallSurfaceScatteringSample;
        mdlData->scatteringFunctions.evalF = mdlData->shaderConfiguration->idxCallSurfaceScatteringEval;
        mdlData->scatteringFunctions.auxF = mdlData->shaderConfiguration->idxCallSurfaceScatteringAuxiliary;

        // thin-walled and looking at the backface and backface.scattering expression available?
        if (mdlData->isThinWalled && !mdlData->isFrontFace && 0 <= mdlData->shaderConfiguration->idxCallBackfaceScatteringSample)
        {
            // Use the backface.scattering BSDF sample and evaluation functions.
            // Apparently the MDL code can handle front- and backfacing calculations appropriately with the original mdlState and the properly setup volume IORs.
            // No need to flip normals to the ray side.
            mdlData->scatteringFunctions.sampleF = mdlData->shaderConfiguration->idxCallBackfaceScatteringSample;
            mdlData->scatteringFunctions.evalF   = mdlData->shaderConfiguration->idxCallBackfaceScatteringEval; // Assumes both are valid.
            mdlData->scatteringFunctions.auxF = mdlData->shaderConfiguration->idxCallBackfaceScatteringAuxiliary;
        }

        mdlData->scatteringFunctions.hasEvaluation = (0 <= mdlData->scatteringFunctions.evalF);
        mdlData->scatteringFunctions.hasSample = (0 <= mdlData->scatteringFunctions.sampleF);
        mdlData->scatteringFunctions.hasAux = (0 <= mdlData->scatteringFunctions.auxF);

        //printf("Scattering evaluation sbt index %d\n"
		//	   "Scattering sample sbt index %d\n"
		//	   "Scattering auxiliary sbt index %d\n\n", 
        //       mdlData->scatteringFunctions.evalF,
        //       mdlData->scatteringFunctions.sampleF,
        //       mdlData->scatteringFunctions.auxF);

    }

    __forceinline__ __device__ void determineEmissionFunctions(MdlData* mdlData)
    {
        // MDL Specs: There is no emission on the back-side unless an EDF is specified with the backface field and thin_walled is set to true.
        if (mdlData->isFrontFace)
        {
            mdlData->emissionFunctions.evalF         = mdlData->shaderConfiguration->idxCallSurfaceEmissionEval;
            mdlData->emissionFunctions.intensityF    = mdlData->shaderConfiguration->idxCallSurfaceEmissionIntensity;
            mdlData->emissionFunctions.modeF         = mdlData->shaderConfiguration->idxCallSurfaceEmissionIntensityMode;

            mdlData->emissionFunctions.intensity     = mdlData->shaderConfiguration->surfaceIntensity;
            mdlData->emissionFunctions.mode          = mdlData->shaderConfiguration->surfaceIntensityMode;
        }
        else if (mdlData->isThinWalled) // && !isFrontFace
        {
            // These can be the same callable indices if the expressions from surface and backface were identical.
            mdlData->emissionFunctions.evalF         = mdlData->shaderConfiguration->idxCallBackfaceEmissionEval;
            mdlData->emissionFunctions.intensityF    = mdlData->shaderConfiguration->idxCallBackfaceEmissionIntensity;
            mdlData->emissionFunctions.modeF         = mdlData->shaderConfiguration->idxCallBackfaceEmissionIntensityMode;

            mdlData->emissionFunctions.intensity     = mdlData->shaderConfiguration->backfaceIntensity;
            mdlData->emissionFunctions.mode          = mdlData->shaderConfiguration->backfaceIntensityMode;
        }

        mdlData->emissionFunctions.hasEmission = 0 <= mdlData->emissionFunctions.evalF;

    }

    __forceinline__ __device__ void initMdl(HitProperties& hitP, MdlData* mdlData)
    {
        //mdl::MdlState state;
        //mdl::MdlResourceData reousrce;
        // Setup the Mdl_mdlState.
        mdlData->state.normal = hitP.nsW;
        mdlData->state.geom_normal = hitP.ngW;
        mdlData->state.position = hitP.position;
        mdlData->state.animation_time = 0.0f; // This renderer implements no support for animations.
        mdlData->state.text_coords = hitP.textureCoordinates;
        mdlData->state.tangent_u = hitP.textureBitangents;
        mdlData->state.tangent_v = hitP.textureTangents;
        float4 texture_results[16]; //TODO add macro
        mdlData->state.text_results = texture_results;
        mdlData->state.ro_data_segment = nullptr;
        mdlData->state.world_to_object = hitP.worldToObject;
        mdlData->state.object_to_world = hitP.objectToWorld;
        mdlData->state.object_id = hitP.instance->geometryDataId; // idObject, this is the sg::Instance node ID.
        mdlData->state.meters_per_scene_unit = 1.0f;

        mdlData->resourceData.shared_data = nullptr;
        mdlData->resourceData.texture_handler = hitP.shader->textureHandler;
        mdlData->shaderConfiguration = hitP.shader->shaderConfiguration;
        mdlData->argBlock = hitP.material->argBlock;
        mdlData->isFrontFace = hitP.isFrontFace;

        // Using a single material init function instead of per distribution init functions.
        // This is always present, even if it just returns.
        optixDirectCall<void>(mdlData->shaderConfiguration->idxCallInit, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);

        // Keeps the material stack from overflowing at silhouettes.
        // Prevents that silhouettes of thin-walled materials use the backface material.
        // Using the true geometry normal attribute as originally defined on the frontface!

        // thin_walled value in case the expression is a constant.

        mdlData->isThinWalled = mdlData->shaderConfiguration->isThinWalled;
        if (0 <= mdlData->shaderConfiguration->idxCallThinWalled)
        {
            optixDirectCall<void>(mdlData->shaderConfiguration->idxCallThinWalled, &mdlData->isThinWalled, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
        }

        determineScatteringFunctions(mdlData);
        determineEmissionFunctions(mdlData);

        if(mdlData->emissionFunctions.hasEmission)
        {
            if (0 <= mdlData->emissionFunctions.intensityF) // Emission intensity is not a constant.
            {
                optixDirectCall<void>(mdlData->emissionFunctions.intensityF, &mdlData->emissionFunctions.intensity, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
            }
            if (0 <= mdlData->emissionFunctions.modeF) // Emission intensity mode is not a constant.
            {
                optixDirectCall<void>(mdlData->emissionFunctions.modeF, &mdlData->emissionFunctions.mode, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
            }

            // Check if the actual point emission is zero.
            if (mdlData->emissionFunctions.intensity == math::vec3f(0.0f)) {
                mdlData->emissionFunctions.hasEmission = false;
            }

        }
    }

    __forceinline__ __device__ math::vec3f getIor(MdlData& mdlData)
    {
        // IOR value in case the material ior expression is constant.
        math::vec3f ior = mdlData.shaderConfiguration->ior;

        if (0 <= mdlData.shaderConfiguration->idxCallIor)
        {
            optixDirectCall<void>(mdlData.shaderConfiguration->idxCallIor, &ior, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
        }

        return ior;
    }

    __forceinline__ __device__ BsdfSampleData sampleBsdf(
        MdlData& mdlData,
        math::vec3f& materialIor,
        const int& stackIndex,
        MaterialStack* stack,
        math::vec3f& outgoingDirection,
        unsigned& seed)
    {
        BsdfSampleData sampleData;
        // Importance sample the BSDF. 
        if (mdlData.scatteringFunctions.hasSample)
        {
            // If the hit is either on the surface or a thin-walled material,
            // the ray is inside the surrounding material and the material ior is on the other side.
            if (mdlData.isFrontFace || mdlData.isThinWalled)
            {
                sampleData.data.ior1 = stack[stackIndex].ior; // From surrounding medium ior
                sampleData.data.ior2 = materialIor;                    // to material ior.
            }
            else
            {
                // When hitting the backface of a non-thin-walled material, 
                // the ray is inside the current material and the surrounding material is on the other side.
                // The material's IOR is the current top-of-stack. We need the one further down!
                sampleData.data.ior1 = materialIor;                    // From material ior 
                sampleData.data.ior2 = stack[math::max(0, stackIndex - 1)].ior; // From surrounding medium ior
            }
            sampleData.data.k1 = outgoingDirection; // == -optixGetWorldRayDirection()
            sampleData.data.xi = rng4(seed);
            optixDirectCall<void>(mdlData.scatteringFunctions.sampleF, &sampleData.data, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
            sampleData.isValid = true;
        }
        else
        {
            // If there is no valid scattering BSDF, it's the black bsdf() which ends the path.
            // This is usually happening with arbitrary mesh lights when only specifying emission.
            sampleData.isValid = false;
        }

        return sampleData;
    }

    __forceinline__ __device__ BsdfEvaluateData evaluateBsdf(
        MdlData& mdlData,
        math::vec3f& materialIor,
        const int& stackIndex,
        MaterialStack* stack,
        math::vec3f& incomingDirection,
        math::vec3f& outgoingDirection)
    {
        BsdfEvaluateData evalData;

        if(mdlData.scatteringFunctions.hasEvaluation)
        {
            // If the hit is either on the surface or a thin-walled material,
			// the ray is inside the surrounding material and the material ior is on the other side.
            if (mdlData.isFrontFace || mdlData.isThinWalled)
            {
                evalData.data.ior1 = stack[stackIndex].ior; // From surrounding medium ior
                evalData.data.ior2 = materialIor;                    // to material ior.
            }
            else
            {
                // When hitting the backface of a non-thin-walled material, 
                // the ray is inside the current material and the surrounding material is on the other side.
                // The material's IOR is the current top-of-stack. We need the one further down!
                evalData.data.ior1 = materialIor;                    // From material ior 
                evalData.data.ior2 = stack[math::max(0, stackIndex - 1)].ior; // From surrounding medium ior
            }

            evalData.data.k1 = outgoingDirection;
            evalData.data.k2 = incomingDirection;

            optixDirectCall<void>(mdlData.scatteringFunctions.evalF, &evalData.data, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
            evalData.isValid = true;
        }
        else
        {
	        evalData.isValid = false;
        }
        

        return evalData;
    }

	__forceinline__ __device__ BsdfAuxiliaryData getAuxiliaryData(
        MdlData& mdlData,
        math::vec3f& materialIor,
        const int& stackIndex,
        MaterialStack* stack,
        math::vec3f& outgoingDirection)
    {
        BsdfAuxiliaryData auxData;

        if(mdlData.scatteringFunctions.hasAux)
        {
            // If the hit is either on the surface or a thin-walled material,
			// the ray is inside the surrounding material and the material ior is on the other side.
            if (mdlData.isFrontFace || mdlData.isThinWalled)
            {
                auxData.data.ior1 = stack[stackIndex].ior; // From surrounding medium ior
                auxData.data.ior2 = materialIor;                    // to material ior.
            }
            else
            {
                // When hitting the backface of a non-thin-walled material, 
                // the ray is inside the current material and the surrounding material is on the other side.
                // The material's IOR is the current top-of-stack. We need the one further down!
                auxData.data.ior1 = materialIor;                    // From material ior 
                auxData.data.ior2 = stack[math::max(0, stackIndex - 1)].ior; // From surrounding medium ior
            }
            auxData.data.k1 = outgoingDirection;

            optixDirectCall<void>(mdlData.scatteringFunctions.auxF, &auxData.data, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
            auxData.isValid = true;
        }
        else
        {
	        auxData.isValid = false;
        }
        

        return auxData;
    }

    __forceinline__ __device__ EmissionEvaluateData evaluateEmission(
        MdlData& mdlData,
        math::vec3f& outgoingDirection)
    {
        EmissionEvaluateData evalData;
        if(mdlData.emissionFunctions.hasEmission)
        {
            // Check if the hit geometry contains any emission.

            evalData.data.k1 = outgoingDirection; // input: outgoing direction (-ray.direction)

            optixDirectCall<void>(mdlData.emissionFunctions.evalF, &evalData.data, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
            evalData.isValid = true;
        }
        else
        {
	        evalData.isValid = false;
        }

        return evalData;
    }

}

#endif
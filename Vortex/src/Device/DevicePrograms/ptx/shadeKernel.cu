#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Device/DevicePrograms/HitPropertiesComputation.h"
#include "Device/DevicePrograms/Mdl/cudaMdlWrapper.h"
#include "Device/DevicePrograms/Mdl/MdlStructs.h"

#undef min
#undef max

// all function types


typedef void (MaterialEvaluationFunction)(mdl::MdlRequest* request, mdl::MaterialEvaluation* result);
extern __constant__ unsigned int     mdl_functions_count;
extern __constant__ MaterialEvaluationFunction* mdl_functions[];

//union Mdl_function_ptr
//{
//    MatExprFunc* expression;
//    BsdfInitFunc* bsdf_init;
//    BsdfSampleFunc* bsdf_sample;
//    BsdfEvaluateFunc* bsdf_evaluate;
//    BsdfAuxiliaryFunc* bsdf_auxiliary;
//    EdfSampleFunc* edf_sample;
//    EdfEvaluateFunc* edf_evaluate;
//};

// function index offset depending on the target code
//extern __constant__ unsigned int     mdl_target_code_offsets[];

// number of generated functions
//extern __constant__ unsigned int     mdl_functions_count;

// the following arrays are indexed by an mdl_function_index
//extern __constant__ MaterialEvaluationFunction* mdl_functions[];
//extern __constant__ unsigned int     mdl_arg_block_indices[];

namespace vtx::mdl
{

    __forceinline__ __device__ void callEvaluateMaterial(int index, MdlRequest* request, MaterialEvaluation* result)
    {
    	mdl_functions[index](request, result);
	}
    //typedef math::vec3ui MdlFunctionIndex;
    //__device__ inline MdlFunctionIndex getMdlFunctionIndex(const math::vec2i& indexPair)
    //{
    //    return {
    //        indexPair.x,   // target_code_index
    //        indexPair.y,   // function_index inside target code
    //        mdl_target_code_offsets[indexPair.x] + indexPair.y // global function index
    //    };
    //}


    //// Init function
    //__device__ inline BsdfInitFunc* asInit(const MdlFunctionIndex& index)
    //{
    //    return mdl_functions[index.z + 0].bsdf_init;
    //}

    //// Expression functions
    //__device__ inline MatExprFunc* asExpression(const MdlFunctionIndex& index)
    //{
    //    return mdl_functions[index.z + 0].expression;
    //}

    //// BSDF functions
    //__device__ inline BsdfSampleFunc* asBsdfSample(const MdlFunctionIndex& index)
    //{
    //    return mdl_functions[index.z + 0].bsdf_sample;
    //}

    //__device__ inline BsdfEvaluateFunc* asBsdfEvaluate(const MdlFunctionIndex& index)
    //{
    //    return mdl_functions[index.z + 1].bsdf_evaluate;
    //}

    ///*__device__ inline Bsdf_pdf_func* as_bsdf_pdf(const MdlFunctionIndex& index)
    //{
    //    return mdl_functions[index.z + 2].bsdf_pdf;
    //}*/

    //__device__ inline BsdfAuxiliaryFunc* asBsdfAuxiliary(const MdlFunctionIndex& index)
    //{
    //    return mdl_functions[index.z + 3].bsdf_auxiliary;
    //}

    //// EDF functions
    //__device__ inline EdfSampleFunc* asEdfSample(const MdlFunctionIndex& index)
    //{
    //    return mdl_functions[index.z + 0].edf_sample;
    //}

    //__device__ inline EdfEvaluateFunc* asEdfEvaluate(const MdlFunctionIndex& index)
    //{
    //    return mdl_functions[index.z + 1].edf_evaluate;
    //}

    /*__device__ inline Edf_pdf_func* as_edf_pdf(const Mdl_function_index& index)
    {
        return mdl_functions[index.z + 2].edf_pdf;
    }

    __device__ inline Edf_auxiliary_func* as_edf_auxiliary(const Mdl_function_index& index)
    {
        return mdl_functions[index.z + 3].edf_auxiliary;
    }*/

//#define BSDF_FRONT_SAMPLE(...) asBsdfSample(getMdlFunctionIndex(mdlFunctions->idxCallSurfaceScatteringSample))(__VA_ARGS__)
//#define BSDF_FRONT_EVALUATE(...) asBsdfEvaluate(getMdlFunctionIndex(mdlFunctions->idxCallSurfaceScatteringEval))(__VA_ARGS__)
//#define BSDF_FRONT_AUXILIARY(...) asBsdfAuxiliary(getMdlFunctionIndex(mdlFunctions->idxCallSurfaceScatteringAuxiliary))(__VA_ARGS__)
//
//
//#define BSDF_BACK_SAMPLE(...) asBsdfSample(getMdlFunctionIndex(mdlFunctions->idxCallBackfaceScatteringSample))(__VA_ARGS__)
//#define BSDF_BACK_EVALUATE(...) asBsdfEvaluate(getMdlFunctionIndex(mdlFunctions->idxCallBackfaceScatteringEval))(__VA_ARGS__)
//#define BSDF_BACK_AUXILIARY(...) asBsdfAuxiliary(getMdlFunctionIndex(mdlFunctions->idxCallBackfaceScatteringAuxiliary))(__VA_ARGS__)
//
//
//#define EDF_FRONT_EVALUATE(...) asEdfEvaluate(getMdlFunctionIndex(mdlFunctions->idxCallSurfaceEmissionEval))(__VA_ARGS__)
//#define EDF_FRONT_INTENSITY(...) asExpression(getMdlFunctionIndex(mdlFunctions->idxCallSurfaceEmissionIntensity))(__VA_ARGS__)
//#define EDF_FRONT_MODE(...) asExpression(getMdlFunctionIndex(mdlFunctions->idxCallSurfaceEmissionIntensityMode))(__VA_ARGS__)
//
//
//#define EDF_BACK_EVALUATE(...) asEdfEvaluate(getMdlFunctionIndex(mdlFunctions->idxCallBackfaceEmissionEval))(__VA_ARGS__)
//#define EDF_BACK_INTENSITY(...) asExpression(getMdlFunctionIndex(mdlFunctions->idxCallBackfaceEmissionIntensity))(__VA_ARGS__)
//#define EDF_BACK_MODE(...) asExpression(getMdlFunctionIndex(mdlFunctions->idxCallBackfaceEmissionIntensityMode))(__VA_ARGS__)
//
//#define MDL_INIT(...) asInit(getMdlFunctionIndex(mdlFunctions->idxCallInit))(__VA_ARGS__)
//#define MDL_THIN_WALLED(...) asExpression(getMdlFunctionIndex(mdlFunctions->idxCallThinWalled))(__VA_ARGS__)
//#define MDL_IOR(...) asExpression(getMdlFunctionIndex(mdlFunctions->idxCallIor))(__VA_ARGS__)
//#define MDL_OPACITY(...) asExpression(getMdlFunctionIndex(mdlFunctions->idxCallGeometryCutoutOpacity))(__VA_ARGS__)

    // Importance sample the BSDF. 
    //__forceinline__ __device__ BsdfSampleResult sampleBsdf(const MdlData* mdlData, math::vec3f& surroundingIor, math::vec3f& outgoingDirection, unsigned seed, const CudaFunctions* mdlFunctions)
    //{
    //    BsdfSampleData data;
    //    // If the hit is either on the surface or a thin-walled material,
    //    // the ray is inside the surrounding material and the material ior is on the other side.
    //    // When hitting the backface of a non-thin-walled material, 
    //    // the ray is inside the current material and the surrounding material is on the other side.
    //    // The material's IOR is the current top-of-stack. We need the one further down!
    //    data.k1 = outgoingDirection; // == -optixGetWorldRayDirection()
    //    data.xi = rng4(seed);
    //    data.pdf = 1.0f;

    //    if (mdlData->isFrontFace || mdlData->isThinWalled)
    //    {
    //        //printf("Evaluating Frontface\n");
    //        data.ior1 = surroundingIor;
    //        data.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
    //    }
    //    else
    //    {
    //        //printf("Evaluating Backface\n");
    //        data.ior1.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;
    //        data.ior2 = surroundingIor;
    //    }

    //    if (mdlData->isFrontFace || !mdlData->isThinWalled)
    //    {
    //        BSDF_FRONT_SAMPLE(&data, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
    //    }
    //    else
    //    {
    //        BSDF_BACK_SAMPLE(&data, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
    //    }

    //    // I've noticed that if the mdl doesn't contain df_specular but roughness is not zero the eventType will always be glossy even for
    //    // specular paths (pdf becomes a large floating number)
    //    if (data.pdf > 1.0f || data.pdf < 0.0f && (data.event_type & mi::neuraylib::BSDF_EVENT_ABSORB) == 0)
    //    {
    //        data.pdf = 1.0f;
    //        if ((data.event_type & mi::neuraylib::BSDF_EVENT_REFLECTION) != 0)
    //        {
    //            data.event_type = mi::neuraylib::BSDF_EVENT_SPECULAR_REFLECTION;
    //        }
    //        else if ((data.event_type & mi::neuraylib::BSDF_EVENT_TRANSMISSION) != 0)
    //        {
    //            data.event_type = mi::neuraylib::BSDF_EVENT_SPECULAR_TRANSMISSION;
    //        }
    //    }

    //    BsdfSampleResult result;
    //    result.isValid = true;
    //    result.isComputed = true;
    //    result.nextDirection = data.k2;
    //    result.bsdfOverPdf = data.bsdf_over_pdf;
    //    result.eventType = data.event_type;
    //    result.pdf = data.pdf;

    //    return result;
    //}

    //__forceinline__ __device__ BsdfEvalResult evaluateBsdf(const MdlData* mdlData, math::vec3f& surroundingIor, math::vec3f& outgoingDirection, math::vec3f& incomingDirection, const CudaFunctions* mdlFunctions)
    //{
    //    BsdfEvaluateData evalData;

    //    evalData.k1 = outgoingDirection;
    //    evalData.k2 = incomingDirection;
    //    evalData.bsdf_diffuse = math::vec3f(0.0f);
    //    evalData.bsdf_glossy = math::vec3f(0.0f);

    //    // If the hit is either on the surface or a thin-walled material,
    //    // the ray is inside the surrounding material and the material ior is on the other side.
    //    if (mdlData->isFrontFace || mdlData->isThinWalled)
    //    {
    //        evalData.ior1 = surroundingIor; // From surrounding medium ior
    //        evalData.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;                    // to material ior.
    //    }
    //    else
    //    {
    //        // When hitting the backface of a non-thin-walled material, 
    //        // the ray is inside the current material and the surrounding material is on the other side.
    //        // The material's IOR is the current top-of-stack. We need the one further down!
    //        evalData.ior1.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;                    // From material ior 
    //        evalData.ior2 = surroundingIor; // From surrounding medium ior
    //    }

    //    if (mdlData->isFrontFace || !mdlData->isThinWalled)
    //    {
    //        BSDF_FRONT_EVALUATE(&evalData, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
    //    }
    //    else
    //    {
    //        BSDF_BACK_EVALUATE(&evalData, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
    //    }

    //    BsdfEvalResult result;
    //    result.isValid = true;
    //    result.pdf = evalData.pdf;
    //    result.diffuse = evalData.bsdf_diffuse;
    //    result.glossy = evalData.bsdf_glossy;

    //    return result;

    //}

    //__forceinline__ __device__ BsdfAuxResult auxiliaryBsdf(const MdlData* mdlData, math::vec3f& surroundingIor, math::vec3f& outgoingDirection, const CudaFunctions* mdlFunctions)
    //{
    //    BsdfAuxiliaryData auxData;

    //    auxData.k1 = outgoingDirection;

    //    // If the hit is either on the surface or a thin-walled material,
    //    // the ray is inside the surrounding material and the material ior is on the other side.
    //    if (mdlData->isFrontFace || mdlData->isThinWalled)
    //    {
    //        auxData.ior1 = surroundingIor; // From surrounding medium ior
    //        auxData.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;                    // to material ior.
    //    }
    //    else
    //    {
    //        // When hitting the backface of a non-thin-walled material, 
    //        // the ray is inside the current material and the surrounding material is on the other side.
    //        // The material's IOR is the current top-of-stack. We need the one further down!
    //        auxData.ior1.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;                    // From material ior 
    //        auxData.ior2 = surroundingIor; // From surrounding medium ior
    //    }


    //    if (mdlData->isFrontFace || !mdlData->isThinWalled)
    //    {
    //        BSDF_FRONT_AUXILIARY(&auxData, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
    //    }
    //    else
    //    {
    //        BSDF_BACK_AUXILIARY(&auxData, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
    //    }

    //    BsdfAuxResult result;
    //    result.isValid = true;
    //    result.albedo = auxData.albedo;
    //    result.normal = auxData.normal;
    //    return result;
    //}

    //__forceinline__ __device__ EdfResult evaluateEmission(const MdlData* mdlData, math::vec3f& outgoingDirection, const CudaFunctions* mdlFunctions)
    //{
    //    EdfEvaluateData evalData;
    //    EdfResult result;

    //    float3 emissionIntensity;
    //    emissionIntensity.x = 0.0f;
    //    emissionIntensity.y = 0.0f;
    //    emissionIntensity.z = 0.0f;

    //    // Emission 
    //    if (mdlData->isFrontFace || !mdlData->isThinWalled)
    //    {
    //        EDF_FRONT_INTENSITY(&emissionIntensity, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);

    //        result.intensity = emissionIntensity;
    //        //printMath("Front Intensity", emissionIntensity);

    //        if (result.intensity == math::vec3f(0.0f)) {
    //            result.isValid = false;
    //            return result;
    //        }

    //        evalData.k1 = outgoingDirection; // input: outgoing direction (-ray.direction)
    //        EDF_FRONT_MODE(&result.mode, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
    //        EDF_FRONT_EVALUATE(&evalData, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
    //    }
    //    else
    //    {
    //        EDF_BACK_INTENSITY(&emissionIntensity, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
    //        result.intensity = emissionIntensity;


    //        if (result.intensity == math::vec3f(0.0f)) {
    //            result.isValid = false;
    //            return result;
    //        }
    //        evalData.k1 = outgoingDirection; // input: outgoing direction (-ray.direction)
    //        EDF_BACK_MODE(&result.mode, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
    //        EDF_BACK_EVALUATE(&evalData, &mdlData->state, &mdlData->resourceData, nullptr, mdlData->argBlock);
    //    }

    //    // Check if the hit geometry contains any emission.
    //    result.isValid = true;
    //    result.cos = evalData.cos;
    //    result.edf = evalData.edf;
    //    result.pdf = evalData.pdf;
    //    return result;
    //}

    //__forceinline__ __device__ MdlData mdlInit(HitProperties* hitP, const CudaFunctions* mdlFunctions)
    //{
    //    MdlData mdlData;

    //    mdlData.state.normal = hitP->nsW;
    //    mdlData.state.geom_normal = hitP->ngW;
    //    mdlData.state.position = hitP->position;
    //    mdlData.state.animation_time = 0.0f;
    //    mdlData.state.text_coords = hitP->textureCoordinates;
    //    mdlData.state.tangent_u = hitP->textureBitangents;
    //    mdlData.state.tangent_v = hitP->textureTangents;
    //    float4 texture_results[16]; //TODO add macro
    //    mdlData.state.text_results = texture_results;
    //    mdlData.state.ro_data_segment = nullptr;
    //    mdlData.state.world_to_object = hitP->wToF4;
    //    mdlData.state.object_to_world = hitP->oTwF4;
    //    mdlData.state.object_id = hitP->instance->instanceId;
    //    mdlData.state.meters_per_scene_unit = 1.0f;

    //    mdlData.resourceData.shared_data = nullptr;
    //    mdlData.resourceData.texture_handler = hitP->material->textureHandler;
    //    mdlData.argBlock = hitP->material->argBlock;
    //    mdlData.isFrontFace = hitP->isFrontFace;

    //    MDL_INIT(&mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
    //    return mdlData;
    //}

    //__forceinline__ __device__ void evaluateMaterial(MdlRequest* request, MaterialEvaluation* result, const CudaFunctions* mdlFunctions)
    //{
    //    MdlData mdlData = mdlInit(request->hitP, mdlFunctions);

    //    /*MDL_THIN_WALLED(&mdlData.isThinWalled, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);

    //    result->isThinWalled = mdlData.isThinWalled;

    //    MDL_IOR(&result->ior, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);

    //    if (request->opacity)
    //    {
    //        MDL_OPACITY(&result->opacity, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
    //    }
    //    if (request->edf)
    //    {
    //        result->edf = evaluateEmission(&mdlData, request->lastRayDirection, mdlFunctions);
    //    }
    //    if (request->bsdfEvaluation)
    //    {
    //        result->bsdfEvaluation = evaluateBsdf(&mdlData, request->surroundingIor, request->lastRayDirection, request->toSampledLight, mdlFunctions);
    //    }
    //    if (request->bsdfSample)
    //    {
    //        result->bsdfSample = sampleBsdf(&mdlData, request->surroundingIor, request->lastRayDirection, request->hitP->seed, mdlFunctions);
    //    }
    //    if (request->auxiliary)
    //    {
    //        result->aux = auxiliaryBsdf(&mdlData, request->surroundingIor, request->lastRayDirection, mdlFunctions);
    //    }*/
    //}
}

namespace vtx
{

    __forceinline__ __device__ void prepareHitProperties(HitProperties* hitP, RayWorkItem& prd, const LaunchParams& optixLaunchParams)
    {
        hitP->position = prd.hitPosition;
        hitP->direction = -prd.direction;
        hitP->baricenter = prd.hitBaricenter;
        hitP->seed = prd.seed;

        utl::getInstanceAndGeometry(hitP, prd.hitInstanceId, optixLaunchParams);
        utl::getVertices(hitP, prd.hitTriangleId);
        utl::setTransform(hitP, &prd);
        utl::computeGeometricHitProperties(hitP, prd.hitTriangleId);
        utl::determineMaterialHitProperties(hitP, prd.hitTriangleId);
    }


    __forceinline__ __device__ void setGeometricAuxiliaryData(RayWorkItem& prd, const HitProperties& hitP)
    {
        //Auxiliary Data
        if (prd.depth == 0)
        {
            prd.colorsTrueNormal = 0.5f * (hitP.ngW + 1.0f);
            prd.colorsUv = hitP.textureCoordinates[0];
            prd.colorsOrientation = hitP.isFrontFace ? math::vec3f(0.0f, 0.0f, 1.0f) : math::vec3f(1.0f, 0.0f, 0.0f);
            prd.colorsTangent = 0.5f * (hitP.tgW + 1.0f);
            prd.colorsBounceDiffuse = math::vec3f(1.0f, 0.0f, 1.0f);
            prd.colorsShadingNormal = 0.5f * (hitP.nsW + 1.0f);
        }
    }

    __forceinline__ __device__ void nextWork(const RayWorkItem& prd, const bool& terminate, LaunchParams& optixLaunchParams)
    {
        if (!terminate)
        {
            optixLaunchParams.radianceTraceQueue->Push(prd);
        }
        else
        {
            optixLaunchParams.accumulationQueue->Push(prd);
        }
    }


    __forceinline__ __device__ void evaluateMaterialAndSampleLight(mdl::MaterialEvaluation* matEval, LightSample* lightSample, HitProperties& hitP, RayWorkItem& prd, LaunchParams& optixLaunchParams)
    {
        mdl::MdlRequest request;

        RendererDeviceSettings::SamplingTechnique& samplingTechnique = optixLaunchParams.settings->samplingTechnique;

        request.auxiliary = true;
        request.ior = true;
        request.edf = true;
        request.lastRayDirection = -prd.direction;
        request.hitP = &hitP;

        request.surroundingIor = prd.mediumIor;

        const bool doSampleLight = samplingTechnique == RendererDeviceSettings::S_DIRECT_LIGHT || samplingTechnique == RendererDeviceSettings::S_MIS;
        const bool doSampleBsdf = samplingTechnique == RendererDeviceSettings::S_MIS || samplingTechnique == RendererDeviceSettings::S_BSDF;

        if (doSampleLight)
        {
            //*lightSample = sampleLight(&prd, &hitP);
            if (lightSample->isValid)
            {
                request.bsdfEvaluation = true;
                request.toSampledLight = lightSample->direction;
            }
        }

        if (doSampleBsdf)
        {
            request.bsdfSample = true;
        }
        if (hitP.meshLightAttributes != nullptr)
        {
            request.edf = true;
        }

        callEvaluateMaterial(hitP.materialConfiguration->idxCallEvaluateMaterialWavefront, &request, matEval);
        //mdl::evaluateMaterial(&request, matEval, hitP.cudaMdlFunctions);

    }


    __forceinline__ __device__ void setAuxiliaryMaterial(const mdl::MaterialEvaluation& matEval, RayWorkItem& prd)
    {
        // Auxiliary Data
        if (prd.depth == 0 && matEval.aux.isValid)
        {
            prd.colorsBounceDiffuse = matEval.aux.albedo;
            prd.colorsShadingNormal = 0.5f * (matEval.aux.normal + 1.0f);
        }
    }

    __forceinline__ __device__ void nextEventEstimation(const mdl::MaterialEvaluation& matEval, LightSample& lightSample, RayWorkItem& prd, LaunchParams& optixLaunchParams)
    {
        RendererDeviceSettings::SamplingTechnique& samplingTechnique = optixLaunchParams.settings->samplingTechnique;

        //Direct Light Sampling
        if ((samplingTechnique == RendererDeviceSettings::S_MIS || samplingTechnique == RendererDeviceSettings::S_DIRECT_LIGHT) && lightSample.isValid)
        {
            //printf("number of tries: %d\n", numberOfTries);
            auto bxdf = math::vec3f(0.0f, 0.0f, 0.0f);
            bxdf += matEval.bsdfEvaluation.diffuse;
            bxdf += matEval.bsdfEvaluation.glossy;

            if (0.0f < matEval.bsdfEvaluation.pdf && bxdf != math::vec3f(0.0f, 0.0f, 0.0f))
            {
                // Pass the current payload registers through to the shadow ray.

                float weightMis = 1.0f;
                if ((lightSample.typeLight == L_MESH || lightSample.typeLight == L_ENV) && samplingTechnique == RendererDeviceSettings::S_MIS)
                {
                    weightMis = utl::heuristic(lightSample.pdf, matEval.bsdfEvaluation.pdf);
                }

                // The sampled emission needs to be scaled by the inverse probability to have selected this light,
                // Selecting one of many lights means the inverse of 1.0f / numLights.
                // This is using the path throughput before the sampling modulated it above.

                prd.radianceDirect += prd.throughput * bxdf * lightSample.radianceOverPdf * weightMis * (float)optixLaunchParams.numberOfLights; // *float(numLights);

                //unsigned escaped = 1;
                //// Note that the sysData.sceneEpsilon is applied on both sides of the shadow ray [t_min, t_max] interval 
                //// to prevent self-intersections with the actual light geometry in the scene.
                //optixTrace(optixLaunchParams.topObject,
                //    prd.hitPosition,
                //    lightSample.direction, // origin, direction
                //    optixLaunchParams.settings->minClamp,
                //    lightSample.distance - optixLaunchParams.settings->minClamp,
                //    0.0f, // tmin, tmax, time
                //    OptixVisibilityMask(0xFF),
                //    OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, // The shadow ray type only uses anyhit programs.
                //    0,
                //    0,
                //    0,
                //    escaped);

                //if (escaped == 1)
                //{
                //    
                //}
            }
        }		//Evauluate Hit Point Emission
    }

    __forceinline__ __device__ void evaluateEmission(mdl::MaterialEvaluation& matEval, RayWorkItem& prd, const HitProperties& hitP, LaunchParams& optixLaunchParams)
    {
        RendererDeviceSettings::SamplingTechnique& samplingTechnique = optixLaunchParams.settings->samplingTechnique;
        if (matEval.edf.isValid)
        {
            const float area = hitP.meshLightAttributes->totalArea;
            matEval.edf.pdf = prd.hitDistance * prd.hitDistance / (area * matEval.edf.cos);
            // Solid angle measure.

            float misWeight = 1.0f;

            // If the last event was diffuse or glossy, calculate the opposite MIS weight for this implicit light hit.
            if (samplingTechnique == RendererDeviceSettings::S_MIS && prd.eventType & (mi::neuraylib::BSDF_EVENT_DIFFUSE | mi::neuraylib::BSDF_EVENT_GLOSSY))
            {
                misWeight = utl::heuristic(prd.pdf, matEval.edf.pdf);
            }
            // Power (flux) [W] divided by light area gives radiant exitance [W/m^2].
            const float factor = (matEval.edf.mode == 0) ? 1.0f : 1.0f / area;
            prd.radiance += prd.throughput * matEval.edf.intensity * matEval.edf.edf * (factor * misWeight);
        }
    }

    __forceinline__ __device__ void russianRoulette(RayWorkItem& prd, bool* terminate, LaunchParams& optixLaunchParams)
    {
        if (optixLaunchParams.settings->useRussianRoulette && 2 <= prd.depth) // Start termination after a minimum number of bounces.
        {
            const float probability = fmaxf(fmaxf(prd.throughput.x, prd.throughput.y), prd.throughput.z);

            if (probability < rng(prd.seed)) // Paths with lower probability to continue are terminated earlier.
            {
                *terminate = true;
            }
            prd.throughput /= probability; // Path isn't terminated. Adjust the throughput so that the average is right again.
        }
    }

    __forceinline__ __device__ void bsdfSample(RayWorkItem& prd, const mdl::MaterialEvaluation& matEval, const HitProperties& hitP, bool* terminate, LaunchParams& optixLaunchParams)
    {
        if (*terminate)
        {
            return;
        }
        RendererDeviceSettings::SamplingTechnique& samplingTechnique = optixLaunchParams.settings->samplingTechnique;
        if ((samplingTechnique == RendererDeviceSettings::S_MIS || samplingTechnique == RendererDeviceSettings::S_BSDF)
            && prd.depth + 1 <= optixLaunchParams.settings->maxBounces
            && matEval.bsdfSample.eventType != mi::neuraylib::BSDF_EVENT_ABSORB)
        {
            prd.origin = prd.hitPosition;
            prd.direction = matEval.bsdfSample.nextDirection; // Continuation direction.
            prd.throughput *= matEval.bsdfSample.bsdfOverPdf;
            // Adjust the path throughput for all following incident lighting.
            prd.pdf = matEval.bsdfSample.pdf;
            // Note that specular events return pdf == 0.0f! (=> Not a path termination condition.)
            prd.eventType = matEval.bsdfSample.eventType;

            if (!matEval.isThinWalled && (prd.eventType & mi::neuraylib::BSDF_EVENT_TRANSMISSION) != 0)
            {
                if (hitP.isFrontFace) // Entered a volume. 
                {
                    prd.mediumIor = matEval.ior;
                    //printf("Entering Volume Ior = %f %f %f\n", matEval.ior.x, matEval.ior.y, matEval.ior.z);
                }
                else // if !isFrontFace. Left a volume.
                {
                    prd.mediumIor = 1.0f;
                }
            }
            if (prd.depth == 0)
            {
                prd.firstHitType = prd.eventType;
            }
            prd.depth++;
            // Unbiased Russian Roulette path termination.

            *terminate = false;
        }
    }

    extern "C" __global__ void shadeKernel(LaunchParams* params)
    {
        LaunchParams& optixLaunchParams = *params;
        const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= optixLaunchParams.frameBuffer.frameSize.x || y >= optixLaunchParams.frameBuffer.frameSize.y)
            return;

        const unsigned int queueWorkId = y * optixLaunchParams.frameBuffer.frameSize.x + x;


        if (optixLaunchParams.shadeQueue->Size() <= queueWorkId)
            return;

        RayWorkItem prd = (*optixLaunchParams.shadeQueue)[queueWorkId];

        HitProperties hitP;

        prepareHitProperties(&hitP, prd, optixLaunchParams);

        setGeometricAuxiliaryData(prd, hitP);


        if (hitP.material != nullptr)
        {
            mdl::MaterialEvaluation matEval{};
            LightSample lightSample{};
            evaluateMaterialAndSampleLight(&matEval, &lightSample, hitP, prd, optixLaunchParams);

            setAuxiliaryMaterial(matEval, prd);

            nextEventEstimation(matEval, lightSample, prd, optixLaunchParams);

            //evaluateEmission(matEval, prd, hitP);

            prd.pdf = 0.0f;

            bool terminate = false;

            russianRoulette(prd, &terminate, optixLaunchParams);

            bsdfSample(prd, matEval, hitP, &terminate, optixLaunchParams);

            nextWork(prd, terminate, optixLaunchParams);

        }
    }
}

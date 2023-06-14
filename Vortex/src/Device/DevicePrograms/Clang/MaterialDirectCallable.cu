#include <optix.h>

#include "Device/DevicePrograms/Mdl/InlineMdlWrapper.h"

namespace vtx::mdl
{
    extern "C" __device__ void __direct_callable__EvaluateMaterial(MdlRequest* request, MaterialEvaluation* result)
    {
        MdlData mdlData = mdlInit(request->hitP);

        thinWalled(&mdlData.isThinWalled, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);

        result->isThinWalled = mdlData.isThinWalled;

        iorEvaluation(&result->ior, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);

        if (request->opacity)
        {
            opacityEvaluation(&result->opacity, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
        }
        if (request->edf)
        {
            result->edf = evaluateEmission(&mdlData, request->lastRayDirection);
        }
        if(request->bsdfEvaluation)
        {
            result->bsdfEvaluation = evaluateBsdf(&mdlData, request->surroundingIor, request->lastRayDirection, request->toSampledLight);
        }
        if (request->bsdfSample)
        {
            result->bsdfSample = sampleBsdf(&mdlData, request->surroundingIor, request->lastRayDirection, request->hitP->seed);
        }
        if (request->auxiliary)
        {
            result->aux = auxiliaryBsdf(&mdlData, request->surroundingIor, request->lastRayDirection);
        }
    }

    extern "C" __device__ void __replace__EvaluateMaterial(MdlRequest * request, MaterialEvaluation * result)
    {
        MdlData mdlData = mdlInit(request->hitP);

        thinWalled(&mdlData.isThinWalled, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);

        result->isThinWalled = mdlData.isThinWalled;

        iorEvaluation(&result->ior, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);

        if (request->opacity)
        {
            opacityEvaluation(&result->opacity, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
        }
        if (request->edf)
        {
            result->edf = evaluateEmission(&mdlData, request->lastRayDirection);
        }
        if (request->bsdfEvaluation)
        {
            result->bsdfEvaluation = evaluateBsdf(&mdlData, request->surroundingIor, request->lastRayDirection, request->toSampledLight);
        }
        if (request->bsdfSample)
        {
            result->bsdfSample = sampleBsdf(&mdlData, request->surroundingIor, request->lastRayDirection, request->hitP->seed);
        }
        if (request->auxiliary)
        {
            result->aux = auxiliaryBsdf(&mdlData, request->surroundingIor, request->lastRayDirection);
        }
    }
}


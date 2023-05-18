#include <optix.h>

#include "Device/DevicePrograms/Mdl/InlineMdlWrapper.h"

namespace vtx::mdl
{
    extern "C" __device__ void __direct_callable__EvaluateMaterial(MdlRequest* request, MaterialEvaluation* result)
    {
        MdlData mdlData = mdlInit(request->hitP);

        if (request->ior)
        {
            iorEvaluation(&result->ior, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
        }
        if (request->opacity)
        {
            opacityEvaluation(&result->opacity, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
        }
        if (request->edf || request->bsdfEvaluation || request->bsdfSample)
        {
            thinWalled(&mdlData.isThinWalled, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
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
}


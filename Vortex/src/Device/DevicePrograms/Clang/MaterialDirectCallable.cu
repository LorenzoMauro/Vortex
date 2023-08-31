#include <optix.h>
#include "Device/DevicePrograms/InlineMdlWrapper.h"

namespace vtx::mdl
{

    __device__ __forceinline__ void evaluate(const MdlRequest* request, MaterialEvaluation* result)
    {
        MdlData mdlData = mdlInit(request);
        thinWalled(&mdlData.isThinWalled, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
    	result->isThinWalled = mdlData.isThinWalled;
        iorEvaluation(&result->ior, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);

        if(request->evalOnNeuralSampling)
        {
            result->neuralBsdfEvaluation = evaluateBsdf(&mdlData, request->surroundingIor, request->outgoingDirection, request->toNeuralSample);
            // TODO: evaluation returs nan values sometimes, I checked the input and they are not nan...
        }
        if (request->bsdfEvaluation)
        {
            result->bsdfEvaluation = evaluateBsdf(&mdlData, request->surroundingIor, request->outgoingDirection, request->toSampledLight);
        }
        if (request->opacity)
        {
            opacityEvaluation(&result->opacity, &mdlData.state, &mdlData.resourceData, nullptr, mdlData.argBlock);
        }
        if (request->edf)
        {
            result->edf = evaluateEmission(&mdlData, request->outgoingDirection);
        }
        if (request->bsdfSample)
        {
            result->bsdfSample = sampleBsdf(&mdlData, request->surroundingIor, request->outgoingDirection, *request->seed);
            correctBsdfSample(result->bsdfSample, mdlData.state.normal);
        }
        if (request->auxiliary)
        {
            result->aux = auxiliaryBsdf(&mdlData, request->surroundingIor, request->outgoingDirection);
        }
    }

    extern "C" __device__ void __direct_callable__EvaluateMaterial(const MdlRequest* request, MaterialEvaluation* result)
    {
        evaluate(request, result);
    }

    extern "C" __device__ void __replace__EvaluateMaterial(const MdlRequest * request, MaterialEvaluation * result)
    {
        evaluate(request, result);
    }
}


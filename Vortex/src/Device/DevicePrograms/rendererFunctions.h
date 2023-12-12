#pragma once
#ifndef RENDERER_FUNCTIONS_H
#define RENDERER_FUNCTIONS_H

#include "Utils.h"
#include "Core/Math.h"
#include "Device/DevicePrograms/randomNumberGenerator.h"
#include "Device/DevicePrograms/MdlStructs.h"
#include "Device/DevicePrograms/ToneMapper.h"
#include "Device/Wrappers/SoaWorkItems.h"
#include "NeuralNetworks/Interface/InferenceQueries.h"
#include "NeuralNetworks/Interface/NetworkInterface.h"
#include "NeuralNetworks/Interface/Paths.h"

#ifdef ARCHITECTURE_OPTIX
#include <optix_device.h>
#else
typedef void (MaterialEvaluationFunction)(vtx::mdl::MdlRequest* request, vtx::mdl::MaterialEvaluation* result);
extern __constant__ unsigned int     mdl_functions_count;
extern __constant__ MaterialEvaluationFunction* mdl_functions[];

namespace vtx::mdl
{
    __forceinline__ __device__ void callEvaluateMaterial(int index, MdlRequest* request, MaterialEvaluation* result)
    {
        mdl_functions[index](request, result);
    }

}
#endif

namespace vtx
{
    enum ArchitectureType {
        A_FULL_OPTIX,
        A_WAVEFRONT_OPTIX_SHADE,
        A_WAVEFRONT_CUDA_SHADE
    };

#define RED     math::vec3f(1.0f, 0.0f, 0.0f)
#define GREEN   math::vec3f(0.0f, 1.0f, 0.0f)
#define BLUE    math::vec3f(0.0f, 0.0f, 1.0f)

#define NEURAL_SAMPLE_DIRECTION(idx) math::normalize(params.replayBuffer->inferenceInputs.action[idx])

    __forceinline__ __device__ bool neuralNetworkActive(const LaunchParams* params)
    {
	    return params->settings.wavefront.active && params->settings.neural.active;
    }

    __forceinline__ __device__ bool neuralSamplingActivated(const LaunchParams* params, const int& depth)
    {
        const bool doSampleNeural =
            neuralNetworkActive(params)
			&& params->settings.neural.doInference
            && params->settings.renderer.iteration >= params->settings.neural.inferenceIterationStart
            && depth + 1 < params->settings.renderer.maxBounces;

        return doSampleNeural;
    }

    __forceinline__ __device__ void evaluateMaterial(const int& programCallId, mdl::MdlRequest* request, mdl::MaterialEvaluation* matEval)
    {
#ifdef ARCHITECTURE_OPTIX
        optixDirectCall<void, mdl::MdlRequest*, mdl::MaterialEvaluation*>(programCallId, request, matEval);
#else
        callEvaluateMaterial(programCallId, request, matEval);
#endif
    }

    __forceinline__ __device__ void nanCheckAdd(const math::vec3f& input, math::vec3f& buffer)
    {
        if (!utl::isNan(input))
        {
            buffer += input;
        }
    }

    __forceinline__ __device__ void nanCheckAddAtomic(const math::vec3f& input, math::vec3f& buffer)
    {
        if (!utl::isNan(input))
        {
            //buffer += input;
            cuAtomicAdd(&buffer.x, input.x);
            cuAtomicAdd(&buffer.y, input.y);
            cuAtomicAdd(&buffer.z, input.z);
        }
    }

    __forceinline__ __device__ void addDebug(const math::vec3f& color, const int pixelId, const LaunchParams* params)
    {
        if (params->settings.renderer.adaptiveSamplingSettings.active && params->settings.renderer.adaptiveSamplingSettings.minAdaptiveSamples <= params->settings.renderer.iteration)
        {
            nanCheckAddAtomic(color, params->frameBuffer.debugColor1[pixelId]);
        }
        else
        {
            nanCheckAdd(color, params->frameBuffer.debugColor1[pixelId]);
        }
    }

    __forceinline__ __device__ void accumulateRay(const AccumulationWorkItem& awi, const LaunchParams* params)
    {
        if (params->settings.renderer.adaptiveSamplingSettings.active && params->settings.renderer.adaptiveSamplingSettings.minAdaptiveSamples <= params->settings.renderer.iteration)
        {
            nanCheckAddAtomic(awi.radiance, params->frameBuffer.radianceAccumulator[awi.originPixel]);
        }
        else
        {
            nanCheckAdd(awi.radiance, params->frameBuffer.radianceAccumulator[awi.originPixel]);
        }
    }


	__forceinline__ __device__ void cleanFrameBuffer(const int id,const LaunchParams* params)
    {
        const bool cleanOnInferenceStart = neuralNetworkActive(params) && params->settings.neural.doInference && params->settings.renderer.iteration == params->settings.neural.inferenceIterationStart && params->settings.neural.clearOnInferenceStart;
        if (params->settings.renderer.iteration <= 0 || !params->settings.renderer.accumulate || cleanOnInferenceStart)
        {
            const FrameBufferData* frameBuffer = &params->frameBuffer;
            frameBuffer->radianceAccumulator[id] = 0.0f;
            frameBuffer->albedoAccumulator[id] = 0.0f;
            frameBuffer->normalAccumulator[id] = 0.0f;
            frameBuffer->tmRadiance[id] = 0.0f;
            frameBuffer->hdriRadiance[id] = 0.0f;
            frameBuffer->normalNormalized[id] = 0.0f;
            frameBuffer->albedoNormalized[id] = 0.0f;
            frameBuffer->trueNormal[id] = 0.0f;
            frameBuffer->tangent[id] = 0.0f;
            frameBuffer->orientation[id] = 0.0f;
            frameBuffer->uv[id] = 0.0f;
            frameBuffer->fireflyPass[id] = 0.0f;
            frameBuffer->samples[id] = 0;
            frameBuffer->gBufferHistory[id].reset();
            frameBuffer->gBuffer[id] = 0.0f;
            reinterpret_cast<math::vec4f*>(frameBuffer->outputBuffer)[id] = math::vec4f(0.0f);
            frameBuffer->noiseBuffer[id].adaptiveSamples = 1;
            if(neuralNetworkActive(params))
            {
                params->networkInterface->debugBuffer1[id] = 0.0f;
                params->networkInterface->debugBuffer3[id] = 0.0f;
                params->networkInterface->paths->pathsAccumulator[id] = 0.0f;
            }
            
        }
        if (neuralNetworkActive(params))
        {
            params->networkInterface->debugBuffer2[id] = 0.0f;
        }
        params->frameBuffer.debugColor1[id] = 0.0f;
    }

    __forceinline__ __device__ void generateCameraRay(int id, const LaunchParams* params, TraceWorkItem& twi) {

        math::vec2f pixel = math::vec2f((float)(id % params->frameBuffer.frameSize.x), (float)(id / params->frameBuffer.frameSize.x));
        math::vec2f screen {(float)params->frameBuffer.frameSize.x, (float)params->frameBuffer.frameSize.y };
        math::vec2f sample = rng2(twi.seed);
        const math::vec2f fragment = pixel + sample;                    // Jitter the sub-pixel location
        const math::vec2f ndc = (fragment / screen) * 2.0f - 1.0f;      // Normalized device coordinates in range [-1, 1].

        const CameraData camera = params->cameraData;

        math::vec3f origin = camera.position;
        math::vec3f direction = camera.horizontal * ndc.x + camera.vertical * ndc.y + camera.direction;
        math::vec3f normalizedDirection = math::normalize(direction);
        
        twi.origin = origin;
        twi.direction = normalizedDirection;
        twi.originPixel = id;
        twi.radiance = math::vec3f(0.0f);
        twi.pdf = 0.0f;
        twi.throughput = math::vec3f(1.0f);
        twi.eventType = mi::neuraylib::BSDF_EVENT_ABSORB; // Initialize for exit. (Otherwise miss programs do not work.)
        twi.depth = 0;
        twi.mediumIor = math::vec3f(1.0f);
        twi.extendRay = true;
        params->frameBuffer.samples[id] += 1;
    }

    __forceinline__ __device__ math::vec3f missShader(EscapedWorkItem& ewi, const LaunchParams* params)
    {
        math::vec3f emission = 0.0f;
        float misWeight = 1.0f;
        if (params->envLight != nullptr)
        {
            const LightData* envLight = params->envLight;
            EnvLightAttributesData attrib = *reinterpret_cast<EnvLightAttributesData*>(envLight->attributes);
            auto texture = attrib.texture;

            math::vec3f dir = math::transformNormal3F(attrib.invTransformation, ewi.direction);

            {
                bool computeOriginalUV = false;
                float u, v;
                if (computeOriginalUV)
                {
                    u = fmodf(atan2f(dir.y, dir.x) * (float)(0.5 / M_PI) + 0.5f, 1.0f);
                    v = acosf(fmax(fminf(-dir.z, 1.0f), -1.0f)) * (float)(1.0 / M_PI);
                }
                else {
                    float theta = acosf(-dir.z);
                    v = theta / (float)M_PI;
                    float phi = atan2f(dir.y, dir.x);// + M_PI / 2.0f; // azimuth angle (theta)
                    u = (phi + (float)M_PI) / (float)(2.0f * M_PI);
                }

                const auto x = math::min<unsigned>((unsigned int)(u * (float)texture->dimension.x), texture->dimension.x - 1);
                const auto y = math::min<unsigned>((unsigned int)(v * (float)texture->dimension.y), texture->dimension.y - 1);
                emission = tex2D<float4>(texture->texObj, u, v);
                //emission = emission; *attrib.envIntensity;

                // to incorporate the point light selection probability
                // If the last surface intersection was a diffuse event which was directly lit with multiple importance sampling,
                // then calculate light emission with multiple importance sampling for this implicit light hit as well.
                bool MiSCondition = (params->settings.renderer.samplingTechnique == S_MIS && (ewi.eventType & (mi::neuraylib::BSDF_EVENT_DIFFUSE | mi::neuraylib::BSDF_EVENT_GLOSSY)));
                float envSamplePdf = attrib.aliasMap[y * texture->dimension.x + x].pdf;
                if (ewi.pdf > 0.0f && MiSCondition)
                {
                    misWeight = utl::heuristic(ewi.pdf, envSamplePdf);
                }
                ewi.radiance += ewi.throughput * emission * misWeight;
                if (ewi.depth == 0)
                {
                    params->frameBuffer.noiseBuffer[ewi.originPixel].adaptiveSamples = -1; //Let's inform adaptive not to sample again a direct miss;
                    nanCheckAdd(math::normalize(emission), params->frameBuffer.albedoAccumulator[ewi.originPixel]);
                }
            }
        }

        if (neuralNetworkActive(params) && ewi.depth > 0)
        {
            params->networkInterface->paths->recordMissBounce(
                ewi.originPixel, ewi.depth,
                params->settings.renderer.maxClamp, emission, misWeight);
        }

        return emission;
    }

    __forceinline__ __device__ LightSample sampleMeshLight(const LightData& light, RayWorkItem& prd, LaunchParams& params)
    {
        MeshLightAttributesData meshLightAttributes = *(MeshLightAttributesData*)(light.attributes);
        LightSample lightSample;

        lightSample.pdf = 0.0f;

        const float* cdfArea = meshLightAttributes.cdfArea;
        const float3 sample3D = rng3(prd.seed);

        // Uniformly sample the triangles over their surface area.
        // Note that zero-area triangles (e.g. at the poles of spheres) are automatically never sampled with this method!
        // The cdfU is one bigger than res.y.
        unsigned int idxTriangle = utl::binarySearchCdf(cdfArea, meshLightAttributes.size, sample3D.z);
        idxTriangle = meshLightAttributes.actualTriangleIndices[idxTriangle];
        unsigned instanceId = meshLightAttributes.instanceId;

        // Barycentric coordinates.
        const float sqrtSampleX = sqrtf(sample3D.x);
        const float alpha = 1.0f - sqrtSampleX;
        const float beta = sample3D.y * sqrtSampleX;
        const float gamma = 1.0f - alpha - beta;
        const auto& baricenter = math::vec3f(alpha, beta, gamma);

        HitProperties hitP;
        hitP.init(instanceId, idxTriangle, baricenter, 0.0f); // zero because position needs to be calculated
        hitP.determineMaterialInfo(&params);
        if (hitP.hasEdf != true)
        {
	        // sampled Triangle doesn't have light, well that's weird, however we return
            return lightSample;
        }
        hitP.calculateForMeshLightSampling(&params, prd.hitProperties.position, &lightSample.direction, &lightSample.distance);
        lightSample.position = hitP.position;
        lightSample.normal = hitP.shadingNormal;
        if (lightSample.distance < DENOMINATOR_EPSILON)
        {
            return lightSample;
        }

        mdl::MdlRequest request;

        request.opacity = true;
        request.outgoingDirection = -lightSample.direction;
        request.surroundingIor = 1.0f;
        request.seed = &prd.seed;
        request.hitProperties = &hitP;
        request.edf = true;

        mdl::MaterialEvaluation matEval;

        evaluateMaterial(hitP.programCall, &request, &matEval);

        if (matEval.opacity <= 0.0f)
        {
            return lightSample;
        }

        if (matEval.edf.isValid)
        {
            const float totArea = meshLightAttributes.totalArea;

        	// Power (flux) [W] divided by light area gives radiant exitance [W/m^2].
            const float factor = (matEval.edf.mode == 0) ? matEval.opacity : matEval.opacity / totArea;

            lightSample.pdf = lightSample.distance * lightSample.distance / (totArea * matEval.edf.cos); // Solid angle measure.
            lightSample.radianceOverPdf = matEval.edf.intensity * matEval.edf.edf * (factor / lightSample.pdf);
            lightSample.isValid = true;
        }

        return lightSample;
    }

    __forceinline__ __device__ LightSample sampleEnvironment(const LightData& light, RayWorkItem& prd, LaunchParams& params)
    {
        EnvLightAttributesData attrib = *reinterpret_cast<EnvLightAttributesData*>(light.attributes);
        auto texture = attrib.texture;

        unsigned width = texture->dimension.x;
        unsigned height = texture->dimension.y;

        LightSample            lightSample;

        // importance sample an envmap pixel using an alias map
        const float3       sample = rng3(prd.seed);
        const unsigned int size = width * height;
        const auto         idx = math::min<unsigned>((unsigned int)(sample.x * (float)size), size - 1);
        unsigned int       envIdx;
        float              sampleY = sample.y;
        if (sampleY < attrib.aliasMap[idx].q) {
            envIdx = idx;
            sampleY /= attrib.aliasMap[idx].q;
        }
        else {
            envIdx = attrib.aliasMap[idx].alias;
            sampleY = (sampleY - attrib.aliasMap[idx].q) / (1.0f - attrib.aliasMap[idx].q);
        }

        const unsigned int py = envIdx / width;
        const unsigned int px = envIdx % width;
        lightSample.pdf = attrib.aliasMap[envIdx].pdf;

        const float u = (float)(px + sampleY) / (float)width;
        //const float phi = (M_PI_2)*(1.0f-u);

        //const float phi = (float)M_PI  -u * (float)(2.0 * M_PI);
        const float phi = u * (float)(2.0 * M_PI) - (float)M_PI;
        float sinPhi, cosPhi;
        sincosf(phi > float(-M_PI) ? phi : (phi + (float)(2.0 * M_PI)), &sinPhi, &cosPhi);
        const float stepTheta = (float)M_PI / (float)height;
        const float theta0 = (float)(py)*stepTheta;
        const float cosTheta = cosf(theta0) * (1.0f - sample.z) + cosf(theta0 + stepTheta) * sample.z;
        const float theta = acosf(cosTheta);
        const float sinTheta = sinf(theta);
        const float v = theta * (float)(1.0 / M_PI);

        float x = cosPhi * sinTheta;
        float y = sinPhi * sinTheta;
        float z = -cosTheta;

        math::vec3f dir{ x, y, z };
        // Now rotate that normalized object space direction into world space. 
        lightSample.direction = math::transformNormal3F(attrib.transformation, dir);
        lightSample.distance = params.settings.renderer.maxClamp; // Environment light.
        lightSample.position = prd.hitProperties.position + lightSample.direction * lightSample.distance;
        lightSample.normal = -lightSample.direction;
        // Get the emission from the spherical environment texture.
        math::vec3f emission = tex2D<float4>(texture->texObj, u, v);
        // For simplicity we pretend that we perfectly importance-sampled the actual texture-filtered environment map
        // and not the Gaussian-smoothed one used to actually generate the CDFs and uniform sampling in the texel.
        // (Note that this does not contain the light.emission which just modulates the texture.)
        if (DENOMINATOR_EPSILON < lightSample.pdf)
        {
            //lightSample.radianceOverPdf = light.emission * emission / lightSample.pdf;
            lightSample.radianceOverPdf = emission / lightSample.pdf;
        }

        lightSample.isValid = true;
        return lightSample;
    }

    __forceinline__ __device__ LightSample sampleLight(RayWorkItem& prd, LaunchParams& params)
    {
        LightSample lightSample;
        if (const int& numLights = params.numberOfLights; numLights > 0)
        {
            //Randomly Selecting a Light
            //TODO, I think here we can do some better selection by giving more importance to lights with greater power

            const int indexLight = (1 < numLights) ? gdt::clamp(static_cast<int>(floorf(rng(prd.seed) * numLights)), 0, numLights - 1) : 0;

            const LightData& light = *(params.lights[indexLight]);

            const LightType& typeLight = light.type;

            switch (typeLight)
            {
            case L_MESH:
            {
                lightSample = sampleMeshLight(light, prd, params);
            }
            break;
            case L_ENV:
            {
                lightSample = sampleEnvironment(light, prd, params);
            }
            break;
            default: {
                lightSample.isValid = false;
                return lightSample;
            }
            }

            lightSample.typeLight = typeLight;

            if (lightSample.isValid) // && dot(lightSample.direction, ngW) >= -0.05f)
            {
                return lightSample;
            }
        }
        lightSample.isValid = false;
        lightSample.pdf = 0.0f;
        return lightSample;
    }

    __forceinline__ __device__ void setAuxiliaryRenderPassData(const RayWorkItem& prd, const mdl::MaterialEvaluation& matEval, const LaunchParams* params)
    {
        //Auxiliary Data
        if (prd.depth == 0)
        {
            const math::vec3f colorsTrueNormal = 0.5f * (prd.hitProperties.trueNormal + 1.0f);
            const math::vec3f colorsUv = prd.hitProperties.uv;
            const math::vec3f colorsOrientation = prd.hitProperties.isFrontFace ? math::vec3f(0.0f, 0.0f, 1.0f) : math::vec3f(1.0f, 0.0f, 0.0f);
            const math::vec3f colorsTangent = 0.5f * (prd.hitProperties.tangent + 1.0f);
            if (matEval.aux.isValid)
            {
                const math::vec3f colorsBounceDiffuse = matEval.aux.albedo;
                const math::vec3f colorsShadingNormal = 0.5f * (matEval.aux.normal + 1.0f);
                if (params->settings.renderer.adaptiveSamplingSettings.active && params->settings.renderer.adaptiveSamplingSettings.minAdaptiveSamples <= params->settings.renderer.iteration)
                {
                    nanCheckAddAtomic(colorsBounceDiffuse, params->frameBuffer.albedoAccumulator[prd.originPixel]);
                    nanCheckAddAtomic(colorsShadingNormal, params->frameBuffer.normalAccumulator[prd.originPixel]);
                }
                else
                {
                    nanCheckAdd(colorsBounceDiffuse, params->frameBuffer.albedoAccumulator[prd.originPixel]);
                    nanCheckAdd(colorsShadingNormal, params->frameBuffer.normalAccumulator[prd.originPixel]);
                }
            }
            if (params->settings.renderer.adaptiveSamplingSettings.active && params->settings.renderer.adaptiveSamplingSettings.minAdaptiveSamples <= params->settings.renderer.iteration)
            {
                nanCheckAddAtomic(colorsTrueNormal, params->frameBuffer.trueNormal[prd.originPixel]);
                nanCheckAddAtomic(colorsTangent, params->frameBuffer.tangent[prd.originPixel]);
                nanCheckAddAtomic(colorsOrientation, params->frameBuffer.orientation[prd.originPixel]);
                nanCheckAddAtomic(colorsUv, params->frameBuffer.uv[prd.originPixel]);
            }
            else
            {
                nanCheckAdd(colorsTrueNormal, params->frameBuffer.trueNormal[prd.originPixel]);
                nanCheckAdd(colorsTangent, params->frameBuffer.tangent[prd.originPixel]);
                nanCheckAdd(colorsOrientation, params->frameBuffer.orientation[prd.originPixel]);
                nanCheckAdd(colorsUv, params->frameBuffer.uv[prd.originPixel]);
            }
        }
    }

    __forceinline__ __device__ void auxiliaryNetworkInference(
        const LaunchParams* params,
        const int& originPixel, const int& depth, const int& shadeQueueIndex,
        const float& samplingFraction,
        const math::vec3f& sample, const float& pdf, const float& cosineDirection)
    {

        if (depth != 0)
        {
            return;
        }
        if(!neuralNetworkActive(params))
        {
            return;
        }
        
        math::vec3f* debugBuffer = params->networkInterface->debugBuffer2;
        InferenceQueries* inferenceQueries = params->networkInterface->inferenceQueries;

        switch (params->settings.renderer.displayBuffer)
        {
        case(FB_NETWORK_INFERENCE_STATE_POSITION):
        {
            debugBuffer[originPixel] = inferenceQueries->getStatePosition(shadeQueueIndex);
        }
        break;
        case(FB_NETWORK_INFERENCE_STATE_NORMAL):
        {
            debugBuffer[originPixel] = (inferenceQueries->getStateNormal(shadeQueueIndex) + 1.0f) * 0.5f;
        }
        break;
        case(FB_NETWORK_INFERENCE_OUTGOING_DIRECTION):
        {
            debugBuffer[originPixel] = (inferenceQueries->getStateDirection(shadeQueueIndex) + 1.0f) * 0.5f;
        }
        break;
        case(FB_NETWORK_INFERENCE_MEAN):
        {
            debugBuffer[originPixel] = (inferenceQueries->getMean(shadeQueueIndex) + 1.0f) * 0.5f;
        }
        break;
        case(FB_NETWORK_INFERENCE_SAMPLE):
        {
            if (sample == math::vec3f(0.0f))
            {
                debugBuffer[originPixel] = math::vec3f(0.0f);
            }
            else
            {
                debugBuffer[originPixel] = (sample + 1.0f) * 0.5f;
            }
        }
        break;
        case(FB_NETWORK_INFERENCE_SAMPLE_DEBUG):
	    {
            if (sample == math::vec3f(0.0f))
            {
                debugBuffer[originPixel] = math::vec3f(0.0f);
                return;
            }
            const math::vec3f mean = inferenceQueries->getMean(shadeQueueIndex);
            const float cosSampleMean = dot(mean, sample);
            const float value = (cosSampleMean + 1.0f) * 0.5f;
            //const math::vec3f color = floatToScientificRGB(value);
            //debugBuffer[originPixel] = color;

            math::vec3f cosineColor = RED* value + GREEN * (1.0f - value);
            cosineColor.z = pdf;
            debugBuffer[originPixel] = cosineColor;
        }
        break;
        case(FB_NETWORK_INFERENCE_CONCENTRATION):
        {
            debugBuffer[originPixel] = floatToScientificRGB(fminf(1.0f, inferenceQueries->getConcentration(shadeQueueIndex)));
        }
        break;
        case(FB_NETWORK_INFERENCE_ANISOTROPY):
        {
            debugBuffer[originPixel] = floatToScientificRGB(fminf(1.0f, inferenceQueries->getAnisotropy(shadeQueueIndex)));
        }
        break;
        case(FB_NETWORK_INFERENCE_SAMPLING_FRACTION):
        {
            debugBuffer[originPixel] = math::vec3f(floatToScientificRGB(samplingFraction));
        }
        break;
        case(FB_NETWORK_INFERENCE_PDF):
        {
            if (pdf == 0.0f)
            {
                debugBuffer[originPixel] = math::vec3f(0.0f);
            }
            else
            {
                
                debugBuffer[originPixel] = math::vec3f(floatToScientificRGB(fminf(1.0f, pdf)));
            }
        }
        break;
        case(FB_NETWORK_INFERENCE_IS_FRONT_FACE):
        {
            debugBuffer[originPixel] = math::vec3f(floatToScientificRGB((cosineDirection + 1.0f) * 0.5f));
        }
        break;
        }
    }

    __forceinline__ __device__ float neuralSamplingFraction(const LaunchParams& params, const int& shadeQueueIndex)
    {
		//if(params.settings.neural.type == network::NT_SAC)
		//{
		//	samplingFraction = params.settings.neural.sac.neuralSampleFraction;
		//}
		//else
		//{
		//}
        const float samplingFraction = params.networkInterface->inferenceQueries->getSamplingFraction(shadeQueueIndex);
        
		return samplingFraction;
	}

    __forceinline__ __device__ void correctBsdfSampling(const LaunchParams& params, mdl::MaterialEvaluation* matEval, const RayWorkItem& prd, const math::vec4f& neuralSample, float samplingFraction, const bool& doNeuralSample, const int& shadeQueueIndex)
    {
	    
        if (matEval->bsdfSample.eventType & mi::neuraylib::BSDF_EVENT_SPECULAR)
        {
            if (doNeuralSample)
            {
                const math::vec3f neuralDirection = { neuralSample.x, neuralSample.y, neuralSample.z };
                const float cosineDirection = dot(neuralDirection, prd.hitProperties.trueNormal);
                auxiliaryNetworkInference(&params, prd.originPixel, prd.depth, shadeQueueIndex, samplingFraction, neuralDirection, neuralSample.w, cosineDirection);
            }
            else
            {
                auxiliaryNetworkInference(&params, prd.originPixel, prd.depth, shadeQueueIndex, samplingFraction, 0.0f, 0.0f, 0.0f);
            }
            return;
        }

        math::vec3f bsdf;
        float bsdfPdf;
        float neuralPdf;
        math::vec3f neuralDirection;
        math::vec3f originalBsdfOverPdf = -1.0f;
        if (doNeuralSample)
        {
            bsdf = matEval->neuralBsdfEvaluation.diffuse + matEval->neuralBsdfEvaluation.glossy;
			neuralDirection = { neuralSample.x, neuralSample.y, neuralSample.z };
            const float cosineDirection = dot(neuralDirection, prd.hitProperties.trueNormal);
            auxiliaryNetworkInference(&params, prd.originPixel, prd.depth, shadeQueueIndex, samplingFraction, neuralDirection, neuralSample.w, cosineDirection);

            if (bsdf == math::vec3f(0.0f))
            {
                matEval->bsdfSample.eventType = mi::neuraylib::BSDF_EVENT_ABSORB;
                return;
            }
            matEval->bsdfSample.eventType = (mdl::BsdfEventType)(mi::neuraylib::BSDF_EVENT_DIFFUSE | mi::neuraylib::BSDF_EVENT_GLOSSY);
            bsdfPdf = matEval->neuralBsdfEvaluation.pdf;
            neuralPdf = neuralSample.w;
            matEval->bsdfSample.nextDirection = neuralDirection;
        }
        else
        {
            auxiliaryNetworkInference(&params, prd.originPixel, prd.depth, shadeQueueIndex, samplingFraction, 0.0f, 0.0f, 0.0f);
            if (matEval->bsdfSample.eventType == mi::neuraylib::BSDF_EVENT_ABSORB)
            {
                return;
            }
            originalBsdfOverPdf = matEval->bsdfSample.bsdfOverPdf;
            bsdf = matEval->bsdfSample.bsdfOverPdf * matEval->bsdfSample.pdf;
            bsdfPdf = matEval->bsdfSample.pdf;
            neuralPdf = params.networkInterface->inferenceQueries->evaluate(shadeQueueIndex, matEval->bsdfSample.nextDirection);
            neuralDirection = 0.0f;
        }
        matEval->bsdfSample.isValid = true;
        matEval->bsdfSample.pdf = bsdfPdf * (1.0f - samplingFraction) + neuralPdf * samplingFraction;
        if (matEval->bsdfSample.pdf == 0.0f)
        {
	        matEval->bsdfSample.eventType = mi::neuraylib::BSDF_EVENT_ABSORB;
			return;
		}
        matEval->bsdfSample.bsdfOverPdf = bsdf / (matEval->bsdfSample.pdf);

        if(math::isNan(matEval->bsdfSample.bsdfOverPdf) || matEval->bsdfSample.pdf < 0.0f)
        {
            matEval->bsdfSample.eventType = mi::neuraylib::BSDF_EVENT_ABSORB;
        }
    }

    __forceinline__ __device__ void correctLightSample(const LaunchParams & params, mdl::MaterialEvaluation * matEval, const math::vec3f & lightDirection, const int& shadeQueueIndex) {
        if (matEval->bsdfEvaluation.isValid) {
            if(matEval->bsdfEvaluation.diffuse+matEval->bsdfEvaluation.glossy == math::vec3f(0.0f))
            {
                matEval->bsdfEvaluation.isValid = false;
                return;
            }
            const float samplingFraction = neuralSamplingFraction(params, shadeQueueIndex);
            const float neuralPdf = params.networkInterface->inferenceQueries->evaluate(shadeQueueIndex, lightDirection);
            matEval->bsdfEvaluation.pdf = matEval->bsdfEvaluation.pdf * (1.0f - samplingFraction) + neuralPdf * samplingFraction;
        }
    }

    __forceinline__ __device__ void evaluateMaterialAndSampleLight(
        mdl::MaterialEvaluation* matEval, LightSample* lightSample, LaunchParams& params,
        RayWorkItem& prd, int shadeQueueIndex)
    {
        mdl::MdlRequest request;
        prd.depth == 0 ? request.auxiliary = true : request.auxiliary = false;
        request.ior = true;
        request.outgoingDirection = -prd.direction;
        request.surroundingIor = prd.mediumIor;
        request.opacity = false;
        request.seed = &prd.seed;
        request.hitProperties = &prd.hitProperties;
        request.edf = prd.hitProperties.hasEdf;

        SamplingTechnique& samplingTechnique = params.settings.renderer.samplingTechnique;
        const bool doSampleLight = samplingTechnique == S_DIRECT_LIGHT || samplingTechnique == S_MIS;
        const bool doSampleBsdf = samplingTechnique == S_MIS || samplingTechnique == S_BSDF;

        if (doSampleLight)
        {
            *lightSample = sampleLight(prd, params);
            if (lightSample->isValid)
            {
                request.bsdfEvaluation = true;
                request.toSampledLight = lightSample->direction;
            }
        }

        math::vec4f neuralSample = math::vec4f(0.0f);
        bool isNeuralSamplingActivated = neuralSamplingActivated(&params, prd.depth);
        bool doNeuralSample = false;
        float samplingFraction = 0.0f;
        if(doSampleBsdf)
        {
            request.bsdfSample = true;
            if(isNeuralSamplingActivated)
            {
                samplingFraction = neuralSamplingFraction(params, shadeQueueIndex);
                if (isnan(samplingFraction) || isinf(samplingFraction) || samplingFraction < 0.0f || samplingFraction > 1.0f)
                {
#ifdef DEBUG
                    printf("Invalid sampling fraction %f\n", samplingFraction);
#endif
                    samplingFraction = 0.0f;
                }
                else
                {
                    if (const float uniformSample = rng(prd.seed); uniformSample > samplingFraction)
                    {
                        doNeuralSample = false;
                    }
                    else
                    {
                        doNeuralSample = true;
                        request.evalOnNeuralSampling = true;
                        neuralSample = params.networkInterface->inferenceQueries->sample(shadeQueueIndex, prd.seed);
                        request.toNeuralSample = { neuralSample.x, neuralSample.y, neuralSample.z };
                        if (utl::isNan(request.toNeuralSample) || utl::isInf(request.toNeuralSample) || math::length(request.toNeuralSample) <= 0.0f || neuralSample.w < 0.0f)
						{
#ifdef DEBUG

							printf("Invalid neural sample %f %f %f Prob %f\n", neuralSample.x, neuralSample.y, neuralSample.z, neuralSample.w);
#endif
                            doNeuralSample = false;
                            request.evalOnNeuralSampling = false;
                            samplingFraction = 0.0f;
						}
                    }
                }
               
            }
        }

        evaluateMaterial(prd.hitProperties.programCall, &request, matEval);

        assert(!(utl::isNan(matEval->bsdfSample.nextDirection) && matEval->bsdfSample.eventType != 0));

        if (isNeuralSamplingActivated)
        {
            if(doSampleBsdf)
            {
                correctBsdfSampling(params, matEval, prd, neuralSample, samplingFraction, doNeuralSample, shadeQueueIndex);
            }
            if(doSampleLight)
            {
                correctLightSample(params, matEval, lightSample->direction, shadeQueueIndex);
            }
        }
            
    }

    __forceinline__ __device__ ShadowWorkItem nextEventEstimation(const mdl::MaterialEvaluation& matEval, LightSample& lightSample, RayWorkItem& prd, LaunchParams& params)
    {
        SamplingTechnique& samplingTechnique = params.settings.renderer.samplingTechnique;
        ShadowWorkItem swi;
        swi.distance = -1; //invalid

        //Direct Light Sampling
        if ((samplingTechnique == S_MIS || samplingTechnique == S_DIRECT_LIGHT) && lightSample.isValid)
        {
			const math::vec3f bxdf = matEval.bsdfEvaluation.diffuse + matEval.bsdfEvaluation.glossy;

            if (0.0f < matEval.bsdfEvaluation.pdf && bxdf != math::vec3f(0.0f, 0.0f, 0.0f))
            {
                float weightMis = 1.0f;
                if ((lightSample.typeLight == L_MESH || lightSample.typeLight == L_ENV) && samplingTechnique == S_MIS)
                {
                    /*printf(
                        " NEE: LightSample prob: %f, bsdfEval prob: %f\n",
                        lightSample.pdf, matEval.bsdfEvaluation.pdf
                    );*/
                    weightMis = utl::heuristic(lightSample.pdf, matEval.bsdfEvaluation.pdf);
                }

                // The sampled emission needs to be scaled by the inverse probability to have selected this light,
                // Selecting one of many lights means the inverse of 1.0f / numLights.
                // This is using the path throughput before the sampling modulated it above.
                // The bxdf from mdl already include the cosine term. We are sampling in path space so the pdf must be over Area.
                // To handle delta lights the pdf is already incorporated into the emission, hence the radianceOverPdf.
                // So once we multiply the throughput by it's multiplier some terms that should belong to the throughput are
                // actually in the radianceOverPdf. This is not a problem since the path terminates after this bounce.
                const math::vec3f throughputMultiplier = bxdf * (float)params.numberOfLights;
                swi.radiance = prd.throughput * weightMis * throughputMultiplier * lightSample.radianceOverPdf;
                swi.direction = lightSample.direction;
                swi.distance = lightSample.distance - params.settings.renderer.minClamp;
                swi.depth = prd.depth + 1;
                swi.originPixel = prd.originPixel;
                swi.origin = prd.hitProperties.position;
                swi.seed = prd.seed;
                swi.mediumIor = prd.mediumIor;

                if (neuralNetworkActive(&params))
                {
                    params.networkInterface->paths->recordLightRayExtension(
                        prd.originPixel, prd.depth,
                        lightSample.position, lightSample.direction, lightSample.normal,
                        throughputMultiplier,lightSample.radianceOverPdf, matEval.bsdfEvaluation.bsdfPdf, lightSample.pdf, weightMis);
                }
            }
        }

        return swi;
    }

    __forceinline__ __device__ void evaluateEmission(mdl::MaterialEvaluation& matEval, RayWorkItem& prd, const LaunchParams& params)
    {
        const SamplingTechnique& samplingTechnique = params.settings.renderer.samplingTechnique;

        math::vec3f emittedRadiance = 0.0f;
        if (matEval.edf.isValid)
        {
            const MeshLightAttributesData* attributes = reinterpret_cast<MeshLightAttributesData*>(prd.hitProperties.lightData->attributes);
            const float area = attributes->totalArea;
            // We compute the solid angle measure of selecting this light to compare with the pdf of the bsdf which is over directions.
            matEval.edf.pdf = prd.hitDistance * prd.hitDistance / (area * matEval.edf.cos * (float)params.numberOfLights);
            float misWeight = 1.0f;
            if (samplingTechnique == S_MIS && prd.eventType & (mi::neuraylib::BSDF_EVENT_DIFFUSE | mi::neuraylib::BSDF_EVENT_GLOSSY))
            {
                misWeight = utl::heuristic(prd.pdf, matEval.edf.pdf);
                /*printf(
                    "Implict Light Hit:"
                    "Hit Distance %f\n"
                    "Area %f\n"
                    "Cos %f\n"
                    "Num Lights %d\n\n"
                    "LightSample prob: %f\n"
                    "bsdfEval prob: %f\n"
                    "Mis weight: %f\n\n"
                    ,
                    prd.hitDistance, area, matEval.edf.cos, params.numberOfLights,
                    matEval.edf.pdf, matEval.bsdfEvaluation.pdf,
                    misWeight
                );*/

                //printf(
				//	"Mis weight: %f\n"
                //    "Light pdf: %f\n"
                //    "Bsdf pdf: %f\n"
                //    "Hit Distance %f\n"
                //    "Area %f\n"
                //    "Cos %f\n"
                //    "Num Lights %d\n\n"
                //    ,
                //    misWeight, matEval.edf.pdf, matEval.bsdfEvaluation.pdf,
                //    prd.hitDistance, area, matEval.edf.cos, params.numberOfLights
                //);
            }
            // Power (flux) [W] divided by light area gives radiant exitance [W/m^2].
            const float factor = (matEval.edf.mode == 0) ? 1.0f : 1.0f / area;
            emittedRadiance = matEval.edf.intensity * matEval.edf.edf * factor;
            prd.radiance += prd.throughput * misWeight * emittedRadiance;
            if (neuralNetworkActive(&params))
            {
                params.networkInterface->paths->recordBounceEmission(
                    prd.originPixel, prd.depth,
                    emittedRadiance, misWeight);
            }
        }

        
    }

    __forceinline__ __device__ bool russianRoulette(RayWorkItem& prd, const LaunchParams& params, float& continuationProbability)
    {
        continuationProbability = 1.0f;
        if (params.settings.renderer.useRussianRoulette && 2 <= prd.depth) // Start termination after a minimum number of bounces.
        {
            const float probability = fmaxf(fmaxf(prd.throughput.x, prd.throughput.y), prd.throughput.z);

            if (probability == 0.0f || probability < rng(prd.seed)) // Paths with lower probability to continue are terminated earlier.
            {
                return false;
            }
            continuationProbability = probability; // Path isn't terminated. Adjust the throughput so that the average is right again.
        }
        return true;
    }

    __forceinline__ __device__ bool bsdfSample(RayWorkItem& prd, const mdl::MaterialEvaluation& matEval, const LaunchParams& params, const float& continuationProbability)
    {
		const SamplingTechnique& samplingTechnique = params.settings.renderer.samplingTechnique;

        const bool doNextBounce = (samplingTechnique == S_MIS || samplingTechnique == S_BSDF) && prd.depth + 1 < params.settings.renderer.maxBounces;
        if (doNextBounce && (matEval.bsdfSample.eventType != mi::neuraylib::BSDF_EVENT_ABSORB))
        {
            prd.direction = matEval.bsdfSample.nextDirection; // Continuation direction.
            const math::vec3f throughputMultiplier = matEval.bsdfSample.bsdfOverPdf/ continuationProbability;
            prd.throughput *= throughputMultiplier;
            prd.pdf = matEval.bsdfSample.pdf;
            prd.eventType = matEval.bsdfSample.eventType;

            if (!matEval.isThinWalled && (prd.eventType & mi::neuraylib::BSDF_EVENT_TRANSMISSION) != 0)
            {
                if (prd.hitProperties.isFrontFace) // Entered a volume. 
                {
                    prd.mediumIor = matEval.ior;
                }
                else // if !isFrontFace. Left a volume.
                {
                    prd.mediumIor = 1.0f;
                }
            }

            if(neuralNetworkActive(&params))
            {
	            params.networkInterface->paths->recordBsdfRayExtension(
                    prd.originPixel, prd.depth,
                    matEval.bsdfSample.nextDirection, throughputMultiplier, matEval.bsdfSample.bsdfPdf, matEval.bsdfSample.pdf);
			}

            return true;
        }

        return false;
    }

    __forceinline__ __device__ void nextWork(const TraceWorkItem& twi, const ShadowWorkItem& swi, const LaunchParams& params)
    {
        if (twi.extendRay)
        {
            params.queues.radianceTraceQueue->Push(twi);
        }
        else
        {
            AccumulationWorkItem awi{};
            awi.radiance = twi.radiance;
            awi.originPixel = twi.originPixel;
            awi.depth = twi.depth;
            params.queues.accumulationQueue->Push(awi);
        }
        if(swi.distance >0.0f)
        {
        	params.queues.shadowQueue->Push(swi);
		}
    }

    __forceinline__ __device__ void setGBuffer(const RayWorkItem& prd, const LaunchParams& params)
    {
	    if (prd.depth == 0)
	    {
            params.frameBuffer.gBufferHistory[prd.originPixel].recordId(prd.hitProperties.instanceId);
            const bool smoothGBuffer = false;
            if (smoothGBuffer)
            {
				const float value = params.frameBuffer.gBuffer[prd.originPixel] * ((float)params.frameBuffer.samples[prd.originPixel] - 1.0f) + (float)prd.hitProperties.instanceId;
				params.frameBuffer.gBuffer[prd.originPixel] = value/(float)params.frameBuffer.samples[prd.originPixel];
            }
            else
            {
                params.frameBuffer.gBuffer[prd.originPixel] = (float)params.frameBuffer.gBufferHistory[prd.originPixel].mostFrequent;
            }
		}
	}
    __forceinline__ __device__ void shade(LaunchParams* params, RayWorkItem& prd, ShadowWorkItem& swi, TraceWorkItem& twi, const int& shadeQueueIndex = 0)
    {
        prd.hitProperties.calculate(params, prd.direction);
        if(neuralNetworkActive(params))
        {
            params->networkInterface->paths->recordBounceHit(
                prd.originPixel, prd.depth,
                prd.hitProperties.position, prd.hitProperties.shadingNormal, -prd.direction, (int)prd.hitProperties.instanceId, (int)prd.hitProperties.triangleId, prd.hitProperties.programCall);
        }

        setGBuffer(prd, *params);

        mdl::MaterialEvaluation matEval{};
        twi.extendRay = false;
        LightSample lightSample{};
        bool extend = false;
        if (prd.hitProperties.hasMaterial)
        {
            evaluateMaterialAndSampleLight(&matEval, &lightSample, *params, prd, shadeQueueIndex);

            swi = nextEventEstimation(matEval, lightSample, prd, *params);

            evaluateEmission(matEval, prd, *params);

            prd.pdf = 0.0f;

            float continuationProbability;
            extend = russianRoulette(prd, *params, continuationProbability);
            if (extend)
            {
                extend = bsdfSample(prd, matEval, *params, continuationProbability);
            }
            if(neuralNetworkActive(params) && !extend)
            {
                params->networkInterface->paths->setBounceAsTerminal(prd.originPixel, prd.depth);
            }

        }
        setAuxiliaryRenderPassData(prd, matEval, params);

        twi.seed = prd.seed;
        twi.originPixel = prd.originPixel;
        twi.depth = prd.depth;
        twi.origin = prd.hitProperties.position;
        twi.direction = prd.direction;
        twi.radiance = prd.radiance;
        twi.throughput = prd.throughput;
        twi.mediumIor = prd.mediumIor;
        twi.eventType = prd.eventType;
        twi.pdf = prd.pdf;
        twi.extendRay = extend;
        twi.depth = prd.depth+1;
    }

    __forceinline__ __device__ bool transparentAnyHit(
        HitProperties& hitProperties,
        const math::vec3f& direction,
        const math::vec3f& mediumIor,
        unsigned seed,
        const LaunchParams* params
    )
    {
        hitProperties.calculate(params, direction);
        mdl::MdlRequest request;
        if (hitProperties.hasOpacity)
        {
            mdl::MaterialEvaluation matEval;
            if (hitProperties.argBlock == nullptr)
            {
                optixIgnoreIntersection();
                return false;
            }
            request.edf = false;
            request.outgoingDirection = -direction;
            request.surroundingIor = mediumIor;
            request.opacity = true;
            request.seed = &seed;
            request.hitProperties = &hitProperties;

            evaluateMaterial(hitProperties.programCall, &request, &matEval);

            // Stochastic alpha test to get an alpha blend effect.
            // No need to calculate an expensive random number if the test is going to fail anyway.
            if (matEval.opacity < 1.0f && matEval.opacity <= rng(seed))
            {
                optixIgnoreIntersection();
                return false;
            }
            return true;
        }
        return false;
    }

    __forceinline__ __device__ bool transparentAnyHit(RayWorkItem* prd, const LaunchParams* params)
    {
        return transparentAnyHit(
            prd->hitProperties,
            prd->direction,
            prd->mediumIor,
            prd->seed,
            params
        );
    }

    __forceinline__ __device__ int getAdaptiveSampleCount(const int& fbIndex, const LaunchParams* params)
    {
        int samplesPerLaunch = 1;
        if (params->settings.renderer.adaptiveSamplingSettings.active)
        {
            if (params->settings.renderer.adaptiveSamplingSettings.minAdaptiveSamples <= params->settings.renderer.iteration)
            {
                samplesPerLaunch = params->frameBuffer.noiseBuffer[fbIndex].adaptiveSamples;
                if (samplesPerLaunch <= 0) //direct miss
                {
                    return 0;
                }
            }
        }
        return samplesPerLaunch;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////// Integrator Kernel Launchers //////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    __forceinline__ __device__ void resetQueues(const LaunchParams* params)
    {
        params->queues.radianceTraceQueue->Reset();
        params->queues.shadeQueue->Reset();
        params->queues.shadowQueue->Reset();
        params->queues.escapedQueue->Reset();
        params->queues.accumulationQueue->Reset();
        if(neuralNetworkActive(params))
        {
			params->networkInterface->inferenceQueries->reset();
        }
    }

    __forceinline__ __device__ void fillCounters(const LaunchParams* params)
    {
	    params->queues.queueCounters->traceQueueCounter = params->queues.radianceTraceQueue->getCounter();
        params->queues.queueCounters->shadeQueueCounter = params->queues.shadeQueue->getCounter();
        params->queues.queueCounters->shadowQueueCounter = params->queues.shadowQueue->getCounter();
        params->queues.queueCounters->escapedQueueCounter = params->queues.escapedQueue->getCounter();
        params->queues.queueCounters->accumulationQueueCounter = params->queues.accumulationQueue->getCounter();
    }

	__forceinline__ __device__ void wfInitRayEntry(const int id, const LaunchParams* params)
    {
        if (id == 0)
        {
            resetQueues(params);
            if(neuralNetworkActive(params))
            {
	            params->networkInterface->paths->resetValidPixels();
            }
            if(params->settings.renderer.iteration <=0)
            {
                fillCounters(params);
            }
        }
            

        cleanFrameBuffer(id, params);
        const int samplesPerLaunch = getAdaptiveSampleCount(id, params);
        if (samplesPerLaunch == 0) return;
        TraceWorkItem                twi;
        for (int i = 0; i < samplesPerLaunch; i++)
        {
            if(neuralNetworkActive(params))
            {
                params->networkInterface->paths->resetPixel(id);
            }
            
            twi.seed = tea<4>(id + i, params->settings.renderer.iteration + *params->frameID);
            generateCameraRay(id, params, twi);
            
            params->queues.radianceTraceQueue->Push(twi);
        }
    }

    __forceinline__ __device__ void handleShading(int queueWorkId, LaunchParams& params)
    {
        if (queueWorkId == 0)
        {
            params.queues.radianceTraceQueue->Reset();
        }

        if (params.queues.shadeQueue->Size() <= queueWorkId)
            return;

        RayWorkItem prd = (*params.queues.shadeQueue)[queueWorkId];
        ShadowWorkItem swi;
        TraceWorkItem twi;
        shade(&params, prd, swi, twi, queueWorkId);
        nextWork(twi, swi, params);
    }

    __forceinline__ __device__ void wfAccumulateEntry(const int queueWorkId, const LaunchParams* params)
    {
        /*if (queueWorkId == 0)
        {
            params->shadeQueue->Reset();
            params->shadowQueue->Reset();
            params->escapedQueue->Reset();
        }*/
        if (queueWorkId >= params->queues.accumulationQueue->Size())
            return;
        const AccumulationWorkItem             awi = (*params->queues.accumulationQueue)[queueWorkId];
        accumulateRay(awi, params);
    }

    __forceinline__ __device__ void wfEscapedEntry(const int id, const LaunchParams* params)
    {
        if (id >= params->queues.escapedQueue->Size())
            return;

        EscapedWorkItem                ewi = (*params->queues.escapedQueue)[id];

        const math::vec3f stepRadiance = missShader(ewi, params);

        AccumulationWorkItem awi;
        awi.originPixel = ewi.originPixel;
        awi.radiance = ewi.radiance;
        awi.depth = ewi.depth;
        params->queues.accumulationQueue->Push(awi);

        
    }




#ifdef ARCHITECTURE_OPTIX

    __forceinline__ __device__ void optixHitProperties(RayWorkItem* prd)
    {
		const float2 baricenter2D = optixGetTriangleBarycentrics();
		prd->hitDistance = optixGetRayTmax();
		const auto baricenter = math::vec3f(1.0f - baricenter2D.x - baricenter2D.y, baricenter2D.x, baricenter2D.y);
		const math::vec3f position = prd->hitProperties.position + prd->direction * prd->hitDistance;
		prd->hitProperties.init(optixGetInstanceId(), optixGetPrimitiveIndex(), baricenter, position);
    }

    template <typename T>
    __forceinline__ __device__ bool trace(math::vec3f& origin, math::vec3f& direction, const float distance, T* rd, const int sbtIndex, LaunchParams& params, const OptixRayFlags flags = OPTIX_RAY_FLAG_NONE)
    {
        math::vec2ui payload = splitPointer(rd);

        optixTrace(
            params.topObject,
            origin,
            direction, // origin, direction
            params.settings.renderer.minClamp,
            distance,
            0.0f, // tmin, tmax, time
            static_cast<OptixVisibilityMask>(0xFF),
            flags,    //OPTIX_RAY_FLAG_NONE,
            sbtIndex,  //SBT Offset
            0,                                // SBT stride
            0, // missSBTIndex
            payload.x,
            payload.y);

        if (payload.x == 0)
        {
            return false;
        }
        return true;
    }

    __forceinline__ __device__ void elaborateShadowTrace(ShadowWorkItem& swi, LaunchParams& params, const ArchitectureType architecture = A_WAVEFRONT_CUDA_SHADE)
    {
        bool hit = trace(swi.origin, swi.direction, swi.distance, &swi, 1, params, OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
        if (hit == false)
        {
            AccumulationWorkItem awi;
            awi.radiance = swi.radiance;
            awi.depth = swi.depth;
            awi.originPixel = swi.originPixel;
            if (architecture == A_FULL_OPTIX)
            {
                accumulateRay(awi, &params);
            }
            else
            {
                params.queues.accumulationQueue->Push(awi);
                if (neuralNetworkActive(&params))
                {
                    params.networkInterface->paths->validateLightSample(swi.originPixel, swi.depth-1); // stored at previous depth, where it was sampled
                }
            }
        }
    }

    __forceinline__ __device__ void elaborateRadianceTrace(TraceWorkItem& twi, LaunchParams& params, ArchitectureType architecture = A_WAVEFRONT_CUDA_SHADE)
    {
        RayWorkItem prd{};
        prd.seed = twi.seed;
        prd.originPixel = twi.originPixel;
        prd.depth = twi.depth;
        prd.hitProperties.position = twi.origin;
        prd.direction = twi.direction;
        prd.radiance = twi.radiance;
        prd.throughput = twi.throughput;
        prd.mediumIor = twi.mediumIor;
        prd.eventType = twi.eventType;
        prd.pdf = twi.pdf;

        bool hit = trace(twi.origin, twi.direction, params.settings.renderer.maxClamp, &prd, 0, params);
        
        if (architecture == A_FULL_OPTIX)
        {
            if(hit)
            {
                ShadowWorkItem swi;
                shade(&params, prd, swi, twi);
                if(swi.distance > 0.0f)
				{
					elaborateShadowTrace(swi, params, A_FULL_OPTIX);
				}
                if (!twi.extendRay)
                {
                    AccumulationWorkItem awi;
                    awi.radiance = prd.radiance;
                    awi.depth = prd.depth;
                    awi.originPixel = prd.originPixel;
                    accumulateRay(awi, &params);
                }
            }
            else
            {
                EscapedWorkItem ewi{};
                ewi.seed = prd.seed;
                ewi.originPixel = prd.originPixel;
                ewi.depth = prd.depth;
                ewi.direction = prd.direction;
                ewi.radiance = prd.radiance;
                ewi.throughput = prd.throughput;
                ewi.eventType = prd.eventType;
                ewi.pdf = prd.pdf;
                missShader(ewi, &params);
                AccumulationWorkItem awi;
                awi.radiance = ewi.radiance;
                awi.depth = ewi.depth;
                awi.originPixel = ewi.originPixel;
                accumulateRay(awi, &params);
                twi.extendRay = false;
            }
		}
        else
        {
            if (hit)
            {
                int shadeQueueIndex = params.queues.shadeQueue->Push(prd);

                if(neuralSamplingActivated(&params, prd.depth))
                {
                    prd.hitProperties.calculateForInferenceQuery(&params);
                    int inferenceQueueIndex = params.networkInterface->inferenceQueries->addInferenceQuery(prd, shadeQueueIndex);
                }
            }
            else
            {
                EscapedWorkItem ewi{};
                ewi.seed = prd.seed;
                ewi.originPixel = prd.originPixel;
                ewi.depth = prd.depth;
                ewi.direction = prd.direction;
                ewi.radiance = prd.radiance;
                ewi.throughput = prd.throughput;
                ewi.eventType = prd.eventType;
                ewi.pdf = prd.pdf;
                params.queues.escapedQueue->Push(ewi);
            }
        }
    }

    __forceinline__ __device__ void wfTraceRadianceEntry(const int queueWorkId, LaunchParams& params)
    {
        if (queueWorkId == 0)
        {
            params.queues.shadeQueue->Reset();
            if(neuralNetworkActive(&params))
            {
				params.networkInterface->inferenceQueries->reset();
            }
        }

        int radianceTraceQueueSize = params.queues.radianceTraceQueue->Size();
        if (radianceTraceQueueSize <= queueWorkId)
            return;

        TraceWorkItem twi = (*params.queues.radianceTraceQueue)[queueWorkId];
        // Shadow Trace
        const int maxTraceQueueSize = params.frameBuffer.frameSize.x * params.frameBuffer.frameSize.y;
        const bool isLongPath = (float)radianceTraceQueueSize <= params.settings.wavefront.longPathPercentage * (float)maxTraceQueueSize;
        if(!(params.settings.wavefront.useLongPathKernel && isLongPath))
        {
            elaborateRadianceTrace(twi, params);
        }
        else
        {
            int remainingBounces = params.settings.renderer.maxBounces - twi.depth;
            for (int i = 0; i < remainingBounces; i++)
            {
                elaborateRadianceTrace(twi, params, A_FULL_OPTIX);
                if (!twi.extendRay)
                {
                    break;
                }
            }
        }
    }

    __forceinline__ __device__ void wfTraceShadowEntry(const int queueWorkId, LaunchParams& params)
    {
        if (params.queues.shadowQueue->Size() <= queueWorkId)
            return;

        ShadowWorkItem swi = (*params.queues.shadowQueue)[queueWorkId];
        // Shadow Trace

        elaborateShadowTrace(swi, params);
    }

#endif
}

#endif
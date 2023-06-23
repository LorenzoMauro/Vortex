#pragma once
#ifndef RENDERER_FUNCTIONS_H
#define RENDERER_FUNCTIONS_H

#include "Utils.h"
#include "Device/DevicePrograms/HitPropertiesComputation.h"
#include "Core/Math.h"
#include "Device/DevicePrograms/randomNumberGenerator.h"
#include "Device/DevicePrograms/Mdl/MdlStructs.h"
#include "Device/DevicePrograms/ToneMapper.h"

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

    __forceinline__ __device__ void getMaterialData(const InstanceData* instance, const GeometryData* geometry, const int& triangleId,
        int& programCall, bool& hasEdf, char** argBlock, TextureHandler** textureHandler)
    {
        const unsigned& materialSlotIndex = geometry->faceAttributeData[triangleId].materialSlotId;
        InstanceData::SlotIds slotIds = instance->materialSlots[materialSlotIndex];

        *textureHandler = slotIds.material->textureHandler;
        *argBlock = slotIds.material->argBlock;

#ifdef ARCHITECTURE_OPTIX
        programCall = slotIds.material->materialConfiguration->idxCallEvaluateMaterialWavefront;
#else
        programCall = slotIds.material->materialConfiguration->idxCallEvaluateMaterialWavefrontCuda;
#endif
        if (slotIds.meshLight != nullptr)
        {
            hasEdf = true;
        }
    }

    __forceinline__ __device__ bool hasMaterial(LaunchParams* params, RayWorkItem& prd)
    {
        InstanceData* instance = params->instances[prd.hitInstanceId];
        GeometryData* geometry = (instance->geometryData);
        const unsigned& materialSlotIndex = geometry->faceAttributeData[prd.hitTriangleId].materialSlotId;
        InstanceData::SlotIds slotIds = instance->materialSlots[materialSlotIndex];
        if (slotIds.material != nullptr)
        {
            return true;
        }
        return false;
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

    __forceinline__ __device__ void addDebug(math::vec3f color, int pixelId, LaunchParams* params)
    {
        nanCheckAdd(color, params->frameBuffer.debugColor1[pixelId]);
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

    __forceinline__ __device__ void accumulateRay(const AccumulationWorkItem& awi, const LaunchParams* params)
    {
        if(params->settings->adaptiveSampling && params->settings->minAdaptiveSamples <= params->settings->iteration)
        {
            nanCheckAddAtomic(awi.radiance, params->frameBuffer.radianceAccumulator[awi.originPixel]);
        }
        else
        {
            nanCheckAdd(awi.radiance, params->frameBuffer.radianceAccumulator[awi.originPixel]);
        }
    }

    __forceinline__ __device__ void cleanFrameBuffer(int id, LaunchParams* params)
    {
        if (params->settings->iteration <= 0)
        {
            FrameBufferData* frameBuffer = &params->frameBuffer;
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
            frameBuffer->debugColor1[id] = 0.0f;
            frameBuffer->fireflyPass[id] = 0.0f;
            frameBuffer->samples[id] = 0;
            reinterpret_cast<math::vec4f*>(frameBuffer->outputBuffer)[id] = math::vec4f(0.0f);
            frameBuffer->noiseBuffer[id].adaptiveSamples = 1;
        }
    }

	__forceinline__ __device__ void generateCameraRay(int id, LaunchParams* params, TraceWorkItem& twi) {

        math::vec2f pixel = math::vec2f((float)(id % params->frameBuffer.frameSize.x), (float)(id / params->frameBuffer.frameSize.x));
        math::vec2f screen {(float)params->frameBuffer.frameSize.x, (float)params->frameBuffer.frameSize.y };
        math::vec2f sample = rng2(twi.seed);
        const math::vec2f fragment = pixel + sample;                    // Jitter the sub-pixel location
        const math::vec2f ndc = (fragment / screen) * 2.0f - 1.0f;      // Normalized device coordinates in range [-1, 1].

        const CameraData camera = params->cameraData;

        math::vec3f origin = camera.position;
        math::vec3f direction = math::normalize<float>(camera.horizontal * ndc.x + camera.vertical * ndc.y + camera.direction);

        twi.origin = origin;
        twi.direction = direction;
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

    __forceinline__ __device__ void missShader(EscapedWorkItem& ewi, LaunchParams* params)
    {
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
                math::vec3f emission = tex2D<float4>(texture->texObj, u, v);
                //emission = emission; *attrib.envIntensity;

                // to incorporate the point light selection probability
                float misWeight = 1.0f;
                // If the last surface intersection was a diffuse event which was directly lit with multiple importance sampling,
                // then calculate light emission with multiple importance sampling for this implicit light hit as well.
                bool MiSCondition = (params->settings->samplingTechnique == RendererDeviceSettings::S_MIS && (ewi.eventType & (mi::neuraylib::BSDF_EVENT_DIFFUSE | mi::neuraylib::BSDF_EVENT_GLOSSY)));
                if (ewi.pdf > 0.0f && MiSCondition)
                {
                    float envSamplePdf = attrib.aliasMap[y * texture->dimension.x + x].pdf;
                    misWeight = utl::heuristic(ewi.pdf, envSamplePdf);
                    //misWeight = ewi.pdf / (ewi.pdf + envSamplePdf);
                }
                ewi.radiance += ewi.throughput * emission * misWeight;
                if(ewi.depth == 0)
                {
                    params->frameBuffer.noiseBuffer[ewi.originPixel].adaptiveSamples = -1; //Let's inform adaptive not to sample again a direct miss;
                    nanCheckAdd(math::normalize(emission), params->frameBuffer.albedoAccumulator[ewi.originPixel]);
                }
            }
        }
    }

    __forceinline__ __device__ LightSample sampleMeshLight(const LightData& light, RayWorkItem& prd, LaunchParams& params)
    {
        MeshLightAttributesData meshLightAttributes = *reinterpret_cast<MeshLightAttributesData*>(light.attributes);
        LightSample lightSample;

        lightSample.pdf = 0.0f;

        const float* cdfArea = meshLightAttributes.cdfArea;
        const float3 sample3D = rng3(prd.seed);

        // Uniformly sample the triangles over their surface area.
        // Note that zero-area triangles (e.g. at the poles of spheres) are automatically never sampled with this method!
        // The cdfU is one bigger than res.y.

        unsigned int idxTriangle = utl::binarySearchCdf(cdfArea, meshLightAttributes.size, sample3D.z);
        idxTriangle = meshLightAttributes.actualTriangleIndices[idxTriangle];

        mdl::MdlRequest request;
        request.instance = params.instances[meshLightAttributes.instanceId];
        request.geometry = request.instance->geometryData;
        int programCallId;
        getMaterialData(request.instance, request.geometry, idxTriangle, programCallId, request.edf, &request.argBlock, &request.textureHandler);
        if (request.edf != true)
        {
            return lightSample;
        }

        // Compute mesh Light hit
	    {
            // Unit square to triangle via barycentric coordinates.
            const float sqrtSampleX = sqrtf(sample3D.x);
            // Barycentric coordinates.
            const float alpha = 1.0f - sqrtSampleX;
            const float beta = sample3D.y * sqrtSampleX;
            const float gamma = 1.0f - alpha - beta;
            request.baricenter = math::vec3f(alpha, beta, gamma);
            //printf("Baricenter Instance %d: %.3f,%.3f,%.3f\n", meshLightAttributes.instanceId, request.baricenter.x, request.baricenter.y, request.baricenter.z);
		    const math::vec3ui  triVerticesIndices = reinterpret_cast<math::vec3ui*>(request.geometry->indicesData)[idxTriangle];
        	graph::VertexAttributes* vertices[3]{ nullptr, nullptr, nullptr };
        	vertices[0] = &(request.geometry->vertexAttributeData[triVerticesIndices.x]);
        	vertices[1] = &(request.geometry->vertexAttributeData[triVerticesIndices.y]);
        	vertices[2] = &(request.geometry->vertexAttributeData[triVerticesIndices.z]);
        	// Object space vertex attributes at the hit point.
        	lightSample.position = vertices[0]->position * request.baricenter.x + vertices[1]->position * request.baricenter.y + vertices[2]->position * request.baricenter.z;
            lightSample.position = math::transformPoint3F(request.instance->transform, lightSample.position);
            //printMath("Sample Transform:", request.instance->transform);
            lightSample.direction = lightSample.position - prd.origin;
        	lightSample.distance = math::length<float>(lightSample.direction);
        	if (lightSample.distance < DENOMINATOR_EPSILON)
        	{
        		return lightSample;
        	}
        	lightSample.direction /= lightSample.distance;
        	//lightSample.direction -= lightSample.distance;
	    }

        request.opacity = true;
        request.outgoingDirection = -lightSample.direction;
        request.surroundingIor = 1.0f;
        request.position = lightSample.position;
        request.seed = &prd.seed;
        request.triangleId = idxTriangle;

        mdl::MaterialEvaluation matEval;

        evaluateMaterial(programCallId, &request, &matEval);

        if (matEval.opacity <= 0.0f)
        {
            return lightSample;
        }

        if (matEval.edf.isValid)
        {
            const float totArea = meshLightAttributes.totalArea;

            // Modulate the emission with the cutout opacity value to get the correct value.
            // The opacity value must not be greater than one here, which could happen for HDR textures.
            float opacity = math::min(matEval.opacity, 1.0f);

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

        lightSample.distance = params.settings->maxClamp; // Environment light.

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

    __forceinline__ __device__ void prepareHitProperties(HitProperties* hitP, RayWorkItem& prd, const LaunchParams& params)
    {
        prd.origin = prd.origin + prd.direction * prd.hitDistance;
        hitP->position = prd.origin;
        hitP->direction = prd.direction;
        hitP->baricenter = prd.hitBaricenter;
        hitP->seed = prd.seed;
        hitP->mediumIor = prd.mediumIor;

        utl::getInstanceAndGeometry(hitP, prd.hitInstanceId, params);
        utl::getVertices(hitP, prd.hitTriangleId);
        utl::setTransform(hitP);
        utl::computeGeometricHitProperties(hitP, prd.hitTriangleId);
        utl::determineMaterialHitProperties(hitP, prd.hitTriangleId);
    }

    __forceinline__ __device__ void setAuxiliaryRenderPassData(RayWorkItem& prd, const mdl::MaterialEvaluation& matEval, LaunchParams* params)
    {
        //Auxiliary Data
        if (prd.depth == 0)
        {
            math::vec3f colorsTrueNormal = 0.5f * (matEval.trueNormal + 1.0f);
            math::vec3f colorsUv = matEval.uv;
            math::vec3f colorsOrientation = matEval.isFrontFace ? math::vec3f(0.0f, 0.0f, 1.0f) : math::vec3f(1.0f, 0.0f, 0.0f);
            math::vec3f colorsTangent = 0.5f * (matEval.tangent + 1.0f);
            if(matEval.aux.isValid)
			{
                math::vec3f colorsBounceDiffuse = matEval.aux.albedo;
                math::vec3f colorsShadingNormal = 0.5f * (matEval.aux.normal + 1.0f);
                if (params->settings->adaptiveSampling && params->settings->minAdaptiveSamples <= params->settings->iteration)
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
            if (params->settings->adaptiveSampling && params->settings->minAdaptiveSamples <= params->settings->iteration)
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

    __forceinline__ __device__ void evaluateMaterialAndSampleLight(
        mdl::MaterialEvaluation* matEval, LightSample* lightSample, LaunchParams& params,
        RayWorkItem& prd)
    {
        mdl::MdlRequest request;
        request.instance = params.instances[prd.hitInstanceId];
        request.geometry = request.instance->geometryData;
        int programCallId;
        getMaterialData(request.instance, request.geometry, prd.hitTriangleId, programCallId, request.edf, &request.argBlock, &request.textureHandler);
        prd.depth==0 ? request.auxiliary = true : request.auxiliary = false;
        request.ior = true;
        request.outgoingDirection = -prd.direction;
        request.surroundingIor = prd.mediumIor;
        request.opacity = false;
        request.position = prd.origin;
        request.baricenter = prd.hitBaricenter;
        request.seed = &prd.seed;
        request.triangleId = prd.hitTriangleId;

        RendererDeviceSettings::SamplingTechnique& samplingTechnique = params.settings->samplingTechnique;
        const bool doSampleLight = samplingTechnique == RendererDeviceSettings::S_DIRECT_LIGHT || samplingTechnique == RendererDeviceSettings::S_MIS;
        const bool doSampleBsdf = samplingTechnique == RendererDeviceSettings::S_MIS || samplingTechnique == RendererDeviceSettings::S_BSDF;

        if (doSampleLight)
        {
            *lightSample = sampleLight(prd, params);
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

        evaluateMaterial(programCallId, &request, matEval);
    }

    __forceinline__ __device__ ShadowWorkItem nextEventEstimation(const mdl::MaterialEvaluation& matEval, LightSample& lightSample, RayWorkItem& prd, LaunchParams& params)
    {
        RendererDeviceSettings::SamplingTechnique& samplingTechnique = params.settings->samplingTechnique;
        ShadowWorkItem shadowWorkItem;
        shadowWorkItem.distance = -1; //invalid

        //Direct Light Sampling
        //prd.shadowTrace = false;
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

                shadowWorkItem.radiance = prd.throughput * bxdf * lightSample.radianceOverPdf * weightMis * (float)params.numberOfLights; // *float(numLights);
                shadowWorkItem.direction = lightSample.direction;
                shadowWorkItem.distance = lightSample.distance - params.settings->minClamp;
                shadowWorkItem.depth = prd.depth+1;
                shadowWorkItem.originPixel = prd.originPixel;
                shadowWorkItem.origin = prd.origin;
            }
        }

        return shadowWorkItem;
    }

    __forceinline__ __device__ void evaluateEmission(mdl::MaterialEvaluation& matEval, RayWorkItem& prd, LaunchParams& params)
    {
        RendererDeviceSettings::SamplingTechnique& samplingTechnique = params.settings->samplingTechnique;
        if (matEval.edf.isValid)
        {
            const InstanceData* instance = params.instances[prd.hitInstanceId];
            const GeometryData* geometry = (instance->geometryData);
            const unsigned& materialSlotIndex = geometry->faceAttributeData[prd.hitTriangleId].materialSlotId;
            const LightData* lightData = instance->materialSlots[materialSlotIndex].meshLight;
            const MeshLightAttributesData* attributes = reinterpret_cast<MeshLightAttributesData*>(lightData->attributes);
            const float area = attributes->totalArea;
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

    __forceinline__ __device__ bool russianRoulette(RayWorkItem& prd, LaunchParams& params)
    {
        if (params.settings->useRussianRoulette && 2 <= prd.depth) // Start termination after a minimum number of bounces.
        {
            const float probability = fmaxf(fmaxf(prd.throughput.x, prd.throughput.y), prd.throughput.z);

            if (probability < rng(prd.seed)) // Paths with lower probability to continue are terminated earlier.
            {
                return false;
            }
            prd.throughput /= probability; // Path isn't terminated. Adjust the throughput so that the average is right again.
        }
        return true;
    }

    __forceinline__ __device__ bool bsdfSample(RayWorkItem& prd, const mdl::MaterialEvaluation& matEval, LaunchParams& params)
    {
        RendererDeviceSettings::SamplingTechnique& samplingTechnique = params.settings->samplingTechnique;
        if ((samplingTechnique == RendererDeviceSettings::S_MIS || samplingTechnique == RendererDeviceSettings::S_BSDF)
            && prd.depth + 1 < params.settings->maxBounces
            && matEval.bsdfSample.eventType != mi::neuraylib::BSDF_EVENT_ABSORB)
        {
            //prd.origin = prd.origin;
            prd.direction = matEval.bsdfSample.nextDirection; // Continuation direction.
            prd.throughput *= matEval.bsdfSample.bsdfOverPdf;
            // Adjust the path throughput for all following incident lighting.
            prd.pdf = matEval.bsdfSample.pdf;
            // Note that specular events return pdf == 0.0f! (=> Not a path termination condition.)
            prd.eventType = matEval.bsdfSample.eventType;

            if (!matEval.isThinWalled && (prd.eventType & mi::neuraylib::BSDF_EVENT_TRANSMISSION) != 0)
            {
                if (matEval.isFrontFace) // Entered a volume. 
                {
                    prd.mediumIor = matEval.ior;
                }
                else // if !isFrontFace. Left a volume.
                {
                    prd.mediumIor = 1.0f;
                }
            }
            return true;
        }
        return false;
    }

    __forceinline__ __device__ void nextWork(const TraceWorkItem& twi, const ShadowWorkItem& swi, LaunchParams& params)
    {
        if (twi.extendRay)
        {
            params.radianceTraceQueue->Push(twi);
        }
        else
        {
            AccumulationWorkItem awi{};
            awi.radiance = twi.radiance;
            awi.originPixel = twi.originPixel;
            awi.depth = twi.depth;
            params.accumulationQueue->Push(awi);
        }
        if(swi.distance >0.0f)
        {
        	params.shadowQueue->Push(swi);
		}
    }

    __forceinline__ __device__ void shade(LaunchParams* params, RayWorkItem& prd, ShadowWorkItem& swi, TraceWorkItem& twi)
    {
        /*HitProperties hitP;

        prepareHitProperties(&hitP, prd, *params);*/

        mdl::MaterialEvaluation matEval{};
        twi.extendRay = false;
        LightSample lightSample{};
        bool extend = false;

        if (hasMaterial(params, prd))
        {
            evaluateMaterialAndSampleLight(&matEval, &lightSample, *params, prd);

            swi = nextEventEstimation(matEval, lightSample, prd, *params);

            evaluateEmission(matEval, prd, *params);

            prd.pdf = 0.0f;

            extend = russianRoulette(prd, *params);
            if (extend)
            {
                extend = bsdfSample(prd, matEval, *params);
            }
        }
        setAuxiliaryRenderPassData(prd, matEval, params);

        twi.seed = prd.seed;
        twi.originPixel = prd.originPixel;
        twi.depth = prd.depth;
        twi.origin = prd.origin;
        twi.direction = prd.direction;
        twi.radiance = prd.radiance;
        twi.throughput = prd.throughput;
        twi.mediumIor = prd.mediumIor;
        twi.eventType = prd.eventType;
        twi.pdf = prd.pdf;
        twi.extendRay = extend;
        twi.depth = prd.depth+1;
    }

    __forceinline__ __device__ bool transparentAnyHit(RayWorkItem* prd, LaunchParams* params)
    {
        mdl::MdlRequest request;
        request.instance = params->instances[prd->hitInstanceId];
        if (request.instance->hasOpacity)
        {
            mdl::MaterialEvaluation matEval;
            request.geometry = request.instance->geometryData;
            int programCallId;
            getMaterialData(request.instance, request.geometry, prd->hitTriangleId, programCallId, request.edf, &request.argBlock, &request.textureHandler);
            if(request.argBlock == nullptr)
            {
                optixIgnoreIntersection();
                return false;
            }
            request.edf = false;
            request.outgoingDirection = -prd->direction;
            request.surroundingIor = prd->mediumIor;
            request.opacity = true;
            request.position = prd->origin;
            request.baricenter = prd->hitBaricenter;
            request.seed = &prd->seed;
            request.triangleId = prd->hitTriangleId;

            evaluateMaterial(programCallId, &request, &matEval);

            // Stochastic alpha test to get an alpha blend effect.
            // No need to calculate an expensive random number if the test is going to fail anyway.
            if (matEval.opacity < 1.0f && matEval.opacity <= rng(prd->seed))
            {
                optixIgnoreIntersection();
                return false;
            }
        	return true;
        }
        return false;
    }

    __forceinline__ __device__ int getAdaptiveSampleCount(const int& fbIndex, LaunchParams* params)
    {
        int samplesPerLaunch = 1;
        if (params->settings->adaptiveSampling)
        {
            if (params->settings->minAdaptiveSamples <= params->settings->iteration)
            {
                samplesPerLaunch = params->frameBuffer.noiseBuffer[fbIndex].adaptiveSamples;
                if(samplesPerLaunch <= 0) //direct miss
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

    __forceinline__ __device__ void wfInitRayEntry(int id, LaunchParams* params)
    {
        if(id==0)
        {
            params->radianceTraceQueue->Reset();
            params->shadeQueue->Reset();
            params->shadowQueue->Reset();
            params->escapedQueue->Reset();
            params->accumulationQueue->Reset();
        }
        cleanFrameBuffer(id, params);
        const int samplesPerLaunch = getAdaptiveSampleCount(id, params);
        if (samplesPerLaunch == 0) return;
        TraceWorkItem                twi;
        for (int i = 0; i < samplesPerLaunch; i++)
        {
            twi.seed = tea<4>(id + i, params->settings->iteration+*params->frameID);
            generateCameraRay(id, params, twi);
            params->radianceTraceQueue->Push(twi);
        }
    }

    __forceinline__ __device__ void handleShading(int queueWorkId, LaunchParams& params)
    {
        if (queueWorkId == 0)
        {
            params.radianceTraceQueue->Reset();
        }

        if (params.shadeQueue->Size() <= queueWorkId)
            return;

        RayWorkItem prd = (*params.shadeQueue)[queueWorkId];
        ShadowWorkItem swi;
        TraceWorkItem twi;
        shade(&params, prd, swi, twi);
        nextWork(twi, swi, params);
    }

    __forceinline__ __device__ void wfAccumulateEntry(int queueWorkId, LaunchParams* params)
    {
        /*if (queueWorkId == 0)
        {
            params->shadeQueue->Reset();
            params->shadowQueue->Reset();
            params->escapedQueue->Reset();
        }*/
        if (queueWorkId >= params->accumulationQueue->Size())
            return;
        const AccumulationWorkItem             awi = (*params->accumulationQueue)[queueWorkId];
        accumulateRay(awi, params);
    }

    __forceinline__ __device__ void wfEscapedEntry(int id, LaunchParams* params)
    {
        if (id >= params->escapedQueue->Size())
            return;

        EscapedWorkItem                ewi = (*params->escapedQueue)[id];

        missShader(ewi, params);

        AccumulationWorkItem awi;
        awi.originPixel = ewi.originPixel;
        awi.radiance = ewi.radiance;
        awi.depth = ewi.depth;
        params->accumulationQueue->Push(awi);
    }


#ifdef ARCHITECTURE_OPTIX

    __forceinline__ __device__ RayWorkItem optixHitProperties(RayWorkItem* prd)
    {
        prd->hitInstanceId = optixGetInstanceId();
        prd->hitTriangleId = optixGetPrimitiveIndex();
        prd->hitDistance = optixGetRayTmax();
        float2 bari = optixGetTriangleBarycentrics();
        prd->hitBaricenter = math::vec3f(1.0f - bari.x - bari.y, bari.x, bari.y);
        prd->origin = prd->origin + prd->direction * prd->hitDistance;

        //const OptixTraversableHandle handle = optixGetTransformListHandle(0);
        //// UNSURE IF THIS IS CORRECT! WE ALWAYS HAVE THE TRANSFORM FROM THE INSTANCE DATA IN CASE
        //const float4* wTo = optixGetInstanceInverseTransformFromHandle(handle);
        //const float4* oTw = optixGetInstanceTransformFromHandle(handle);

        //prd->hitOTW = math::affine3f(oTw);
        //prd->hitWTO = math::affine3f(wTo);
    }

    template <typename T>
    __forceinline__ __device__ bool trace(math::vec3f& origin, math::vec3f& direction, float distance, T* rd, int sbtIndex, LaunchParams& params, OptixRayFlags flags = OPTIX_RAY_FLAG_NONE)
    {
        math::vec2ui payload = splitPointer(rd);

        optixTrace(params.topObject,
            origin,
            direction, // origin, direction
            params.settings->minClamp,
            distance,
            0.0f, // tmin, tmax, time
            static_cast<OptixVisibilityMask>(0xFF),
            flags,    //OPTIX_RAY_FLAG_NONE,
            sbtIndex,  //SBT Offset
            0,                                // SBT stride
            sbtIndex, // missSBTIndex
            payload.x,
            payload.y);

        if (payload.x == 0)
        {
            return false;
        }
        return true;
    }

    __forceinline__ __device__ void elaborateShadowTrace(ShadowWorkItem& swi, LaunchParams& params, ArchitectureType architecture = A_WAVEFRONT_CUDA_SHADE)
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
                params.accumulationQueue->Push(awi);
            }
            //printf("Shadow Trace Hit\n");
            addDebug(RED, swi.originPixel, &params);
        }
        else
        {
            //printf("Shadow Trace Miss\n");
            addDebug(BLUE, swi.originPixel, &params);
        }
        /*if (hit == false)
        {
            AccumulationWorkItem awi;
            awi.radiance = math::vec3f(1.0f, 0.0f, 0.0f);
            awi.depth = swi.depth;
            awi.originPixel = swi.originPixel;
            if (architecture == A_FULL_OPTIX)
            {
                accumulateRay(awi, &params);
            }
            else
            {
                params.accumulationQueue->Push(awi);
            }
        }
        else
        {
            AccumulationWorkItem awi;
            awi.radiance = math::vec3f(0.0f, 0.0f, 1.0f);
            awi.depth = swi.depth;
            awi.originPixel = swi.originPixel;
            if (architecture == A_FULL_OPTIX)
            {
                accumulateRay(awi, &params);
            }
            else
            {
                params.accumulationQueue->Push(awi);
            }
        }*/
    }

    __forceinline__ __device__ void elaborateRadianceTrace(TraceWorkItem& twi, LaunchParams& params, ArchitectureType architecture = A_WAVEFRONT_CUDA_SHADE)
    {
        RayWorkItem prd{};
        prd.seed = twi.seed;
        prd.originPixel = twi.originPixel;
        prd.depth = twi.depth;
        prd.origin = twi.origin;
        prd.direction = twi.direction;
        prd.radiance = twi.radiance;
        prd.throughput = twi.throughput;
        prd.mediumIor = twi.mediumIor;
        prd.eventType = twi.eventType;
        prd.pdf = twi.pdf;

        bool hit = trace(twi.origin, twi.direction, params.settings->maxClamp, &prd, 0, params, OPTIX_RAY_FLAG_DISABLE_ANYHIT);
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
                params.shadeQueue->Push(prd);
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
                params.escapedQueue->Push(ewi);
            }
        }
    }

    __forceinline__ __device__ void wfTraceRadianceEntry(int queueWorkId, LaunchParams& params)
    {
        if (queueWorkId == 0)
        {
            params.shadeQueue->Reset();
        }

        int radianceTraceQueueSize = params.radianceTraceQueue->Size();
        if (radianceTraceQueueSize <= queueWorkId)
            return;

        TraceWorkItem twi = (*params.radianceTraceQueue)[queueWorkId];
        // Shadow Trace
        bool isLongPath = (float)radianceTraceQueueSize <= params.settings->longPathPercentage * (float)params.settings->maxTraceQueueSize;
        if(!params.settings->useLongPathKernel || !isLongPath)
        {
            elaborateRadianceTrace(twi, params);
        }
        else
        {
            int remainingBounces = params.settings->maxBounces - twi.depth;
            for (int i = 0; i < remainingBounces; i++)
            {
                //printf("Remaining Bounces");
                elaborateRadianceTrace(twi, params, A_FULL_OPTIX);
                if (!twi.extendRay)
                {
                    break;
                }
            }
        }
    }

    __forceinline__ __device__ void wfTraceShadowEntry(int queueWorkId, LaunchParams& params)
    {
        if (params.shadowQueue->Size() <= queueWorkId)
            return;

        ShadowWorkItem swi = (*params.shadowQueue)[queueWorkId];
        // Shadow Trace

        elaborateShadowTrace(swi, params);
    }

#endif
}

#endif
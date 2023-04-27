#ifndef OPTIXCODE
#define OPTIXCODE
#endif

#include <optix_device.h>
#include "RayData.h"
#include "DataFetcher.h"
#include "mdlDeviceWrapper.h"

namespace vtx {


    __forceinline__ __device__ uint32_t makeColor(const math::vec3f& radiance)
    {

        const auto r = static_cast<uint8_t>(radiance.x * 255.0f);
        const auto g = static_cast<uint8_t>(radiance.y * 255.0f);
        const auto b = static_cast<uint8_t>(radiance.z * 255.0f);
        const auto a = static_cast<uint8_t>(1.0f * 255.0f);
        const uint32_t returnColor = (a << 24) | (b << 16) | (g << 8) | r;
        return returnColor;


    }
    //------------------------------------------------------------------------------
    // closest hit and anyhit programs for radiance-type rays.
    //
    // Note eventually we will have to create one pair of those for each
    // ray type and each geometry type we want to render; but this
    // simple example doesn't use any actual geometries yet, so we only
    // create a single, dummy, set of them (we do have to have at least
    // one group of them to set up the SBT)
    //------------------------------------------------------------------------------

    extern "C" __global__ void __exception__all()
    {
        //const uint3 theLaunchDim     = optixGetLaunchDimensions(); 
        const uint3 theLaunchIndex = optixGetLaunchIndex();
        const int   theExceptionCode = optixGetExceptionCode();
        const char* exceptionLineInfo = optixGetExceptionLineInfo();

        printf("Optix Exception: \n"
			   "    Code: %d\n"
			   "    LineInfo: %s\n"
			   "    at launch Index (pixel): x = %u y = %u\n",
               theExceptionCode, exceptionLineInfo, theLaunchIndex.x, theLaunchIndex.y);

        // FIXME This only works for render strategies where the launch dimension matches the outputBuffer resolution.
        //float4* buffer = reinterpret_cast<float4*>(sysData.outputBuffer);
        //const unsigned int index = theLaunchIndex.y * theLaunchDim.x + theLaunchIndex.x;

        //buffer[index] = make_float4(1000000.0f, 0.0f, 1000000.0f, 1.0f); // super magenta
    }

    __forceinline__ __device__ void computeHitProperties(HitProperties* hitP,const PerRayData& prd,const vtxID& instanceId,const vtxID& primitiveId)
    {
        hitP->instance = getData<InstanceData>(instanceId);
        hitP->geometry = getData<GeometryData>(hitP->instance->geometryDataId);

        const float4* wTo = optixGetInstanceInverseTransformFromHandle(optixGetTransformListHandle(hitP->geometry->traversable));
        const float4* oTw = optixGetInstanceTransformFromHandle(optixGetTransformListHandle(hitP->geometry->traversable));
        //
        //const float4* wTo = optixGetInstanceInverseTransformFromHandle(optixGetTransformListHandle(0));
        //const float4* oTw = optixGetInstanceTransformFromHandle(optixGetTransformListHandle(0));
        hitP->objectToWorld = math::affine3f(oTw);
        hitP->worldToObject = math::affine3f(wTo);

        hitP->distance = optixGetRayTmax();
        hitP->position = prd.position + prd.wi * hitP->distance;

        const math::vec3ui  triVerticesIndices = reinterpret_cast<math::vec3ui*>(hitP->geometry->indicesData)[primitiveId];

        const graph::VertexAttributes   vertex0         = hitP->geometry->vertexAttributeData[triVerticesIndices.x];
        const graph::VertexAttributes   vertex1         = hitP->geometry->vertexAttributeData[triVerticesIndices.y];
        const graph::VertexAttributes   vertex2         = hitP->geometry->vertexAttributeData[triVerticesIndices.z];
        const unsigned&                 materialSlot    = hitP->geometry->faceAttributeData[primitiveId].materialSlotId;

        const math::vec2f baricenter = optixGetTriangleBarycentrics();
        const float alpha = 1.0f - baricenter.x - baricenter.y;


        hitP->nsO = vertex0.normal * alpha + vertex1.normal * baricenter.x + vertex2.normal * baricenter.y;
        hitP->ngO = cross(vertex1.position - vertex0.position, vertex2.position - vertex0.position);
        hitP->tgO = vertex0.tangent * alpha + vertex1.tangent * baricenter.x + vertex2.tangent * baricenter.y;

        // TODO we already have the inverse so there can be some OPTIMIZATION here
        hitP->nsW = math::normalize(transformNormal3F(hitP->objectToWorld, hitP->nsO));
        hitP->ngW = math::normalize(transformNormal3F(hitP->objectToWorld, hitP->ngO));
        hitP->tgW = math::normalize(transformVector3F(hitP->objectToWorld, hitP->tgO));

        math::vec3f bt = math::normalize(cross(hitP->nsW, hitP->tgW));
        hitP->tgW = cross(bt, hitP->nsW);
        
        hitP->textureCoordinates[0] = vertex0.texCoord * alpha + vertex1.texCoord * baricenter.x + vertex2.texCoord * baricenter.y;
        hitP->textureBitangents[0] = bt;
        hitP->textureTangents[0] = hitP->tgW;

        hitP->textureCoordinates[1] = hitP->textureCoordinates[0];
        hitP->textureBitangents[1] = bt;
        hitP->textureTangents[1] = hitP->tgW;

        // Explicitly include edge-on cases as frontface condition!
        hitP->isFrontFace = 0.0f <= dot(prd.wo, hitP->ngW);

        if(hitP->instance->numberOfMaterials > 0)
        {
        	hitP->material = getData<MaterialData>(hitP->instance->materialsDataId[materialSlot]);
			hitP->shader = getData<ShaderData>(hitP->material->shaderId);
            hitP->shaderConfiguration = hitP->shader->shaderConfiguration;

            vtxID meshLightId = hitP->instance->meshLightDataId[materialSlot];
            if (meshLightId != INVALID_INDEX)
            {
                const LightData* lightData = getData<LightData>(meshLightId);
                const MeshLightAttributes* attributes = reinterpret_cast<MeshLightAttributes*>(lightData->attributes);
                hitP->meshLight = lightData;
                hitP->meshLightAttributes = attributes;
			}
			else
			{
				hitP->meshLight = nullptr;
                hitP->meshLightAttributes = nullptr;
			}
		}
		else
		{
			hitP->material = nullptr;
			hitP->shader = nullptr;
			hitP->shaderConfiguration = nullptr;
            hitP->meshLight = nullptr;
            hitP->meshLightAttributes = nullptr;
		}
    }

    extern "C" __global__ void __closesthit__radiance()
    {
        PerRayData* prd = mergePointer(optixGetPayload_0(), optixGetPayload_1());
        const unsigned int instanceId = optixGetInstanceId();
        const unsigned int primitiveId = optixGetPrimitiveIndex();

        HitProperties hitP;
        computeHitProperties(&hitP, *prd, instanceId, primitiveId);

        prd->distance = hitP.distance;
        prd->position = hitP.position;

        if(hitP.material!=nullptr)
        {
            mdl::MdlData mdlData;
            mdl::initMdl(hitP, &mdlData);

            math::vec3f ior = mdl::getIor(mdlData);

            if (prd->depth == 0)
            {
                prd->colors.trueNormal = 0.5f*(hitP.ngW+1.0f);
                prd->colors.orientation = hitP.isFrontFace ? math::vec3f(0.0f, 0.0f, 1.0f) : math::vec3f(1.0f, 0.0f, 0.0f);

                mdl::BsdfAuxiliaryData auxiliaryData = mdl::getAuxiliaryData(mdlData, ior, prd->idxStack, prd->stack, hitP.ngW);

                if (auxiliaryData.isValid)
                {
                    prd->colors.diffuse = auxiliaryData.data.albedo;
                    prd->colors.shadingNormal = auxiliaryData.data.normal;
                }
                else
                {
                    prd->colors.diffuse = math::vec3f(1.0f, 0.0f, 1.0f);
                    prd->colors.shadingNormal = hitP.ngW;
                }
            }

            //Evauluate Hit Point Emission
            if(hitP.meshLightAttributes!=nullptr && mdlData.emissionFunctions.hasEmission)
            {
                mdl::EmissionEvaluateData evalData = mdl::evaluateEmission(mdlData, prd->wo);
                if (evalData.isValid)
                {
                    const float area = hitP.meshLightAttributes->totalArea;
                    evalData.data.pdf = prd->distance * prd->distance / (area * evalData.data.cos); // Solid angle measure.

                    float MisWeight = 1.0f;

                    // If the last event was diffuse or glossy, calculate the opposite MIS weight for this implicit light hit.
                    //if (sysData.directLighting && (prd->eventType & (mi::neuraylib::BSDF_EVENT_DIFFUSE | mi::neuraylib::BSDF_EVENT_GLOSSY)))
                    //{
                    //    weightMIS = balanceHeuristic(prd->pdf, evalData.pdf);
                    //}

                    // Power (flux) [W] divided by light area gives radiant exitance [W/m^2].
                    const float factor = (mdlData.emissionFunctions.mode == 0) ? 1.0f : 1.0f / area;

                    prd->radiance += prd->throughput * mdlData.emissionFunctions.intensity * evalData.data.edf * (factor * MisWeight);
                    //prd->radiance += prd->throughput * mdlData.emissionFunctions.intensity * evalData.data.edf * (1.0f * MisWeight);
                    if(prd->depth==0)
                    {
                        prd->colors.debugColor1 = mdlData.emissionFunctions.intensity;
                        prd->colors.debugColor2 = evalData.data.edf;
                        prd->colors.debugColor3 = prd->throughput;
                    }



                    //prd->radiance += math::vec3f(100.0f);
                    //printMath("Radiance: ", prd->radiance);
                }
                else
                {
                    const float a = 0.0f;
                }
                
            }

            //Importance Sampling the Bsdf
            prd->pdf = 0.0f;
            mdl::BsdfSampleData sampleData = mdl::sampleBsdf(mdlData, ior, prd->idxStack, prd->stack, prd->wo, prd->seed);
            if (sampleData.isValid)
            {
                prd->wi = sampleData.data.k2;            // Continuation direction.
                prd->throughput *= sampleData.data.bsdf_over_pdf; // Adjust the path throughput for all following incident lighting.
                prd->pdf = sampleData.data.pdf;           // Note that specular events return pdf == 0.0f! (=> Not a path termination condition.)
                prd->eventType = sampleData.data.event_type;    // This replaces the PRD flags used inside the other examples.
            }
            else
            {
                return;
                // None of the following code will have any effect in that case.
            }


            // Fetch Auxiliary Data
            



#ifdef MIS

            // CODE SNIPPETS FOR BSDF EVALUATION
            math::vec3f lightSampleDirection = sampleData.data.k2; // temporary
            math::vec3f evalIncoming = -(prd->wo - 2 * dot(prd->wo, ngW) * ngW);
            math::vec3f evalOutgoing = prd->wo;

            mdl::BsdfEvaluateData evalData = mdl::evaluateBsdf(mdlData, ior, prd->idxStack, prd->stack, evalIncoming, evalOutgoing);

            prd->radiance = prd->throughput;
            float factor = 1.0f / (dot(evalIncoming, ngW));
            prd->colors.diffuse = evalData.data.bsdf_diffuse;
            prd->colors.diffuse *= factor;
            prd->colors.specular = evalData.data.bsdf_glossy;
            prd->colors.specular *= factor;
            prd->colors.shadingNormal = mdlData.state.normal;

            evalIncoming = ngW;
            evalOutgoing = ngW;
            evalData.data = mdl::evaluateBsdf(mdlData, ior, prd->idxStack, prd->stack, evalIncoming, evalOutgoing);
            prd->colors.debugColor1 = evalData.data.bsdf_diffuse;


#endif
        }

    }

    extern "C" __global__ void __anyhit__radiance()
    { /*! for this simple example, this will remain empty */
    }



    //------------------------------------------------------------------------------
    // miss program that gets called for any ray that did not have a
    // valid intersection
    //
    // as with the anyhit/closest hit programs, in this example we only
    // need to have _some_ dummy function to set up a valid SBT
    // ------------------------------------------------------------------------------

    extern "C" __global__ void __miss__radiance()
    { /*! for this simple example, this will remain empty */
        PerRayData* prd = mergePointer(optixGetPayload_0(), optixGetPayload_1());
        prd->colors.debugColor1 = math::vec3f(0.2f, 0.2f, 0.2f);
        if (prd->depth == 0)
        {
            prd->colors.diffuse = prd->colors.debugColor1;
            prd->colors.orientation = prd->colors.debugColor1;
            prd->colors.shadingNormal = prd->colors.debugColor1;
            prd->colors.trueNormal = prd->colors.debugColor1;
            prd->colors.debugColor2 = prd->colors.debugColor1;
            prd->colors.debugColor3 = prd->colors.debugColor1;

        }
        //printVector(thePrd->color, "MISS COLOR");
    }

    __forceinline__ __device__ math::vec3f integrator(PerRayData& prd)
    {
        // The integrator starts with black radiance and full path throughput.
        prd.radiance = math::vec3f(0.0f);
        prd.pdf = 0.0f;
        prd.throughput = math::vec3f(1.0f);
        prd.sigmaT = math::vec3f(0.0f); // Extinction coefficient: sigma_a + sigma_s.
        prd.walk = 0;                 // Number of random walk steps taken through volume scattering. 
        prd.eventType = mi::neuraylib::BSDF_EVENT_ABSORB; // Initialize for exit. (Otherwise miss programs do not work.)
        
        prd.idxStack = 0; // Nested material handling. 
        // Small stack of four entries of which the first is vacuum.
        prd.stack[0].ior = math::vec3f(1.0f); // No effective IOR.
        prd.stack[0].absorption = math::vec3f(0.0f); // No volume absorption.
        prd.stack[0].scattering = math::vec3f(0.0f); // No volume scattering.
        prd.stack[0].bias = 0.0f;              // Isotropic volume scattering.
        prd.depth = 0;

        int depth = 0;
        int maxDepth = optixLaunchParams.settings->maxBounces;
        float maxDistance = 1000.0f;
        float minDistance = 0.00001f;

        math::vec2ui payload = splitPointer(&prd);

        while (prd.depth < maxDepth)
        {
			prd.wo = -prd.wi;
            prd.distance = maxDistance;
            prd.flags = 0;

            optixTrace(optixLaunchParams.topObject,
                       prd.position,
                       prd.wi, // origin, direction
                       minDistance,
                       maxDistance,
                       0.0f, // tmin, tmax, time
                       static_cast<OptixVisibilityMask>(0xFF),
                       OPTIX_RAY_FLAG_DISABLE_ANYHIT, //OPTIX_RAY_FLAG_NONE,
                       TYPE_RAY_RADIANCE, //SBT Offset
                       NUM_RAY_TYPES, // SBT stride
                       TYPE_RAY_RADIANCE, // missSBTIndex
                       payload.x,
                       payload.y);


            // Path termination by miss shader or sample() routines.
            if ((prd.eventType == mi::neuraylib::BSDF_EVENT_ABSORB) || (prd.throughput == math::vec3f(0.0f)))
            {
                break;
            }

            ++prd.depth; // Next path segment.

        }

        return prd.radiance;

    }


    //------------------------------------------------------------------------------
    // ray gen program - the actual rendering happens in here
    //------------------------------------------------------------------------------
    extern "C" __global__ void __raygen__renderFrame()
    {
        const FrameBufferData* frameBuffer = getData<FrameBufferData>();
	    const math::vec2ui& frameSize = frameBuffer->frameSize;
        

        const int ix = optixGetLaunchIndex().x;
        const int iy = optixGetLaunchIndex().y;
        const uint32_t fbIndex = ix + iy * frameSize.x;

        PerRayData prd;



        const math::vec2f pixel = math::vec2f(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
        //const math::vec2f sample = rng2(prd.seed);
        const math::vec2f sample = math::vec2f{0.5f, 0.5f};
        const math::vec2f screen{ static_cast<float>(frameSize.x), static_cast<float>(frameSize.y) };

        const LensRay cameraRay = optixDirectCall<LensRay, const math::vec2f, const math::vec2f, const math::vec2f>(0, screen, pixel, sample);

        prd.position = cameraRay.org;
        prd.wi = cameraRay.dir;
#define RANDOM_SAMPLING
#ifdef RANDOM_SAMPLING
        prd.seed = tea<4>(fbIndex, *optixLaunchParams.frameID); // PERF This template really generates a lot of instructions.
#else
        prd.seed = 0.5f; // Initialize the random number generator.
#endif

        //const int r = static_cast<int>(255.99f * prd.debugColor.x);
        //const int g = static_cast<int>(255.99f * prd.debugColor.y);
        //const int b = static_cast<int>(255.99f * prd.debugColor.z);
        
        // convert to 32-bit rgba value (we explicitly set alpha to 0xff
        // to make stb_image_write happy ...
        //const uint32_t rgba = 0xff000000
         //   | (r << 0) | (g << 8) | (b << 16);
        
        // and write to frame buffer ...
        
        // and write to frame buffer ...

        math::vec3f radiance = integrator(prd);

        math::vec4f* outputBuffer = reinterpret_cast<math::vec4f*>(frameBuffer->outputBuffer); // This is a per device launch sized buffer in this renderer strategy.
        const char* accumulate = optixLaunchParams.settings->accumulate ? "true" : "false";
        //printf("ACCUMULATE : %s \tITERATION : % d\n", accumulate, optixLaunchParams.settings->iteration);
        switch (frameBuffer->frameBufferType)
        {
        case(FrameBufferData::FrameBufferType::FB_NOISY) :
	        {
            //frameBuffer->radianceBuffer[fbIndex] = frameBuffer->radianceBuffer[fbIndex] + ((1.0f / float(getFrameId() + 1)) * (radiance - outputBuffer[fbIndex]));
            if(!optixLaunchParams.settings->accumulate || optixLaunchParams.settings->iteration == 0)
            {
                frameBuffer->radianceBuffer[fbIndex] = radiance;
            }
            else
            {
				frameBuffer->radianceBuffer[fbIndex] += radiance;
            }
            //outputBuffer[fbIndex] = makeColor(frameBuffer->radianceBuffer[fbIndex]);
            //outputBuffer[fbIndex] = makeColor(prd.radiance);
            outputBuffer[fbIndex] = math::vec4f(frameBuffer->radianceBuffer[fbIndex] / static_cast<float>(optixLaunchParams.settings->iteration), 1.0f);
            //outputBuffer[fbIndex] = math::vec4f(frameBuffer->radianceBuffer[fbIndex], 1.0f);
	        }
	        break;
        case(FrameBufferData::FrameBufferType::FB_DIFFUSE) :
	        {
		        outputBuffer[fbIndex] = math::vec4f(prd.colors.diffuse, 1.0f);
	        }
	        break;
        case(FrameBufferData::FrameBufferType::FB_ORIENTATION) :
	        {
		        outputBuffer[fbIndex] = math::vec4f(prd.colors.orientation, 1.0f);
	        }
	        break;
        case(FrameBufferData::FrameBufferType::FB_TRUE_NORMAL) :
	        {
		        outputBuffer[fbIndex] = math::vec4f(prd.colors.trueNormal, 1.0f);
	        }
	        break;
        case(FrameBufferData::FrameBufferType::FB_SHADING_NORMAL) :
	        {
		        outputBuffer[fbIndex] = math::vec4f(prd.colors.shadingNormal, 1.0f);
	        }
	        break;
        case(FrameBufferData::FrameBufferType::FB_DEBUG_1):
        {
            outputBuffer[fbIndex] = math::vec4f(prd.colors.debugColor1, 1.0f);
        }
        break;
        case(FrameBufferData::FrameBufferType::FB_DEBUG_2):
        {
            outputBuffer[fbIndex] = math::vec4f(prd.colors.debugColor2, 1.0f);
        }
        break;
        case(FrameBufferData::FrameBufferType::FB_DEBUG_3):
        {
            outputBuffer[fbIndex] = math::vec4f(prd.colors.debugColor3, 1.0f);
        }
        break;
        }
    }
}

#include <optix_device.h>
#include "RayData.h"
#include "DataFetcher.h"
#define TEX_SUPPORT_NO_VTABLES
#define TEX_SUPPORT_NO_DUMMY_SCENEDATA
#include "texture_lookup.h"

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

        printf("Exception %d at (%u, %u)\n", theExceptionCode, theLaunchIndex.x, theLaunchIndex.y);

        // FIXME This only works for render strategies where the launch dimension matches the outputBuffer resolution.
        //float4* buffer = reinterpret_cast<float4*>(sysData.outputBuffer);
        //const unsigned int index = theLaunchIndex.y * theLaunchDim.x + theLaunchIndex.x;

        //buffer[index] = make_float4(1000000.0f, 0.0f, 1000000.0f, 1.0f); // super magenta
    }

    extern "C" __global__ void __closesthit__radiance()
    { /*! for this simple example, this will remain empty */

        const unsigned int instanceId = optixGetInstanceId();
        const unsigned int primitiveId = optixGetPrimitiveIndex();

        const InstanceData* instance = getData<InstanceData>(instanceId);
        const GeometryData* geometry = getData<GeometryData>(instance->geometryDataId);

        vtxID* materialIds = nullptr;
        const MaterialData* material = nullptr;
        if (instance->numberOfMaterials > 0)
        {
            materialIds = instance->materialsDataId;
            material = getData<MaterialData>(instance->materialsDataId[0]);
        }

        PerRayData* prd = mergePointer(optixGetPayload_0(), optixGetPayload_1());
        prd->distance = optixGetRayTmax();
        prd->position += prd->wi * prd->distance;

        math::vec3ui triangleIndices = geometry->indicesData[primitiveId];

        const math::vec3ui* indices = reinterpret_cast<math::vec3ui*>(geometry->indicesData);
        const math::vec3ui  tri = indices[primitiveId];

        const graph::VertexAttributes vertex0 = geometry->vertexAttributeData[tri.x];
        const graph::VertexAttributes vertex1 = geometry->vertexAttributeData[tri.y];
        const graph::VertexAttributes vertex2 = geometry->vertexAttributeData[tri.z];

        const math::vec2f baricenter = optixGetTriangleBarycentrics();
        const float alpha = 1.0f - baricenter.x - baricenter.y;

        const float4* wTo = optixGetInstanceInverseTransformFromHandle(optixGetTransformListHandle(geometry->traversable));
        const float4* oTw = optixGetInstanceTransformFromHandle(optixGetTransformListHandle(geometry->traversable));

        const math::affine3f objectToWorld(oTw);
        const math::affine3f worldToObject(wTo);
        math::vec3f ns = vertex0.normal * alpha + vertex1.normal * baricenter.x + vertex2.normal * baricenter.y;
        math::vec3f ng = cross(vertex1.position - vertex0.position, vertex2.position - vertex0.position);
        math::vec3f tg = vertex0.tangent * alpha + vertex1.tangent * baricenter.x + vertex2.tangent * baricenter.y;
        ns = math::normalize(transformNormal3F(worldToObject, ns));
        ng = math::normalize(transformNormal3F(worldToObject, ng));
        tg = math::normalize(transformVector3F(worldToObject, tg));
        math::vec3f bt = math::normalize(cross(ns, tg));
        tg = cross(bt, ns);
        math::vec3f textureCoordinate = vertex0.texCoord * alpha + vertex1.texCoord * baricenter.x + vertex2.texCoord * baricenter.y;
        math::vec3f textureBitangent = bt;
        math::vec3f textureTangent = tg;

        prd->debugColor = 0.5f * (ns + 1.0f);
        //prd->debugColor = prd->position*0.25f;

        //typedef mi::neuraylib::Shading_state_material MdlState;
        //MdlState mdlState;
        //mdlState.normal;
        //mdlState.position;
        //printVector(thePrd->color, "HIT COLOR");
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
        PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());
        thePrd->debugColor = math::vec3f(0.2f, 0.2f, 0.2f);
        //printVector(thePrd->color, "MISS COLOR");
    }




    //------------------------------------------------------------------------------
    // ray gen program - the actual rendering happens in here
    //------------------------------------------------------------------------------
    extern "C" __global__ void __raygen__renderFrame()
    {
        const FrameBufferData* frameBuffer = getData<FrameBufferData>();
	    const math::vec2ui& frameSize = frameBuffer->frameSize;
        
    	const math::vec2f pixel = math::vec2f(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
        const math::vec2f sample{ 0.5f, 0.5f };
        const math::vec2f screen{ static_cast<float>(frameSize.x), static_cast<float>(frameSize.y) };
        
        const LensRay cameraRay = optixDirectCall<LensRay, const math::vec2f, const math::vec2f, const math::vec2f>(0, screen, pixel, sample);
        
        PerRayData prd;
        prd.radiance = math::vec3f(0.0f, 0.0f, 0.0f);
        prd.position = cameraRay.org;
        prd.wi = cameraRay.dir;
        math::vec2ui payload = splitPointer(&prd);
        
        float minClamp = 0.00001f;
        float maxClamp = 1000;
        
        optixTrace(optixLaunchParams.topObject,
                   prd.position,
                   prd.wi, // origin, direction
                   minClamp, 
                   maxClamp,
                   0.0f, // tmin, tmax, time
                   static_cast<OptixVisibilityMask>(0xFF), 
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT, //OPTIX_RAY_FLAG_NONE,
                   TYPE_RAY_RADIANCE, //SBT Offset
                   NUM_RAY_TYPES, // SBT stride
                   TYPE_RAY_RADIANCE, // missSBTIndex
                   payload.x,
                   payload.y);

        //const int r = static_cast<int>(255.99f * prd.debugColor.x);
        //const int g = static_cast<int>(255.99f * prd.debugColor.y);
        //const int b = static_cast<int>(255.99f * prd.debugColor.z);
        
        // convert to 32-bit rgba value (we explicitly set alpha to 0xff
        // to make stb_image_write happy ...
        //const uint32_t rgba = 0xff000000
         //   | (r << 0) | (g << 8) | (b << 16);
        
        // and write to frame buffer ...
        
        const int ix = optixGetLaunchIndex().x;
        const int iy = optixGetLaunchIndex().y;
        
        // and write to frame buffer ...
        const uint32_t fbIndex = ix + iy * frameSize.x;
        uint32_t* colorBuffer = reinterpret_cast<uint32_t*>(frameBuffer->colorBuffer); // This is a per device launch sized buffer in this renderer strategy.
        colorBuffer[fbIndex] = makeColor(prd.debugColor);
    }
}
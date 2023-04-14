#pragma once

#include <optix.h>
#include <optix_device.h>
#include <cuda_runtime.h>
#include "Device/LaunchParams.h"
#include "RayData.h"

namespace vtx {
	
    /*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
    extern "C" __constant__ LaunchParams optixLaunchParams;


    __device__ void printVector(const math::vec3f& vec, const char* message = nullptr) {
        printf("%s (%f %f %f)\n", message, vec.x, vec.y, vec.z);
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
        unsigned int instanceID = optixGetInstanceId();
        unsigned int primitveID = optixGetPrimitiveIndex();
        PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());
        thePrd->color = math::randomColor(instanceID+ primitveID);
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
        thePrd->color = math::vec3f(1.0f, 1.0f, 1.0f);
        //printVector(thePrd->color, "MISS COLOR");
    }




    //------------------------------------------------------------------------------
    // ray gen program - the actual rendering happens in here
    //------------------------------------------------------------------------------
    extern "C" __global__ void __raygen__renderFrame()
    {
        const int frameID = optixLaunchParams.frameID;
        if (frameID == 0 &&
            optixGetLaunchIndex().x == 0 &&
            optixGetLaunchIndex().y == 0) {
            // we could of course also have used optixGetLaunchDims to query
            // the launch size, but accessing the optixLaunchParams here
            // makes sure they're not getting optimized away (because
            // otherwise they'd not get used)
            printf("############################################\n");
            printf("Hello world from OptiX 7 raygen program!\n(within a %ix%i-sized launch)\n",
                   optixLaunchParams.fbSize.x,
                   optixLaunchParams.fbSize.y);
            printf("############################################\n");
        }

        // ------------------------------------------------------------------
        // for this example, produce a simple test pattern:
        // ------------------------------------------------------------------

        // compute a test pattern based on pixel ID
        const int ix = optixGetLaunchIndex().x;
        const int iy = optixGetLaunchIndex().y;

        math::vec2f pixel = math::vec2f(ix, iy);
        math::vec2f sample = math::vec2f(0.5f, 0.5f);
        math::vec2f screen = math::vec2f(optixLaunchParams.fbSize.x, optixLaunchParams.fbSize.y);

        const LensRay ray = optixDirectCall<LensRay, const math::vec2f, const math::vec2f, const math::vec2f>(0, screen, pixel, sample);

        PerRayData prd;
        prd.color = math::vec3f(0.0f, 0.0f, 0.0f);
        prd.position = ray.org;
        prd.wi = ray.dir;
        uint2 payload = splitPointer(&prd);

        float minClamp = 0.00001f;
        float maxClamp = 1000;

        //VTX_INFO("Ray Origin: %f %f %f", ray.org.x, ray.org.y, ray.org.z);
        //printVector(ray.org, "ray origin: ");
        //printVector(ray.dir, "ray direction: ");
        //printVector(camera.position, "Camera Position: ");
        //printVector(camera.right, "Camera Right: ");
        //printVector(camera.up, "Camera Up: ");
        //printVector(camera.direction, "Camera Direction: ");
        //printVector(prd.position, "PRD Position: ");
        //printVector(prd.wi, "PRD Right: ");

        optixTrace(optixLaunchParams.topObject,
                   prd.position,
                   prd.wi, // origin, direction
                   minClamp, 
                   maxClamp,
                   0.0f, // tmin, tmax, time
                   OptixVisibilityMask(0xFF), 
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT, //OPTIX_RAY_FLAG_NONE,
                   TYPE_RAY_RADIANCE, //SBT Offset
                   NUM_RAY_TYPES, // SBT stride
                   TYPE_RAY_RADIANCE, // missSBTIndex
                   payload.x,
                   payload.y);

        //const int r = ((ix + frameID) % 256);
        //const int g = ((iy + frameID) % 256);
        //const int b = ((ix + iy + frameID) % 256);

        //const int r = 50;
        //const int g = 50;
        //const int b = 50;


        //const int r = int(prd.color.x * 256);
        //const int g = int(prd.color.y * 256);
        //const int b = int(prd.color.z * 256);

        const int r = int(255.99f * prd.color.x);
        const int g = int(255.99f * prd.color.y);
        const int b = int(255.99f * prd.color.z);

        // convert to 32-bit rgba value (we explicitly set alpha to 0xff
        // to make stb_image_write happy ...
        const uint32_t rgba = 0xff000000
            | (r << 0) | (g << 8) | (b << 16);

        // and write to frame buffer ...
        const uint32_t fbIndex = ix + iy * optixLaunchParams.fbSize.x;
        uint32_t* colorBuffer = reinterpret_cast<uint32_t*>(optixLaunchParams.colorBuffer); // This is a per device launch sized buffer in this renderer strategy.
        colorBuffer[fbIndex] = rgba;
    }
}
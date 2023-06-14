#pragma once
#ifndef WORK_QUEUE_H
#define WORK_QUEUE_H
#include <cuda_runtime.h>
#include <mi/neuraylib/target_code_types.h>

#include "Core/Math.h"
#include "DevicePrograms/nvccUtils.h"

namespace vtx::mdl
{
    typedef mi::neuraylib::Bsdf_event_type BsdfEventType;

}
namespace vtx
{

    struct HitData
    {
        math::vec3f                  position;
        float                        distance;
        math::vec3f                  baricenter;
        unsigned                     instanceId;
        unsigned                     triangleId;
        math::affine3f         wTo;
        math::affine3f         oTw;
    };

    struct Colors
    {
        math::vec3f trueNormal;
        math::vec3f shadingNormal;
        math::vec3f bounceDiffuse;
        math::vec3f tangent;
        math::vec3f uv;
        math::vec3f orientation;
        math::vec3f debugColor1;
    };

    struct RayData
    {
        int    originPixel;
        unsigned seed;
        math::vec3f origin;
        math::vec3f direction;
        int depth;

        float pdf;
        math::vec3f radiance;
        math::vec3f throughput;
        math::vec3f mediumIor;
        mdl::BsdfEventType  eventType;

        HitData lastHit;

        Colors colors;
	};

    __forceinline__ __device__ void enqueueRayData(RayData** slots, RayData* job, int* size, const int* maxQueueSize)
    {
        if (*size >= *maxQueueSize)
        {
            printf("WorkQueue is full \n");
        }
        const int position = cuAtomicAdd(size, 1);
        slots[position] = job;
    }

    class WorkQueue
    {
    public:
        __forceinline__ __device__ void WorkQueue::addJob(RayData* job, const char* name, bool actuallySetValue = true)
        {
            if (size >= maxSize)
            {
                printf("WorkQueue %s is full \n", name);
	            
            }
            else
            {
                clock_t start = clock();
                int position = cuAtomicAdd(&size, 1);
                if (actuallySetValue)
                {
                    data[position] = job;
                }
            }
        }

        __forceinline__ __device__ void reset()
        {
	        size = 0;
        }

        RayData** data;
        unsigned maxSize;
        int 	 size = 0;

    };

}

#endif
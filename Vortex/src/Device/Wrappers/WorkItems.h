#pragma once
#ifndef WORK_ITEMS_H
#define WORK_ITEMS_H
#include "Core/Math.h"
#include <mi/neuraylib/target_code_types.h>
#include "Device/DevicePrograms/HitProperties.h"

namespace vtx::mdl
{
    typedef mi::neuraylib::Bsdf_event_type BsdfEventType;

}

namespace vtx
{

    struct ShadowWorkItem
    {
        math::vec3f origin;
        float distance = -1;
        math::vec3f direction;
        int originPixel;
        math::vec3f radiance;
        int depth;
        math::vec3f mediumIor;
        unsigned seed;
    };

    struct RayWorkItem
    {
        unsigned seed;
        int originPixel;
        int depth;
        math::vec3f direction;
        math::vec3f radiance;
        math::vec3f throughput;
        math::vec3f mediumIor;
        mdl::BsdfEventType eventType;
        float pdf;
        float hitDistance;
        HitProperties hitProperties;
    };

    struct TraceWorkItem
    {
        unsigned seed;
        int originPixel;
        int depth;
        math::vec3f origin;
        math::vec3f direction;
        math::vec3f radiance;
        math::vec3f throughput;
        math::vec3f mediumIor;
        mdl::BsdfEventType eventType;
        float pdf;
        bool extendRay;
    };

    struct EscapedWorkItem
    {
        unsigned seed;
        int originPixel;
        int depth;
        math::vec3f direction;
        math::vec3f radiance;
        math::vec3f throughput;
        mdl::BsdfEventType eventType;
        float pdf;
    };

    struct AccumulationWorkItem
    {
        int originPixel;
        math::vec3f radiance;
        int depth;
    };
}



#endif
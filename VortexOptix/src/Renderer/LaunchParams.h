#pragma once
#include "Core/math.h"
#include "Core/VortexID.h"
#include "cuda.h"
#include "optix.h"

namespace vtx {

    // Camera
    struct CameraData {
        math::vec3f  position;
        math::vec3f  vertical;
        math::vec3f  horizontal;
        math::vec3f  direction;
    };

    // This struct are uploaded
    struct GeometryInstanceData {
        vtxID						InstanceId;
        CUdeviceptr					Vertexattributes;
        CUdeviceptr					indices;
        CUdeviceptr					MaterialArray;
    };

    struct LaunchParams
    {
        int                         frameID{ 0 };
        CUdeviceptr                 colorBuffer;
        math::vec2i                 fbSize;

        OptixTraversableHandle      topObject;
        GeometryInstanceData*       geometryInstanceData; // Attributes, indices, idMaterial, idLight, idObject per instance.
        CameraData                  CameraData;
    };

    enum TypeRay
    {
        TYPE_RAY_RADIANCE = 0,
        NUM_RAY_TYPES
    };

    struct LensRay
    {
        math::vec3f org;
        math::vec3f dir;
    };

} // ::osc
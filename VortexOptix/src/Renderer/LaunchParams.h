#pragma once
#include "Core/math.h"
#include "Core/VortexID.h"
#include "cuda.h"
#include "optix.h"

namespace vtx {

    // Camera
    struct CameraData {
        math::vec3f  position;
        math::vec3f  up;
        math::vec3f  right;
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
        CameraData*                 CameraData;
    };

    struct LensRay
    {
        math::vec3f org;
        math::vec3f dir;
    };

} // ::osc
#pragma once
#include "Core/math.h"
#include "Core/VortexID.h"
#include "cuda.h"
#include "optix.h"

namespace vtx {

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
    };

} // ::osc
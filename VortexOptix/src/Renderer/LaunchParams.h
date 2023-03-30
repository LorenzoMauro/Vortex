#pragma once
#include "Core/math.h"

namespace vtx {

    struct LaunchParams
    {
        int             frameID{ 0 };
        CUdeviceptr     colorBuffer;
        math::vec2i     fbSize;
    };

} // ::osc
#ifndef OPTIXCODE
#define OPTIXCODE
#endif

#include "../LaunchParams.h"
#include "Core/Math.h"


namespace vtx {

    extern "C" __constant__ LaunchParams optixLaunchParams;

}
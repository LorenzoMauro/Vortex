#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Device/DevicePrograms/rendererFunctions.h"

namespace vtx
{

    extern "C" __global__ void wfShadeEntry(LaunchParams * params)
    {
        const unsigned int queueWorkId = blockIdx.x * blockDim.x + threadIdx.x;

        handleShading(queueWorkId, *params);
    }
}

#include <optix_device.h>
#include <cuda_runtime.h>
#include "Renderer/LaunchParams.h"
#include "Core/Math.h"
#include <optix.h>

namespace vtx {

    extern "C" __constant__ LaunchParams optixLaunchParams;

    extern "C" __device__ LensRay __direct_callable__pinhole(const math::vec2f screen, const math::vec2f pixel, const math::vec2f sample) {
        const math::vec2f fragment = pixel + sample;                    // Jitter the sub-pixel location
        const math::vec2f ndc = (fragment / screen) * 2.0f - 1.0f;      // Normalized device coordinates in range [-1, 1].

        const CameraData camera = optixLaunchParams.CameraData;

        LensRay ray;

        ray.org = camera.position;
        ray.dir = math::normalize<float>(camera.horizontal * ndc.x + camera.vertical * ndc.y + camera.direction);
        return ray;
    }
}
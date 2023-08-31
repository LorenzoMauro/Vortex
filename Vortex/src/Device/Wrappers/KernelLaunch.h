// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_GPU_UTIL_H
#define PBRT_GPU_UTIL_H

#include <map>
#include <typeindex>
#include <utility>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "KernelTimings.h"
#include "Core/Log.h"
#include "Device/CUDAChecks.h"


namespace vtx {

    template <typename F>
    inline int GetBlockSize(const char* description, F kernel) {
        // Note: this isn't reentrant, but that's fine for our purposes...
        static std::map<std::type_index, int> kernelBlockSizes;

        std::type_index index = std::type_index(typeid(F));

        auto iter = kernelBlockSizes.find(index);
        if (iter != kernelBlockSizes.end())
            return iter->second;

        int minGridSize, blockSize;
        CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0));
        kernelBlockSizes[index] = blockSize;
        VTX_INFO("[{}]: block size {}", description, blockSize);

        return blockSize;
    }

    template <typename F>
    __global__ void Kernel(F func, int nItems) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= nItems)
            return;

        func(tid);
    }

    // GPU Launch Function Declarations
    template <typename F>
    void gpuParallelFor(const char* description, int nItems, F func);

    template <typename F>
    void gpuParallelFor(const char* description, int nItems, F func) {
        auto kernel = &Kernel<F>;

        int                                       blockSize = GetBlockSize(description, kernel);
		const std::pair<cudaEvent_t, cudaEvent_t> events    = GetProfilerEvents(description);

        cudaEventRecord(events.first);
        int gridSize = (nItems + blockSize - 1) / blockSize;
        kernel<<<gridSize, blockSize>>>(func, nItems);
        cudaEventRecord(events.second);
        CUDA_SYNC_CHECK();
    }

    template <typename F>
    void Do(const char* description, F&& func) {
        gpuParallelFor(description, 1, [=] __device__(int) mutable { func(); });
    }
}

#endif 

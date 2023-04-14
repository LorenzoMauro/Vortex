#pragma once
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <sstream>
#include <stdexcept>
#include "Core/Log.h"

enum OptixLogLevels {
    OPTIX_CALLBACK_START_ZERO_IGNORE,
    OPTIX_CALLBACK_FATAL,
    OPTIX_CALLBACK_ERROR,
    OPTIX_CALLBACK_WARNING,
    OPTIX_CALLBACK_INFO
};

static void checkCudaError(cudaError_t rc) {
    VTX_ASSERT_CLOSE(rc == cudaSuccess, "CUDA Driver API Error {} ({})", cudaGetErrorName(rc), cudaGetErrorString(rc));
}

static void checkCuResultError(CUresult rc) {
    const char* errorName = nullptr;
    const char* errorString = nullptr;

    cuGetErrorName(rc, &errorName);
    cuGetErrorString(rc, &errorString);

    VTX_ASSERT_CLOSE(rc == CUDA_SUCCESS, "CUDA Driver API Error {} ({})", errorName, errorString);
}

static void checkOptixError(OptixResult res, const std::string& call_str, int line) {
    VTX_ASSERT_CLOSE(res == OPTIX_SUCCESS, "Optix call ({}) failed with code {} (line {})", call_str, res, line);
}

static void cudaSynchonize(std::string file, int line) {
    cudaDeviceSynchronize();                                            
    cudaError_t error = cudaGetLastError();    
    VTX_ASSERT_CLOSE(error == cudaSuccess, "error (%s: line %d): %s\n", file, line, cudaGetErrorString(error));
}


static void context_log_cb(unsigned int level,
    const char* tag,
    const char* message,
    void*)
{
    switch ((int)level) {
        case OPTIX_CALLBACK_INFO:
			VTX_INFO("{}{}: {}", (int)level, tag, message);
			return;
		case OPTIX_CALLBACK_WARNING:
            VTX_WARN("{}{}: {}", (int)level, tag, message);
            return;
        case OPTIX_CALLBACK_ERROR:
            VTX_ERROR("{}{}: {}", (int)level, tag, message);
            return;
        case OPTIX_CALLBACK_FATAL:
            VTX_ERROR("{}{}: {}", (int)level, tag, message);
			return;
    }
}

#define OPTIX_CHECK(call) checkOptixError(call, #call, __LINE__)

#define CUDA_CHECK(call) checkCudaError(call)

#define CU_CHECK(call) checkCuResultError(call)

#define CUDA_SYNC_CHECK() cudaSynchonize(__FILE__, __LINE__)
  
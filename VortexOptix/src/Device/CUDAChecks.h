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

static void checkCudaError(cudaError_t rc, std::string file, int line) {
    VTX_ASSERT_CLOSE(rc == cudaSuccess, "CUDA Runtime API Error, ({}: line {}), {} ({})", file, line, cudaGetErrorName(rc), cudaGetErrorString(rc));
}

static void checkCuResultError(CUresult rc, std::string file, int line) {
    const char* errorName = nullptr;
    const char* errorString = nullptr;

    cuGetErrorName(rc, &errorName);
    cuGetErrorString(rc, &errorString);

    VTX_ASSERT_CLOSE(rc == CUDA_SUCCESS, "CUDA Driver API Error, ({}: line {}),  {} ({})", file, line, errorName, errorString);
}

static void checkOptixError(OptixResult res, const std::string& call_str, int line) {
    VTX_ASSERT_CLOSE(res == OPTIX_SUCCESS, "Optix Error Check: call ({}) failed with code {} (line {})", call_str, res, line);
}

static void cudaSynchonize(std::string file, int line) {
    cudaDeviceSynchronize();                                            
    cudaError_t error = cudaGetLastError();    
    VTX_ASSERT_CLOSE(error == cudaSuccess, "error ({}: line {}): %s\n", file, line, cudaGetErrorString(error));
}


static void context_log_cb(unsigned int level,
    const char* tag,
    const char* message,
    void*)
{
    if(strlen(message) == 0)
    {
        return;
    }
    switch (level) {
        case OPTIX_CALLBACK_INFO:
			VTX_INFO("Optix Context: {} message: {}", tag, message);
			return;
		case OPTIX_CALLBACK_WARNING:
            VTX_WARN("Optix Context: {} message: {}", tag, message);
            return;
        case OPTIX_CALLBACK_ERROR:
            VTX_ERROR("Optix Context: {} message: {}", tag, message);
            return;
        case OPTIX_CALLBACK_FATAL:
            VTX_ERROR("Optix Context: {} message: {}", tag, message);
			return;
    }
}

#define OPTIX_CHECK(call) checkOptixError(call, #call, __LINE__)

#define CUDA_CHECK(call) checkCudaError(call,__FILE__, __LINE__)

#define CU_CHECK(call) checkCuResultError(call,__FILE__, __LINE__)

#define CUDA_SYNC_CHECK() cudaSynchonize(__FILE__, __LINE__)
  
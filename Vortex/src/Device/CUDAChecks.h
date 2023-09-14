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

static void checkCudaError(cudaError_t rc, std::string file, int line, bool close=true) {
    if(close) 
        VTX_ASSERT_CLOSE(rc == cudaSuccess, "CUDA Runtime API Error, ({}: line {}), {} ({})", file, line, cudaGetErrorName(rc), cudaGetErrorString(rc));
    else
        VTX_ASSERT_CONTINUE(rc == cudaSuccess, "CUDA Runtime API Error, ({}: line {}), {} ({})", file, line, cudaGetErrorName(rc), cudaGetErrorString(rc));
}

static void checkCuResultError(CUresult rc, std::string file, int line, bool close = true) {
    const char* errorName = nullptr;
    const char* errorString = nullptr;

    cuGetErrorName(rc, &errorName);
    cuGetErrorString(rc, &errorString);

    if (close)
        VTX_ASSERT_CLOSE(rc == CUDA_SUCCESS, "CUDA Driver API Error, ({}: line {}),  {} ({})", file, line, errorName, errorString);
    else
        VTX_ASSERT_CONTINUE(rc == CUDA_SUCCESS, "CUDA Driver API Error, ({}: line {}),  {} ({})", file, line, errorName, errorString);
}

static void checkOptixError(OptixResult res, const std::string& call_str, std::string file, int line, bool close = true) {
    if (close)
    {
        VTX_ASSERT_CLOSE(res == OPTIX_SUCCESS, "Optix Error Check: file: {} line: {}\n\tCall ({}) failed with code {}", file, line, call_str, (int)res);
    }
    else
        VTX_ASSERT_CONTINUE(res == OPTIX_SUCCESS, "Optix Error Check: file: {} line: {}\n\tCall ({}) failed with code {}", file, line, call_str, (int)res);
}

static void cudaSynchronize(std::string file, int line, bool close = true) {
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (close)
        VTX_ASSERT_CLOSE(error == cudaSuccess, "CUDA synchronization error: File {}: line {}:\n\t {}", file, line, cudaGetErrorString(error));
    else
        VTX_ASSERT_CONTINUE(error == cudaSuccess, "CUDA synchronization error: File {}: line {}:\n\t {}", file, line, cudaGetErrorString(error));
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

#define OPTIX_CHECK(call) checkOptixError(call, #call, __FILE__, __LINE__)

#define CUDA_CHECK(call) checkCudaError(call,__FILE__, __LINE__)

#define CU_CHECK(call) checkCuResultError(call,__FILE__, __LINE__)

#define OPTIX_CHECK_CONTINUE(call) checkOptixError(call, #call, __FILE__, __LINE__, false)

#define CUDA_CHECK_CONTINUE(call) checkCudaError(call,__FILE__, __LINE__, false)

#define CU_CHECK_CONTINUE(call) checkCuResultError(call,__FILE__, __LINE__, false)

#ifdef CUDA_CHECK_ALSO_ON_RELEASE
#define CUDA_SYNC_CHECK() cudaSynchronize(__FILE__, __LINE__, true)

#define CUDA_SYNC_CHECK_CONTINUE() cudaSynchronize(__FILE__, __LINE__, false)
#else
#ifdef DEBUG

#define CUDA_SYNC_CHECK() cudaSynchronize(__FILE__, __LINE__, true)

#define CUDA_SYNC_CHECK_CONTINUE() cudaSynchronize(__FILE__, __LINE__, false)
#else

#define CUDA_SYNC_CHECK()

#define CUDA_SYNC_CHECK_CONTINUE()

#endif
#endif
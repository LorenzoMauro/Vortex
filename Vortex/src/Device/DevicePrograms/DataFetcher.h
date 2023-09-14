#pragma once
#ifndef DATA_FETCHER_H
#define DATA_FETCHER_H
#include "LaunchParams.h"
#include "CudaDebugHelper.h"

namespace vtx
{
    /*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
    extern "C" __constant__ LaunchParams optixLaunchParams;

    
    /*template<typename T>
    __forceinline__ __device__ const T* getData(const CudaMap<vtxID, T>* dataMap, const vtxID index, const char* dataName)
    {
        if (dataMap == nullptr)
        {
            CUDA_ERROR_PRINT("%s Data Map is null!\n", dataName);
            return nullptr;
        }
        if (dataMap->contains(index))
        {
	        const T* data = &((*dataMap)[index]);
            CUDA_DEBUG_PRINT("%s Data with Vortex Id %d found!, Returning %p data pointer!\n", dataName, index, data);
            return data;
        }
        else
        {
            CUDA_ERROR_PRINT("%s Data with Vortex Id %d not found!\n", dataName, index);
			return nullptr;
        }
    }*/


    /*template<typename T>
    const T* getData(const vtxID nodeId);*/


    /*template<>
    __forceinline__ __device__ const InstanceData* getData(const vtxID nodeId) {
        return getData(optixLaunchParams.instances, nodeId, "Instance");
    }*/
    
    template<typename T>
    __forceinline__ __device__ const T* getData();

    template<>
    __forceinline__ __device__ const CameraData* getData() {
        return &optixLaunchParams.cameraData;
    }

    template<>
    __forceinline__ __device__ const FrameBufferData* getData() {
        if (optixLaunchParams.frameBuffer.outputBuffer == 0)
        {
            CUDA_ERROR_PRINT("FrameBuffer color Buffer is not set!\n");
            return nullptr;
        }
        return &optixLaunchParams.frameBuffer;
    }

    template<>
    __forceinline__ __device__ const OnDeviceSettings* getData() {
        if (optixLaunchParams.settings == nullptr)
        {
            CUDA_ERROR_PRINT("renderer Settings is not set!\n");
            return nullptr;
        }
        return optixLaunchParams.settings;
    }

    __forceinline__ __device__ int getFrameId()
    {
        if(optixLaunchParams.frameID == nullptr)
        {
            CUDA_ERROR_PRINT("FrameId is not set(nullptr)!\n");
            return 0;
        }
        return *optixLaunchParams.frameID;
	}

    __forceinline__ __device__ OptixTraversableHandle getTopTraversable()
    {
        if (optixLaunchParams.topObject == 0)
        {
            CUDA_ERROR_PRINT("OptixTraversableHandle top Traversal is not set(0)?\n");
        }
        return *optixLaunchParams.frameID;
    }

    __forceinline__ __device__ const BsdfSamplingPartData* getBsdfPart(const BsdfData* bsdfData, const int part)
    {
        switch (part) {
            case 0:
                if (bsdfData->hasReflectionBsdf)
                {
                    return bsdfData->reflectionBsdf;
                }
                break;
            case 1:
                if (bsdfData->hasTransmissionBsdf)
                {
                    return bsdfData->transmissionBsdf;
                }
                break;
            default: return nullptr;
        }
        return nullptr;
    }

}

#endif

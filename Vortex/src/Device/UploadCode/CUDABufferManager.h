#pragma once
#ifndef CUDA_BUFFER_MANAGER_H
#define CUDA_BUFFER_MANAGER_H
#include "Device/UploadCode/CUDABuffer.h"

namespace vtx
{
    class CUDABufferManager {
    public:

        CUDABufferManager() = delete;
        template <typename T>
        static T* allocate(const size_t count) {
            std::lock_guard<std::mutex> lock(mutex); // Protect against concurrent access

            CUDABuffer* buffer = new CUDABuffer();
            buffer->alloc(sizeof(T) * count);

            // Store the buffer pointer for later deallocation
            buffers.push_back(buffer);

            return buffer->castedPointer<T>();
        }

        template <typename T>
        static CUDABuffer* allocateReturnBuffer(const size_t count) {
            std::lock_guard<std::mutex> lock(mutex); // Protect against concurrent access

            CUDABuffer* buffer = new CUDABuffer();
            buffer->alloc(sizeof(T) * count);

            // Store the buffer pointer for later deallocation
            buffers.push_back(buffer);

            return buffer;
        }

        static void deallocateAll();

    private:
        // Vector to keep track of all allocated CUDABuffers
        static std::vector<CUDABuffer*> buffers;

        // Mutex to protect against concurrent access
        static std::mutex mutex;
    };

}



#endif
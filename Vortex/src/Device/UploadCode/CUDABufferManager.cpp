#include "CUDABufferManager.h"


namespace vtx
{
    // Initialize static members
    std::vector<CUDABuffer*> CUDABufferManager::buffers;
    std::mutex CUDABufferManager::mutex;

    void CUDABufferManager::deallocateAll() {
        std::lock_guard<std::mutex> lock(mutex); // Protect against concurrent access

        // Deallocate all CUDABuffers and delete their pointers
        for (CUDABuffer* buffer : buffers) {
            buffer->free();
            delete buffer;
        }

        // Clear the buffer pointers vector
        buffers.clear();
    }
	
}


#include "CUDABuffer.h"

namespace vtx
{

    /* Get the raw cuda pointer */
    CUdeviceptr CUDABuffer::dPointer() const
    {
        return (CUdeviceptr)d_ptr;
    }

    /* Free the allocated memory */

    void CUDABuffer::free()
    {
        if (d_ptr)
        {
            const auto err = cudaFree(d_ptr);
            checkCudaError(err, __FILE__, __LINE__, false);
            d_ptr = nullptr;
            sizeInBytes = 0;
        }
    }

    //! re-size buffer to given number of bytes

    void CUDABuffer::resize(const size_t size)
    {
        if (size != sizeInBytes)
        {
            free();
            alloc(size);
        }
    }

    //! allocate to given number of bytes

    void CUDABuffer::alloc(const size_t size)
    {
        VTX_ASSERT_CLOSE(d_ptr == nullptr, "CudaBuffer: trying to allocate memory when device pointer is not null");
        sizeInBytes = size;
        CUDA_CHECK(cudaMalloc((void**)&d_ptr, sizeInBytes));
    }

    size_t CUDABuffer::bytesSize() const
    {
        return sizeInBytes;
    }

    /*Internal Function to manage allocation, resize and upload on upload request*/

    void CUDABuffer::uploadImplementation(const void* uploadData, const size_t newSize)
    {
        if (newSize == 0)
        {
            VTX_ERROR("CudaBuffer: trying to upload a zero size Data?");
            return;
            //waitAndClose();
        }
        if (d_ptr == nullptr)
        {
            alloc(newSize);
        }
        else if (newSize != sizeInBytes)
        {
            resize(newSize);
        }

        VTX_ASSERT_CLOSE(d_ptr != nullptr, "CudaBuffer: Some Error occured, the device pointer is still null");
        CUDA_CHECK(cudaMemcpy(d_ptr, uploadData, sizeInBytes, cudaMemcpyHostToDevice));
    }

}
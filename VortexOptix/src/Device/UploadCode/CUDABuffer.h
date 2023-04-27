#pragma once
#include "Device/CUDAChecks.h"
#include <vector>
#include <assert.h>
#include "Core/Log.h"

namespace vtx {

    /*! simple wrapper for creating, and managing a device-side CUDA
        buffer */
    struct CUDABuffer {

        /* Get the raw cuda pointer */
        __host__ inline CUdeviceptr dPointer() const
        {
            return (CUdeviceptr)d_ptr;
        }

        /* Get the cuda pointer casted to the specific type */
        template<typename T>
        __host__ T* castedPointer() const
        {
        	return reinterpret_cast<T*>(d_ptr);
		}

        /* Free the allocated memory */
        __host__ void free()
        {
            if (d_ptr)
            {
	            const auto err = cudaFree(d_ptr);
				CUDA_CHECK(err);
				d_ptr = nullptr;
				sizeInBytes = 0;
            }
        }

        /*upload a generic T type, the function takes care of eventual resizing and allocation*/
        template<typename T>
        __host__ void upload(const T& uploadData)
        {
	        const size_t newSize = sizeof(T);
            uploadImplementation(&uploadData, newSize);
        }

        /*upload a std::vector type, the function takes care of eventual resizing and allocation*/
        template<typename T>
        __host__ void upload(const std::vector<T>& uploadVector)
        {
            const size_t newSize = sizeof(T)*uploadVector.size();
            uploadImplementation(uploadVector.data(), newSize);
        }

        template<typename T>
        __host__ void upload(T* uploadArray, size_t count)
        {
            const size_t newSize = sizeof(T) * count;
            uploadImplementation(uploadArray, newSize);
        }
        /*Download Data, no checks are made about the requested type if not that the stored size is a multiple of the size of the requested type
         * It returns the count of the data
         */
        template<typename T>
        __host__ uint32_t download(T* t)
        {
            VTX_ASSERT_CLOSE(d_ptr != nullptr, "CudaBuffer: trying to download a resource not previously allocated!");
            VTX_ASSERT_CLOSE(sizeInBytes % sizeof(T) == 0, 
                             "CudaBuffer: Trying to download a CUDABuffer but the storage size on the device is not a multiple of the requested type size."
							 "Are you sure you are requesting the proper data type?");
            CUDA_CHECK(cudaMemcpy((void*)t, d_ptr, sizeInBytes, cudaMemcpyDeviceToHost));
            return 0;
        }

        //! re-size buffer to given number of bytes
        __host__ void resize(const size_t size)
        {
            free();
            alloc(size);
        }

        //! allocate to given number of bytes
        __host__ void alloc(const size_t size)
        {
            VTX_ASSERT_CLOSE(d_ptr == nullptr, "CudaBuffer: trying to allocate memory when device pointer is not null");
            sizeInBytes = size;
            CUDA_CHECK(cudaMalloc((void**)&d_ptr, sizeInBytes));
        }

        size_t bytesSize() const
        {
            return sizeInBytes;
        }
    private:
        /*Internal Function to manage allocation, resize and upload on upload request*/
        __host__ void uploadImplementation(const void* uploadData, const size_t newSize)
        {
            VTX_ASSERT_CLOSE(newSize != 0, "CudaBuffer: trying to upload a zero size Data?");
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

        size_t sizeInBytes{ 0 };
        void* d_ptr{ nullptr };
    };

}

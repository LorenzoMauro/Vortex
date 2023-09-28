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
        __host__ T* upload(const T& uploadData)
        {
	        const size_t newSize = sizeof(T);
            uploadImplementation(&uploadData, newSize);
            return castedPointer<T>();
        }

        template<typename T>
        __host__ T* uploadAtPtr(const T& uploadData, T* ptr)
        {
	        const size_t dataSize = sizeof(T);
            const char* endOfData = (char*)ptr + dataSize;
            const char* allocationEnd = (char*)d_ptr + sizeInBytes;
            if (endOfData > allocationEnd)
            {
				VTX_ERROR("CudaBuffer: trying to upload data at a pointer that is not inside the allocated memory");
                return nullptr;
			}
			CUDA_CHECK(cudaMemcpy((void*)ptr, &uploadData, dataSize, cudaMemcpyHostToDevice));
			return ptr;
		}

        /*upload a std::vector type, the function takes care of eventual resizing and allocation*/
        template<typename T>
        __host__ T* upload(const std::vector<T>& uploadVector)
        {
            const size_t newSize = sizeof(T)*uploadVector.size();
            uploadImplementation(uploadVector.data(), newSize);
            return castedPointer<T>();
        }

        template<typename T>
        __host__ T* upload(T* uploadArray, const size_t count)
        {
            const size_t newSize = sizeof(T) * count;
            uploadImplementation(uploadArray, newSize);
            return castedPointer<T>();
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


        template<typename T>
        __host__ T* alloc(const int count)
        {
            const size_t newSize = sizeof(T) * count;
            resize(newSize);
            return castedPointer<T>();
        }

        //! re-size buffer to given number of bytes
        __host__ void resize(const size_t size)
        {
            if(size != sizeInBytes)
            {
                free();
                alloc(size);
            }
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
            if (newSize == 0)
            {
                VTX_ERROR("CudaBuffer: trying to upload a zero size Data?");
                waitAndClose();
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

        size_t sizeInBytes{ 0 };
        void* d_ptr{ nullptr };
    };

}

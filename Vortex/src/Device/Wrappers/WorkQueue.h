#pragma once
#ifndef WORK_QUEUE_H
#define WORK_QUEUE_H

#include "Device/DevicePrograms/nvccUtils.h"
#include "cuda_runtime.h"
#include "Soa.h"
#include "Device/UploadCode/CUDABufferManager.h"

namespace vtx
{
    // WorkQueueSOA Definition
    template <typename WorkItem>
    class WorkQueueSOA : public SOA<WorkItem> {
    public:
        // WorkQueueSOA Public Methods
        WorkQueueSOA() = default;
        WorkQueueSOA(int n, const std::string& queueName) : SOA<WorkItem>(n), nAlloc(n)
        {
            CUDABuffer* nameBuffer = CUDABufferManager::allocateReturnBuffer<char>(queueName.size());

            nameBuffer->upload(queueName.c_str(), queueName.size());

            name = nameBuffer->castedPointer<char>();
        }

        WorkQueueSOA& operator=(const WorkQueueSOA& w) {
            SOA<WorkItem>::operator=(w);
            size = w.size;
            return *this;
        }
        __forceinline__ __device__ void setCounter(int* counter)
        {
            size = counter;
        }
        __forceinline__ __device__ int Size() const {
            return *size;
        }

        __forceinline__ __device__ void Reset() {
            *size = 0;
        }

        __forceinline__ __device__ int Push(WorkItem w) {
            int index = AllocateEntry();
            (*this)[index] = w;
            return index;
        }

        __forceinline__ __device__ int maxSize()
        {
            return nAlloc;
        }


    protected:
        // WorkQueueSOA Protected Methods
        __forceinline__ __device__ int AllocateEntry() {
            return cuAtomicAdd(size, 1);
        }

    private:
        // WorkQueueSOA Private Members
        int* size = nullptr;
        int nAlloc;
        char* name;
    };
}

#endif
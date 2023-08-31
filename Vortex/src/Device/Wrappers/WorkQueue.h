#pragma once
#ifndef WORK_QUEUE_H
#define WORK_QUEUE_H

#include "Device/DevicePrograms/nvccUtils.h"
#include "Soa.h"
#include "Device/UploadCode/CUDABufferManager.h"

namespace vtx
{
#ifdef DEBUG
#define localAssert(A) if (!(A)) { printf("Assertion failed: %s\n In WorkQueue %s", #A, name); }
#else
#define localAssert(A)
#endif

    // WorkQueueSOA Definition
    template <typename WorkItem>
    class WorkQueueSOA : public SOA<WorkItem> {
    public:
        // WorkQueueSOA Public Methods
        WorkQueueSOA() = default;
        WorkQueueSOA(int n, const std::string& queueName = "unnamed") : SOA<WorkItem>(n), nAlloc(n)
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
        __forceinline__ __device__ int* getCounter()
        {
            return &size;
        }
        __forceinline__ __device__ int Size() const {
            return size;
        }

        __forceinline__ __device__ void Reset() {
            size = 0;
        }

        __forceinline__ __device__ int Push(WorkItem w) {
            int index = AllocateEntry();
            localAssert(index < nAlloc);
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
            return cuAtomicAdd(&size, 1);
        }

    private:
        // WorkQueueSOA Private Members
        int size = 0;
        int nAlloc = 0;
        char* name;
    };
}

#endif
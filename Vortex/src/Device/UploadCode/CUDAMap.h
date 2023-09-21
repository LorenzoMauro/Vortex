#pragma once
#ifndef CUDAMAP_H
#define CUDAMAP_H
#include <cuda_runtime.h>
#include "CUDABuffer.h"
#include "Device/DevicePrograms/CudaDebugHelper.h"

namespace vtx {

    __host__ __device__ inline uint32_t stringHash(const char* str) {
        uint32_t hash = 5381;
        int c;

        while ((c = *str++))
            hash = ((hash << 5) + hash) + c; // hash * 33 + c

        return hash;
    }

    template <typename TKey, typename TValue>
    class CudaMap {
    public:
        TKey* keys;
        TValue* values;
        int size;
        int capacity;
        bool isUpdated;
//#ifndef __CUDACC__
        CUDABuffer keyBuffer;
        CUDABuffer valueBuffer;
        CUDABuffer mapBuffer;
//#endif


        __host__ CudaMap() :
    		keys(nullptr),
    		values(nullptr),
    		size(0),
    		capacity(0),
			isUpdated(true)
    	{}

        __host__ ~CudaMap() {
            delete[] keys;
            delete[] values;

            // With this trick we should be able to free cuda resources once the CudaMap on the host ends its life
            if(keyBuffer.dPointer())
            {
	            keyBuffer.free();
            }
            if(valueBuffer.dPointer())
            {
	            valueBuffer.free();
			}
        }


        __host__ void insert(const TKey& key, const TValue& value) {
            int keyIndex = find_index(key);
            if(keyIndex > -1)
            {
	            values[keyIndex] = value;
                isUpdated = true;
			}
            else
            {
                if (size >= capacity) {
                    int new_capacity = capacity == 0 ? 1 : capacity * 2;
                    resize(new_capacity);
                }
                keys[size] = key;
                values[size] = value;
                ++size;
                isUpdated = true;
            }
        }

        __host__ __device__ TValue& operator[](const TKey& key) {
            int index = find_index(key);
            if(index<0)
            {
                CUDA_ERROR_PRINT("Trying to access a non valid index %d!", key);
            }
            TValue& value = values[index];
            //CUDA_DEBUG_PRINT("Value %p Index %d on CudaMap operator []!\n", &value, key);
            return value;
        }

        __host__ __device__ const TValue& operator[](const TKey& key) const {
            int index = find_index(key);
            if (index < 0)
            {
                CUDA_ERROR_PRINT("Trying to access a non valid index %d!", key);
            }
            TValue& value = values[index];
            //CUDA_DEBUG_PRINT("Value %p Index %d found on CudaMap operator []!\n", &value, key);
            return value;
        }

        __host__ __device__ bool contains(const TKey& key) const {
            return find_index(key) >= 0;
        }

        __host__ CudaMap* upload() {
            // Copy the map to the device and iterate over all elements
            finalize();
            if(size>0)
            {
        	    keyBuffer.upload(keys, size);
                valueBuffer.upload(values, size);

                CudaMap<TKey, TValue>  tempMap;
                tempMap.keys = keyBuffer.castedPointer<TKey>();
                tempMap.values = valueBuffer.castedPointer<TValue>();
                VTX_ASSERT_CONTINUE(tempMap.keys, "Cuda pointer of Cuda Map keys seems to be null!");
                VTX_ASSERT_CONTINUE(tempMap.values, "Cuda pointer of Cuda Map values seems to be null!");
                tempMap.size = size;
                tempMap.capacity = capacity;

                mapBuffer.upload(tempMap);

                // DO this so that when the destructor is called it won't try to free cuda memory...
                tempMap.keys = nullptr;
                tempMap.values = nullptr;

                isUpdated = false;

                return mapBuffer.castedPointer<CudaMap<TKey, TValue>>();
            }
            return nullptr;
        }

        class iterator {
        public:
            TKey* key_ptr;
            TValue* value_ptr;

            __host__ __device__ iterator(TKey* key_ptr, TValue* value_ptr) : key_ptr(key_ptr), value_ptr(value_ptr) {}

            __host__ __device__ iterator& operator++() {
                ++key_ptr;
                ++value_ptr;
                return *this;
            }

            __host__ __device__ bool operator!=(const iterator& other) const {
                return key_ptr != other.key_ptr;
            }

            __host__ __device__ std::pair<TKey, TValue> operator*() const {
                return std::make_pair(*key_ptr, *value_ptr);
            }
        };

        __host__ __device__ iterator begin() {
            return iterator(keys, values);
        }

        __host__ __device__ iterator end() {
            return iterator(keys + size, values + size);
        }

        __host__ void finalize() {
            if (size < capacity) {
                resize(size);
            }
        }
    private:
        __host__ void resize(const int new_capacity) {
            TKey* new_keys = new TKey[new_capacity];
            TValue* new_values = new TValue[new_capacity];

            for (int i = 0; i < size; ++i) {
                new_keys[i] = keys[i];
                new_values[i] = values[i];
            }

            delete[] keys;
            delete[] values;

            keys = new_keys;
            values = new_values;
            capacity = new_capacity;
        }

        __host__ __device__ int find_index(const TKey& key) const {
            for (int i = 0; i < size; ++i) {
                if (keys[i] == key) {
                    //CUDA_DEBUG_PRINT("Index %d found!\n", key);
                    return i;
                }
            }
            //CUDA_DEBUG_PRINT("Index %d Not found!\n", key);
            return -1;
        }

    };

}

#endif // CUDAMAP_H
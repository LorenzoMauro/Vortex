#pragma once

#include <cuda_runtime.h>
#include "CUDABuffer.h"

namespace vtx {
    template <typename TKey, typename TValue>
    class CudaMap {
    public:
        TKey* keys;
        TValue* values;
        int size;
        int capacity;

        __host__ CudaMap() : keys(nullptr), values(nullptr), size(0), capacity(0) {}

        __host__ ~CudaMap() {
            delete[] keys;
            delete[] values;
        }


        __host__ void insert(const TKey& key, const TValue& value) {
            if (size >= capacity) {
                int new_capacity = capacity == 0 ? 1 : capacity * 2;
                resize(new_capacity);
            }
            keys[size] = key;
            values[size] = value;
            ++size;
        }


        __host__ __device__ TValue& operator[](const TKey& key) {
            int index = find_index(key);
            return values[index];
        }

        __host__ __device__ const TValue& operator[](const TKey& key) const {
            int index = find_index(key);
            return values[index];
        }

        __host__ __device__ bool contains(const TKey& key) const {
            return find_index(key) >= 0;
        }

        __host__ CudaMap* allocAndUpload() {
            // Copy the map to the device and iterate over all elements
            finalize();

            CUDABuffer keysBuffer;
            CUDABuffer valuesBuffer;

            std::vector<TKey> keysVec(keys, keys + size);
            std::vector<TValue> valuesVec(values, values + size);

            keysBuffer.alloc_and_upload(keysVec);
            valuesBuffer.alloc_and_upload(valuesVec);

            CudaMap<TKey, TValue>  temp_map;

            temp_map.keys = reinterpret_cast<TKey*>(keysBuffer.d_pointer());
            temp_map.values = reinterpret_cast<TValue*>(valuesBuffer.d_pointer());
            temp_map.size = size;
            temp_map.capacity = capacity;

            CudaMap<TKey, TValue>* d_map;

            cudaMalloc((void**)&d_map, sizeof(CudaMap<TKey, TValue>));
            cudaMemcpy(d_map, &temp_map, sizeof(CudaMap<TKey, TValue>), cudaMemcpyHostToDevice);

            // DO this so that when the destructor is called it won't try to free cuda memory...
            temp_map.keys = nullptr;
            temp_map.values = nullptr;

            return d_map;

            //
            //TKey* d_keys;
            //TValue* d_values;
            //cudaMalloc(&d_keys, size * sizeof(TKey));
            //cudaMalloc(&d_values, size * sizeof(TValue));
            //cudaMemcpy(d_keys, keys, size * sizeof(TKey), cudaMemcpyHostToDevice);
            //cudaMemcpy(d_values, values, size * sizeof(TValue), cudaMemcpyHostToDevice);
            //
            //cudaMemcpy(&((*d_map)->keys), &d_keys, sizeof(TKey*), cudaMemcpyHostToDevice);
            //cudaMemcpy(&((*d_map)->values), &d_values, sizeof(TValue*), cudaMemcpyHostToDevice);
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

    private:
        __host__ void resize(int new_capacity) {
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
                    return i;
                }
            }
            return -1;
        }

        __host__ void finalize() {
            if (size < capacity) {
                resize(size);
            }
        }
    };

}


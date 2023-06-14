#pragma once
#ifndef SOA_H
#define SOA_H

#include <mi/neuraylib/target_code_types.h>
#include "Device/UploadCode/CUDABuffer.h"
#include "Core/Math.h"
#include "Device/DevicePrograms/nvccUtils.h"

namespace vtx::mdl
{
    typedef mi::neuraylib::Bsdf_event_type BsdfEventType;

}

namespace vtx
{
    class CUDABufferManager {
    public:

        CUDABufferManager() = delete;
        template <typename T>
        static T* allocate(size_t count) {
            std::lock_guard<std::mutex> lock(mutex); // Protect against concurrent access

            CUDABuffer* buffer = new CUDABuffer();
            buffer->alloc(sizeof(T) * count);

            // Store the buffer pointer for later deallocation
            buffers.push_back(buffer);

            return buffer->castedPointer<T>();
        }


        template <typename T>
        static CUDABuffer* allocateReturnBuffer(size_t count) {
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


    template <typename T> struct SOA
    {
        int nAlloc;
    };

    template<>
    struct SOA<math::vec3f> {
        SOA() = default;

        SOA(int n) : nAlloc(n) {
            this->x = CUDABufferManager::allocate<float>(n);
            this->y = CUDABufferManager::allocate<float>(n);
            this->z = CUDABufferManager::allocate<float>(n);
        }

        SOA& operator=(const SOA& s) {
            nAlloc = s.nAlloc;
            this->x = s.x;
            this->y = s.y;
            this->z = s.z;
            return *this;
        }

        struct Proxy {
            // Conversion operator to retrieve math::vec3f from SOA
            __device__ __host__ operator math::vec3f() const {
                math::vec3f r;
                r.x = soa->x[i];
                r.y = soa->y[i];
                r.z = soa->z[i];
                return r;
            }

            // Assignment operator to set values in SOA from math::vec3f
            __device__ __host__ void operator=(const math::vec3f& a) {
                soa->x[i] = a.x;
                soa->y[i] = a.y;
                soa->z[i] = a.z;
            }

            SOA* soa;
            int i;
        };

        __device__ __host__ Proxy operator[](int i) {
            assert(i < nAlloc); // Ensure the index is in range
            return Proxy{ this, i };
        }

        __device__ __host__ math::vec3f operator[](int i) const {
            assert(i < nAlloc); // Ensure the index is in range
            math::vec3f r;
            r.x = this->x[i];
            r.y = this->y[i];
            r.z = this->z[i];
            return r;
        }

        int nAlloc;
        float* __restrict x;
        float* __restrict y;
        float* __restrict z;
    };

    template<>
    struct SOA<math::affine3f> {
        SOA() = default;

        SOA(int n) : nAlloc(n) {
            // Linear Part
            m00 = CUDABufferManager::allocate<float>(n);
            m01 = CUDABufferManager::allocate<float>(n);
            m02 = CUDABufferManager::allocate<float>(n);

            m10 = CUDABufferManager::allocate<float>(n);
            m11 = CUDABufferManager::allocate<float>(n);
            m12 = CUDABufferManager::allocate<float>(n);

            m20 = CUDABufferManager::allocate<float>(n);
            m21 = CUDABufferManager::allocate<float>(n);
            m22 = CUDABufferManager::allocate<float>(n);

            // Affine Part
            m30 = CUDABufferManager::allocate<float>(n);
            m31 = CUDABufferManager::allocate<float>(n);
            m32 = CUDABufferManager::allocate<float>(n);
        }

        SOA& operator=(const SOA& s) {
            nAlloc = s.nAlloc;
            m00 = s.m00;
            m01 = s.m01;
            m02 = s.m02;

            m10 = s.m10;
            m11 = s.m11;
            m12 = s.m12;

            m20 = s.m20;
            m21 = s.m21;
            m22 = s.m22;

            m30 = s.m30;
            m31 = s.m31;
            m32 = s.m32;
            return *this;
        }

        struct Proxy {
            // Conversion operator to retrieve math::affine3f from SOA
            __device__ __host__ operator math::affine3f() const {
                math::affine3f r;

                math::vec3f& c0 = r.l.vx;
                math::vec3f& c1 = r.l.vy;
                math::vec3f& c2 = r.l.vz;
                math::vec3f& c3 = r.p;

                c0.x = soa->m00[i]; c1.x = soa->m10[i]; c2.x = soa->m20[i]; c3.x = soa->m30[i];
                c0.y = soa->m01[i]; c1.y = soa->m11[i]; c2.y = soa->m21[i]; c3.y = soa->m31[i];
                c0.z = soa->m02[i]; c1.z = soa->m12[i]; c2.z = soa->m22[i]; c3.z = soa->m32[i];

                return r;
            }

            // Assignment operator to set values in SOA from math::affine3f
            __device__ __host__ void operator=(const math::affine3f& a) {
                soa->m00[i] = a.l.vx.x;
                soa->m01[i] = a.l.vx.y;
                soa->m02[i] = a.l.vx.z;

                soa->m10[i] = a.l.vy.x;
                soa->m11[i] = a.l.vy.y;
                soa->m12[i] = a.l.vy.z;

                soa->m20[i] = a.l.vz.x;
                soa->m21[i] = a.l.vz.y;
                soa->m22[i] = a.l.vz.z;

                soa->m30[i] = a.p.x;
                soa->m31[i] = a.p.y;
                soa->m32[i] = a.p.z;
            }

            SOA* soa;
            int i;
        };


        __device__ __host__ Proxy operator[](int i) {
            assert(i < nAlloc); // Ensure the index is in range
            return Proxy{ this, i };
        }

        __device__ __host__ math::affine3f operator[](int i) const {
            assert(i < nAlloc); // Ensure the index is in range
            math::affine3f r;

            math::vec3f& c0 = r.l.vx;
            math::vec3f& c1 = r.l.vy;
            math::vec3f& c2 = r.l.vz;
            math::vec3f& c3 = r.p;

            c0.x = this->m00[i]; c1.x = this->m10[i]; c2.x = this->m20[i]; c3.x = this->m30[i];
            c0.y = this->m01[i]; c1.y = this->m11[i]; c2.y = this->m21[i]; c3.y = this->m31[i];
            c0.z = this->m02[i]; c1.z = this->m12[i]; c2.z = this->m22[i]; c3.z = this->m32[i];

            return r;
        }

        int nAlloc;
        // Linear Part
        float* __restrict m00;
        float* __restrict m01;
        float* __restrict m02;

        float* __restrict m10;
        float* __restrict m11;
        float* __restrict m12;

        float* __restrict m20;
        float* __restrict m21;
        float* __restrict m22;

        // Affine Part
        float* __restrict m30;
        float* __restrict m31;
        float* __restrict m32;
    };


    // WorkQueueSOA Definition
    template <typename WorkItem>
    class WorkQueueSOA : public SOA<WorkItem> {
    public:
        // WorkQueueSOA Public Methods
        WorkQueueSOA() = default;
        WorkQueueSOA(int n, std::string QueueName) : SOA<WorkItem>(n), nAlloc(n)
        {
            CUDABuffer* nameBuffer = CUDABufferManager::allocateReturnBuffer<char>(QueueName.size());

            nameBuffer->upload(QueueName.c_str(), QueueName.size());

	        name = nameBuffer->castedPointer<char>();
        }
        WorkQueueSOA& operator=(const WorkQueueSOA& w) {
            SOA<WorkItem>::operator=(w);
            size = w.size;
            return *this;
        }

        __device__ __host__ int Size() const {
            return size;
        }
        __device__ __host__ void Reset() {
            size = 0;
        }

        __device__ __host__ int Push(WorkItem w) {
            int index = AllocateEntry();
            if (index >= nAlloc)
            {
                printf("WorkQueueSOA::Push() - %s - index %d >= nAlloc %d\n", name, index, nAlloc);
                return -1;
            }
            (*this)[index] = w;
            return index;
        }


    protected:
        // WorkQueueSOA Protected Methods
        __device__ __host__ int AllocateEntry() {
            return cuAtomicAdd(&size, 1);
        }

    private:
        // WorkQueueSOA Private Members
        int size = 0;
        int nAlloc;
        char* name;
    };

    struct alignas(16) RayWorkItem
    {
        // Assuming math::affine3f is large, place these at the top
        math::affine3f hitWTO;
        math::affine3f hitOTW;

        // Assuming math::vec3f is 12 bytes, group them with smaller members
        math::vec3f origin;
        float pdf;

        math::vec3f direction;
        float hitDistance;

        math::vec3f radiance;
        unsigned seed;

        math::vec3f throughput;
        unsigned hitInstanceId;

        math::vec3f mediumIor;
        unsigned hitTriangleId;

        math::vec3f hitPosition;
        int originPixel;

        math::vec3f hitBaricenter;
        int depth;

        math::vec3f colorsDirectLight;
        mdl::BsdfEventType eventType;

        math::vec3f colorsTrueNormal;
        mdl::BsdfEventType firstHitType;

        math::vec3f colorsShadingNormal;
        bool shadowTrace; // you can still name it padding if no other variable fits here

        math::vec3f colorsBounceDiffuse;
        float padding2;

        math::vec3f colorsTangent;
        float padding3;

        math::vec3f colorsUv;
        float padding4;

        math::vec3f colorsOrientation;
        float padding5;

        math::vec3f colorsDebugColor1;
        float       padding6;

        math::vec3f radianceDirect;
	};

    template <> struct SOA<RayWorkItem> {
        SOA() = default;

        SOA(int n) : nAlloc(n) {

			originPixel = CUDABufferManager::allocate<int>(n);
			seed        = CUDABufferManager::allocate<unsigned>(n);
			origin      = SOA<math::vec3f>(n);
			direction   = SOA<math::vec3f>(n);
			depth       = CUDABufferManager::allocate<int>(n);
			pdf         = CUDABufferManager::allocate<float>(n);
            radiance    = SOA<math::vec3f>(n);
            radianceDirect    = SOA<math::vec3f>(n);
            shadowTrace = CUDABufferManager::allocate<bool>(n);
            throughput  = SOA<math::vec3f>(n);
			mediumIor   = SOA<math::vec3f>(n);
			eventType   = CUDABufferManager::allocate<mdl::BsdfEventType>(n);

			hitPosition   = SOA<math::vec3f>(n);
			hitDistance   = CUDABufferManager::allocate<float>(n);
			hitBaricenter = SOA<math::vec3f>(n);
			hitInstanceId = CUDABufferManager::allocate<unsigned>(n);
			hitTriangleId = CUDABufferManager::allocate<unsigned>(n);
			hitWTO        = SOA<math::affine3f>(n);
			hitOTW        = SOA<math::affine3f>(n);

            firstHitType        = CUDABufferManager::allocate<mdl::BsdfEventType>(n);
            colorsDirectLight   = SOA<math::vec3f>(n);
            colorsTrueNormal    = SOA<math::vec3f>(n);
			colorsShadingNormal = SOA<math::vec3f>(n);
			colorsBounceDiffuse = SOA<math::vec3f>(n);
			colorsTangent       = SOA<math::vec3f>(n);
			colorsUv            = SOA<math::vec3f>(n);
			colorsOrientation   = SOA<math::vec3f>(n);
			colorsDebugColor1   = SOA<math::vec3f>(n);
        }

        SOA& operator=(const SOA& s) {
            nAlloc = s.nAlloc;

            this->originPixel = s.originPixel;
            this->seed = s.seed;
            this->origin = s.origin;
            this->direction = s.direction;
            this->depth = s.depth;
            this->pdf = s.pdf;
            this->radiance = s.radiance;
            this->throughput = s.throughput;
            this->mediumIor = s.mediumIor;
            this->eventType = s.eventType;

            this->hitPosition = s.hitPosition;
            this->hitDistance = s.hitDistance;
            this->hitBaricenter = s.hitBaricenter;
            this->hitInstanceId = s.hitInstanceId;
            this->hitTriangleId = s.hitTriangleId;
            this->hitWTO = s.hitWTO;
            this->hitOTW = s.hitOTW;

            this->firstHitType = s.firstHitType;
            this->colorsDirectLight = s.colorsDirectLight;
            this->colorsTrueNormal = s.colorsTrueNormal;
            this->colorsShadingNormal = s.colorsShadingNormal;
            this->colorsBounceDiffuse = s.colorsBounceDiffuse;
            this->colorsTangent = s.colorsTangent;
            this->colorsUv = s.colorsUv;
            this->colorsOrientation = s.colorsOrientation;
            this->colorsDebugColor1 = s.colorsDebugColor1;

            return *this;
        }

    	struct Proxy {
            __device__ __host__ operator RayWorkItem() const {
                RayWorkItem r;
                r.originPixel = soa->originPixel[i];
                r.seed = soa->seed[i];
                r.origin = soa->origin[i];
                r.direction = soa->direction[i];
                r.depth = soa->depth[i];
                r.pdf = soa->pdf[i];
                r.radiance = soa->radiance[i];
                r.throughput = soa->throughput[i];
                r.mediumIor = soa->mediumIor[i];
                r.eventType = soa->eventType[i];
                r.shadowTrace = soa->shadowTrace[i];
                r.radianceDirect = soa->radianceDirect[i];

                r.hitPosition = soa->hitPosition[i];
                r.hitDistance = soa->hitDistance[i];
                r.hitBaricenter = soa->hitBaricenter[i];
                r.hitInstanceId = soa->hitInstanceId[i];
                r.hitTriangleId = soa->hitTriangleId[i];
                r.hitWTO = soa->hitWTO[i];
                r.hitOTW = soa->hitOTW[i];

                r.firstHitType = soa->firstHitType[i];
                r.colorsDirectLight = soa->colorsDirectLight[i];
                r.colorsTrueNormal = soa->colorsTrueNormal[i];
                r.colorsShadingNormal = soa->colorsShadingNormal[i];
                r.colorsBounceDiffuse = soa->colorsBounceDiffuse[i];
                r.colorsTangent = soa->colorsTangent[i];
                r.colorsUv = soa->colorsUv[i];
                r.colorsOrientation = soa->colorsOrientation[i];
                r.colorsDebugColor1 = soa->colorsDebugColor1[i];
                return r;
            }

        	__device__ __host__ void operator=(const RayWorkItem& a) {
                assert(i < soa->nAlloc); // Ensure the index is in range
                soa->originPixel[i] = a.originPixel;
                soa->seed[i] = a.seed;
                soa->origin[i] = a.origin;
                soa->direction[i] = a.direction;
                soa->depth[i] = a.depth;
                soa->pdf[i] = a.pdf;
                soa->radiance[i] = a.radiance;
                soa->throughput[i] = a.throughput;
                soa->mediumIor[i] = a.mediumIor;
                soa->eventType[i] = a.eventType;
                soa->shadowTrace[i] = a.shadowTrace;
                soa->radianceDirect[i] = a.radianceDirect;

                soa->hitPosition[i] = a.hitPosition;
                soa->hitDistance[i] = a.hitDistance;
                soa->hitBaricenter[i] = a.hitBaricenter;
                soa->hitInstanceId[i] = a.hitInstanceId;
                soa->hitTriangleId[i] = a.hitTriangleId;
                soa->hitWTO[i] = a.hitWTO;
                soa->hitOTW[i] = a.hitOTW;

                soa->firstHitType[i] = a.firstHitType;
                soa->colorsDirectLight[i] = a.colorsDirectLight;
                soa->colorsTrueNormal[i] = a.colorsTrueNormal;
                soa->colorsShadingNormal[i] = a.colorsShadingNormal;
                soa->colorsBounceDiffuse[i] = a.colorsBounceDiffuse;
                soa->colorsTangent[i] = a.colorsTangent;
                soa->colorsUv[i] = a.colorsUv;
                soa->colorsOrientation[i] = a.colorsOrientation;
                soa->colorsDebugColor1[i] = a.colorsDebugColor1;
            }

            SOA* soa;
            int i;
        };

        __device__ __host__ Proxy operator[](int i) {
            assert(i < nAlloc); // Ensure the index is in range
            return Proxy{ this, i };
        }
        __device__ __host__ RayWorkItem operator[](int i) const {
            assert(i < nAlloc); // Ensure the index is in range
            RayWorkItem r;

            r.originPixel = this->originPixel[i];
            r.seed = this->seed[i];
            r.origin = this->origin[i];
            r.direction = this->direction[i];
            r.depth = this->depth[i];
            r.pdf = this->pdf[i];
            r.radiance = this->radiance[i];
            r.throughput = this->throughput[i];
            r.mediumIor = this->mediumIor[i];
            r.eventType = this->eventType[i];
            r.shadowTrace = this->shadowTrace[i];
            r.radianceDirect = this->radianceDirect[i];

            r.hitPosition = this->hitPosition[i];
            r.hitDistance = this->hitDistance[i];
            r.hitBaricenter = this->hitBaricenter[i];
            r.hitInstanceId = this->hitInstanceId[i];
            r.hitTriangleId = this->hitTriangleId[i];
            r.hitWTO = this->hitWTO[i];
            r.hitOTW = this->hitOTW[i];

            r.firstHitType = this->firstHitType[i];
            r.colorsDirectLight = this->colorsDirectLight[i];
            r.colorsTrueNormal = this->colorsTrueNormal[i];
            r.colorsShadingNormal = this->colorsShadingNormal[i];
            r.colorsBounceDiffuse = this->colorsBounceDiffuse[i];
            r.colorsTangent = this->colorsTangent[i];
            r.colorsUv = this->colorsUv[i];
            r.colorsOrientation = this->colorsOrientation[i];
            r.colorsDebugColor1 = this->colorsDebugColor1[i];

            return r;
        }


        int nAlloc;
        int* __restrict                originPixel;
		unsigned* __restrict           seed;
		SOA<math::vec3f>               origin;
		SOA<math::vec3f>               direction;
		int* __restrict                depth;
		float* __restrict              pdf;
		SOA<math::vec3f>               radiance;
		SOA<math::vec3f>               throughput;
		SOA<math::vec3f>               mediumIor;
		mdl::BsdfEventType* __restrict eventType;

        SOA<math::vec3f>  radianceDirect;
        bool* __restrict shadowTrace;

		SOA<math::vec3f>     hitPosition;
		float* __restrict    hitDistance;
		SOA<math::vec3f>     hitBaricenter;
		unsigned* __restrict hitInstanceId;
		unsigned* __restrict hitTriangleId;
		SOA<math::affine3f>  hitWTO;
		SOA<math::affine3f>  hitOTW;

        mdl::BsdfEventType* __restrict firstHitType;
        SOA<math::vec3f> colorsDirectLight;
        SOA<math::vec3f> colorsTrueNormal;
		SOA<math::vec3f> colorsShadingNormal;
		SOA<math::vec3f> colorsBounceDiffuse;
		SOA<math::vec3f> colorsTangent;
		SOA<math::vec3f> colorsUv;
		SOA<math::vec3f> colorsOrientation;
		SOA<math::vec3f> colorsDebugColor1;
    };


    struct PixelWorkItem
    {
        int         pixelId;
    };

    template <> struct SOA<PixelWorkItem> {
        SOA() = default;

        SOA(int n) : nAlloc(n) {

            originPixel = CUDABufferManager::allocate<int>(n);
        }

        SOA& operator=(const SOA& s) {
            nAlloc = s.nAlloc;

            this->originPixel = s.originPixel;
            return *this;
        }

        struct Proxy {
            __device__ __host__ operator PixelWorkItem() const {
                PixelWorkItem r;
                r.pixelId = soa->originPixel[i];
                return r;
            }
            __device__ __host__ void operator=(const PixelWorkItem& a) {
                assert(i < soa->nAlloc); // Ensure the index is in range
                soa->originPixel[i] = a.pixelId;
            }

            SOA* soa;
            int i;
        };

        __device__ __host__ Proxy operator[](int i) {
            assert(i < nAlloc); // Ensure the index is in range
            return Proxy{ this, i };
        }
        __device__ __host__ PixelWorkItem operator[](int i) const {
            assert(i < nAlloc); // Ensure the index is in range
            PixelWorkItem r;

            r.pixelId = this->originPixel[i];

            return r;
        }

        int nAlloc;

        int* __restrict                originPixel;
    };

}
#endif
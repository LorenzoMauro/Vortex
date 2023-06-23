﻿#pragma once
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

    __forceinline__ __both__ void assertF(bool condition, char* message="")
    {
	    if (!condition)
	    {
	    	printf("Assertion failed! %s\n", message);
		}
    }

#define USE_ASSERT
#ifdef USE_ASSERT
#define vtxAssert(condition) assertF(condition)
#define vtxAssertMsg(condition, msg) assertF(condition, msg)
#else
#define vtxAssert(condition)
#define vtxAssertMsg(condition, msg)
#endif

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

    struct AccumulationWorkItem
    {
        int originPixel;
        math::vec3f radiance;
        int depth;
    };


    struct ShadowWorkItem
    {
        math::vec3f origin;
        float distance = -1;
        math::vec3f direction;
        int originPixel;
        math::vec3f radiance;
        int depth;
    };

    struct RayWorkItem
    {
        unsigned seed;
        int originPixel;
        int depth;
        math::vec3f origin;
        math::vec3f direction;
        math::vec3f radiance;
        math::vec3f throughput;
        math::vec3f mediumIor;
        math::vec3f hitBaricenter;
        unsigned hitInstanceId;
        unsigned hitTriangleId;
        mdl::BsdfEventType eventType;
        float pdf;
        float hitDistance;
    };

    struct TraceWorkItem
    {
        unsigned seed;
        int originPixel;
        int depth;
        math::vec3f origin;
        math::vec3f direction;
        math::vec3f radiance;
        math::vec3f throughput;
        math::vec3f mediumIor;
        mdl::BsdfEventType eventType;
        float pdf;
        bool extendRay;
    };

    struct EscapedWorkItem
    {
        unsigned seed;
        int originPixel;
        int depth;
        math::vec3f direction;
        math::vec3f radiance;
        math::vec3f throughput;
        mdl::BsdfEventType eventType;
        float pdf;
    };

	template <>
	struct SOA<math::vec3f>
	{
		SOA() = default;

		SOA(int n) : nAlloc(n)
		{
			this->x = CUDABufferManager::allocate<float>(n * 3);
			this->y = this->x + n;
			this->z = this->y + n;
		}

		/*SOA(float* ptr, int n) : nAlloc(n)
		{
			this->x = ptr;
			this->y = this->x + n;
			this->z = this->y + n;
		}*/

		SOA& operator=(const SOA& s)
		{
			nAlloc  = s.nAlloc;
			this->x = s.x;
			this->y = s.y;
			this->z = s.z;
			return *this;
		}

		struct Proxy
		{
			// Conversion operator to retrieve math::vec3f from SOA
			__forceinline__ __device__ operator math::vec3f() const
			{
				//vtxAssert(i < soa->nAlloc); // Ensure the index is in range
				//math::vec3f r;
				//r.x = soa->x[i];
				//r.y = soa->y[i];
				//r.z = soa->z[i];
				return {soa->x[i], soa->y[i], soa->z[i]};
			}

			// Assignment operator to set values in SOA from math::vec3f
			__forceinline__ __device__ void operator=(const math::vec3f& a)
			{
				//vtxAssert(i < soa->nAlloc); // Ensure the index is in range
				soa->x[i] = a.x;
				soa->y[i] = a.y;
				soa->z[i] = a.z;
			}

			SOA* soa;
			int  i;
		};

		__forceinline__ __device__ Proxy operator[](int i)
		{
			vtxAssert(i < nAlloc); // Ensure the index is in range
			return Proxy{this, i};
		}

		__forceinline__ __device__ math::vec3f operator[](int i) const
		{
			//vtxAssert(i < nAlloc); // Ensure the index is in range
			//math::vec3f r;
			//r.x = this->x[i];
			//r.y = this->y[i];
			//r.z = this->z[i];
			return { this->x[i], this->y[i], this->z[i] };
		}

		int               nAlloc;
		float* __restrict x;
		float* __restrict y;
		float* __restrict z;
	};

    template<>
    struct SOA<math::affine3f> {
        SOA() = default;

        SOA(int n) : nAlloc(n) {
            // Linear Part
            this->m00 = CUDABufferManager::allocate<float>(n * 12);
            this->m01 = this->m00 + n;
            this->m02 = this->m01 + n;
            this->m10 = this->m02 + n;
            this->m11 = this->m10 + n;
            this->m12 = this->m11 + n;
            this->m20 = this->m12 + n;
            this->m21 = this->m20 + n;
            this->m22 = this->m21 + n;
            this->m30 = this->m22 + n;
            this->m31 = this->m30 + n;
            this->m32 = this->m31 + n;
        }

        /*SOA(float* ptr, int n) : nAlloc(n)
        {
            this->m00 = ptr;
            this->m01 = this->m00 + n;
            this->m02 = this->m01 + n;
            this->m10 = this->m02 + n;
            this->m11 = this->m10 + n;
            this->m12 = this->m11 + n;
            this->m20 = this->m12 + n;
            this->m21 = this->m20 + n;
            this->m22 = this->m21 + n;
            this->m30 = this->m22 + n;
            this->m31 = this->m30 + n;
            this->m32 = this->m31 + n;
        }*/

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
            __forceinline__ __device__ operator math::affine3f() const {
                vtxAssert(i < soa->nAlloc); // Ensure the index is in range
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
            __forceinline__ __device__ void operator=(const math::affine3f& a) {
                vtxAssert(i < soa->nAlloc); // Ensure the index is in range
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


        __forceinline__ __device__ Proxy operator[](int i) {
            vtxAssert(i < nAlloc); // Ensure the index is in range
            return Proxy{ this, i };
        }

        __forceinline__ __device__ math::affine3f operator[](int i) const {
            vtxAssert(i < nAlloc); // Ensure the index is in range
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

    template <> struct SOA<AccumulationWorkItem> {
        SOA() = default;

        SOA(int n) : nAlloc(n) {

            originPixel = CUDABufferManager::allocate<int>(n);
            depth = CUDABufferManager::allocate<int>(n);
            radiance = SOA<math::vec3f>(n);
        }

        SOA& operator=(const SOA& s) {
            nAlloc = s.nAlloc;

            this->originPixel = s.originPixel;
            this->depth = s.depth;
            this->radiance = s.radiance;

            return *this;
        }

        struct Proxy {
            __forceinline__ __device__ operator AccumulationWorkItem() const {
                AccumulationWorkItem r;
                r.originPixel = soa->originPixel[i];
                r.radiance = soa->radiance[i];
                r.depth = soa->depth[i];
                return r;
            }

            __forceinline__ __device__ void operator=(const AccumulationWorkItem& a) {
                vtxAssert(i < soa->nAlloc); // Ensure the index is in range
                soa->originPixel[i] = a.originPixel;
                soa->radiance[i] = a.radiance;
                soa->depth[i] = a.depth;
            }

            SOA* soa;
            int i;
        };

        __forceinline__ __device__ Proxy operator[](int i) {
            vtxAssert(i < nAlloc); // Ensure the index is in range
            return Proxy{ this, i };
        }
        __forceinline__ __device__ AccumulationWorkItem operator[](int i) const {
            vtxAssert(i < nAlloc); // Ensure the index is in range
            AccumulationWorkItem r;

            r.originPixel = this->originPixel[i];
            r.radiance = this->radiance[i];
            r.depth = this->depth[i];
            return r;
        }


        int nAlloc;
        int* __restrict                originPixel;
        int* __restrict                depth;
        SOA<math::vec3f>               radiance;
    };

    template <> struct SOA<TraceWorkItem> {
        SOA() = default;

        SOA(int n) : nAlloc(n) {
            originPixel = CUDABufferManager::allocate<int>(n);
            depth = CUDABufferManager::allocate<int>(n);
            pdf = CUDABufferManager::allocate<float>(n);
            seed = CUDABufferManager::allocate<unsigned>(n);
            extendRay = CUDABufferManager::allocate<bool>(n);
            eventType = CUDABufferManager::allocate<mdl::BsdfEventType>(n);
            origin = SOA<math::vec3f>(n);
            direction = SOA<math::vec3f>(n);
            radiance = SOA<math::vec3f>(n);
            throughput = SOA<math::vec3f>(n);
            mediumIor = SOA<math::vec3f>(n);
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
            this->extendRay = s.extendRay;

            return *this;
        }

        struct Proxy {
            __forceinline__ __device__ operator TraceWorkItem() const {
                TraceWorkItem r{};
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
                r.extendRay = soa->extendRay[i];

                return r;
            }

            __forceinline__ __device__ void operator=(const TraceWorkItem& a) {
                vtxAssert(i < soa->nAlloc); // Ensure the index is in range
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
                soa->extendRay[i] = a.extendRay;
            }

            SOA* soa;
            int i;
        };

        __forceinline__ __device__ Proxy operator[](int i) {
            vtxAssert(i < nAlloc); // Ensure the index is in range
            return Proxy{ this, i };
        }
        __forceinline__ __device__ TraceWorkItem operator[](int i)const {
            vtxAssert(i < nAlloc); // Ensure the index is in range
            TraceWorkItem r;
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
            r.extendRay = this->extendRay[i];
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
        bool* __restrict extendRay;
    };

    template <> struct SOA<ShadowWorkItem> {
        SOA() = default;

        SOA(int n) : nAlloc(n) {

            distance = CUDABufferManager::allocate<float>(n);
            originPixel = CUDABufferManager::allocate<int>(n);
            depth = CUDABufferManager::allocate<int>(n);

            radiance = SOA<math::vec3f>(n);
            direction = SOA<math::vec3f>(n);
            origin = SOA<math::vec3f>(n);
        }

        SOA& operator=(const SOA& s) {
            nAlloc = s.nAlloc;

            this->originPixel = s.originPixel;
            this->distance = s.distance;
            this->depth = s.depth;
            this->radiance = s.radiance;
            this->direction = s.direction;
            this->origin = s.origin;

            return *this;
        }

        struct Proxy {
            __forceinline__ __device__ operator ShadowWorkItem() const {
                ShadowWorkItem r;
                r.originPixel = soa->originPixel[i];
                r.origin = soa->origin[i];
                r.direction = soa->direction[i];
                r.radiance = soa->radiance[i];
                r.depth = soa->depth[i];
                r.distance = soa->distance[i];
                return r;
            }

            __forceinline__ __device__ void operator=(const ShadowWorkItem& a) {
                vtxAssert(i < soa->nAlloc); // Ensure the index is in range
                soa->originPixel[i] = a.originPixel;
                soa->origin[i] = a.origin;
                soa->direction[i] = a.direction;
                soa->radiance[i] = a.radiance;
                soa->depth[i] = a.depth;
                soa->distance[i] = a.distance;
            }

            SOA* soa;
            int i;
        };

        __forceinline__ __device__ Proxy operator[](int i) {
            vtxAssert(i < nAlloc); // Ensure the index is in range
            return Proxy{ this, i };
        }
        __forceinline__ __device__ ShadowWorkItem operator[](int i) const {
            vtxAssert(i < nAlloc); // Ensure the index is in range
            ShadowWorkItem r;

            r.originPixel = this->originPixel[i];
            r.origin = this->origin[i];
            r.direction = this->direction[i];
            r.radiance = this->radiance[i];
            r.depth = this->depth[i];
            r.distance = this->distance[i];
            return r;
        }


        int nAlloc;
        int* __restrict                originPixel;
        int* __restrict                depth;
        float* __restrict              distance;
        SOA<math::vec3f>               radiance;
        SOA<math::vec3f>               origin;
        SOA<math::vec3f>               direction;
    };

    template <> struct SOA<RayWorkItem> {
        SOA() = default;

        SOA(int n) : nAlloc(n) {

            originPixel = CUDABufferManager::allocate<int>(n);
            depth = CUDABufferManager::allocate<int>(n);

            pdf = CUDABufferManager::allocate<float>(n);
            hitDistance = CUDABufferManager::allocate<float>(n);
            //shadowDistance = CUDABufferManager::allocate<float>(n);

            seed = CUDABufferManager::allocate<unsigned>(n);
            hitInstanceId = CUDABufferManager::allocate<unsigned>(n);
            hitTriangleId = CUDABufferManager::allocate<unsigned>(n);

            //shadowTrace = CUDABufferManager::allocate<bool>(n);
            //extendRay = CUDABufferManager::allocate<bool>(n);

            eventType = CUDABufferManager::allocate<mdl::BsdfEventType>(n);
            //firstHitType = CUDABufferManager::allocate<mdl::BsdfEventType>(n);

            /*hitWTO = SOA<math::affine3f>(n);
            hitOTW = SOA<math::affine3f>(n);*/

            origin = SOA<math::vec3f>(n);
            direction = SOA<math::vec3f>(n);
            radiance = SOA<math::vec3f>(n);
            throughput = SOA<math::vec3f>(n);
            mediumIor = SOA<math::vec3f>(n);
            //hitPosition = SOA<math::vec3f>(n);
            hitBaricenter = SOA<math::vec3f>(n);
            //radianceDirect = SOA<math::vec3f>(n);
            //shadowDirection = SOA<math::vec3f>(n);

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

            //this->hitPosition = s.hitPosition;
            this->hitDistance = s.hitDistance;
            this->hitBaricenter = s.hitBaricenter;
            this->hitInstanceId = s.hitInstanceId;
            this->hitTriangleId = s.hitTriangleId;
            /*this->hitWTO = s.hitWTO;
            this->hitOTW = s.hitOTW;*/

            //this->firstHitType = s.firstHitType;

            //this->radianceDirect = s.radianceDirect;
            //this->shadowTrace = s.shadowTrace;
            //this->shadowDirection = s.shadowDirection;
            //this->shadowDistance = s.shadowDistance;

            //this->extendRay = s.extendRay;

            return *this;
        }

        struct Proxy {
            __forceinline__ __device__ operator RayWorkItem() const {
                RayWorkItem r{};
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

                r.hitDistance = soa->hitDistance[i];
                r.hitBaricenter = soa->hitBaricenter[i];
                r.hitInstanceId = soa->hitInstanceId[i];
                r.hitTriangleId = soa->hitTriangleId[i];

                return r;
            }

            __forceinline__ __device__ void operator=(const RayWorkItem& a) {
                vtxAssert(i < soa->nAlloc); // Ensure the index is in range

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

                soa->hitDistance[i] = a.hitDistance;
                soa->hitBaricenter[i] = a.hitBaricenter;
                soa->hitInstanceId[i] = a.hitInstanceId;
                soa->hitTriangleId[i] = a.hitTriangleId;
            }

            SOA* soa;
            int i;
        };

        __forceinline__ __device__ Proxy operator[](int i) {
            vtxAssert(i < nAlloc); // Ensure the index is in range
            return Proxy{ this, i };
        }
        __forceinline__ __device__ RayWorkItem operator[](int i)const {
            vtxAssert(i < nAlloc); // Ensure the index is in range
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

            r.hitDistance = this->hitDistance[i];
            r.hitBaricenter = this->hitBaricenter[i];
            r.hitInstanceId = this->hitInstanceId[i];
            r.hitTriangleId = this->hitTriangleId[i];

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


        float* __restrict    hitDistance;
        SOA<math::vec3f>     hitBaricenter;
        unsigned* __restrict hitInstanceId;
        unsigned* __restrict hitTriangleId;

        SOA<math::vec3f>     compressedHit;
    };

    template <> struct SOA<EscapedWorkItem> {
        SOA() = default;

        SOA(int n) : nAlloc(n) {

            originPixel = CUDABufferManager::allocate<int>(n);
            depth = CUDABufferManager::allocate<int>(n);
            pdf = CUDABufferManager::allocate<float>(n);
            seed = CUDABufferManager::allocate<unsigned>(n);
            eventType = CUDABufferManager::allocate<mdl::BsdfEventType>(n);
            direction = SOA<math::vec3f>(n);
            radiance = SOA<math::vec3f>(n);
            throughput = SOA<math::vec3f>(n);

        }

        SOA& operator=(const SOA& s) {
            nAlloc = s.nAlloc;

            this->originPixel = s.originPixel;
            this->seed = s.seed;
            this->direction = s.direction;
            this->depth = s.depth;
            this->pdf = s.pdf;
            this->radiance = s.radiance;
            this->throughput = s.throughput;
            this->eventType = s.eventType;
            return *this;
        }

        struct Proxy {
            __forceinline__ __device__ operator EscapedWorkItem() const {
                EscapedWorkItem r{};
                r.originPixel = soa->originPixel[i];
                r.seed = soa->seed[i];
                r.direction = soa->direction[i];
                r.depth = soa->depth[i];
                r.pdf = soa->pdf[i];
                r.radiance = soa->radiance[i];
                r.throughput = soa->throughput[i];
                r.eventType = soa->eventType[i];
                return r;
            }

            __forceinline__ __device__ void operator=(const EscapedWorkItem& a) {
                vtxAssert(i < soa->nAlloc); // Ensure the index is in range

                soa->originPixel[i] = a.originPixel;
                soa->seed[i] = a.seed;
                soa->direction[i] = a.direction;
                soa->depth[i] = a.depth;
                soa->pdf[i] = a.pdf;
                soa->radiance[i] = a.radiance;
                soa->throughput[i] = a.throughput;
                soa->eventType[i] = a.eventType;
            }

            SOA* soa;
            int i;
        };

        __forceinline__ __device__ Proxy operator[](int i) {
            vtxAssert(i < nAlloc); // Ensure the index is in range
            return Proxy{ this, i };
        }
        __forceinline__ __device__ EscapedWorkItem operator[](int i)const {
            vtxAssert(i < nAlloc); // Ensure the index is in range
            EscapedWorkItem r;

            r.originPixel = this->originPixel[i];
            r.seed = this->seed[i];
            r.direction = this->direction[i];
            r.depth = this->depth[i];
            r.pdf = this->pdf[i];
            r.radiance = this->radiance[i];
            r.throughput = this->throughput[i];
            r.eventType = this->eventType[i];

            return r;
        }


        int nAlloc;
        int* __restrict                originPixel;
        unsigned* __restrict           seed;
        SOA<math::vec3f>               direction;
        int* __restrict                depth;
        float* __restrict              pdf;
        SOA<math::vec3f>               radiance;
        SOA<math::vec3f>               throughput;
        mdl::BsdfEventType* __restrict eventType;
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
            //printf("Pushing %s %d\n", name, index);
            vtxAssertMsg(index < nAlloc, name); // Ensure the index is in range
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
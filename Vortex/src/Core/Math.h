#pragma once
#ifndef H_MATH
#define H_MATH
#include "Math/vec.h"
#include "Math/AffineSpace.h"
#include <tuple>
#include <cuda_runtime.h>

namespace vtx::math
{

#define PI 3.1415926535897932384626433832795
#define PI_180 0.01745329251994329576923690768489
#define _180_PI 57.295779513082320876798154814105


	template<typename T>
	constexpr auto length = gdt::length<T>;

	template<typename T>
	__both__ T max(T a , T b)
	{
		return a > b ? a : b;
	}

	template<typename T>
	__both__ T min(T a, T b)
	{
		return a < b ? a : b;
	}

	using OneTy = gdt::OneTy;
	static OneTy Identity = gdt::one;

	template<typename T>
	T toRadians(T degree) {
		return degree * T(PI_180);
	}

	template<typename T>
	T toDegrees(T  radian) {
		return radian * _180_PI;
	}

	///////////////////////////////////////////////////////////////////////////
	///// Vector Definitions //////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////
	
	template<typename T, int N>
	using vec_t = gdt::vec_t<T, N>;

	template<typename T>
	using vec3a_t = gdt::vec3a_t<T>;
		
#define DEFINE_VEC_TYPES(T,t)		\
		using vec2##t = vec_t<T,2>;			\
		using vec3##t = vec_t<T,3>;			\
		using vec4##t = vec_t<T,4>;			\
		using vec3##t##a = vec3a_t<T>;		\
		
	DEFINE_VEC_TYPES(int8_t, c);
	DEFINE_VEC_TYPES(int16_t, s);
	DEFINE_VEC_TYPES(int32_t, i);
	DEFINE_VEC_TYPES(int64_t, l);
	DEFINE_VEC_TYPES(uint8_t, uc);
	DEFINE_VEC_TYPES(uint16_t, us);
	DEFINE_VEC_TYPES(uint32_t, ui);
	DEFINE_VEC_TYPES(uint64_t, ul);
	DEFINE_VEC_TYPES(float, f);
	DEFINE_VEC_TYPES(double, d);

	template<typename T>
	static inline __both__ vec_t<T, 3> normalize(const vec_t<T, 3>& v) {
		return gdt::normalize(v);
	}

	__inline__ __both__ float saturate(float f)
	{
		return min(1.f, max(0.f, f));
	}

	static inline __both__ vec3f saturate(const vec3f& v) {
		return vec3f(saturate(v.x), saturate(v.y), saturate(v.z));
	}

	inline __both__ vec3f randomColor(int i)
	{
		return gdt::randomColor(i);
	}
		
	///////////////////////////////////////////////////////////////////////////
		///// Linear Space Definitions ////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////

	template<typename T>
	using LinearSpace2 = gdt::LinearSpace2<T>;

	template<typename T>
	using LinearSpace3 = gdt::LinearSpace3<T>;

	/*! Shortcuts for common linear spaces. */
	using LinearSpace2f = LinearSpace2<vec2f>;
	using LinearSpace3f = LinearSpace3<vec3f>;
	using LinearSpace3fa = LinearSpace3<vec3fa>;

	using linear2f = LinearSpace2f;
	using linear3f = LinearSpace3f;


	///////////////////////////////////////////////////////////////////////////
		///// Quaternion Definitions //////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////

	template<typename T>
	using QuaternionT = gdt::QuaternionT<T>;

	/*! default template instantiations */
	typedef QuaternionT<float>  Quaternion3f;
	typedef QuaternionT<double> Quaternion3d;

	///////////////////////////////////////////////////////////////////////////
		///// Affine Space Definition	///////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////

	/// NB : AffineSpace to float[12] assignment performed added in gdt/AffineSpace.h


	struct rowLinearSpace
	{
		float4 l[3];
	};

#define Scalar_t typename T::vector_t::scalar_t

	template<typename T>
	struct AffineSpaceT : public gdt::AffineSpaceT<T> {

		using gdt::AffineSpaceT<T>::AffineSpaceT; // Inherit all constructors

		///* Assign values to a transform matrix expressed as float[12] with a conversion operator*/
		//inline __both__  operator Scalar_t* () const {
		//	auto* m = (Scalar_t*)alloca(12 * sizeof(Scalar_t));
		//	m[0] = (Scalar_t)l.vx.x; m[1] = (Scalar_t)l.vy.x; m[2] = (Scalar_t)l.vz.x; m[3] = (Scalar_t)p.x;
		//	m[4] = (Scalar_t)l.vx.y; m[5] = (Scalar_t)l.vy.y; m[6] = (Scalar_t)l.vz.y; m[7] = (Scalar_t)p.y;
		//	m[8] = (Scalar_t)l.vx.z; m[9] = (Scalar_t)l.vy.z; m[10] = (Scalar_t)l.vz.z; m[11] = (Scalar_t)p.z;
		//	return m;
		//}

		//inline __both__ operator float4* () const {
		//	auto* m = (float4*)alloca(3 * sizeof(float4));
		//	m[0].x = (float)l.vx.x; m[0].y = (float)l.vx.y; m[0].z = (float)l.vx.z; m[0].w = 0.0f;
		//	m[1].x = (float)l.vy.x; m[1].y = (float)l.vy.y; m[1].z = (float)l.vy.z; m[1].w = 0.0f;
		//	m[2].x = (float)l.vz.x; m[2].y = (float)l.vz.y; m[2].z = (float)l.vz.z; m[2].w = 0.0f;
		//	return m;
		//}

		inline __host__ __device__ void toFloat(float* out) const
		{
			out[0] = (float)l.vx.x; out[1] = (float)l.vy.x; out[2] = (float)l.vz.x; out[3] = (float)p.x;
			out[4] = (float)l.vx.y; out[5] = (float)l.vy.y; out[6] = (float)l.vz.y; out[7] = (float)p.y;
			out[8] = (float)l.vx.z; out[9] = (float)l.vy.z; out[10] = (float)l.vz.z; out[11] = (float)p.z;
		}

		inline __host__ __device__ void toFloat4(float4* out) const {
			out[0].x = (float)l.vx.x; out[0].y = (float)l.vx.y; out[0].z = (float)l.vx.z; out[0].w = 0.0f;
			out[1].x = (float)l.vy.x; out[1].y = (float)l.vy.y; out[1].z = (float)l.vy.z; out[1].w = 0.0f;
			out[2].x = (float)l.vz.x; out[2].y = (float)l.vz.y; out[2].z = (float)l.vz.z; out[2].w = 0.0f;
		}

		//Constructor from Scalar_t[12] array in row major order
		__both__ AffineSpaceT(const Scalar_t* rowMajor)
		{
			l.vx.x = rowMajor[0]; l.vy.x = rowMajor[1]; l.vz.x = rowMajor[2]; p.x = rowMajor[3];
			l.vx.y = rowMajor[4]; l.vy.y = rowMajor[5]; l.vz.y = rowMajor[6]; p.y = rowMajor[7];
			l.vx.z = rowMajor[8]; l.vy.z = rowMajor[9]; l.vz.z = rowMajor[10]; p.z = rowMajor[11];
		}
//#ifdef __CUDACC__
		//Constructor from float4 array in row major order
		__both__  AffineSpaceT(const float4* rowMajor)
		{
			l.vx.x = rowMajor[0].x;
			l.vx.y = rowMajor[1].x;
			l.vx.z = rowMajor[2].x;

			l.vy.x = rowMajor[0].y;
			l.vy.y = rowMajor[1].y;
			l.vy.z = rowMajor[2].y;

			l.vz.x = rowMajor[0].z;
			l.vz.y = rowMajor[1].z;
			l.vz.z = rowMajor[2].z;

			p.x = rowMajor[0].w;
			p.y = rowMajor[1].w;
			p.z = rowMajor[2].w;
		}
//#endif
	};

	template<typename T>
	AffineSpaceT<T> AffineFromEuler(const vec_t<Scalar_t, 3>& euler) {
		// Transform euler to quaternion

		QuaternionT<Scalar_t> quat = QuaternionT<Scalar_t>(euler.z, euler.y, euler.z);
		T L = T(quat);
		auto Affine = AffineSpaceT<T>(L);
		return Affine;
	}


	template<typename T>
	void VectorFromAffine(AffineSpaceT<T> affine, vec_t<Scalar_t, 3>& translation, vec_t<Scalar_t, 3> scale, vec_t<Scalar_t, 3>& euler) {
		translation = affine.p;

		scale.x = length<Scalar_t>(affine.l.vx);
		scale.y = length<Scalar_t>(affine.l.vy);
		scale.z = length<Scalar_t>(affine.l.vz);

		vec_t<Scalar_t, 3> reverseScale = 1.0f / scale;
		vec_t<Scalar_t, 3> rVx = affine.l.vx * reverseScale.x;
		vec_t<Scalar_t, 3> rVy = affine.l.vy * reverseScale.y;
		vec_t<Scalar_t, 3> rVz = affine.l.vz * reverseScale.z;

		Scalar_t& m00 = affine.l.vx.x;
		Scalar_t& m10 = affine.l.vx.y;
		Scalar_t& m20 = affine.l.vx.z;

		Scalar_t& m01 = affine.l.vy.x;
		Scalar_t& m11 = affine.l.vy.y;
		Scalar_t& m21 = affine.l.vy.z;

		Scalar_t& m02 = affine.l.vz.x;
		Scalar_t& m12 = affine.l.vz.y;
		Scalar_t& m22 = affine.l.vz.z;

		if (m10 > 0.998f) { // singularity at north pole
			euler.y = atan2(m02, m22);
			euler.x = M_PI * 0.5f;
			euler.z = 0;
		}
		else if (m10 < -0.998f) { // singularity at south pole
			euler.y = atan2(m02, m22);
			euler.x = M_PI * 0.5f;
			euler.z = 0;
		}
		else {
			euler.y = atan2(-m20, m00);
			euler.z = atan2(-m12, m11);
			euler.x = asin(m10);
		}
		return;
	}
#undef ScalarT

	using AffineSpace2f = AffineSpaceT<LinearSpace2f>;
	using AffineSpace3f = AffineSpaceT<LinearSpace3f>;
	using AffineSpace3fa = AffineSpaceT<LinearSpace3fa>;
	using OrthonormalSpace3f = AffineSpaceT<Quaternion3f>;

	using affine2f = AffineSpace2f;
	using affine3f = AffineSpace3f;

	__device__ __host__ static inline vec3f transformVector3F(const affine3f& affine, const vec3f& vec) {
		return gdt::xfmVector<LinearSpace3f>(affine, vec);
	}

	__device__ __host__ static inline vec3f transformPoint3F(const affine3f& affine, const vec3f& vec) {
		return gdt::xfmPoint<LinearSpace3f>(affine, vec);
	}

	__device__ __host__ static inline vec3f transformNormal3F(const affine3f& affine, const vec3f& vec) {
		return gdt::xfmNormal<LinearSpace3f>(affine, vec);
	}

	/////////////////////////////////////////////////////////////
		/////////////////// Some Utils Definition ///////////////////
		/////////////////////////////////////////////////////////////

	static vec3f xAxis = vec3f{ 1.0f, 0.0f, 0.0f };
	static vec3f yAxis = vec3f{ 0.0f, 1.0f, 0.0f };
	static vec3f zAxis = vec3f{ 0.0f, 0.0f, 1.0f };
	static vec3f origin = vec3f{ 0.0f, 0.0f, 1.0f };

}

#endif // !__GDT_MATH_H__
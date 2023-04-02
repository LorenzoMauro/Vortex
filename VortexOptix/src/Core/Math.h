#pragma once
#include "gdt/math/vec.h"
#include "gdt/math/AffineSpace.h"
#include <tuple>

namespace vtx {
	namespace math {


		template<typename T>
		constexpr auto lenght = gdt::length<T>;

		using OneTy = gdt::OneTy;
		static OneTy Identity = gdt::one;


		///////////////////////////////////////////////////////////////////////////
		///// Vector Definitions //////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////

		template<typename T, int N>
		using vec_t = gdt::vec_t<T, N>;
		
		template<typename T>
		using vec3a_t = gdt::vec3a_t<T>;
		
		#define define_vec_types(T,t)		\
		using vec2##t = vec_t<T,2>;			\
		using vec3##t = vec_t<T,3>;			\
		using vec4##t = vec_t<T,4>;			\
		using vec3##t##a = vec3a_t<T>;		\
		
		define_vec_types(int8_t, c);
		define_vec_types(int16_t, s);
		define_vec_types(int32_t, i);
		define_vec_types(int64_t, l);
		define_vec_types(uint8_t, uc);
		define_vec_types(uint16_t, us);
		define_vec_types(uint32_t, ui);
		define_vec_types(uint64_t, ul);
		define_vec_types(float, f);
		define_vec_types(double, d);
		
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

#define Scalar_t typename T::vector_t::scalar_t

		template<typename T>
		struct AffineSpaceT : public gdt::AffineSpaceT<T> {

			using gdt::AffineSpaceT<T>::AffineSpaceT; // Inherit all constructors

			/* Assign values to a transform matrix expressed as float[12] with a conversion operator*/
			inline operator Scalar_t* () const {
				Scalar_t* m = (Scalar_t*)alloca(12 * sizeof(Scalar_t));
				m[0] = (Scalar_t)l.vx.x; m[1] = (Scalar_t)l.vy.x; m[2] = (Scalar_t)l.vz.x; m[3] = (Scalar_t)p.x;
				m[4] = (Scalar_t)l.vx.y; m[5] = (Scalar_t)l.vy.y; m[6] = (Scalar_t)l.vz.y; m[7] = (Scalar_t)p.y;
				m[8] = (Scalar_t)l.vx.z; m[9] = (Scalar_t)l.vy.z; m[10] = (Scalar_t)l.vz.z; m[11] = (Scalar_t)p.z;
				return m;
			}
		};

		template<typename T>
		AffineSpaceT<T> AffineFromEuler(const vec_t<Scalar_t, 3>& euler) {
			// Transform euler to quaternion

			QuaternionT<Scalar_t> quat = QuaternionT<Scalar_t>(euler.z, euler.y, euler.z);
			T L = T(quat);
			AffineSpaceT<T> Affine = AffineSpaceT<T>(L);
			return Affine;
		}


		template<typename T>
		void VectorFromAffine(AffineSpaceT<T> affine, vec_t<Scalar_t, 3>& translation, vec_t<Scalar_t, 3> scale, vec_t<Scalar_t, 3>& euler) {
			translation = affine.p;

			scale.x = length(affine.l.vx);
			scale.y = length(affine.l.vy);
			scale.z = length(affine.l.vz);

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

			if (m10 > 0.998) { // singularity at north pole
				euler.y = atan2(m02, m22);
				euler.x = M_PI * 0.5f;
				euler.z = 0;
			}
			else if (m10 < -0.998) { // singularity at south pole
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
	}
}


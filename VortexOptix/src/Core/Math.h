#pragma once
#include "gdt/math/vec.h"
#include "gdt/math/AffineSpace.h"
#include <tuple>

namespace vtx {
	namespace math {

		using vec2i = gdt::vec2i;
		using vec3f = gdt::vec3f;
		using vec4f = gdt::vec4f;
		using LinearSpace3f = gdt::LinearSpace3f;
		using Quaternionf = gdt::Quaternion3f;
		using Affine3f = gdt::AffineSpace3f;

		template<typename T>
		constexpr auto lenght = gdt::length<T>;

		template<typename T>
		using vec_3 = gdt::vec_t<T,3>;

		template<typename T>
		using QuaternionT = gdt::QuaternionT<T>;

		template<typename T>
		using LinearSpace3T = gdt::LinearSpace3<vec_3<T>>;

		template<typename T>
		using Affine3T = gdt::AffineSpaceT<LinearSpace3T<T>>;

		template<typename T>
		Affine3T<T> AffineFromEuler(const vec_3<T>& euler) {
			// Transform euler to quaternion
			
			QuaternionT<T> quat = QuaternionT<T>(euler.z, euler.y, euler.z);
			LinearSpace3T<T> L = LinearSpace3T<T>(quat);
			Affine3T<T> Affine = Affine3T<T>(L);
			return Affine;
		}

		template<typename T>
		void VectorFromAffine(Affine3T<T> affine, vec_3<T>& translation, vec_3<T> scale, vec_3<T>& euler) {
			translation = affine.p;

			scale.x = length(affine.l.vx);
			scale.y = length(affine.l.vy);
			scale.z = length(affine.l.vz);

			vec_3<T> reverseScale = 1.0f / scale;
			vec_3<T> rVx = affine.l.vx * reverseScale.x;
			vec_3<T> rVy = affine.l.vy * reverseScale.y;
			vec_3<T> rVz = affine.l.vz * reverseScale.z;

			T& m00 = affine.l.vx.x;
			T& m10 = affine.l.vx.y;
			T& m20 = affine.l.vx.z;

			T& m01 = affine.l.vy.x;
			T& m11 = affine.l.vy.y;
			T& m21 = affine.l.vy.z;

			T& m02 = affine.l.vz.x;
			T& m12 = affine.l.vz.y;
			T& m22 = affine.l.vz.z;

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
	}
}


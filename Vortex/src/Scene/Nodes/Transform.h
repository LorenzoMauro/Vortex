#pragma once
#include "Scene/Node.h"
#include "Core/Math.h"

namespace vtx::graph
{
	class Transform : public Node {
	public:
		Transform();

		 ~Transform() override;

		math::vec3f transformVector(const math::vec3f& vector);

		math::vec3f transformNormal(const math::vec3f& vector);

		math::vec3f transformPoint(const math::vec3f& vector);

		void applyLocalTransform(const math::affine3f& transform);
		void applyGlobalSpaceTransform(const math::affine3f& transform);

		void setAffine(const math::affine3f& affine);

		/* Scale utility given float */
		void scale(const math::vec3f& scale);

		/* Translation utility given vector */
		void translate(const math::vec3f& translation);

		/* Translation utility given axis and amount */
		void translate(const math::vec3f& direction, float amount);

		/* Rotation utility for axis angle in radians */
		void rotate(const math::vec3f& axis, float radian);

		/* Rotation utility for axis angle in degree */
		void rotateDegree(const math::vec3f& axis, float degree);

		/* Rotation utility for axis angle around point in radians */
		void rotateAroundPoint(const math::vec3f& point, const math::vec3f& axis, float radian);

		/* Rotation utility for axis angle around point in degree */
		void rotateAroundPointDegree(const math::vec3f& point, const math::vec3f& axis, float degree);

		/* Rotation utility for rotation expressed as quaternion */
		void rotateQuaternion(const math::Quaternion3f& quaternion);

		/* Rotation utility for rotations expressed as orbit*/
		void rotateOrbit(float pitch, math::vec3f xAxis, float yaw, math::vec3f zAxis);

		/* Update the transformation given the vector representation*/
		void updateFromVectors();

		/* Update the vector representation given the affine matrix*/
		void updateFromAffine();

		void accept(NodeVisitor& visitor) override;

	public:
		math::vec3f scaleVector{ 1.0f };
		math::vec3f translation{ 0.0f };
		math::vec3f eulerAngles{ 0.0f };
		math::affine3f rcpAffineTransform = math::affine3f(math::Identity);
		math::affine3f affineTransform = math::affine3f(math::Identity);
		math::affine3f globalTransform = math::affine3f(math::Identity);
		math::affine3f parentGlobalTransform = math::affine3f(math::Identity);
		math::affine3f reciprocalParentGlobalTransform = math::affine3f(math::Identity);

	};
}

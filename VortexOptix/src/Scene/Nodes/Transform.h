#pragma once
#include "Scene/Node.h"
#include "Core/Math.h"

namespace vtx::graph
{

	struct TransformAttribute {
		math::vec3f scale{ 1.0f };
		math::vec3f translation{ 0.0f };
		math::vec3f eulerAngles{ 0.0f };
		math::affine3f affineTransform = math::affine3f(math::Identity);

		/* Update the transformation given the vector representation*/
		void updateFromVectors() {
			affineTransform = math::affine3f::translate(translation) * math::AffineFromEuler<math::LinearSpace3f>(eulerAngles) * math::affine3f::scale(scale);
		}

		/* Update the vector representation given the affine matrix*/
		void updateFromAffine() {
			math::VectorFromAffine<math::LinearSpace3f>(affineTransform, translation, scale, eulerAngles);
		}
	};

	class Transform : public Node {
	public:
		Transform();

		math::vec3f transformVector(const math::vec3f& vector);

		math::vec3f transformNormal(const math::vec3f& vector);

		math::vec3f transformPoint(const math::vec3f& vector);

		void setAffine(const math::affine3f& affine);

		/* Scale utility given float */
		void scale(float scale);

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

		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;

		void accept(std::shared_ptr<NodeVisitor> visitor) override;

	public:
		TransformAttribute transformationAttribute;
	};
}

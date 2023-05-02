#include "Transform.h"
#include "Scene/Traversal.h"

namespace vtx::graph
{
	Transform::Transform() : Node(NT_TRANSFORM) {}

	math::vec3f Transform::transformVector(const math::vec3f& vector) {
		return transformVector3F(transformationAttribute.affineTransform, vector);
	}

	math::vec3f Transform::transformNormal(const math::vec3f& vector) {
		return transformNormal3F(transformationAttribute.affineTransform, vector);
	}

	math::vec3f Transform::transformPoint(const math::vec3f& vector) {
		return transformPoint3F(transformationAttribute.affineTransform, vector);
	}

	void Transform::setAffine(const math::affine3f& affine)
	{
		transformationAttribute.affineTransform = affine;
		transformationAttribute.updateFromAffine();
	}

	/* Translation utility given vector */

	void Transform::scale(float scale)
	{
		const math::affine3f scaleMatrix = math::affine3f::scale(scale);
		transformationAttribute.affineTransform = scaleMatrix * transformationAttribute.affineTransform;
		transformationAttribute.updateFromAffine();
	}

	void Transform::translate(const math::vec3f& translation) {
		const math::affine3f translationMatrix = math::affine3f::translate(translation);
		transformationAttribute.affineTransform = translationMatrix * transformationAttribute.affineTransform;
		transformationAttribute.updateFromAffine();
	}

	/* Translation utility given axis and ammount */

	void Transform::translate(const math::vec3f& direction, const float amount) {
		translate(direction * amount);
	}

	/* Rotation utility for axis angle in radians */

	void Transform::rotate(const math::vec3f& axis, const float radian) {
		const math::affine3f rotationMatrix = math::affine3f::rotate(axis, radian);
		transformationAttribute.affineTransform = rotationMatrix * transformationAttribute.affineTransform;
		transformationAttribute.updateFromAffine();
	}

	/* Rotation utility for axis angle in degree */

	void Transform::rotateDegree(const math::vec3f& axis, const float degree) {
		rotate(axis, math::toRadians(degree));
	}

	/* Rotation utility for axis angle around point in radians */

	void Transform::rotateAroundPoint(const math::vec3f& point, const math::vec3f& axis, const float radian) {
		const math::affine3f transformation = math::affine3f::rotate(point, axis, radian);
		transformationAttribute.affineTransform = transformation * transformationAttribute.affineTransform;
		transformationAttribute.updateFromAffine();
	}

	/* Rotation utility for axis angle around point in degree */

	void Transform::rotateAroundPointDegree(const math::vec3f& point, const math::vec3f& axis, const float degree) {
		rotateAroundPoint(point, axis, math::toRadians(degree));
	}

	void Transform::rotateQuaternion(const math::Quaternion3f& quaternion) {
		auto rotationMatrix = math::LinearSpace3f(quaternion);
		auto transformation = math::affine3f(rotationMatrix);
		transformationAttribute.affineTransform = transformation * transformationAttribute.affineTransform;
		transformationAttribute.updateFromAffine();
	}

	void Transform::rotateOrbit(float pitch, math::vec3f xAxis, float yaw, math::vec3f zAxis) {
		math::affine3f rotationPitchMatrix = math::affine3f::rotate(xAxis, pitch);
		math::affine3f rotationYawMatrix = math::affine3f::rotate(zAxis, yaw);
		transformationAttribute.affineTransform = rotationPitchMatrix * rotationYawMatrix * transformationAttribute.affineTransform;

		transformationAttribute.updateFromAffine();
	}
	void Transform::traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors)
	{
		ACCEPT(visitors)
	}
	void Transform::accept(const std::shared_ptr<NodeVisitor> visitor)
	{
		visitor->visit(sharedFromBase<Transform>());
	}
}


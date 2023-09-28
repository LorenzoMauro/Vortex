#include "Transform.h"
#include "Scene/SIM.h"
#include "Scene/Traversal.h"

namespace vtx::graph
{
	Transform::Transform() : Node(NT_TRANSFORM)
	{
		typeID = SIM::get()->getTypeId<Transform>();
	}

	Transform::~Transform()
	{
		SIM::get()->releaseTypeId<Transform>(typeID);
	}

	math::vec3f Transform::transformVector(const math::vec3f& vector) {
		return transformVector3F(affineTransform, vector);
	}

	math::vec3f Transform::transformNormal(const math::vec3f& vector) {
		return transformNormal3F(affineTransform, vector);
	}

	math::vec3f Transform::transformPoint(const math::vec3f& vector) {
		return transformPoint3F(affineTransform, vector);
	}

	void Transform::applyLocalTransform(const math::affine3f& transform)
	{
		affineTransform = transform * affineTransform;
		updateFromAffine();
	}

	void Transform::applyGlobalSpaceTransform(const math::affine3f& transform)
	{
		const math::affine3f newTransform = reciprocalParentGlobalTransform * transform * parentGlobalTransform * affineTransform;
		affineTransform = newTransform;
		updateFromAffine();
	}

	void Transform::setAffine(const math::affine3f& affine)
	{
		affineTransform = affine;
		updateFromAffine();
	}

	/* Translation utility given vector */

	void Transform::scale(const math::vec3f& scale)
	{
		const math::affine3f scaleMatrix = math::affine3f::scale(scale);
		affineTransform = scaleMatrix * affineTransform;
		updateFromAffine();
	}

	void Transform::translate(const math::vec3f& translation) {
		const math::affine3f translationMatrix = math::affine3f::translate(translation);
		affineTransform = translationMatrix * affineTransform;
		updateFromAffine();
	}

	/* Translation utility given axis and ammount */

	void Transform::translate(const math::vec3f& direction, const float amount) {
		translate(direction * amount);
	}

	/* Rotation utility for axis angle in radians */

	void Transform::rotate(const math::vec3f& axis, const float radian) {
		const math::affine3f rotationMatrix = math::affine3f::rotate(axis, radian);
		affineTransform = rotationMatrix * affineTransform;
		updateFromAffine();
	}

	/* Rotation utility for axis angle in degree */

	void Transform::rotateDegree(const math::vec3f& axis, const float degree) {
		rotate(axis, math::toRadians(degree));
	}

	/* Rotation utility for axis angle around point in radians */

	void Transform::rotateAroundPoint(const math::vec3f& point, const math::vec3f& axis, const float radian) {
		const math::affine3f transformation = math::affine3f::rotate(point, axis, radian);
		affineTransform = transformation * affineTransform;
		updateFromAffine();
	}

	/* Rotation utility for axis angle around point in degree */

	void Transform::rotateAroundPointDegree(const math::vec3f& point, const math::vec3f& axis, const float degree) {
		rotateAroundPoint(point, axis, math::toRadians(degree));
	}

	void Transform::rotateQuaternion(const math::Quaternion3f& quaternion) {
		auto rotationMatrix = math::LinearSpace3f(quaternion);
		auto transformation = math::affine3f(rotationMatrix);
		affineTransform = transformation * affineTransform;
		updateFromAffine();
	}

	void Transform::rotateOrbit(float pitch, math::vec3f xAxis, float yaw, math::vec3f zAxis) {
		math::affine3f rotationPitchMatrix = math::affine3f::rotate(xAxis, pitch);
		math::affine3f rotationYawMatrix = math::affine3f::rotate(zAxis, yaw);
		affineTransform = rotationPitchMatrix * rotationYawMatrix * affineTransform;

		updateFromAffine();
	}

	/* Update the transformation given the vector representation*/

	void Transform::updateFromVectors() {
		affineTransform = math::affine3f::translate(translation) * math::AffineFromEuler<math::LinearSpace3f>(eulerAngles) * math::affine3f::scale(scaleVector);
		rcpAffineTransform = rcp(affineTransform);
		state.updateOnDevice = true;
	}

	/* Update the vector representation given the affine matrix*/

	void Transform::updateFromAffine() {
		math::VectorFromAffine<math::LinearSpace3f>(affineTransform, translation, scaleVector, eulerAngles);
		rcpAffineTransform = rcp(affineTransform);
		state.updateOnDevice = true;
	}
	void Transform::accept(NodeVisitor& visitor)
	{
		visitor.visit(as<Transform>());
	}
}


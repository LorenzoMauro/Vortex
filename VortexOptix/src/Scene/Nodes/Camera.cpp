#include "Camera.h"
#include "Scene/Traversal.h"
#include "Scene/Utility/Operations.h"

namespace vtx::graph
{

	Camera::Camera() :
		Node(NT_CAMERA),
		position(math::vec3f{ 0.0f, 0.0f, 0.0f }),
		direction(math::vec3f{ 0.0f, 0.0f, -1.0f }),
		vertical(math::vec3f{ 0.0f, 1.0f, 0.0f }),
		horizontal(math::vec3f{ 1.0f, 0.0f, 0.0f }),
		fovY(45.0f),
		aspect(static_cast<float>(getOptions()->width) / static_cast<float>(getOptions()->height)),
		resolution(math::vec2i{ getOptions()->width, getOptions()->height }),
		mousePosition(math::vec2f{ 0.0f, 0.0f }),
		mouseDelta(math::vec2f{ 0.0f, 0.0f }),
		movementSensibility(100.0f),
		rotationSensibility(3.0f),
		zoomSensibility(0.01f),
		updated(true)
	{
		transform = ops::createNode<Transform>();
	}

	void Camera::updateDirections() {
		//horizontal = transform->transformationAttribute.AffineTransform.l.vx;
		//vertical = transform->transformationAttribute.AffineTransform.l.vy;
		//direction = transform->transformationAttribute.AffineTransform.l.vz;
		//position = transform->transformationAttribute.AffineTransform.p;

		horizontal = transform->transformVector(math::vec3f{ 1.0f, 0.0f, 0.0f });
		vertical = transform->transformVector(math::vec3f{ 0.0f, 1.0f, 0.0f });
		direction = transform->transformVector(math::vec3f{ 0.0f, 0.0f, -1.0f });
		position = transform->transformPoint(math::vec3f{ 0.0f, 0.0f, 0.0f });
	}

	void Camera::resize(const uint32_t width, const uint32_t height) {
		resolution.x = width;
		resolution.y = height;
		aspect = static_cast<float>(width) / static_cast<float>(height);
		updated = true;
	}

	void Camera::onUpdate(const float ts) {
		math::vec2f lastMousePosition = mousePosition;
		mousePosition = Input::GetMousePosition();
		mouseDelta = (mousePosition - lastMousePosition);
		//VTX_INFO("Mouse Delta: {0}, {1}", mouseDelta.x, mouseDelta.y);
		orbitNavigation(ts);
	}

	void Camera::orbitNavigation(const float ts) {
		if (Input::MouseWheel() != 0.0f && navigationActive) {
			updated = true;
			fovY += Input::MouseWheel() * zoomSensibility;
			if (fovY < 1.0f)
			{
				fovY = 1.0f;
			}
			else if (fovY > 179.0f)
			{
				fovY = 179.0f;
			}
		}

		math::vec2f delta = math::vec2f(mouseDelta.x / resolution.x, mouseDelta.y / resolution.y);

		if (Input::IsMouseButtonDown(MouseButton::Middle) && navigationActive && !(delta.x == 0.0f && delta.y == 0.0f)) {
			updated = true;
			if (Input::IsKeyDown(KeyCode::LeftShift) || Input::IsKeyDown(KeyCode::RightShift)) {
				navigationMode = NAV_PAN;
			}
			else if (Input::IsKeyDown(KeyCode::LeftControl) || Input::IsKeyDown(KeyCode::RightControl)) {
				navigationMode = NAV_DOLLY;
			}
			else {
				navigationMode = NAV_ORBIT;
			}
		}
		else {
			navigationActive = false;
			navigationMode = NAV_NONE;
			Input::SetCursorMode(CursorMode::Normal);
			return;
		}

		Input::SetCursorMode(CursorMode::Locked);

		switch (navigationMode) {
			case NAV_PAN:
			{
				math::vec3f translation = ts * movementSensibility * (-horizontal * delta.x + vertical * delta.y);
				transform->translate(translation);
				position = transform->transformPoint(math::vec3f{ 0.0f, 0.0f, 0.0f });
			}
			break;
			case NAV_DOLLY:
			{
				math::vec3f translation = -ts * movementSensibility * direction * delta.y;
				transform->translate(translation);
				position = transform->transformPoint(math::vec3f{ 0.0f, 0.0f, 0.0f });
			}
			break;
			case NAV_ORBIT:
			{
				float pitchDelta = -rotationSensibility * delta.y;
				float YawDelta = -rotationSensibility * delta.x;

				transform->rotateOrbit(pitchDelta, horizontal, YawDelta, math::zAxis);

				//math::Quaternion3f pitchRotationQuat = math::Quaternion3f(0.0f, pitchDelta, YawDelta);
				//transform->rotateQuaternion(pitchRotationQuat);
				updateDirections();
			}
			break;
		default: ;
		}
	}

	void Camera::traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors)
	{
		ACCEPT(orderedVisitors)
	}

	void Camera::accept(std::shared_ptr<NodeVisitor> visitor)
	{
		visitor->visit(sharedFromBase<Camera>());
	}
}

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
		zoomSensibility(0.1f)
	{
		transform = ops::createNode<Transform>();
	}

	void Camera::updateDirections() {
		//horizontal = inputSockets[nodeNames[NT_TRANSFORM]].getNodeTransformTransfomr>->transformationAttribute.AffineTransform.l.vx;
		//vertical = inputSockets[nodeNames[NT_TRANSFORM]].getNodeTransformTransfomr>->transformationAttribute.AffineTransform.l.vy;
		//direction = inputSockets[nodeNames[NT_TRANSFORM]].getNodeTransformTransfomr>->transformationAttribute.AffineTransform.l.vz;
		//position = inputSockets[nodeNames[NT_TRANSFORM]].getNodeTransformTransfomr>->transformationAttribute.AffineTransform.p;

		horizontal = transform->transformVector(math::vec3f{ 1.0f, 0.0f, 0.0f });
		vertical = transform->transformVector(math::vec3f{ 0.0f, 1.0f, 0.0f });
		direction = transform->transformVector(math::vec3f{ 0.0f, 0.0f, -1.0f });
		position = transform->transformPoint(math::vec3f{ 0.0f, 0.0f, 0.0f });
	}

	void Camera::resize(const uint32_t width, const uint32_t height) {
		resolution.x = width;
		resolution.y = height;
		aspect = static_cast<float>(width) / static_cast<float>(height);
		isUpdated = true;
	}

	void Camera::onUpdate(const float ts) {
		if (lockCamera)
		{
			return;
		}
		math::vec2f lastMousePosition = mousePosition;
		mousePosition = Input::GetMousePosition();
		mouseDelta = (mousePosition - lastMousePosition);
		//VTX_INFO("Mouse Delta: {0}, {1}", mouseDelta.x, mouseDelta.y);
		orbitNavigation(ts);
	}

	void Camera::orbitNavigation(const float ts) {

		math::vec2f delta = math::vec2f(mouseDelta.x / resolution.x, mouseDelta.y / resolution.y);

		if (Input::IsMouseButtonDown(MouseButton::Middle) && navigationActive)
		{
			if (navigationActive && !(delta.x == 0.0f && delta.y == 0.0f)) {
				isUpdated = true;
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
		}
		else if (Input::MouseWheel() != 0.0f && navigationActive)
		{
			isUpdated = true;
			if (Input::IsKeyDown(KeyCode::LeftAlt) || Input::IsKeyDown(KeyCode::RightAlt))
			{
				navigationMode = NAV_FOV;
				delta = math::vec2f(0.0f, Input::MouseWheel() * zoomSensibility);
			}
			else {
				navigationMode = NAV_DOLLY;
				delta = math::vec2f(0.0f, -Input::MouseWheel() * zoomSensibility);
			}
		}
		else {
			navigationActive = false;
			navigationMode = NAV_NONE;
			Input::SetCursorMode(CursorMode::Normal);
			//VTX_INFO("Setting cursor mode to normal");
			return;
		}

		Input::SetCursorMode(CursorMode::Locked);
		//VTX_INFO("Setting cursor mode to locked");
		switch (navigationMode) {
		case NAV_PAN:
		{
			const math::vec3f translation = ts * movementSensibility * (-horizontal * delta.x + vertical * delta.y);
			transform->translate(translation);
			position = transform->transformPoint(math::vec3f{ 0.0f, 0.0f, 0.0f });
		}
		break;
		case NAV_DOLLY:
		{
			const math::vec3f translation = -ts * movementSensibility * direction * delta.y;
			transform->translate(translation);
			position = transform->transformPoint(math::vec3f{ 0.0f, 0.0f, 0.0f });
		}
		break;
		case NAV_ORBIT:
		{
			const float pitchDelta = -rotationSensibility * delta.y;
			const float yawDelta = -rotationSensibility * delta.x;
			transform->rotateOrbit(pitchDelta, horizontal, yawDelta, math::zAxis);
				updateDirections();
			}
			break;
			case NAV_FOV:
				{
				fovY -= delta.y;
				fovY = fmaxf(1.0f, fminf(fovY, 179.0f));
			}
			break;
		default: ;
		}
	}

	std::vector<std::shared_ptr<Node>> Camera::getChildren() const
	{
		return { transform };
	}

	void Camera::accept(NodeVisitor& visitor)
	{
		visitor.visit(as<Camera>());
	}
}

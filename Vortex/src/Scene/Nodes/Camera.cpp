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
		state.updateOnDevice= true;
	}

	Camera::~Camera()
	{
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
		if (width == 0 || height == 0)
		{
						return;
		}
		if (width == resolution.x && height == resolution.y)
		{
			return;
		}
		resolution.x = width;
		resolution.y = height;
		aspect = static_cast<float>(width) / static_cast<float>(height);
		state.updateOnDevice= true;
	}

	void Camera::onUpdate(const float ts) {
		if (lockCamera)
		{
			return;
		}
		const math::vec2f lastMousePosition = mousePosition;
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
				state.updateOnDevice= true;
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
			state.updateOnDevice= true;
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

	const math::vec2f Camera::project(const math::vec3f& worldPosition, bool flipY) const
	{
		const float       cameraSpaceX        = math::dot(horizontal, worldPosition - position);
		const float       cameraSpaceY        = math::dot(vertical, worldPosition - position);
		const float       cameraSpaceZ        = math::dot(direction, worldPosition - position);
		const math::vec3f cameraSpacePosition = math::vec3f(cameraSpaceX, cameraSpaceY, cameraSpaceZ);
		//const math::vec3f cameraSpacePosition = math::transformVector3F(transform->rcpAffineTransform, worldPosition);
		const float tanFovY = tanf((fovY*M_PI )/ (180.0f*2.0f));
		const float ndcX    = cameraSpaceX / (cameraSpaceZ * tanFovY * aspect);
		const float ndcY    = cameraSpaceY / (cameraSpaceZ * tanFovY);
		// Transform to screen coordinates
		const float screenX = (ndcX + 1.0f) * 0.5f * (float)resolution.x;
		float screenY = (ndcY + 1.0f) * 0.5f * (float)resolution.y;
		if(flipY)
		{
			screenY = (float)resolution.y - screenY;
		}

		return math::vec2f{ screenX, screenY };

	}

	const math::vec3f Camera::projectPixelAtPointDepth(const math::vec2f& pixel, const math::vec3f& worldPoint) const
	{
		const float cameraSpaceZ = math::dot(direction, worldPoint - position);

		const float       tanFovY      = tanf((fovY * M_PI) / (180.0f * 2.0f));
		const math::vec2f ndc          = (pixel / math::vec2f((float)resolution.x,(float)resolution.y)) * 2.0f - 1.0f;
		const float       cameraSpaceX = ndc.x * (cameraSpaceZ * tanFovY * aspect);
		const float       cameraSpaceY = ndc.y * (cameraSpaceZ * tanFovY);

		const math::vec3f worldProjection = position + horizontal * cameraSpaceX + vertical * cameraSpaceY + direction * cameraSpaceZ;

		return worldProjection;

	}

	void Camera::accept(NodeVisitor& visitor)
	{
		visitor.visit(as<Camera>());
	}
}



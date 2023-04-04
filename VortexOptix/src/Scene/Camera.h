#pragma once
#include "SceneGraph.h"
#include "Core/Options.h"
#include "Core/Input/Input.h"
#include "Core/Log.h"

namespace vtx {
	namespace scene {

		enum NavigationType {
			NAV_NONE,
			NAV_PAN,
			NAV_ORBIT,
			NAV_ZOOM,
			NAV_DOLLY
		};

		class Camera : public Node
		{
		public:
			Camera() :
				Node(NT_CAMERA),
				horizontal(math::vec3f{ 1.0f, 0.0f, 0.0f }),
				vertical(math::vec3f{ 0.0f, 1.0f, 0.0f }),
				direction(math::vec3f{ 0.0f, 0.0f, -1.0f }),
				position(math::vec3f{ 0.0f, 0.0f, 0.0f }),
				resolution(math::vec2i{ options.width, options.height }),
				FovY(45.0f),
				aspect(float(options.width)  /float(options.height)),
				zoomSensibility(0.01f),
				movementSensibility(100.0f),
				rotationSensibility(3.0f),
				mousePosition(math::vec2f{ 0.0f, 0.0f }),
				mouseDelta(math::vec2f{ 0.0f, 0.0f })
			{}
			

			void UpdateDirections() {
				horizontal = transform->TransformVector(math::vec3f{ 1.0f, 0.0f, 0.0f });
				vertical = transform->TransformVector(math::vec3f{ 0.0f, 1.0f, 0.0f });
				direction = transform->TransformVector(math::vec3f{ 0.0f, 0.0f, -1.0f });
				position = transform->TransformPoint(math::vec3f{ 0.0f, 0.0f, 0.0f });
			}

			void Resize(uint32_t width, uint32_t height) {
				resolution.x = width;
				resolution.y = height;
				aspect = float(width) / float(height);
			}

			void OnUpdate(float ts) {
				math::vec2f lastMousePosition = mousePosition;
				mousePosition = Input::GetMousePosition();
				mouseDelta = (mousePosition - lastMousePosition);
				//VTX_INFO("Mouse Delta: {0}, {1}", mouseDelta.x, mouseDelta.y);
				OrbitNavigation(ts);
			}

			void OrbitNavigation(float ts) {
				if (Input::MouseWheel() != 0.0f && NavigationActive) {
					FovY += float(Input::MouseWheel()) * zoomSensibility;
					if (FovY < 1.0f)
					{
						FovY = 1.0f;
					}
					else if (179.0 < FovY)
					{
						FovY = 179.0f;
					}
				}

				math::vec2f Delta = math::vec2f(mouseDelta.x / resolution.x, mouseDelta.y / resolution.y);

				if (Input::IsMouseButtonDown(MouseButton::Middle) && NavigationActive && !(Delta.x == 0.0f && Delta.y == 0.0f)) {
					if (Input::IsKeyDown(KeyCode::LeftShift) || Input::IsKeyDown(KeyCode::RightShift)) {
						NavigationMode = NAV_PAN;
					}
					else if (Input::IsKeyDown(KeyCode::LeftControl) || Input::IsKeyDown(KeyCode::RightControl)) {
						NavigationMode = NAV_DOLLY;
					}
					else {
						NavigationMode = NAV_ORBIT;
					}
				}
				else {
					NavigationActive = false;
					NavigationMode = NAV_NONE;
					Input::SetCursorMode(CursorMode::Normal);
					return;
				}

				Input::SetCursorMode(CursorMode::Locked);

				switch (NavigationMode) {
					case NAV_PAN: {
						math::vec3f translation = ts * movementSensibility * (-horizontal * Delta.x + vertical * Delta.y);
						transform->translate(translation);
						position = transform->TransformPoint(math::vec3f{ 0.0f, 0.0f, 0.0f });
					}
								break;
					case NAV_DOLLY: {
						math::vec3f translation = -ts * movementSensibility * direction * Delta.y;
						transform->translate(translation);
						position = transform->TransformPoint(math::vec3f{ 0.0f, 0.0f, 0.0f });
					}
								  break;
					case NAV_ORBIT: {
						float pitchDelta = -rotationSensibility * Delta.y;
						float YawDelta = -rotationSensibility * Delta.x;

						transform->rotateOrbit(pitchDelta, horizontal, YawDelta, math::zAxis);

						//math::Quaternion3f pitchRotationQuat = math::Quaternion3f(0.0f, pitchDelta, YawDelta);
						//transform->rotateQuaternion(pitchRotationQuat);
						UpdateDirections();
					}
					break;
				}
			}

		public:
			std::shared_ptr<Transform> transform;
			math::vec3f position;
			math::vec3f direction;
			math::vec3f vertical;
			math::vec3f horizontal;
			float FovY;
			float aspect;
			math::vec2i resolution;
			math::vec2f mousePosition;
			math::vec2f mouseDelta;
			bool	NavigationActive = false;
			NavigationType NavigationMode = NAV_ORBIT;
			float movementSensibility;
			float rotationSensibility;
			float zoomSensibility;
		};
	}

}
#include "Camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/constants.hpp>
#include "Walnut/Input/Input.h"
using namespace Walnut;

Camera::Camera(float verticalFOV, float nearClip, float farClip)
	: m_VerticalFov(verticalFOV), m_NearClip(nearClip), m_FarClip(farClip)
{
	m_Position = glm::vec3(0, 6, 0);
	RecalculateView();
	RecalculateRayDirections();
}

bool Camera::OnUpdate(float ts) {
	if (m_NavStyle == "FPS") {
		return FPSNavigation(ts);
	}
	else if (m_NavStyle == "Orbit Z" || m_NavStyle == "Free Orbit") {
		return OrbitNavigation(ts);
	}
	else {
		return false;
	}
}
bool Camera::FPSNavigation(float ts)
{
	glm::vec2 mousePos = Input::GetMousePosition();
	glm::vec2 delta = (mousePos - m_LastMousePosition) * 0.002f;
	m_LastMousePosition = mousePos;

	if (!Input::IsMouseButtonDown(MouseButton::Right))
	{
		Input::SetCursorMode(CursorMode::Normal);
		return false;
	}

	Input::SetCursorMode(CursorMode::Locked);

	bool moved = false;
	constexpr glm::vec3 upDirection(0.f, 0.0f, 1.0f);
	glm::vec3 rightDirection = glm::cross(m_ForwardDirection, upDirection);

	float speed = 5.0f;

	// Movement
	if (Input::IsKeyDown(KeyCode::W))
	{
		m_Position += m_ForwardDirection * speed * ts;
		moved = true;
	}
	else if (Input::IsKeyDown(KeyCode::S))
	{
		m_Position -= m_ForwardDirection * speed * ts;
		moved = true;
	}
	if (Input::IsKeyDown(KeyCode::A))
	{
		m_Position -= rightDirection * speed * ts;
		moved = true;
	}
	else if (Input::IsKeyDown(KeyCode::D))
	{
		m_Position += rightDirection * speed * ts;
		moved = true;
	}
	if (Input::IsKeyDown(KeyCode::Q))
	{
		m_Position -= upDirection * speed * ts;
		moved = true;
	}
	else if (Input::IsKeyDown(KeyCode::E))
	{
		m_Position += upDirection * speed * ts;
		moved = true;
	}

	// Rotation
	if (delta.x != 0.0f || delta.y != 0.0f)
	{
		float pitchDelta = delta.y * GetRotationSpeed();
		float yawDelta = delta.x * GetRotationSpeed();

		glm::quat q = glm::normalize(glm::cross(glm::angleAxis(-pitchDelta, rightDirection), glm::angleAxis(-yawDelta, glm::vec3(0.f, 0.0f, 1.0f))));
		m_ForwardDirection = glm::rotate(q, m_ForwardDirection);

		moved = true;
	}

	if (moved)
	{
		RecalculateView();
		RecalculateRayDirections();
	}

	return moved;
}
bool Camera::OrbitNavigation(float ts)
{
	glm::vec2 mousePos = Input::GetMousePosition();
	glm::vec2 delta = (mousePos - m_LastMousePosition) * 0.002f;
	m_LastMousePosition = mousePos;

	if (!Input::IsMouseButtonDown(MouseButton::Middle))
	{
		Input::SetCursorMode(CursorMode::Normal);
		return false;
	}

	Input::SetCursorMode(CursorMode::Locked);

	bool moved = false;
	
	if (Input::IsKeyDown(KeyCode::LeftShift) || Input::IsKeyDown(KeyCode::RightShift)) {
		if (Input::IsMouseButtonDown(MouseButton::Middle)) {
			// Pan
			if (delta.x != 0.0f || delta.y != 0.0f) {
				float HorizzontalPanDelta = -delta.x;
				float VerticalPanDelta = delta.y;

				m_Position += (m_RightDirection * HorizzontalPanDelta + m_UpDirection * VerticalPanDelta) * GetSpeed() * ts;
				moved = true;
			}
		}
	}
	else if (Input::IsKeyDown(KeyCode::LeftControl) || Input::IsKeyDown(KeyCode::RightControl)) {
		if (Input::IsMouseButtonDown(MouseButton::Middle)) {
			// Zoom
			if (delta.y != 0.0f) {
				float ZoomDelta = -delta.y;

				m_Position += m_ForwardDirection * ZoomDelta * GetSpeed() * ts;
				moved = true;
			}
		}
	}
	else {
		if (Input::IsMouseButtonDown(MouseButton::Middle)) {
			// Orbit
			if (delta.x != 0.0f || delta.y != 0.0f) {
				
				glm::vec3 UpAxis = glm::vec3(0.f, 0.0f, 1.0f);
				if (m_NavStyle == "Free Orbit") {
					UpAxis = m_UpDirection;
				}
				
				float pitchDelta =  delta.y * GetRotationSpeed();
				glm::quat pitchRotation = glm::angleAxis(-pitchDelta, m_RightDirection);
				float yawDelta = delta.x * GetRotationSpeed();
				glm::quat yawRotation = glm::angleAxis(-yawDelta, UpAxis);
				
				auto CROSS = glm::cross(pitchRotation, yawRotation);
				glm::quat q = glm::normalize(CROSS);
				
				m_Position = glm::rotate(q, m_Position);
				m_ForwardDirection = glm::rotate(q, m_ForwardDirection);
				m_UpDirection = glm::rotate(q, m_UpDirection);
				m_RightDirection = glm::cross(m_ForwardDirection, m_UpDirection);

				moved = true;
			}
		}
	}

	if (moved)
	{
		RecalculateView();
		RecalculateRayDirections();
	}

	return moved;
}
;

void Camera::OnProjectionChange(uint32_t width, uint32_t height) {
	if (width == m_ViewportWidth && height == m_ViewportHeight && m_VerticalFov == m_OldVerticalFOV) {
		return;
	}

	m_ViewportWidth = width;
	m_ViewportHeight = height;
	m_OldVerticalFOV = m_VerticalFov;


	RecalculateProjection();
	RecalculateRayDirections();
};

float Camera::GetRotationSpeed()
{
	return m_RotationSpeed;
}

float Camera::GetSpeed()
{
	return m_Speed;
}

void Camera::RecalculateProjection()
{
	m_Projection = glm::perspectiveFov(glm::radians(m_VerticalFov), (float)m_ViewportWidth, (float)m_ViewportHeight, m_NearClip, m_FarClip);
	m_InverseProjection = glm::inverse(m_Projection);
}

void Camera::RecalculateView()
{
	m_View = glm::lookAt(m_Position, m_Position + m_ForwardDirection, m_UpDirection);
	m_InverseView = glm::inverse(m_View);
	ComputeEulerAngles();
}

void Camera::ComputeEulerAngles() {
	// not sure if correct plus need a way to update the camera rotation in the editor
	m_EulerAngles = ((180.0f / glm::pi<float>()))*glm::eulerAngles(glm::quat_cast(m_View));
}

void Camera::RecalculateRayDirections()
{
	m_RayDirections.resize(m_ViewportWidth * m_ViewportHeight);

	for (uint32_t y = 0; y < m_ViewportHeight; y++)
	{
		for (uint32_t x = 0; x < m_ViewportWidth; x++)
		{
			glm::vec2 coord = { (float)x / (float)m_ViewportWidth, (float)y / (float)m_ViewportHeight };
			coord = coord * 2.0f - 1.0f;

			glm::vec4 target = m_InverseProjection * glm::vec4(coord.x, coord.y, 1, 1); // Camera Space
			glm::vec3 rayDirection = glm::vec3(m_InverseView * glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0)); // World space
			m_RayDirections[x + y * m_ViewportWidth] = rayDirection;
		}
	}
}
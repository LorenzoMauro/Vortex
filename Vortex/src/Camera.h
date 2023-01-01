#pragma once
#include <glm/glm.hpp>
#include <vector>

class Camera
{
public:

	Camera(float verticalFOV, float nearClip, float farClip);

	bool OnUpdate(float ts);

	bool FPSNavigation(float ts);
	bool OrbitNavigation(float ts);
	
	void OnProjectionChange(uint32_t width, uint32_t height);

	const glm::mat4& GetProjection() const { return m_Projection; }
	const glm::mat4& GetInverseProjection() const { return m_InverseProjection; }
	const glm::mat4& GetView() const { return m_View; }
	const glm::mat4& GetInverseView() const { return m_InverseView; }

	const glm::vec3& GetPosition() const { return m_Position; }
	const glm::vec3& GetDirection() const { return m_ForwardDirection; }

	const std::vector<glm::vec3>& GetRayDirections() const { return m_RayDirections; }

	float GetRotationSpeed();
	float GetSpeed();

	void ComputeEulerAngles();

	void RecalculateProjection();

	void RecalculateView();

	void RecalculateRayDirections();
	


public:
	float m_VerticalFov = 55.0f;
	const char* m_NavStyleNames[3] = { "Orbit Z", "Free Orbit", "FPS" };
	char* m_NavStyle = (char*)m_NavStyleNames[0];
	int m_NavStyleIndex = 0;
	glm::vec3 m_EulerAngles = { 0, 0, 0 };
	float m_RotationSpeed = 2.f;
	float m_Speed = 30.0f;
private:
	float m_OldVerticalFOV = 55.0f;
	float m_NearClip = 0.1f;
	float m_FarClip = 100.0f;

	glm::vec3 m_Position{ 0.0f, 0.0f, 0.0f };
	
	glm::vec3 m_ForwardDirection{ 0.0f, -1.0f, 0.0f };
	glm::vec3 m_RightDirection{ 1.0f, 0.0f, 0.0f };
	glm::vec3 m_UpDirection{ 0.0f, 0.0f, 1.0f };
	
	float m_RollAngle = 0.0f;

	glm::mat4 m_Projection{ 1.0f };
	glm::mat4 m_View{ 1.0f };
	glm::mat4 m_InverseProjection{ 1.0f };
	glm::mat4 m_InverseView{ 1.0f };

	std::vector<glm::vec3> m_RayDirections;
	glm::vec2 m_LastMousePosition{ 0.0f, 0.0f };
	uint32_t m_ViewportWidth = 0, m_ViewportHeight = 0;
};



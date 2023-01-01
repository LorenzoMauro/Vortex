#pragma once
#include <glm/gtc/type_ptr.hpp>
#include "Scene.h"
#include "Renderer.h"
#include "Walnut/Application.h"

class EngineUI : public Walnut::Layer
{
public:
	virtual void OnUpdate(float ts) override;
	virtual void OnUIRender() override;
	virtual void OnAttach() override;

	void SettingsWindow();
	
	void RenderStatusWindow();
	
	void ViewportWindow();

	void MaterialProperties(int i);

	void MaterialSettings(int i);
	
	void MaterialUI();
	
	void SphereSettings(int i);

	void SphereUI();
	
	void LightingUI();

	void RenderUI();

	void CameraUI();

	void RestartRenderMonitor();
	template <typename T = bool>
	void RestartRenderMonitor(T UpdateBoolean);
	
private:
	bool m_IsRendering = false;
	bool m_RestartRender = false;
	
	Scene m_Scene;
	Camera& m_Camera = m_Scene.Camera;
	std::vector<Sphere>& m_Spheres = m_Scene.Spheres;
	std::vector<Material>& m_Materials = m_Scene.Materials;
	std::vector<std::string>& m_MaterialNames = m_Scene.MaterialNames;
	float m_Fov = m_Camera.m_VerticalFov;
	
	Renderer m_Renderer = Renderer(m_Scene);
	Renderer::RenderSettings& m_RendererSettings = m_Renderer.GetSettings();
	
private:
	uint32_t m_ViewportWidth = 0, m_ViewportHeight = 0;
	float m_Fps = 0.0f;
	float m_MillisPerFrame = 0.0f;
	const ImVec4 button_color_pressed = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
	const ImVec4 button_color_normal = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

	int m_MaterialNameEditingID = -1;
	std::tuple<int, bool> m_MaterialNameEditing = { -1, false };
	
};


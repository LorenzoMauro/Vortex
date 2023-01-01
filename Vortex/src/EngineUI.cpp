#include "EngineUI.h"
#include "Walnut/Image.h"
#include "Walnut/Timer.h"
#include "Renderer.h"
#include "Camera.h"
#include "Scene.h"
#include "IconsFontAwesome6.h"
#include "Utils.h"
#include <filesystem>


void EngineUI::OnUpdate(float ts)
{
	Walnut::Timer timer;
	RestartRenderMonitor(m_Camera.OnUpdate(ts));

	if (m_IsRendering) {
		if (m_RestartRender == true) {
			m_Renderer.RestartRender();
			m_RestartRender = false;
		}
		m_Renderer.OnResize(m_ViewportWidth, m_ViewportHeight);
		m_Camera.OnProjectionChange(m_ViewportWidth, m_ViewportHeight);
		m_Renderer.Render();
	}
	m_MillisPerFrame = timer.ElapsedMillis();
	m_Fps = 1000.0f / m_MillisPerFrame;


}

void EngineUI::OnUIRender()
{
	
	
	//ImGui::PushFont(roboto_mono);
	RenderStatusWindow();
	SettingsWindow();
	ViewportWindow();
	ImGui::ShowDemoWindow();

}

void EngineUI::OnAttach()
{
	std::filesystem::path p = "fonts\\fa-regular-400.ttf";
	std::filesystem::path absolute_path = std::filesystem::absolute(p);
	Utils::AddIcons(absolute_path.string().data());
}

inline void EngineUI::SettingsWindow() {
	ImGui::Begin("Settings");
	SphereUI();
	MaterialUI();
	LightingUI();
	CameraUI();
	RenderUI();
	ImGui::End();
}

inline void EngineUI::RenderStatusWindow() {
	ImGui::Begin("Render Status");
	ImGui::Text("FPS: %.2f Fps", m_Fps);
	ImGui::Text("Time Per Frame: %.2f ms", m_MillisPerFrame);
	ImGui::Text("Samples PerPixel: %i spp", m_Renderer.GetSamplesPerPixel());

	ImGui::Text("Render");
	ImGui::SameLine(); 
	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
	
	bool playbutton = ImGui::Button(ICON_FA_CIRCLE_PLAY);
	ImGui::SameLine();
	bool pausebutton = ImGui::Button(ICON_FA_CIRCLE_PAUSE);
	ImGui::SameLine();
	bool stopbutton = ImGui::Button(ICON_FA_CIRCLE_STOP);
	ImGui::PopStyleColor();
	//ImGui::SameLine();
	//bool reloadbutton = ImGui::Button(ICON_FA_CIRCLE_EXCLAMATION);

	if (playbutton) { m_IsRendering = true; }
	if (pausebutton){ m_IsRendering = false; }
	if (stopbutton)
	{
		m_Renderer.ClearImageData();
		m_IsRendering = false;
	}
	
	//bool render_button_pressed = ImGui::Button("Render");
	//if (render_button_pressed)
	//{
	//	// change Value of Boolean variable m_isRender
	//	m_IsRendering = !m_IsRendering;
	//}

	ImGui::End();
}

inline void EngineUI::ViewportWindow() {
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
	ImGui::Begin("Viewport");

	m_ViewportWidth = ImGui::GetContentRegionAvail().x;
	m_ViewportHeight = ImGui::GetContentRegionAvail().y;

	auto image = m_Renderer.GetFinalImage();
	if (image)
	{
		ImGui::Image(image->GetDescriptorSet(), { (float)image->GetWidth(), (float)image->GetHeight() }, ImVec2(0, 1), ImVec2(1, 0));
	}

	ImGui::End();
	ImGui::PopStyleVar();
}

void EngineUI::MaterialProperties(int i) {
	bool isEdited;
	ImGui::PushID(i);
	Material& mat = m_Materials[i];
	RestartRenderMonitor(ImGui::ColorEdit3("Albedo", glm::value_ptr(mat.Albedo), 0.1f));
	RestartRenderMonitor(ImGui::DragFloat("Roughness", &mat.Roughness, 0.01f, .0f, 1.f));
	RestartRenderMonitor(ImGui::DragFloat("Metallic", &mat.Metallic, 0.01f, .0f, 1.f));

	if (ImGui::Button("Delete Material"))
	{
		m_Scene.DeleteMaterial(i);
		RestartRenderMonitor();

	}

	ImGui::PopID();
}

void EngineUI::MaterialSettings(int i) {

	ImGui::PushID(i);
	Material& mat = m_Materials[i];
	char tempName[256];
	strncpy(tempName, m_MaterialNames[i].c_str(), sizeof(tempName));

	bool isEditingName = false;
	if (i == m_MaterialNameEditingID) {
		isEditingName = true;
	}

	if (!isEditingName) {
		ImGui::LabelText("Name", tempName);
		if (ImGui::IsItemHovered()) {
			// The mouse is hovering over the button
			if (ImGui::IsMouseDoubleClicked(0)) {
				// Create an input field for the text
				m_MaterialNameEditingID = i;
			}
		}
	}
	else {
		ImGui::InputText("Name Input Text", tempName, sizeof(tempName));
		if (ImGui::IsItemDeactivatedAfterEdit()) {
			std::string newName = Utils::FindAvailableName(tempName, m_MaterialNames);
			m_Materials[i].Name = newName;
			m_MaterialNames[i] = newName;
			m_MaterialNameEditingID = -1;
		}
	}

	MaterialProperties(i);

	ImGui::PopID();
	ImGui::Separator();
}

void EngineUI::MaterialUI() {

	if (ImGui::CollapsingHeader("Materials Settings")) {
		if (ImGui::Button("Create Material"))
		{
			m_Scene.AddMaterial();
			RestartRenderMonitor();
		}
		ImGui::Separator();
		for (size_t i = 0; i < m_Materials.size(); i++)
		{
			MaterialSettings(i);
		}
	}
}

void EngineUI::SphereSettings(int i) {

	ImGui::PushID(i);
	if (ImGui::TreeNode("Sphere"))
	{
		Sphere& Sphere = m_Spheres[i];
		RestartRenderMonitor(ImGui::DragFloat3("Position", glm::value_ptr(Sphere.Origin), 0.1f));
		RestartRenderMonitor(ImGui::DragFloat("Radius", &Sphere.Radius, 0.1f, 0.f));

		if (ImGui::TreeNode("Material")) {
			if (ImGui::BeginCombo("", m_MaterialNames[Sphere.MaterialIndex].c_str())) {
				for (int i = 0; i < m_MaterialNames.size(); i++) {
					bool isSelected = (Sphere.MaterialIndex == i);
					if (ImGui::Selectable(m_MaterialNames[i].c_str(), isSelected)) {
						Sphere.MaterialIndex = i;
						RestartRenderMonitor();
					}
					if (isSelected) {
						ImGui::SetItemDefaultFocus();
					}
				}
				ImGui::EndCombo();
			}
			//ImGui::BeginChild("Material", ImVec2(0, 0), true, ImGuiWindowFlags_AlwaysAutoResize);
			MaterialProperties(i);
			//ImGui::EndChild();
			ImGui::TreePop();

		}
		if (ImGui::Button("Delete Sphere")) {
			m_Scene.DeleteSphere(i);
			RestartRenderMonitor();
		}
		ImGui::Separator();
		ImGui::TreePop();
	}
	ImGui::PopID();
}

void EngineUI::SphereUI() {

	if (ImGui::CollapsingHeader("Spheres Settings")) {
		if (ImGui::Button("Create Sphere"))
		{
			m_Scene.AddSphere();
			RestartRenderMonitor();

		}
		ImGui::Separator();
		for (size_t i = 0; i <m_Spheres.size(); i++)
		{
			SphereSettings(i);
		}
	}
}

inline void EngineUI::LightingUI() {
	if (ImGui::CollapsingHeader("Lighting Settings")) {
		RestartRenderMonitor(ImGui::ColorEdit3("Sun Color", glm::value_ptr(m_Scene.SunLight.Color), 0.1f));
		RestartRenderMonitor(ImGui::DragFloat3("Sun Direction", glm::value_ptr(m_Scene.SunLight.Direction), 0.1f));
		RestartRenderMonitor(ImGui::ColorEdit3("Ambient Color", glm::value_ptr(m_Scene.World.AmbientColor), 0.1f));
	}
}

inline void EngineUI::RenderUI() {
	if (ImGui::CollapsingHeader("Render Settings")) {
		RestartRenderMonitor(ImGui::DragInt("Bounces", &m_RendererSettings.MaxBounces, 0.1f));
		ImGui::Checkbox("Accumulate", &m_RendererSettings.Accumulate);
		RestartRenderMonitor(ImGui::Checkbox("MultiThread", &m_RendererSettings.MultiThread));
		//RestartRenderMonitor(ImGui::DragInt("Number of Threads", &m_RendererSettings.MaxThreads, 1.f));
	}
}

inline void EngineUI::CameraUI() {
	if (ImGui::CollapsingHeader("Camera Settings")) {
		RestartRenderMonitor(ImGui::DragFloat("FOV", &m_Camera.m_VerticalFov, 0.1f));
		int item_current = 0;
		if (ImGui::Combo("Navigation", &m_Camera.m_NavStyleIndex, m_Camera.m_NavStyleNames, IM_ARRAYSIZE(m_Camera.m_NavStyleNames))) {
			m_Camera.m_NavStyle = (char*)m_Camera.m_NavStyleNames[m_Camera.m_NavStyleIndex];
		}
		ImGui::InputFloat3("Euler Angles", glm::value_ptr(m_Camera.m_EulerAngles), "%.2f");
		ImGui::DragFloat("Move Sensibility", &m_Camera.m_Speed, 0.1f);
		ImGui::DragFloat("Rotation Sensibility", &m_Camera.m_RotationSpeed, 0.1f);

	}
}

void EngineUI::RestartRenderMonitor()
{
	m_RestartRender = true;

}

template<typename T>
void EngineUI::RestartRenderMonitor(T UpdateBoolean) {
	if (UpdateBoolean) {
		m_RestartRender = true;
	}

}

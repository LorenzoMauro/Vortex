#include "ViewportLayer.h"
#include "imgui.h"

namespace vtx {
	ViewportLayer::ViewportLayer(Renderer* _Renderer)
	{
        renderer = _Renderer;

	}

    void ViewportLayer::OnAttach()
    {
    }

    void ViewportLayer::OnDetach()
    {
    }

    void ViewportLayer::OnUpdate(float ts)
    {
    }

    void ViewportLayer::OnUIRender() {
        ImGui::Begin("Viewport");
        uint32_t m_Width = ImGui::GetContentRegionAvail().x;
        uint32_t m_Height = ImGui::GetContentRegionAvail().y;
        renderer->Resize(m_Width, m_Height);
        GLuint frameAttachment = renderer->GetFrame();
        ImGui::Image((void*)frameAttachment, ImVec2{ (float)m_Width, (float)m_Height }, ImVec2{ 0, 1 }, ImVec2{ 1, 0 });
        ImGui::End();
    }
}

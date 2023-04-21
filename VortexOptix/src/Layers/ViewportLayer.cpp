#include "ViewportLayer.h"
#include "imgui.h"
namespace vtx {
	ViewportLayer::ViewportLayer(std::shared_ptr<graph::Renderer> _Renderer)
	{
        renderer = _Renderer;
        deviceVisitor = std::make_shared<device::DeviceVisitor>();
        hostVisitor = std::make_shared<HostVisitor>();
    }

    void ViewportLayer::OnAttach()
    {
    }

    void ViewportLayer::OnDetach()
    {
    }

    void ViewportLayer::OnUpdate(float ts)
    {
        renderer->camera->onUpdate(ts);
        renderer->traverse({std::dynamic_pointer_cast<NodeVisitor>(hostVisitor)});
        renderer->traverse({std::dynamic_pointer_cast<NodeVisitor>(deviceVisitor)});
        device::incrementFrame();
		device::finalizeUpload();


        renderer->render();
    }

    void ViewportLayer::OnUIRender() {
        ImGui::Begin("Viewport");
        const uint32_t width = ImGui::GetContentRegionAvail().x;
        const uint32_t height = ImGui::GetContentRegionAvail().y;
        renderer->resize(width, height);
        const GLuint frameAttachment = renderer->getFrame();
        ImGui::Image(reinterpret_cast<void*>(frameAttachment), ImVec2{ static_cast<float>(width), static_cast<float>(height) }, ImVec2{ 0, 1 }, ImVec2{ 1, 0 });
        if (!renderer->camera->navigationActive) {
            renderer->camera->navigationActive = ImGui::IsItemHovered();
        }
        ImGui::End();
    }
}

#include "ViewportLayer.h"
#include "imgui.h"
#include "GuiElements/RendererNodeGui.h"
#include "GuiElements/MaterialNodeGui.h"

namespace vtx {
	ViewportLayer::ViewportLayer(std::shared_ptr<graph::Renderer> _Renderer)
	{
        renderer = _Renderer;
        deviceVisitor = std::make_shared<device::DeviceVisitor>();
        hostVisitor = std::make_shared<HostVisitor>();
        materialGui = gui::MaterialGui();
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
        if (renderer->camera->updated)
        {
            renderer->settings.iteration = -1;
            renderer->settings.isUpdated = true;
        }
        if (renderer->isReady() && renderer->settings.iteration<= renderer->settings.maxSamples)
        {
            renderer->settings.iteration++;
            renderer->settings.isUpdated = true;
            renderer->traverse({ std::dynamic_pointer_cast<NodeVisitor>(hostVisitor) });
            renderer->traverse({ std::dynamic_pointer_cast<NodeVisitor>(deviceVisitor) });
            device::incrementFrame();
            device::finalizeUpload();
            renderer->threadedRender();
            //renderer->render();
        }
        
    }

    void ViewportLayer::OnUIRender() {

		gui::rendererNodeGui(renderer);
        materialGui.materialGui();
        ImGui::Begin("Viewport");
        const uint32_t width = ImGui::GetContentRegionAvail().x;
        const uint32_t height = ImGui::GetContentRegionAvail().y;
        renderer->resize(width, height);
        GlFrameBuffer& bf = renderer->getFrame();
        ImGui::Image(reinterpret_cast<void*>(bf.colorAttachment), ImVec2{ static_cast<float>(bf.width), static_cast<float>(bf.height) }, ImVec2{ 0, 1 }, ImVec2{ 1, 0 });
        if (!renderer->camera->navigationActive) {
            renderer->camera->navigationActive = ImGui::IsItemHovered();
        }
        ImGui::End();
    }
}

#include "ViewportLayer.h"
#include "imgui.h"
#include "Core/Timer.h"

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
        if(renderer->camera->updated)
        {
            renderer->settings.iteration = 0;
            renderer->settings.isUpdated = true;
        }
        else
        {
	        renderer->settings.iteration++;
            renderer->settings.isUpdated = true;
        }
        renderer->traverse({std::dynamic_pointer_cast<NodeVisitor>(hostVisitor)});
        renderer->traverse({std::dynamic_pointer_cast<NodeVisitor>(deviceVisitor)});
        device::incrementFrame();
		device::finalizeUpload();

        renderer->render();
    }

    void ViewportLayer::OnUIRender() {
        ImGui::Begin("Renderer Settings");

        const char* items[] = { "Noisy", "Diffuse", "Orientation", "True Normal", "Shading Normal", "Debug1", "Debug2" , "Debug3" };
        static int currentItem = 0; // You can set this to the index of the initial selected item

        if (ImGui::Combo("Frame Buffer Type", &currentItem, items, IM_ARRAYSIZE(items)))
        {
	        renderer->frameBufferType = static_cast<FrameBufferData::FrameBufferType>(currentItem);
            renderer->isUpdated= true;
        }

        ImGui::Separator();
        if(ImGui::SliderInt("Max Samples", &(renderer->settings.maxSamples), 0, 10000, "%d")){
            renderer->settings.iteration = -1;
            renderer->settings.isUpdated = true;
        };
        if (ImGui::SliderInt("Max Bounces", &(renderer->settings.maxBounces), 0, 35, "%d")) {
            renderer->settings.iteration = -1;
            renderer->settings.isUpdated = true;
        };
        if (ImGui::Checkbox("Accumulate", &(renderer->settings.accumulate))) {
            renderer->settings.iteration = -1;
            renderer->settings.isUpdated = true;
        };

        ///////// INFO /////////////
        ////////////////////////////
        ImGui::Separator();
        ImGui::Text("Total Time:            %.3f s",    renderer->totalTimeSeconds);
        ImGui::Text("Samples Per Pixels:    %d",        renderer->settings.iteration);
        ImGui::Text("SPP per Seconds:       %.3f",      renderer->sppS);
        ImGui::Text("Frame Time:            %.3f ms",   renderer->averageFrameTime);
        ImGui::Text("Fps:                   %.3f",      renderer->fps);
        ImGui::Separator();

		ImGui::End();
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

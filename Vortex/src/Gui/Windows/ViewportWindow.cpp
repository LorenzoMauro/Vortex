#include "ViewportWindow.h"
#include "imgui.h"
#include "Scene/Scene.h"

namespace vtx {
    ViewportWindow::ViewportWindow()
    {
        renderer      = graph::Scene::getScene()->renderer;
        name          = "Viewport";
        useToolbar = false;
        isBorderLess = true;
    }

    void ViewportWindow::OnUpdate(const float ts)
    {
        renderer->camera->onUpdate(ts);
        if (renderer->camera->isUpdated)
        {
            renderer->settings.iteration = -1;
            renderer->settings.isUpdated = true;
        }
        if(renderer->settings.runOnSeparateThread)
        {
            if (renderer->isReady() && renderer->settings.iteration <= renderer->settings.maxSamples)
            {
                renderer->settings.iteration++;
                renderer->settings.isUpdated = true;

                //This step speed up the material computation, but is not really coherent with the rest of the code
                graph::computeMaterialsMultiThreadCode();
                renderer->traverse(hostVisitor);
                renderer->traverse(deviceVisitor);
                device::incrementFrame();
                device::finalizeUpload();
                renderer->threadedRender();
                //renderer->render();
            }
        }
        else
        {
	        if (renderer->settings.iteration <= renderer->settings.maxSamples)
	        {
	        	renderer->settings.iteration++;
				renderer->settings.isUpdated = true;
                //This step speed up the material computation, but is not really coherent with the rest of the code
				graph::computeMaterialsMultiThreadCode();
                renderer->traverse(hostVisitor);
                renderer->traverse(deviceVisitor);
				device::incrementFrame();
				device::finalizeUpload();
				renderer->render();
			}
        }

    }

    void ViewportWindow::mainContent()
    {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        const int width = (int)ImGui::GetContentRegionAvail().x;
        const int height = (int)ImGui::GetContentRegionAvail().y;
        renderer->resize(width, height);
        const GlFrameBuffer& bf = renderer->getFrame();
        ImGui::Image((ImTextureID)bf.colorAttachment, ImVec2{ static_cast<float>(bf.width), static_cast<float>(bf.height) }, ImVec2{ 0, 1 }, ImVec2{ 1, 0 });
        if (!renderer->camera->navigationActive) {
            renderer->camera->navigationActive = ImGui::IsItemHovered();
        }
        if (ImGui::IsItemHovered() && ImGui::IsItemClicked()){
            if (ImGui::GetIO().KeyCtrl)
            {
                selectedId = 0;
            }
            else
            {
                const ImVec2 mousePos = ImGui::GetMousePos();
                const ImVec2 windowPos = ImGui::GetWindowPos();

                const int mouseXRelativeToWindow = static_cast<int>(mousePos.x - windowPos.x);
                const int mouseYRelativeToWindow = static_cast<int>(bf.height) - static_cast<int>(mousePos.y - windowPos.y);
                const int pixelIndex = mouseYRelativeToWindow * width + mouseXRelativeToWindow;

                selectedId = renderer->getInstanceIdOnClick(pixelIndex);
            }
			
            windowManager->selectedNodes["ViewportWindow"] = { selectedId };
            renderer->selectedId = selectedId;
        }

        ImGui::PopStyleVar();
    }

    void ViewportWindow::preRender()
    {
        if (renderer->isSizeLocked)
        {
            const float titleBarHeight = ImGui::GetFontSize() + ImGui::GetStyle().FramePadding.y * 2;
            const int height = renderer->height;
            const int width = renderer->width;
            const float adjustedHeight = width + titleBarHeight;// +menu_bar_height;

            ImGui::SetNextWindowSize(ImVec2(height, adjustedHeight), ImGuiCond_Always);
            windowFlags |= ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoResize;
        }
    }
}

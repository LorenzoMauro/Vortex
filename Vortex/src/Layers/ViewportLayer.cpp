#include "ViewportLayer.h"
#include "imgui.h"
#include "GuiElements/RendererNodeGui.h"
#include "GuiElements/MaterialNodeGui.h"
#include "GuiElements/SceneGraph.h"
#include "Scene/Scene.h"
#include "Scene/Nodes/Material.h"

namespace vtx {
    ViewportLayer::ViewportLayer()
    {
        renderer      = graph::Scene::getScene()->renderer;
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
                //graph::computeMaterialsMultiThreadCode();
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
				//graph::computeMaterialsMultiThreadCode();
                renderer->traverse(hostVisitor);
                renderer->traverse(deviceVisitor);
				device::incrementFrame();
				device::finalizeUpload();
				renderer->render();
			}
        }
    }

    void ViewportLayer::OnUIRender() {
		gui::SceneGraphGui::draw();

        ImGui::Begin("Renderer Settings");
		gui::rendererNodeGui(renderer);
        ImGui::End();

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));


        //materialGui.materialGui();
        if(renderer->isSizeLocked)
        {
            const float titleBarHeight = ImGui::GetFontSize() + ImGui::GetStyle().FramePadding.y * 2;
            const int height = renderer->height;
            const int width = renderer->width;
            const float adjustedHeight = width + titleBarHeight;// +menu_bar_height;

	        ImGui::SetNextWindowSize(ImVec2(height, adjustedHeight), ImGuiCond_Always);
            ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoResize);
        }
        else
        {
            ImGui::Begin("Viewport");
        }

        const int width = (int)ImGui::GetContentRegionAvail().x;
        const int height = (int)ImGui::GetContentRegionAvail().y;
        renderer->resize(width, height);
		const GlFrameBuffer& bf = renderer->getFrame();
        ImGui::Image((ImTextureID)bf.colorAttachment, ImVec2{ static_cast<float>(bf.width), static_cast<float>(bf.height) }, ImVec2{ 0, 1 }, ImVec2{ 1, 0 });
        if (!renderer->camera->navigationActive) {
            renderer->camera->navigationActive = ImGui::IsItemHovered();
        }
        ImGui::End();
        ImGui::PopStyleVar();

    }
}

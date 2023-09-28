#include "ViewportWindow.h"
#include "imgui.h"
#include "Core/CustomImGui/CustomImGui.h"
#include "Device/UploadCode/DeviceDataCoordinator.h"
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
        if(renderer->settings.runOnSeparateThread)
        {
            if (renderer->isReady() && renderer->settings.iteration <= renderer->settings.maxSamples)
            {
                renderer->settings.iteration++;
                renderer->settings.isUpdated = true;

                //This step speed up the material computation, but is not really coherent with the rest of the code
                graph::computeMaterialsMultiThreadCode();
                renderer->traverse(hostVisitor);
                onDeviceData->sync();
                onDeviceData->incrementFrameIteration();
                renderer->threadedRender();
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
                onDeviceData->sync();
                onDeviceData->incrementFrameIteration();
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

        bool isTransforming = false;
        if (ImGui::IsItemHovered())
        {
            renderer->camera->navigationActive = true;

            if (renderer->camera->navigationMode == graph::NAV_NONE)
            {
                transformUI.monitorTransformUI(renderer->camera);
                isTransforming = transformUI.isTransforming();

                // activate selection interaction only if not transforming
                if (!isTransforming && ImGui::IsItemClicked()) {
                    if (ImGui::GetIO().KeyCtrl)
                    {
                        //We remove all selected instances
                        graph::Scene::getScene()->removeInstancesFromSelection();
                    }
                    else
                    {
                        const ImVec2 mousePos = ImGui::GetMousePos();
                        const ImVec2 windowPos = ImGui::GetWindowPos();

                        const int mouseXRelativeToWindow = static_cast<int>(mousePos.x - windowPos.x);
                        const int mouseYRelativeToWindow = static_cast<int>(bf.height) - static_cast<int>(mousePos.y - windowPos.y);
                        const int pixelIndex = mouseYRelativeToWindow * width + mouseXRelativeToWindow;

                        vtxID selected = renderer->getInstanceIdOnClick(pixelIndex);

                        if (ImGui::GetIO().KeyShift)
                        {
                            graph::Scene::getScene()->addNodesToSelection({ selected });
                        }
                        else
                        {
                            graph::Scene::getScene()->setSelected({ selected });

                        }
                    }
                }
            }
        }

        // Draw pivots and transform line
        const std::set<std::shared_ptr<graph::Instance>>& instances = graph::Scene::getScene()->getSelectedInstances();
        if (!instances.empty())
        {
            math::vec3f meanPivot = 0.0f;
            math::vec2f projectedMeanPivot;
            for (const auto& instance: instances)
            {
                const math::vec3f pivot = instance->transform->globalTransform.p;
                math::vec2f onScreenPivot = renderer->camera->project(pivot, true);
                // add circle
                vtxImGui::drawOrigin(onScreenPivot);
                meanPivot += pivot;
            }
            if(instances.size() > 1)
            {
	            meanPivot /= instances.size();
                projectedMeanPivot = renderer->camera->project(meanPivot, true);
				vtxImGui::drawOrigin(projectedMeanPivot);
            }
            else
            {
                projectedMeanPivot = renderer->camera->project(meanPivot, true);
            }

            if(isTransforming)
            {
                const ImVec2 mousePos = ImGui::GetMousePos();
                const ImVec2 windowPos = ImGui::GetWindowPos();
                vtxImGui::drawDashedLine(projectedMeanPivot, { mousePos.x - windowPos.x, mousePos.y - windowPos.y });
            }
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

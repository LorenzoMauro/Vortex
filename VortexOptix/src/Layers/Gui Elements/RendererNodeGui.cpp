#include "RendererNodeGui.h"

#include "imgui.h"
#include "Scene/Nodes/Renderer.h"
#include "Device/DevicePrograms/LaunchParams.h"

namespace vtx::gui
{
    void rendererNodeGui(std::shared_ptr<graph::Renderer> renderNode)
    {
        ImGui::Begin("Renderer Settings");

        int displayBufferItem = renderNode->settings.displayBuffer;

        if (ImGui::Combo("Display Buffer Type", &displayBufferItem, RendererDeviceSettings::displayBufferNames, RendererDeviceSettings::DisplayBuffer::FB_COUNT))
        {
            renderNode->settings.displayBuffer = static_cast<RendererDeviceSettings::DisplayBuffer>(displayBufferItem);
            renderNode->settings.isUpdated = true;
        }

        int samplingTechniqueItem = renderNode->settings.samplingTechnique;

        if (ImGui::Combo("Sampling Technique", &samplingTechniqueItem, RendererDeviceSettings::samplingTechniqueNames, RendererDeviceSettings::SamplingTechnique::S_COUNT))
        {
            renderNode->settings.samplingTechnique = static_cast<RendererDeviceSettings::SamplingTechnique>(samplingTechniqueItem);
            renderNode->settings.iteration = -1;
            renderNode->settings.isUpdated = true;
        }

        ImGui::Separator();
        if (ImGui::SliderInt("Max Samples", &(renderNode->settings.maxSamples), 0, 10000, "%d")) {
            renderNode->settings.iteration = -1;
            renderNode->settings.isUpdated = true;
        };
        if (ImGui::SliderInt("Max Bounces", &(renderNode->settings.maxBounces), 0, 35, "%d")) {
            renderNode->settings.iteration = -1;
            renderNode->settings.isUpdated = true;
        };
        if (ImGui::Checkbox("Accumulate", &(renderNode->settings.accumulate))) {
            renderNode->settings.iteration = -1;
            renderNode->settings.isUpdated = true;
        };
        if (ImGui::SliderFloat("Min Clip", &(renderNode->settings.minClamp), 0, 1000, "%.3f")) {
            renderNode->settings.iteration = -1;
            renderNode->settings.isUpdated = true;
        };
        if (ImGui::SliderFloat("Max Clip", &(renderNode->settings.maxClamp), 0, 10000, "%.3f")) {
            renderNode->settings.iteration = -1;
            renderNode->settings.isUpdated = true;
        };

        ///////// INFO /////////////
        ////////////////////////////
        ImGui::Separator();
        ImGui::Text("Total Time:            %.3f s", renderNode->totalTimeSeconds);
        ImGui::Text("Samples Per Pixels:    %d", renderNode->settings.iteration);
        ImGui::Text("SPP per Seconds:       %.3f", renderNode->sppS);
        ImGui::Text("Frame Time:            %.3f ms", renderNode->averageFrameTime);
        ImGui::Text("Fps:                   %.3f", renderNode->fps);
        ImGui::Separator();

        ImGui::End();
    }
}


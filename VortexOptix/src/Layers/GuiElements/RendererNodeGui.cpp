#include "RendererNodeGui.h"

#include "imgui.h"
#include "Core/ImGuiOp.h"
#include "Scene/Nodes/Renderer.h"
#include "Device/DevicePrograms/LaunchParams.h"

namespace vtx::gui
{
    void rendererNodeGui(std::shared_ptr<graph::Renderer> renderNode)
    {
        ImGui::Begin("Renderer Settings");

        int displayBufferItem = renderNode->settings.displayBuffer;

        if (ImGui::Combo(labelPrefix("Display Buffer Type").c_str(), &displayBufferItem, RendererDeviceSettings::displayBufferNames, RendererDeviceSettings::DisplayBuffer::FB_COUNT))
        {
            renderNode->settings.displayBuffer = static_cast<RendererDeviceSettings::DisplayBuffer>(displayBufferItem);
            renderNode->settings.isUpdated = true;
        }

        int samplingTechniqueItem = renderNode->settings.samplingTechnique;

        if (ImGui::Combo(labelPrefix("Sampling Technique").c_str(), &samplingTechniqueItem, RendererDeviceSettings::samplingTechniqueNames, RendererDeviceSettings::SamplingTechnique::S_COUNT))
        {
            renderNode->settings.samplingTechnique = static_cast<RendererDeviceSettings::SamplingTechnique>(samplingTechniqueItem);
            renderNode->settings.iteration = -1;
            renderNode->settings.isUpdated = true;
        }

        ImGui::Separator();
        if (ImGui::SliderInt(labelPrefix("Max Samples").c_str(), &(renderNode->settings.maxSamples), 0, 100000, "%d")) {
            //renderNode->settings.iteration = -1;
            renderNode->settings.isUpdated = true;
        };
        if (ImGui::SliderInt(labelPrefix("Max Bounces").c_str(), &(renderNode->settings.maxBounces), 0, 35, "%d")) {
            renderNode->settings.iteration = -1;
            renderNode->settings.isUpdated = true;
        };
        if (ImGui::Checkbox(labelPrefix("Accumulate").c_str(), &(renderNode->settings.accumulate))) {
            renderNode->settings.iteration = -1;
            renderNode->settings.isUpdated = true;
        };
        if (ImGui::SliderFloat(labelPrefix("Min Clip").c_str(), &(renderNode->settings.minClamp), 0, 1000, "%.3f")) {
            renderNode->settings.iteration = -1;
            renderNode->settings.isUpdated = true;
        };
        if (ImGui::SliderFloat(labelPrefix("Max Clip").c_str(), &(renderNode->settings.maxClamp), 0, 10000, "%.3f")) {
            renderNode->settings.iteration = -1;
            renderNode->settings.isUpdated = true;
        };

        ImGui::Separator();
        if (ImGui::Checkbox(labelPrefix("Adaptive Sampling").c_str(), &(renderNode->settings.adaptiveSampling))) {
            renderNode->settings.iteration = -1;
            renderNode->settings.isUpdated = true;
        }
        if (ImGui::DragInt(labelPrefix("Noise Kernel Size").c_str(), &(renderNode->settings.noiseKernelSize), 1, 0, 40)) {
            renderNode->settings.iteration = -1;
        }
        if (ImGui::DragInt(labelPrefix("Start Adaptive Sample Sample").c_str(), &(renderNode->settings.minAdaptiveSamples), 1, 0, 500)) {
            renderNode->settings.iteration = -1;
            renderNode->settings.isUpdated = true;
        }
        if (ImGui::DragInt(labelPrefix("Adaptive Min Pixel Sample").c_str(), &(renderNode->settings.minPixelSamples), 1, 0, 20)) {
            renderNode->settings.iteration = -1;
            renderNode->settings.isUpdated = true;
        }
        if (ImGui::DragInt(labelPrefix("Adaptive Max Pixel Sample").c_str(), &(renderNode->settings.maxPixelSamples), 1, 1, 1000)) {
            renderNode->settings.iteration = -1;
            renderNode->settings.isUpdated = true;
        }
        if (ImGui::DragFloat(labelPrefix("Albedo-Normal Noise Influence").c_str(), &(renderNode->settings.albedoNormalNoiseInfluence), 0.01f, 0.0f, 1.0f)) {
            renderNode->settings.iteration = -1;
        }
        if (ImGui::DragFloat(labelPrefix("Noise Threshold").c_str(), &(renderNode->settings.noiseCutOff), 0.00001f, 0.0f, 1.0f)) {
            renderNode->settings.iteration = -1;
        }

        ImGui::Separator();
        if (ImGui::ColorEdit3(labelPrefix("White Point").c_str(), &(renderNode->toneMapperSettings.whitePoint.x))) {
            renderNode->toneMapperSettings.isUpdated = true;
        }
        if (ImGui::ColorEdit3(labelPrefix("Color balance").c_str(), &(renderNode->toneMapperSettings.colorBalance.x))) {
            renderNode->toneMapperSettings.isUpdated = true;
        }
        if (ImGui::DragFloat(labelPrefix("Burn Highlights").c_str(), &(renderNode->toneMapperSettings.burnHighlights), 0.01f, 0.0f, 1.0f)) {
            renderNode->toneMapperSettings.isUpdated = true;
        }
        if (ImGui::DragFloat(labelPrefix("Crush Blacks").c_str(), &(renderNode->toneMapperSettings.crushBlacks), 0.01f, 0.0f, 1.0f)) {
            renderNode->toneMapperSettings.isUpdated = true;
        }
        if (ImGui::DragFloat(labelPrefix("Saturation").c_str(), &(renderNode->toneMapperSettings.saturation), 0.01f, 0.0f, 2.0f)) {
            renderNode->toneMapperSettings.isUpdated = true;
        }
        if (ImGui::DragFloat(labelPrefix("Gamma").c_str(), &(renderNode->toneMapperSettings.gamma), 0.01f, 0.1f, 5.0f)) {
            renderNode->toneMapperSettings.isUpdated = true;
        }

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


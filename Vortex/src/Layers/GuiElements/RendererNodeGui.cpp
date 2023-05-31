#include "RendererNodeGui.h"

#include "imgui.h"
#include "Core/ImGuiOp.h"
#include "Core/CustomImGui/CustomImGui.h"
#include "Scene/Nodes/Renderer.h"
#include "Device/DevicePrograms/LaunchParams.h"

namespace vtx::gui
{
	typedef bool (*ComboFuncType)(const char*, int*, const char* const [], int, int);

    void rendererNodeGui(std::shared_ptr<graph::Renderer> renderNode)
    {
        ImGui::Begin("Renderer Settings");

		float availableWidth = ImGui::GetContentRegionAvail().x;

		ImGui::PushItemWidth(availableWidth); // Set the width of the next widget to 200

		std::string hiddenLabel = "##hidden";
        int displayBufferItem = renderNode->settings.displayBuffer;

        if (vtxImGui::HalfSpaceWidget("Display Buffer Type", (ComboFuncType)ImGui::Combo, (hiddenLabel + "_Display Buffer Type").c_str(), &displayBufferItem, RendererDeviceSettings::displayBufferNames, RendererDeviceSettings::DisplayBuffer::FB_COUNT, -1))
        {
            renderNode->settings.displayBuffer = static_cast<RendererDeviceSettings::DisplayBuffer>(displayBufferItem);
            renderNode->settings.isUpdated = true;
        }

        int samplingTechniqueItem = renderNode->settings.samplingTechnique;

		if (vtxImGui::HalfSpaceWidget("Sampling Technique", (ComboFuncType)ImGui::Combo, (hiddenLabel + "_Sampling Technique").c_str(), &samplingTechniqueItem, RendererDeviceSettings::samplingTechniqueNames, RendererDeviceSettings::SamplingTechnique::S_COUNT, -1))
        {
            renderNode->settings.samplingTechnique = static_cast<RendererDeviceSettings::SamplingTechnique>(samplingTechniqueItem);
            renderNode->settings.iteration = -1;
            renderNode->settings.isUpdated = true;
        }

		ImGui::Separator();

		if (vtxImGui::HalfSpaceWidget("Max Samples", ImGui::SliderInt,(hiddenLabel + "_Max Samples").c_str(), &(renderNode->settings.maxSamples), 0, 100000, "%d", 0))
		{
			//renderNode->settings.iteration = -1;
			renderNode->settings.isUpdated = true;
		};
		if (vtxImGui::HalfSpaceWidget("Max Bounces", ImGui::SliderInt, (hiddenLabel + "_Max Bounces").c_str(), &(renderNode->settings.maxBounces), 0, 35, "%d", 0))
		{
			renderNode->settings.iteration = -1;
			renderNode->settings.isUpdated = true;
		};
		if (vtxImGui::HalfSpaceWidget("Accumulate", ImGui::Checkbox, (hiddenLabel + "_Accumulate").c_str(), &(renderNode->settings.accumulate)))
		{
			renderNode->settings.iteration = -1;
			renderNode->settings.isUpdated = true;
		};
		if (vtxImGui::HalfSpaceWidget("Min Clip", ImGui::DragFloat, (hiddenLabel + "_Min Clip").c_str(), &(renderNode->settings.minClamp), 0.01, 0, 1000, "%.3f", 0))
		{
			renderNode->settings.iteration = -1;
			renderNode->settings.isUpdated = true;
		};
		if (vtxImGui::HalfSpaceWidget("Max Clip", ImGui::DragFloat, (hiddenLabel + "_Max Clip").c_str(), &(renderNode->settings.maxClamp), 1, 0, 10000, "%.3f", 0))
		{
			renderNode->settings.iteration = -1;
			renderNode->settings.isUpdated = true;
		};

		ImGui::Separator();
		ImGui::SetNextItemOpen(true, ImGuiCond_Once);
		if (ImGui::CollapsingHeader("Adaptive Sampling Settings"))
		{
			if (vtxImGui::HalfSpaceWidget("Adaptive Sampling", ImGui::Checkbox, (hiddenLabel + "_Adaptive Sampling").c_str(), &(renderNode->settings.adaptiveSampling)))
			{
				renderNode->settings.iteration = -1;
					renderNode->settings.isUpdated = true;
			}
			if (vtxImGui::HalfSpaceWidget("Noise Kernel Size", ImGui::DragInt, (hiddenLabel + "_Noise Kernel Size").c_str(), &(renderNode->settings.noiseKernelSize), 1, 0, 40, "%d", 0))
			{
				renderNode->settings.iteration = -1;
			}
			if (vtxImGui::HalfSpaceWidget("Start Adaptive Sample Sample", ImGui::DragInt, (hiddenLabel + "_Start Adaptive Sample Sample").c_str(), &(renderNode->settings.minAdaptiveSamples), 1, 0, 500, "%d", 0))
			{
				renderNode->settings.iteration = -1;
				renderNode->settings.isUpdated = true;
			}
			if (vtxImGui::HalfSpaceWidget("Adaptive Min Pixel Sample", ImGui::DragInt, (hiddenLabel + "_Adaptive Min Pixel Sample").c_str(), &(renderNode->settings.minPixelSamples), 1, 0, 20, "%d", 0))
			{
				renderNode->settings.iteration = -1;
				renderNode->settings.isUpdated = true;
			}
			if (vtxImGui::HalfSpaceWidget("Adaptive Max Pixel Sample", ImGui::DragInt, (hiddenLabel + "_Adaptive Max Pixel Sample").c_str(), &(renderNode->settings.maxPixelSamples), 1, 1, 1000, "%d", 0))
			{
				renderNode->settings.iteration = -1;
				renderNode->settings.isUpdated = true;
			}
			if (vtxImGui::HalfSpaceWidget("Albedo-Normal Noise Influence", ImGui::DragFloat, (hiddenLabel + "_Albedo-Normal Noise Influence").c_str(), &(renderNode->settings.albedoNormalNoiseInfluence), 0.01f, 0.0f, 1.0f, "%.3f", 0))
			{
				renderNode->settings.iteration = -1;
			}
			if (vtxImGui::HalfSpaceWidget("Noise Threshold", ImGui::DragFloat, (hiddenLabel + "_Noise Threshold").c_str(), &(renderNode->settings.noiseCutOff), 0.00001f, 0.0f, 1.0f, "%.3f", 0))
			{
				renderNode->settings.iteration = -1;
			}
		}

		ImGui::Separator();
		ImGui::SetNextItemOpen(true, ImGuiCond_Once);
		if(ImGui::CollapsingHeader("Post Processing"))
		{
			ImGui::SetNextItemOpen(true, ImGuiCond_Once);
			if (ImGui::CollapsingHeader("Firefly Filtering"))
			{
				if (vtxImGui::HalfSpaceWidget("FireFly Kernel Size", ImGui::DragInt, (hiddenLabel + "FireFly Kernel Size").c_str(), &(renderNode->settings.fireflyKernelSize), 1, 1, 30, "%d", 0))
				{
				}
				if (vtxImGui::HalfSpaceWidget("FireFly Threshold", ImGui::DragFloat, (hiddenLabel + "FireFly Threshold").c_str(), &(renderNode->settings.fireflyThreshold), 0.01f, 1.0f, 5.0f, "%.3f", 0))
				{
				}
				if(vtxImGui::HalfSpaceWidget("Enable Fireflies Filtering", ImGui::Checkbox, (hiddenLabel + "Enable Fireflies Filtering").c_str(), &(renderNode->settings.removeFireflies)))
				{
					renderNode->settings.isUpdated = true;
				}
			}

			ImGui::SetNextItemOpen(true, ImGuiCond_Once);
			if (ImGui::CollapsingHeader("Denoiser Settings"))
			{
				if (vtxImGui::HalfSpaceWidget("Enable Denoiser", ImGui::Checkbox, (hiddenLabel + "Enable Denoiser").c_str(), &(renderNode->settings.enableDenoiser)))
				{
					renderNode->settings.isUpdated = true;
				}
				if (vtxImGui::HalfSpaceWidget("Denoiser Iteration Start", ImGui::DragInt, (hiddenLabel + "Denoiser Iteration Start").c_str(), &(renderNode->settings.denoiserStart), 1, 1, 1000, "%d", 0))
				{
				}
				if (vtxImGui::HalfSpaceWidget("Denoiser Blend", ImGui::DragFloat, (hiddenLabel + "Denoiser Blend").c_str(), &(renderNode->settings.denoiserBlend), 0.01f, 0.0f, 1.0f, "%.3f", 0))
				{
				}
			}

			ImGui::SetNextItemOpen(true, ImGuiCond_Once);
			if (ImGui::CollapsingHeader("Tone Mapper Settings"))
			{
				if (vtxImGui::HalfSpaceWidget("White Point", vtxImGui::colorPicker, (hiddenLabel + "_White Point").c_str(), &(renderNode->toneMapperSettings.whitePoint.x)))
				{
					renderNode->toneMapperSettings.isUpdated = true;
				}
				if (vtxImGui::HalfSpaceWidget("Color balance", vtxImGui::colorPicker, (hiddenLabel + "_Color balance").c_str(), &(renderNode->toneMapperSettings.colorBalance.x)))
				{
					renderNode->toneMapperSettings.isUpdated = true;
				}
				if (vtxImGui::HalfSpaceWidget("Burn Highlights", ImGui::DragFloat, (hiddenLabel + "_Burn Highlights").c_str(), &(renderNode->toneMapperSettings.burnHighlights), 0.01f, 0.0f, 1.0f, "%.3f", 0))
				{
					renderNode->toneMapperSettings.isUpdated = true;
				}
				if (vtxImGui::HalfSpaceWidget("Crush Blacks", ImGui::DragFloat, (hiddenLabel + "_Crush Blacks").c_str(), &(renderNode->toneMapperSettings.crushBlacks), 0.01f, 0.0f, 1.0f, "%.3f", 0))
				{
					renderNode->toneMapperSettings.isUpdated = true;
				}
				if (vtxImGui::HalfSpaceWidget("Saturation", ImGui::DragFloat, (hiddenLabel + "_Saturation").c_str(), &(renderNode->toneMapperSettings.saturation), 0.01f, 0.0f, 2.0f, "%.3f", 0))
				{
					renderNode->toneMapperSettings.isUpdated = true;
				}
				if (vtxImGui::HalfSpaceWidget("Gamma", ImGui::DragFloat, (hiddenLabel + "_Gamma").c_str(), &(renderNode->toneMapperSettings.gamma), 0.01f, 0.1f, 5.0f, "%.3f", 0))
				{
					renderNode->toneMapperSettings.isUpdated = true;
				}
			}
		}
		

        ///////// INFO /////////////
        ////////////////////////////
        ImGui::Separator();
        vtxImGui::HalfSpaceWidget("Total Time:", vtxImGui::booleanText, "%.3f s", renderNode->totalTimeSeconds);
		vtxImGui::HalfSpaceWidget("Samples Per Pixels:", vtxImGui::booleanText, "%d", renderNode->settings.iteration);
		vtxImGui::HalfSpaceWidget("SPP per Seconds:", vtxImGui::booleanText, "%.3f", renderNode->sppS);
		vtxImGui::HalfSpaceWidget("Frame Time:", vtxImGui::booleanText, "%.3f ms", renderNode->averageFrameTime);
		vtxImGui::HalfSpaceWidget("Fps:", vtxImGui::booleanText, "%.3f", renderNode->fps);

		float toPercent = 100.0f / (1000.0f * renderNode->totalTimeSeconds);
		vtxImGui::HalfSpaceWidget("Noise   Computation % ", vtxImGui::booleanText, "%.2f", (toPercent*renderNode->noiseComputationTime ));
		vtxImGui::HalfSpaceWidget("Trace   Computation % ", vtxImGui::booleanText, "%.2f", (toPercent*renderNode->traceComputationTime ));
		vtxImGui::HalfSpaceWidget("Post	   Computation % ", vtxImGui::booleanText, "%.2f", (toPercent*renderNode->postProcessingComputationTime ));
		vtxImGui::HalfSpaceWidget("Display Computation % ", vtxImGui::booleanText, "%.2f", (toPercent*renderNode->displayComputationTime ));
        ImGui::Separator();

		ImGui::PopItemWidth();
        ImGui::End();
    }
}


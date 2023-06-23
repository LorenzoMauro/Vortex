#include "RendererNodeGui.h"

#include "imgui.h"
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

		if (vtxImGui::HalfSpaceWidget("Separate Thread", ImGui::Checkbox, (hiddenLabel + "_Separate Thread").c_str(), &(renderNode->settings.runOnSeparateThread)))
		{
			renderNode->settings.iteration = -1;
			renderNode->settings.isUpdated = true;
		}

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

		if (vtxImGui::HalfSpaceWidget("Max Samples", ImGui::SliderInt, (hiddenLabel + "_Max Samples").c_str(), &(renderNode->settings.maxSamples), 0, 100000, "%d", 0))
		{
			//renderNode->settings.iteration = -1;
			renderNode->settings.isUpdated = true;
		};
		if (vtxImGui::HalfSpaceWidget("Max Bounces", ImGui::SliderInt, (hiddenLabel + "_Max Bounces").c_str(), &(renderNode->settings.maxBounces), 1, 35, "%d", 0))
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
		if (ImGui::CollapsingHeader("WaveFront Settings"))
		{
			if (vtxImGui::HalfSpaceWidget("Use WaveFront", ImGui::Checkbox, (hiddenLabel + "_Use Wavefront").c_str(), &(renderNode->settings.useWavefront)))
			{
				renderNode->settings.iteration = -1;
				renderNode->settings.isUpdated = true;
			}

			if (vtxImGui::HalfSpaceWidget("Long Path Mega Kernel", ImGui::Checkbox, (hiddenLabel + "_Long Path Mega Kernel").c_str(), &(renderNode->settings.useLongPathKernel)))
			{
				renderNode->settings.iteration = -1;
				renderNode->settings.isUpdated = true;
			}

			if (vtxImGui::HalfSpaceWidget("Wavefront Long Path Kernel Start", ImGui::DragFloat, (hiddenLabel + "_Wavefront Long Path Kernel Start").c_str(), &(renderNode->settings.longPathPercentage), 0.01, 0, 1, "%.3f", 0))
			{
				renderNode->settings.iteration = -1;
				renderNode->settings.isUpdated = true;
			}

			if (vtxImGui::HalfSpaceWidget("Fit Kernel Launch", ImGui::Checkbox, (hiddenLabel + "_Fit Kernel Launch").c_str(), &(renderNode->settings.fitWavefront)))
			{
				renderNode->settings.iteration = -1;
				renderNode->settings.isUpdated = true;
			}

			if (vtxImGui::HalfSpaceWidget("Use Optix Shader Kernel", ImGui::Checkbox, (hiddenLabel + "_Use Optix Shader Kernel").c_str(), &(renderNode->settings.optixShade)))
			{
				renderNode->settings.iteration = -1;
				renderNode->settings.isUpdated = true;
			}

			if (vtxImGui::HalfSpaceWidget("Russian Roulette", ImGui::Checkbox, (hiddenLabel + "_Use Russian Roulette").c_str(), &(renderNode->settings.useRussianRoulette)))
			{
				renderNode->settings.iteration = -1;
				renderNode->settings.isUpdated = true;
			}
		}

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
			/*if (vtxImGui::HalfSpaceWidget("Adaptive Min Pixel Sample", ImGui::DragInt, (hiddenLabel + "_Adaptive Min Pixel Sample").c_str(), &(renderNode->settings.minPixelSamples), 1, 0, 20, "%d", 0))
			{
				renderNode->settings.iteration = -1;
				renderNode->settings.isUpdated = true;
			}
			if (vtxImGui::HalfSpaceWidget("Adaptive Max Pixel Sample", ImGui::DragInt, (hiddenLabel + "_Adaptive Max Pixel Sample").c_str(), &(renderNode->settings.maxPixelSamples), 1, 1, 1000, "%d", 0))
			{
				renderNode->settings.iteration = -1;
				renderNode->settings.isUpdated = true;
			}*/
			if (vtxImGui::HalfSpaceWidget("Albedo-Normal Noise Influence", ImGui::DragFloat, (hiddenLabel + "_Albedo-Normal Noise Influence").c_str(), &(renderNode->settings.albedoNormalNoiseInfluence), 0.01f, 0.0f, 1.0f, "%.3f", 0))
			{
				renderNode->settings.iteration = -1;
			}
			/*if (vtxImGui::HalfSpaceWidget("Noise Threshold", ImGui::DragFloat, (hiddenLabel + "_Noise Threshold").c_str(), &(renderNode->settings.noiseCutOff), 0.00001f, 0.0f, 1.0f, "%.3f", 0))
			{
				renderNode->settings.iteration = -1;
			}*/
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
		ImGui::SetNextItemOpen(true, ImGuiCond_Once);
		if (ImGui::CollapsingHeader("Statistics"))
		{
			vtxImGui::HalfSpaceWidget("Total Time:", vtxImGui::booleanText, "%.3f s", renderNode->totalTimeSeconds);
			vtxImGui::HalfSpaceWidget("Samples Per Pixels:", vtxImGui::booleanText, "%d", renderNode->settings.iteration);
			vtxImGui::HalfSpaceWidget("SPP per Seconds:", vtxImGui::booleanText, "%.3f", renderNode->sppS);
			vtxImGui::HalfSpaceWidget("Frame Time:", vtxImGui::booleanText, "%.3f ms", renderNode->averageFrameTime);
			vtxImGui::HalfSpaceWidget("Fps:", vtxImGui::booleanText, "%.3f", renderNode->fps);
			ImGui::Separator();

			float internalFps = ((1000.0f) * (float)(renderNode->internalIteration)) / renderNode->overallTime;
			float totTimeInternal = renderNode->overallTime / 1000.0f;
			vtxImGui::HalfSpaceWidget("CPU Fps:", vtxImGui::booleanText, "%.3f", internalFps);
			vtxImGui::HalfSpaceWidget("CPU Tot Time:", vtxImGui::booleanText, "%.3f", totTimeInternal);
			ImGui::Separator();

			KernelTimes& kernelTimes = renderNode->getWaveFrontTimes();
			int actualLaunches = renderNode->getWavefrontLaunches();
			float factor = 1.0f / (float)actualLaunches;

			vtxImGui::HalfSpaceWidget("Renderer Noise				", vtxImGui::booleanText, "%.2f ms", factor* renderNode->noiseComputationTime);
			vtxImGui::HalfSpaceWidget("Renderer Trace				", vtxImGui::booleanText, "%.2f ms", factor* renderNode->traceComputationTime);
			vtxImGui::HalfSpaceWidget("Renderer Post				", vtxImGui::booleanText, "%.2f ms", factor* renderNode->postProcessingComputationTime);
			vtxImGui::HalfSpaceWidget("Renderer Display				", vtxImGui::booleanText, "%.2f ms", factor* renderNode->displayComputationTime);
			ImGui::Separator();
			vtxImGui::HalfSpaceWidget("WaveFront Generate Ray		", vtxImGui::booleanText, "%.2f ms", factor* kernelTimes.genCameraRay);
			vtxImGui::HalfSpaceWidget("WaveFront Trace				", vtxImGui::booleanText, "%.2f ms", factor* kernelTimes.traceRadianceRay);
			vtxImGui::HalfSpaceWidget("WaveFront Shade				", vtxImGui::booleanText, "%.2f ms", factor* kernelTimes.shadeRay);
			vtxImGui::HalfSpaceWidget("WaveFront Escaped			", vtxImGui::booleanText, "%.2f ms", factor* kernelTimes.handleEscapedRay);
			vtxImGui::HalfSpaceWidget("WaveFront Accumulate			", vtxImGui::booleanText, "%.2f ms", factor* kernelTimes.accumulateRay);
			vtxImGui::HalfSpaceWidget("WaveFront Reset				", vtxImGui::booleanText, "%.2f ms", factor* kernelTimes.reset);
			vtxImGui::HalfSpaceWidget("WaveFront Fetch Queue Size	", vtxImGui::booleanText, "%.2f ms", factor * kernelTimes.fetchQueueSize);
		}
        
		ImGui::PopItemWidth();
        ImGui::End();
    }
}


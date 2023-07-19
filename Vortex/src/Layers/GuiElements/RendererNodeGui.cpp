#include "RendererNodeGui.h"

#include "imgui.h"
#include "Core/CustomImGui/CustomImGui.h"
#include "Device/KernelInfos.h"
#include "Scene/Nodes/Renderer.h"
#include "Device/DevicePrograms/LaunchParams.h"
#include "implot.h"
#include <variant>

#include "NeuralNetworks/NeuralNetworkGraphs.h"

namespace vtx::gui
{
	typedef bool (*ComboFuncType)(const char*, int*, const char* const [], int, int);

#define Q1_LOSS_COLOR IM_COL32(255, 50, 20, 255)
#define Q2_LOSS_COLOR IM_COL32(200, 100, 20, 255)
#define POLICY_LOSS_COLOR IM_COL32(100, 100, 200, 255)
#define ALPHA_LOSS_COLOR IM_COL32(100, 200, 100, 255)


	using DataType = std::variant<std::vector<int>, std::vector<float>>;

	struct PlotInfo
	{
		std::vector<DataType> data;
		std::vector<ImU32> color ;
		std::vector<std::string> name;

		std::string xLabel = "X";
		std::string yLabel = "Y";
		std::string title = "Plot";

		void addPlot(const DataType& _data, const ImU32 _color = IM_COL32_WHITE, std::string _name = "")
		{
			data.push_back(_data);
			color.push_back(_color);

			_name = (_name.empty()) ? "Plot_" + std::to_string(name.size()) : _name;
			name.push_back(_name);
		}
	};

	void plotLines(const PlotInfo& lines, const ImVec2& quadrantSize = ImGui::GetContentRegionAvail())
	{
		ImGui::BeginChild(lines.title.c_str(), quadrantSize, false);
		if (ImPlot::BeginPlot(lines.title.c_str(), quadrantSize)) {
			ImPlotAxisFlags flags = ImPlotAxisFlags_AutoFit;
			flags |= ImPlotAxisFlags_NoLabel;
			ImPlot::SetupAxis(ImAxis_X1, lines.xLabel.c_str(), flags);
			ImPlot::SetupAxis(ImAxis_Y1, lines.yLabel.c_str(), flags);

			for (int i = 0; i < lines.color.size(); i++)
			{
				ImPlot::PushStyleColor(ImPlotCol_Line, lines.color[i]);
				std::visit([&](auto&& arg) {
					using T = std::decay_t<decltype(arg)>;
					if constexpr (std::is_same_v<T, std::vector<int>>)
					{
						ImPlot::PlotLine(lines.name[i].c_str(), arg.data(), arg.size());
					}
					else if constexpr (std::is_same_v<T, std::vector<float>>)
					{
						ImPlot::PlotLine(lines.name[i].c_str(), arg.data(), arg.size());
					}
					}, lines.data[i]);
				ImPlot::PopStyleColor();
			}

			ImPlot::EndPlot();
		}
		ImGui::EndChild();
	}

	void gridPlot(std::vector<PlotInfo> plots)
	{
		const int numberOfPlots = plots.size();
		int xNumberPlots = std::ceil(std::sqrt((float)numberOfPlots));
		int yNumberPlots = std::ceil((float)numberOfPlots / xNumberPlots);


		ImGuiStyle& style = ImGui::GetStyle();
		math::vec2f windowSize = math::vec2f(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y);
		math::vec2f itemSpacing = math::vec2f(style.ItemSpacing.x, style.ItemSpacing.y);
		math::vec2f windowPadding = math::vec2f(style.WindowPadding.x, style.WindowPadding.y);

		math::vec2f quadrantSize = math::vec2f(windowSize) / math::vec2f(xNumberPlots, yNumberPlots) - itemSpacing * math::vec2f(xNumberPlots-1, yNumberPlots-1); //- windowPadding 

		//VTX_INFO("Window Size: {}-{}\n Item Spacing: {}-{}\n Window Padding: {}-{}\n Quadrant Size: {}-{}\n", windowSize.x, windowSize.y, itemSpacing.x, itemSpacing.y, windowPadding.x, windowPadding.y, quadrantSize.x, quadrantSize.y);

		for (int y = 0; y < yNumberPlots; y++){
			for (int x = 0; x < xNumberPlots; x++){
				int index = x + y * xNumberPlots;

				if (index < numberOfPlots) // make sure index is within the bounds
				{
					math::vec2f actualSize = quadrantSize;
					if (index == numberOfPlots - 1) // if last plot
					{
						int remainingXSpaces = xNumberPlots - (x + 1);
						int remainingYSpaces = yNumberPlots - (y + 1);

						actualSize.x += (float)remainingXSpaces * (quadrantSize.x + itemSpacing.x);
						actualSize.y += (float)remainingYSpaces * (quadrantSize.y + itemSpacing.y);

						//VTX_INFO("Remaining X Spaces: {}\n Remaining Y Spaces: {}\n Quadrant Size = {}-{}", remainingXSpaces, remainingYSpaces, quadrantSize.x, quadrantSize.y);
					}

					plotLines(plots[index], { actualSize.x, actualSize.y });
					if (x < xNumberPlots- 1)
					{
						ImGui::SameLine();
					}
				}
			}
		}
	}

	void sacPlots(vtx::network::Network& network)
	{
		network::GraphsData& graphsData = network.getGraphs();

		PlotInfo q1q2Losses;
		q1q2Losses.title = "Q1Q2Losses";
		q1q2Losses.xLabel = "Batches";
		q1q2Losses.yLabel = "Loss";
		q1q2Losses.addPlot(graphsData.graphs[network::G_Q1_LOSS], Q1_LOSS_COLOR, "Q1 Loss");
		q1q2Losses.addPlot(graphsData.graphs[network::G_Q2_LOSS], Q2_LOSS_COLOR, "Q2 Loss");


		PlotInfo alphaLosses;
		alphaLosses.title = "AlphaLosses";
		alphaLosses.xLabel = "Batches";
		alphaLosses.yLabel = "Loss";
		alphaLosses.addPlot(graphsData.graphs[network::G_ALPHA_LOSS], ALPHA_LOSS_COLOR, "Alpha Loss");

		PlotInfo alphaValues;
		alphaValues.title = "AlphaValues";
		alphaValues.xLabel = "Batches";
		alphaValues.yLabel = "Alpha";
		alphaValues.addPlot(graphsData.graphs[network::G_ALPHA_VALUES], ALPHA_LOSS_COLOR, "Alpha Value");

		PlotInfo policyLosses;
		policyLosses.title = "PolicyLosses";
		policyLosses.xLabel = "Batches";
		policyLosses.yLabel = "Loss";
		policyLosses.addPlot(graphsData.graphs[network::G_POLICY_LOSS], POLICY_LOSS_COLOR, "Policy Loss");

		PlotInfo rewards;
		rewards.title = "Replay buffer Rewards";
		rewards.xLabel = "Batches";
		rewards.yLabel = "Reward";
		rewards.addPlot(graphsData.graphs[network::G_DATASET_REWARDS], POLICY_LOSS_COLOR, "Replay buffer Rewards");
		rewards.addPlot(graphsData.graphs[network::G_Q1_VALUES], Q1_LOSS_COLOR, "Q1");
		rewards.addPlot(graphsData.graphs[network::G_Q2_VALUES], Q2_LOSS_COLOR, "Q2");

		PlotInfo inferenceConcentration;
		inferenceConcentration.title = "Inference Concentration";
		inferenceConcentration.xLabel = "Batches";
		inferenceConcentration.yLabel = "Concentration";
		inferenceConcentration.addPlot(graphsData.graphs[network::G_INFERENCE_CONCENTRATION], POLICY_LOSS_COLOR, "Inference Concentration");

		gridPlot({ q1q2Losses, alphaLosses, alphaValues, policyLosses, rewards, inferenceConcentration });

	}

	void ngpPlots(vtx::network::Network& network)
	{
		network::GraphsData& graphsData = network.getGraphs();
	}

	void sacSettingsGui(network::SacSettings& settings, network::Network& network, const std::shared_ptr<graph::Renderer>& renderNode)
	{
		const std::string hiddenLabel = "##hidden";

		if (vtxImGui::HalfSpaceWidget("Policy Lr", ImGui::DragFloat, (hiddenLabel + "_Policy Lr").c_str(), &(settings.policyLr), 0.00001, 0, 1, "%.10f", 0))
		{
			network.reset();
		}

		if (vtxImGui::HalfSpaceWidget("Q Lr", ImGui::DragFloat, (hiddenLabel + "_Q Lr").c_str(), &(settings.qLr), 0.00001, 0, 1, "%.10f", 0))
		{
			network.reset();
		}

		if (vtxImGui::HalfSpaceWidget("Alpha Lr", ImGui::DragFloat, (hiddenLabel + "_Alpha Lr").c_str(), &(settings.alphaLr), 0.00001, 0, 1, "%.10f", 0))
		{
			network.reset();
		}

		if (vtxImGui::HalfSpaceWidget("Gamma ", ImGui::DragFloat, (hiddenLabel + "_Learning Rate").c_str(), &(settings.gamma), 0.00001, 0, 1, "%.6f", 0))
		{
			network.reset();
		}

		if (vtxImGui::HalfSpaceWidget("Polyak Update Factor", ImGui::DragFloat, (hiddenLabel + "_Polyak Update Factor").c_str(), &(settings.polyakFactor), 0.00001, 0, 1, "%.6f", 0))
		{
		}

		if (vtxImGui::HalfSpaceWidget("Neural Sampling Fraction", ImGui::DragFloat, (hiddenLabel + "_Neural Sampling Fraction").c_str(), &(settings.neuralSampleFraction), 0.001, 0, 1, "%.3f", 0))
		{
			renderNode->settings.isUpdated = true;
		}
	}

	void ngpSettingsGui(network::NgpSettings& settings, network::Network& network, const std::shared_ptr<graph::Renderer>& renderNode)
	{

	}

	void neuralNetworkGui(vtx::network::Network& network, const std::shared_ptr<graph::Renderer>& renderNode)
	{
		const std::string hiddenLabel = "##hidden";

		ImGui::Begin("Neural Net Settings");
		// Get available width and calculate sizes for each child window
		const float availableWidth = ImGui::GetContentRegionAvail().x;
		const float plotWidth      = availableWidth * 0.8f; // 75% for the plot
		const float settingsWidth  = availableWidth * 0.2f; // 25% for the settings

		ImGui::BeginChild("Plot Child", ImVec2(plotWidth, 0), true);

		if(network.type == network::NT_SAC)
		{
			sacPlots(network);
		}
		else if (network.type == network::NT_NGP)
		{
			ngpPlots(network);
		}

		ImGui::EndChild();

		ImGui::SameLine();  // Position the next child window on the same line to the right

		ImGui::BeginChild("Settings Child", ImVec2(settingsWidth, 0), true);
		ImGui::PushItemWidth(settingsWidth); // Set the width of the next widget to 200


		int networkType = network.type;
		if (vtxImGui::HalfSpaceWidget("Network Type", (ComboFuncType)ImGui::Combo, (hiddenLabel + "_Network Type").c_str(), &networkType, network::networkNames, network::NT_COUNT, -1))
		{
			network.type = static_cast<network::NetworkType>(networkType);
			network.initNetworks();
			renderNode->settings.isUpdated = true;
		}

		network::NetworkSettings& nnSettings = network.getNeuralNetSettings();
		if (vtxImGui::HalfSpaceWidget("BatchSize", ImGui::DragInt, (hiddenLabel + "_BatchSize").c_str(), &(nnSettings.batchSize), 1, 0, 10000, "%d", 0))
		{
		}
		if (vtxImGui::HalfSpaceWidget("Max Training Steps Per Frame", ImGui::DragInt, (hiddenLabel + "_BatchSize").c_str(), &(nnSettings.maxTrainingStepPerFrame), 1, 0, 10000, "%d", 0))
		{
		}

		if (vtxImGui::HalfSpaceWidget("Do Inference", ImGui::Checkbox, (hiddenLabel + "_Do Inference").c_str(), &(nnSettings.doInference)))
		{
			renderNode->settings.isUpdated = true;
		}
		if (vtxImGui::HalfSpaceWidget("Inference Start", ImGui::DragInt, (hiddenLabel + "_Inference Start").c_str(), &(nnSettings.inferenceIterationStart), 1, 0, 10000, "%d", 0))
		{
			renderNode->settings.isUpdated = true;
		}
		if (vtxImGui::HalfSpaceWidget("Clear buffer on Inference Start", ImGui::Checkbox, (hiddenLabel + "_Inference Start").c_str(), &(nnSettings.clearOnInferenceStart)))
		{
			renderNode->settings.isUpdated = true;
		}

		if(network.type == network::NT_SAC)
		{
			network::SacSettings& sacSettings = static_cast<network::SacSettings&>(nnSettings);

			sacSettingsGui(sacSettings, network, renderNode);
		}
		else if (network.type == network::NT_NGP)
		{
			network::NgpSettings& ngpSettings = static_cast<network::NgpSettings&>(nnSettings);
			ngpSettingsGui(ngpSettings, network, renderNode);
		}

		// reset half sapce widget Button
		bool resetNetwork = false;
		if (vtxImGui::HalfSpaceWidget("Reset Network", ImGui::Checkbox, (hiddenLabel + "_Reset Network").c_str(), &(resetNetwork)))
		{
			network.reset();
			renderNode->settings.iteration = -1;
			renderNode->settings.isUpdated = true;
		}

		ImGui::EndChild();

		ImGui::End();

	}

    void rendererNodeGui(const std::shared_ptr<graph::Renderer>& renderNode)
    {
        ImGui::Begin("Renderer Settings");

		const float availableWidth = ImGui::GetContentRegionAvail().x;

		ImGui::PushItemWidth(availableWidth); // Set the width of the next widget to 200

		const std::string hiddenLabel = "##hidden";
        int displayBufferItem = renderNode->settings.displayBuffer;

		if (vtxImGui::HalfSpaceWidget("Separate Thread", ImGui::Checkbox, (hiddenLabel + "_Separate Thread").c_str(), &(renderNode->settings.runOnSeparateThread)))
		{
			renderNode->settings.iteration = -1;
			renderNode->settings.isUpdated = true;
		}

        if (vtxImGui::HalfSpaceWidget("Display Buffer Type", (ComboFuncType)ImGui::Combo, (hiddenLabel + "_Display Buffer Type").c_str(), &displayBufferItem, RendererDeviceSettings::displayBufferNames, FB_COUNT, -1))
        {
            renderNode->settings.displayBuffer = static_cast<DisplayBuffer>(displayBufferItem);
            renderNode->settings.isUpdated = true;
        }

        int samplingTechniqueItem = renderNode->settings.samplingTechnique;

		if (vtxImGui::HalfSpaceWidget("Sampling Technique", (ComboFuncType)ImGui::Combo, (hiddenLabel + "_Sampling Technique").c_str(), &samplingTechniqueItem, RendererDeviceSettings::samplingTechniqueNames, S_COUNT, -1))
        {
            renderNode->settings.samplingTechnique = static_cast<SamplingTechnique>(samplingTechniqueItem);
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
			if (vtxImGui::HalfSpaceWidget("Use Network", ImGui::Checkbox, (hiddenLabel + "_Use SAC").c_str(), &(renderNode->settings.useNetwork)))
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
			const CudaEventTimes cuTimes = getCudaEventTimes();
			const int actualLaunches = getLaunches();
			float totTimeSeconds = (cuTimes.trace + cuTimes.noiseComputation + cuTimes.postProcessing + cuTimes.display)/1000.0f;
			float sppS = (renderNode->width * renderNode->height * actualLaunches) / totTimeSeconds;
			float averageFrameTime = totTimeSeconds/ actualLaunches;
			float fps = 1.0f / averageFrameTime;

			vtxImGui::HalfSpaceWidget("Total Time:", vtxImGui::booleanText, "%.3f s", totTimeSeconds);
			vtxImGui::HalfSpaceWidget("Samples Per Pixels:", vtxImGui::booleanText, "%d", renderNode->settings.iteration);
			vtxImGui::HalfSpaceWidget("SPP per Seconds:", vtxImGui::booleanText, "%.3f", sppS);
			vtxImGui::HalfSpaceWidget("Frame Time:", vtxImGui::booleanText, "%.3f ms", averageFrameTime);
			vtxImGui::HalfSpaceWidget("Fps:", vtxImGui::booleanText, "%.3f", fps);
			ImGui::Separator();

			float internalFps = ((1000.0f) * (float)(renderNode->internalIteration)) / renderNode->overallTime;
			float totTimeInternal = renderNode->overallTime / 1000.0f;
			vtxImGui::HalfSpaceWidget("CPU Fps:", vtxImGui::booleanText, "%.3f", internalFps);
			vtxImGui::HalfSpaceWidget("CPU Tot Time:", vtxImGui::booleanText, "%.3f", totTimeInternal);
			ImGui::Separator();

			const float factor = 1.0f / (float)actualLaunches;

			vtxImGui::HalfSpaceWidget("Renderer Noise				", vtxImGui::booleanText, "%.2f ms", factor* cuTimes.noiseComputation);
			vtxImGui::HalfSpaceWidget("Renderer Trace				", vtxImGui::booleanText, "%.2f ms", factor* cuTimes.trace);
			vtxImGui::HalfSpaceWidget("Renderer Post				", vtxImGui::booleanText, "%.2f ms", factor* cuTimes.postProcessing);
			vtxImGui::HalfSpaceWidget("Renderer Display				", vtxImGui::booleanText, "%.2f ms", factor* cuTimes.display);
			ImGui::Separator();
			vtxImGui::HalfSpaceWidget("WaveFront Generate Ray		", vtxImGui::booleanText, "%.2f ms", factor* cuTimes.genCameraRay);
			vtxImGui::HalfSpaceWidget("WaveFront Trace				", vtxImGui::booleanText, "%.2f ms", factor* cuTimes.traceRadianceRay);
			vtxImGui::HalfSpaceWidget("WaveFront Shade				", vtxImGui::booleanText, "%.2f ms", factor* cuTimes.shadeRay);
			vtxImGui::HalfSpaceWidget("WaveFront Escaped			", vtxImGui::booleanText, "%.2f ms", factor* cuTimes.handleEscapedRay);
			vtxImGui::HalfSpaceWidget("WaveFront Accumulate			", vtxImGui::booleanText, "%.2f ms", factor* cuTimes.accumulateRay);
			vtxImGui::HalfSpaceWidget("WaveFront Reset				", vtxImGui::booleanText, "%.2f ms", factor* cuTimes.reset);
			vtxImGui::HalfSpaceWidget("WaveFront Fetch Queue Size	", vtxImGui::booleanText, "%.2f ms", factor* cuTimes.fetchQueueSize);
			ImGui::Separator();
			vtxImGui::HalfSpaceWidget("Neural Shuffle Dataset		", vtxImGui::booleanText, "%.2f ms", factor* cuTimes.nnShuffleDataset);
			vtxImGui::HalfSpaceWidget("Neural Network Train			", vtxImGui::booleanText, "%.2f ms", factor * cuTimes.nnTrain);
			vtxImGui::HalfSpaceWidget("Neural Network Infer			", vtxImGui::booleanText, "%.2f ms", factor * cuTimes.nnInfer);

		}
        
		ImGui::PopItemWidth();
        ImGui::End();

		if (renderNode->settings.useNetwork)
		{
			neuralNetworkGui(renderNode->waveFrontIntegrator.network, renderNode);
		}
    }
}


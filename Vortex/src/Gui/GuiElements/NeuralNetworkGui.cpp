#include "NeuralNetworkGui.h"
#include "Core/CustomImGui/CustomImGui.h"
#include "NeuralNetworks/NeuralNetworkGraphs.h"
#include "PlottingWrapper.h"
#include "NeuralNetworks/NeuralNetwork.h"

namespace vtx::gui
{
	bool encodingSettingsEditorGui(network::EncodingSettings& settings, const std::string& encodedFeatureName)
	{
		if (ImGui::CollapsingHeader((encodedFeatureName).c_str()))
		{
			int encodingType = settings.type;
			ImGui::Text((encodedFeatureName + "Encoding:").c_str());
			if (vtxImGui::halfSpaceWidget("Encoding Type", (ComboFuncType)ImGui::Combo, (hiddenLabel + "_Input Encoding Type" + encodedFeatureName).c_str(), &encodingType, network::encodingNames, network::E_COUNT, -1))
			{
				settings.isUpdated = true;
				settings.type = static_cast<network::EncodingType>(encodingType);
			}

			if (vtxImGui::halfSpaceWidget("Feature Size", ImGui::DragInt, (hiddenLabel + "_Feature Size" + encodedFeatureName).c_str(), &settings.features, 1.0f, 1, 50, "%d", 0))
			{
				settings.isUpdated = true;
			}
		}
		return settings.isUpdated;
	}

	void encodingSettingsDisplay(network::EncodingSettings& settings, const std::string& encodedFeatureName)
	{
		if (ImGui::CollapsingHeader((encodedFeatureName + " :").c_str()))
		{
			vtxImGui::halfSpaceWidget("Encoding Type", vtxImGui::booleanText, "%s", network::encodingNames[settings.type]);
			vtxImGui::halfSpaceWidget("Feature Size", vtxImGui::booleanText, "%d", settings.features);
		}
	}

	bool networkInputSettingsEditor(network::InputSettings& settings)
	{
		if (ImGui::CollapsingHeader("Network Input Settings"))
		{
			ImGui::Indent();
			settings.isUpdated |= encodingSettingsEditorGui(settings.positionEncoding, "Position");
			settings.isUpdated |= encodingSettingsEditorGui(settings.woEncoding, "Outgoing Direction");
			settings.isUpdated |= encodingSettingsEditorGui(settings.normalEncoding, "Normal");
			ImGui::Unindent();
		}
		return settings.isUpdated;
	}

	void networkInputSettingsDisplay(network::InputSettings& settings)
	{
		if (ImGui::CollapsingHeader("Network Input Settings"))
		{
			encodingSettingsDisplay(settings.positionEncoding, "Position");
			encodingSettingsDisplay(settings.woEncoding, "Outgoing Direction");
			encodingSettingsDisplay(settings.normalEncoding, "Normal");
		}
	}

	bool pathGuidingNetworkSettingsEditor(network::PathGuidingNetworkSettings& settings)
	{
		if (ImGui::CollapsingHeader("Path Guiding Network Settings"))
		{
			if (vtxImGui::halfSpaceWidget("Hidden Dimension", ImGui::DragInt, (hiddenLabel + "_Hidden Dimension").c_str(), &(settings.hiddenDim), 1.0f, 1, 500, "%d", 0))
			{
				settings.isUpdated = true;
			}

			if (vtxImGui::halfSpaceWidget("Number of Hidden Layers", ImGui::DragInt, (hiddenLabel + "_Number of Hidden Layers").c_str(), &(settings.numHiddenLayers), 1.0f, 1, 10, "%d", 0))
			{
				settings.isUpdated = true;
			}

			int distributionType = settings.distributionType;
			if (vtxImGui::halfSpaceWidget("Distribution Type", (ComboFuncType)ImGui::Combo, (hiddenLabel + "_Distribution Type").c_str(), &distributionType, network::distributionNames, network::D_COUNT, -1))
			{
				settings.distributionType = (network::DistributionType)distributionType;
				settings.isUpdated = true;
			}


			if (vtxImGui::halfSpaceWidget("Mixture Size", ImGui::DragInt, (hiddenLabel + "_Mixture Size").c_str(), &(settings.mixtureSize), 1.0f, 1, 10, "%d", 0))
			{
				settings.isUpdated = true;
			}
		}

		return settings.isUpdated;
	}

	void pathGuidingNetworkSettingsDisplay(network::PathGuidingNetworkSettings& settings)
	{
		if (ImGui::CollapsingHeader("Path Guiding Network Settings"))
		{
			vtxImGui::halfSpaceWidget("Hidden Dimension", vtxImGui::booleanText, "%d", settings.hiddenDim);
			vtxImGui::halfSpaceWidget("Number of Hidden Layers", vtxImGui::booleanText, "%d", settings.numHiddenLayers);
			vtxImGui::halfSpaceWidget("Distribution Type", vtxImGui::booleanText, "%s", network::distributionNames[settings.distributionType]);
		}
	}

	bool sacSettingsEditorGui(network::SacSettings& settings)
	{
		if (ImGui::CollapsingHeader("Soft Actor Critic Settings"))
		{
			if (vtxImGui::halfSpaceWidget("Policy Lr", ImGui::DragFloat, (hiddenLabel + "_Policy Lr").c_str(), &settings.policyLr, 0.00001, 0, 1, "%.10f", 0))
			{
				settings.isUpdated = true;
			}

			if (vtxImGui::halfSpaceWidget("Q Lr", ImGui::DragFloat, (hiddenLabel + "_Q Lr").c_str(), &settings.qLr, 0.00001, 0, 1, "%.10f", 0))
			{
				settings.isUpdated = true;
			}

			if (vtxImGui::halfSpaceWidget("Alpha Lr", ImGui::DragFloat, (hiddenLabel + "_Alpha Lr").c_str(), &settings.alphaLr, 0.00001, 0, 1, "%.10f", 0))
			{
				settings.isUpdated = true;
			}

			if (vtxImGui::halfSpaceWidget("Gamma ", ImGui::DragFloat, (hiddenLabel + "_Learning Rate").c_str(), &settings.gamma, 0.00001, 0, 1, "%.6f", 0))
			{
				settings.isUpdated = true;
			}

			if (vtxImGui::halfSpaceWidget("Polyak Update Factor", ImGui::DragFloat, (hiddenLabel + "_Polyak Update Factor").c_str(), &settings.polyakFactor, 0.00001, 0, 1, "%.6f", 0))
			{
				settings.isUpdated = true;
			}

			if (vtxImGui::halfSpaceWidget("Neural Sampling Fraction", ImGui::DragFloat, (hiddenLabel + "_Neural Sampling Fraction").c_str(), &settings.neuralSampleFraction, 0.001, 0, 1, "%.3f", 0))
			{
				settings.isUpdated = true;
			}

			if (vtxImGui::halfSpaceWidget("Log Alpha Start", ImGui::DragFloat, (hiddenLabel + "_Log Alpha Start").c_str(), &settings.logAlphaStart, 0.001, 0, 1, "%.3f", 0))
			{
				settings.isUpdated = true;
			}
		}

		return settings.isUpdated;
	}

	void sacSettingsDisplayGui(network::SacSettings& settings)
	{
		if (ImGui::CollapsingHeader("Soft Actor Critic Settings"))
		{
			vtxImGui::halfSpaceWidget("Policy Lr", vtxImGui::booleanText, "%.5f", settings.policyLr);
			vtxImGui::halfSpaceWidget("Q Lr", vtxImGui::booleanText, "%.5f", settings.qLr);
			vtxImGui::halfSpaceWidget("Alpha Lr", vtxImGui::booleanText, "%.5f", settings.alphaLr);
			vtxImGui::halfSpaceWidget("Gamma ", vtxImGui::booleanText, "%.5f", settings.gamma);
			vtxImGui::halfSpaceWidget("Polyak Update Factor", vtxImGui::booleanText, "%.5f", settings.polyakFactor);
			vtxImGui::halfSpaceWidget("Neural Sampling Fraction", vtxImGui::booleanText, "%.5f", settings.neuralSampleFraction);
			vtxImGui::halfSpaceWidget("Log Alpha Start", vtxImGui::booleanText, "%.5f", settings.logAlphaStart);
		}
	}

	bool npgSettingsEditorGui(network::NpgSettings& settings)
	{
		if (ImGui::CollapsingHeader("Neural Path Guiding Settings"))
		{
			if (vtxImGui::halfSpaceWidget("Learning Rate", ImGui::DragFloat, (hiddenLabel + "_Learning Rate").c_str(), &settings.learningRate, 0.00001, 0, 1, "%.10f", 0))
			{
				settings.isUpdated = true;
			}

			if (vtxImGui::halfSpaceWidget("Loss Blending Factor", ImGui::DragFloat, (hiddenLabel + "_Loss Blending Factor").c_str(), &settings.e, 0.00001, 0, 1, "%.10f", 0))
			{
				settings.isUpdated = true;
			}

			if (vtxImGui::halfSpaceWidget("Constant Loss Blend Factor", ImGui::Checkbox, (hiddenLabel + "_Constant Loss Blend Factor").c_str(), &settings.constantBlendFactor))
			{
				settings.isUpdated = true;
			}

			if (vtxImGui::halfSpaceWidget("Sampling Fraction Blend", ImGui::Checkbox, (hiddenLabel + "_Sampling Fraction Blend").c_str(), &settings.samplingFractionBlend))
			{
				settings.isUpdated = true;
			}
			int lossType = settings.lossType;
			if (vtxImGui::halfSpaceWidget("Loss Type", (ComboFuncType)ImGui::Combo, (hiddenLabel + "_Loss Type").c_str(), &lossType, network::lossNames, network::L_COUNT, -1))
			{
				settings.lossType = (network::LossType)lossType;
				settings.isUpdated = true;
			}

			if (vtxImGui::halfSpaceWidget("Reduce to Mean", ImGui::Checkbox, (hiddenLabel + "_Reduce to Mean").c_str(), &settings.meanLoss))
			{
				settings.isUpdated = true;
			}


			if (vtxImGui::halfSpaceWidget("Take Absolute Loss", ImGui::Checkbox, (hiddenLabel + "_Take Abs Loss").c_str(), &settings.absoluteLoss))
			{
				settings.isUpdated = true;
			}

		}

		return settings.isUpdated;
	}

	void npgSettingsDisplayGui(network::NpgSettings& settings)
	{
		if (ImGui::CollapsingHeader("Neural Path Guiding Settings"))
		{
			vtxImGui::halfSpaceWidget("Learning Rate", vtxImGui::booleanText, "%.10f", settings.learningRate);
			vtxImGui::halfSpaceWidget("Blending Weight", vtxImGui::booleanText, "%.10f", settings.e);
		}
	}

	bool trainingBatchGenerationEditorGui(network::TrainingBatchGenerationSettings& settings)
	{
		if (ImGui::CollapsingHeader("Training Batch Generation Settings"))
		{
			if (vtxImGui::halfSpaceWidget("Light Sampling Prob", ImGui::DragFloat, (hiddenLabel + "_Light Sampling Prob").c_str(), &settings.lightSamplingProb, 0.00001, 0, 1, "%.10f", 0))
			{
				settings.isUpdated = true;
			}
			if (vtxImGui::halfSpaceWidget("Weight Contribution", ImGui::Checkbox, (hiddenLabel + "_Weight Contribution").c_str(), &settings.weightByMis))
			{
				settings.isUpdated = true;
			}
			int strategy = settings.strategy;
			if (vtxImGui::halfSpaceWidget("Sampling Strategy", (ComboFuncType)ImGui::Combo, (hiddenLabel + "_Sampling Strategy").c_str(), &strategy, network::samplingStrategyNames, network::SS_COUNT, -1))
			{
				settings.strategy = (network::SamplingStrategy)strategy;
				settings.isUpdated = true;
			}
		}
		
		return settings.isUpdated;
	}
	bool networkSettingsEditorGui(network::NetworkSettings& settings)
	{
		if (ImGui::CollapsingHeader("Network Settings"))
		{
			if (vtxImGui::halfSpaceWidget("Enable", ImGui::Checkbox, (hiddenLabel + "_Enable").c_str(), &settings.active))
			{
				settings.isUpdated = true;
			}
			if (vtxImGui::halfSpaceWidget("Reset Network", ImGui::Checkbox, (hiddenLabel + "_Reset Network").c_str(), &settings.isUpdated))
			{
				settings.isUpdated = true;
			}
			int type = settings.type;
			if (vtxImGui::halfSpaceWidget("Network Type", (ComboFuncType)ImGui::Combo, (hiddenLabel + "_Network Type").c_str(), &type, network::networkNames, network::NT_COUNT, -1))
			{
				settings.type = (network::NetworkType)type;
				settings.isUpdated = true;
			}
			if (vtxImGui::halfSpaceWidget("BatchSize", ImGui::DragInt, (hiddenLabel + "_BatchSize").c_str(), &settings.batchSize, 1, 1, 10000, "%d", 0))
			{
				if (settings.batchSize == 0)
				{
					settings.batchSize = 1;
				}
				settings.isDatasetSizeUpdated = true;
			}
			if (vtxImGui::halfSpaceWidget("Max Training Steps Per Frame", ImGui::DragInt, (hiddenLabel + "_BatchSize").c_str(), &settings.maxTrainingStepPerFrame, 1, 0, 10000, "%d", 0))
			{
				settings.isDatasetSizeUpdated = true;
			}

			if (vtxImGui::halfSpaceWidget("Do Training", ImGui::Checkbox, (hiddenLabel + "_Do Training").c_str(), &settings.doTraining))
			{
				settings.isUpdated = true;
			}
			if (vtxImGui::halfSpaceWidget("Max Training Steps", ImGui::DragInt, (hiddenLabel + "_Max Training Steps").c_str(), &settings.maxTrainingSteps, 1, 0, 10000, "%d", 0))
			{
				settings.isUpdated = true;
			}
			if (vtxImGui::halfSpaceWidget("Do Inference", ImGui::Checkbox, (hiddenLabel + "_Do Inference").c_str(), &settings.doInference))
			{
				settings.isUpdated = true;
			}
			if (vtxImGui::halfSpaceWidget("Inference Start", ImGui::DragInt, (hiddenLabel + "_Inference Start").c_str(), &settings.inferenceIterationStart, 1, 0, 10000, "%d", 0))
			{
				settings.isUpdated = true;
			}
			if (vtxImGui::halfSpaceWidget("Clear buffer on Inference Start", ImGui::Checkbox, (hiddenLabel + "_Inference Start").c_str(), &settings.clearOnInferenceStart))
			{
				settings.isUpdated = true;
			}
			if (vtxImGui::halfSpaceWidget("Depth To Debug", ImGui::DragInt, (hiddenLabel + "_Depth To Debug").c_str(), &settings.depthToDebug, 1, 0, 5, "%d", 0))
			{
				settings.isUpdated = true;
			}


			ImGui::Indent();
			settings.isUpdated |= trainingBatchGenerationEditorGui(settings.trainingBatchGenerationSettings);
			settings.isUpdated |= networkInputSettingsEditor(settings.inputSettings);
			settings.isUpdated |= pathGuidingNetworkSettingsEditor(settings.pathGuidingSettings);

			if (settings.type == network::NT_SAC)
			{
				settings.isUpdated |= sacSettingsEditorGui(settings.sac);
			}
			else if (settings.type == network::NT_NGP)
			{
				settings.isUpdated |= npgSettingsEditorGui(settings.npg);
			}
			ImGui::Unindent();

		}


		return settings.isUpdated;
	}

	void networkSettingsDisplayGui(network::NetworkSettings& settings)
	{
		if (ImGui::CollapsingHeader("Network Settings"))
		{
			vtxImGui::halfSpaceWidget("BatchSize", vtxImGui::booleanText, "%d", settings.batchSize);
			vtxImGui::halfSpaceWidget("Max Training Steps Per Frame", vtxImGui::booleanText, "%d", settings.maxTrainingStepPerFrame);
			vtxImGui::halfSpaceWidget("Do Inference", vtxImGui::booleanText, "%d", settings.doInference);
			vtxImGui::halfSpaceWidget("Inference Start", vtxImGui::booleanText, "%d", settings.inferenceIterationStart);
			vtxImGui::halfSpaceWidget("Clear buffer on Inference Start", vtxImGui::booleanText, "%d", settings.clearOnInferenceStart);
			networkInputSettingsDisplay(settings.inputSettings);
			pathGuidingNetworkSettingsDisplay(settings.pathGuidingSettings);
			if (settings.type == network::NT_SAC)
			{
				sacSettingsDisplayGui(settings.sac);
			}
			else if (settings.type == network::NT_NGP)
			{
				npgSettingsDisplayGui(settings.npg);
			}
		}
	}

	std::vector<PlotInfo> ngpPlots(network::GraphsData& graphsData, const network::DistributionType& dt)
	{
		PlotInfo loss;
		loss.title = "Loss";
		loss.xLabel = "Batches";
		loss.yLabel = "Loss";
		loss.addPlot(graphsData.graphs[network::G_NGP_T_LOSS], "Loss");
		loss.addPlot(graphsData.graphs[network::G_NGP_T_LOSS_Q], "Loss non Blended");
		loss.addPlot(graphsData.graphs[network::G_NGP_T_LOSS_BLENDED_Q], "Loss Blended");

		PlotInfo target;
		target.title = "Probabilities";
		target.xLabel = "Batches";
		target.yLabel = "Target";
		target.addPlot(graphsData.graphs[network::G_NGP_T_TARGET_P], "Target");
		target.addPlot(graphsData.graphs[network::G_NGP_T_NEURAL_P], "Neural Pdf");

		PlotInfo bsdf;
		bsdf.title = "Bsdf";
		bsdf.xLabel = "Batches";
		bsdf.yLabel = "Bsdf";
		bsdf.addPlot(graphsData.graphs[network::G_NGP_T_BSDF_P], "Bsdf Pdf");
		bsdf.addPlot(graphsData.graphs[network::G_NGP_T_BLENDED_P], "Blended Pdf Target");

		PlotInfo samplingFraction;
		samplingFraction.title = "Sampling Fraction";
		samplingFraction.xLabel = "Batches";
		samplingFraction.yLabel = "Sampling Fraction";
		samplingFraction.addPlot(graphsData.graphs[network::G_NGP_T_SAMPLING_FRACTION], "Sampling Fraction Train");
		samplingFraction.addPlot(graphsData.graphs[network::G_NGP_I_SAMPLING_FRACTION], "Sampling Fraction Inference");
		samplingFraction.addPlot(graphsData.graphs[network::G_NGP_TAU], "Tau");

		std::vector<PlotInfo> plots = { loss,target, samplingFraction, bsdf};

		if(dt == network::D_SPHERICAL_GAUSSIAN)
		{
			PlotInfo concentration;
			concentration.title = "Concentration";
			concentration.xLabel = "Batches";
			concentration.yLabel = "Concentration";
			concentration.addPlot(graphsData.graphs[network::G_SPHERICAL_GAUSSIAN_T_K], "Concentration Train");
			concentration.addPlot(graphsData.graphs[network::G_SPHERICAL_GAUSSIAN_I_K], "Concentration Inference");

			plots.push_back(concentration);
		}

		if(dt==network::D_NASG_ANGLE || dt == network::D_NASG_TRIG|| dt == network::D_NASG_AXIS_ANGLE)
		{
			PlotInfo lambda;
			lambda.title = "Lambda";
			lambda.xLabel = "Batches";
			lambda.yLabel = "Lambda";
			lambda.addPlot(graphsData.graphs[network::G_NASG_T_LAMBDA], "Lambda Train");
			lambda.addPlot(graphsData.graphs[network::G_NASG_I_LAMBDA], "Lambda Inference");

			PlotInfo a;
			a.title = "A";
			a.xLabel = "Batches";
			a.yLabel = "A";
			a.addPlot(graphsData.graphs[network::G_NASG_T_A], "Anisotropy Train");
			a.addPlot(graphsData.graphs[network::G_NASG_I_A], "Anisotropy Inference");

			plots.push_back(lambda);
			plots.push_back(a);
		}
		

		return plots;
	}

	std::vector<PlotInfo> sacPlots(network::GraphsData& graphsData)
	{
		PlotInfo q1q2Losses;
		q1q2Losses.title = "Q1Q2Losses";
		q1q2Losses.xLabel = "Batches";
		q1q2Losses.yLabel = "Loss";
		q1q2Losses.addPlot(graphsData.graphs[network::G_Q1_LOSS], "Q1 Loss");
		q1q2Losses.addPlot(graphsData.graphs[network::G_Q2_LOSS], "Q2 Loss");


		PlotInfo alphaLosses;
		alphaLosses.title = "AlphaLosses";
		alphaLosses.xLabel = "Batches";
		alphaLosses.yLabel = "Loss";
		alphaLosses.addPlot(graphsData.graphs[network::G_ALPHA_LOSS], "Alpha Loss");

		PlotInfo alphaValues;
		alphaValues.title = "AlphaValues";
		alphaValues.xLabel = "Batches";
		alphaValues.yLabel = "Alpha";
		alphaValues.addPlot(graphsData.graphs[network::G_ALPHA_VALUES], "Alpha Value");

		PlotInfo policyLosses;
		policyLosses.title = "PolicyLosses";
		policyLosses.xLabel = "Batches";
		policyLosses.yLabel = "Loss";
		policyLosses.addPlot(graphsData.graphs[network::G_POLICY_LOSS], "Policy Loss");

		PlotInfo rewards;
		rewards.title = "Replay buffer Rewards";
		rewards.xLabel = "Batches";
		rewards.yLabel = "Reward";
		rewards.addPlot(graphsData.graphs[network::G_DATASET_REWARDS], "Replay buffer Rewards");
		rewards.addPlot(graphsData.graphs[network::G_Q1_VALUES], "Q1");
		rewards.addPlot(graphsData.graphs[network::G_Q2_VALUES], "Q2");

		PlotInfo inferenceConcentration;
		inferenceConcentration.title = "Inference Concentration";
		inferenceConcentration.xLabel = "Batches";
		inferenceConcentration.yLabel = "Concentration";
		inferenceConcentration.addPlot(graphsData.graphs[network::G_INFERENCE_CONCENTRATION], "Inference Concentration");

		return { q1q2Losses, alphaLosses, alphaValues, policyLosses, rewards, inferenceConcentration };

	}
	std::vector<PlotInfo> neuralNetworkPlots(network::Network& network)
	{
		if (network.settings.type == network::NT_SAC)
		{
			return sacPlots(network.getGraphs());
		}
		if (network.settings.type == network::NT_NGP)
		{
			return ngpPlots(network.getGraphs(), network.settings.pathGuidingSettings.distributionType);
		}
		return {};
	}
}


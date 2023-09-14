#pragma once
#include "NeuralNetworks/NetworkSettings.h"
#include <yaml-cpp/yaml.h>

namespace YAML
{
	template<>
	struct convert<vtx::network::NetworkType>
	{
		static Node encode(const vtx::network::NetworkType& rhs)
		{
			return Node(vtx::network::networkNames[rhs]);
		}
		static bool decode(const Node& node, vtx::network::NetworkType& rhs)
		{
			const auto name = node.as<std::string>();
			for (int i = 0; i < vtx::network::NT_COUNT; i++) {
				if (name == vtx::network::networkNames[i]) {
					rhs = static_cast<vtx::network::NetworkType>(i);
					return true;
				}
			}
			return false;
		}
	};

	template<>
	struct convert<vtx::network::EncodingType>
	{
		static Node encode(const vtx::network::EncodingType& rhs)
		{
			return Node(vtx::network::encodingNames[rhs]);
		}
		static bool decode(const Node& node, vtx::network::EncodingType& rhs)
		{
			const auto name = node.as<std::string>();
			for (int i = 0; i < vtx::network::E_COUNT; i++)
			{
				if (name == vtx::network::encodingNames[i])
				{
					rhs = static_cast<vtx::network::EncodingType>(i);
					return true;
				}
			}
			return false;
		}
	};

	template<>
	struct convert<vtx::network::DistributionType>
	{
		static Node encode(const vtx::network::DistributionType& rhs)
		{
			return Node(vtx::network::distributionNames[rhs]);
		}

		static bool decode(const Node& node, vtx::network::DistributionType& rhs)
		{
			const auto name = node.as<std::string>();
			for (int i = 0; i < vtx::network::D_COUNT; i++)
			{
				if (name == vtx::network::distributionNames[i])
				{
					rhs = static_cast<vtx::network::DistributionType>(i);
					return true;
				}
			}
			return false;
		}
	};

	template<>
	struct convert<vtx::network::LossType>
	{
		static Node encode(const vtx::network::LossType& rhs)
		{
			return Node(vtx::network::lossNames[rhs]);
		}

		static bool decode(const Node& node, vtx::network::LossType& rhs)
		{
			const auto name = node.as<std::string>();
			for (int i = 0; i < vtx::network::L_COUNT; i++)
			{
				if (name == vtx::network::lossNames[i])
				{
					rhs = static_cast<vtx::network::LossType>(i);
					return true;
				}
			}
			return false;
		}
	};


	template<>
	struct convert<vtx::network::SamplingStrategy>
	{
		static Node encode(const vtx::network::SamplingStrategy& rhs)
		{
			return Node(vtx::network::samplingStrategyNames[rhs]);
		}

		static bool decode(const Node& node, vtx::network::SamplingStrategy& rhs)
		{
			const auto name = node.as<std::string>();
			for (int i = 0; i < vtx::network::SS_COUNT; i++)
			{
				if (name == vtx::network::samplingStrategyNames[i])
				{
					rhs = static_cast<vtx::network::SamplingStrategy>(i);
					return true;
				}
			}
			return false;
		}
	};

	template<>
	struct convert<vtx::network::EncodingSettings>
	{
		static Node encode(const vtx::network::EncodingSettings& rhs)
		{
			Node node;
			node["type"] = rhs.type;
			node["features"] = rhs.features;
			return node;
		}

		static bool decode(const Node& node, vtx::network::EncodingSettings& rhs)
		{
			if (!node.IsMap()) {
				return false;
			}
			rhs.type = node["type"].as<vtx::network::EncodingType>();
			rhs.features = node["features"].as<int>();
			rhs.isUpdated = true;
			return true;
		}
	};

	template<>
	struct convert<vtx::network::InputSettings>
	{
		static Node encode(const vtx::network::InputSettings& rhs)
		{
			Node node;
			node["positionEncoding"] = rhs.positionEncoding;
			node["woEncoding"] = rhs.woEncoding;
			node["normalEncoding"] = rhs.normalEncoding;
			return node;
		}
		static bool decode(const Node& node, vtx::network::InputSettings& rhs)
		{
			if (!node.IsMap()) {
				return false;
			}
			rhs.positionEncoding = node["positionEncoding"].as<vtx::network::EncodingSettings>();
			rhs.woEncoding = node["woEncoding"].as<vtx::network::EncodingSettings>();
			rhs.normalEncoding = node["normalEncoding"].as<vtx::network::EncodingSettings>();
			rhs.isUpdated = true;
			return true;
		}
	};


	template<>
	struct convert<vtx::network::PathGuidingNetworkSettings>
	{
		static Node encode(const vtx::network::PathGuidingNetworkSettings& rhs)
		{
			Node node;
			node["hiddenDim"] = rhs.hiddenDim;
			node["numHiddenLayers"] = rhs.numHiddenLayers;
			node["distributionType"] = rhs.distributionType;
			node["produceSamplingFraction"] = rhs.produceSamplingFraction;
			node["mixtureSize"] = rhs.mixtureSize;
			
			return node;
		}
		static bool decode(const Node& node, vtx::network::PathGuidingNetworkSettings& rhs)
		{
			if (!node.IsMap()) {
				return false;
			}
			rhs.hiddenDim = node["hiddenDim"].as<int>();
			rhs.numHiddenLayers = node["numHiddenLayers"].as<int>();
			rhs.distributionType = node["distributionType"].as<vtx::network::DistributionType>();
			rhs.produceSamplingFraction = node["produceSamplingFraction"].as<bool>();\
			rhs.mixtureSize = node["mixtureSize"].as<int>();
			rhs.isUpdated = true;
			return true;
		}
	};

	template<>
	struct convert<vtx::network::SacSettings>
	{
		static Node encode(const vtx::network::SacSettings& rhs)
		{
			Node node;
			node["polyakFactor"] = rhs.polyakFactor;
			node["logAlphaStart"] = rhs.logAlphaStart;
			node["gamma"] = rhs.gamma;
			node["neuralSampleFraction"] = rhs.neuralSampleFraction;
			node["policyLr"] = rhs.policyLr;
			node["qLr"] = rhs.qLr;
			node["alphaLr"] = rhs.alphaLr;
			return node;
		}
		static bool decode(const Node& node, vtx::network::SacSettings& rhs)
		{
			if (!node.IsMap()) {
				return false;
			}
			rhs.polyakFactor = node["polyakFactor"].as<float>();
			rhs.logAlphaStart = node["logAlphaStart"].as<float>();
			rhs.gamma = node["gamma"].as<float>();
			rhs.neuralSampleFraction = node["neuralSampleFraction"].as<float>();
			rhs.policyLr = node["policyLr"].as<float>();
			rhs.qLr = node["qLr"].as<float>();
			rhs.alphaLr = node["alphaLr"].as<float>();
			rhs.isUpdated = true;
			return true;
		}
	};


	template<>
	struct convert<vtx::network::TrainingBatchGenerationSettings>
	{
		static Node encode(const vtx::network::TrainingBatchGenerationSettings& rhs)
		{
			Node node;
			node["weightByMis"] = rhs.weightByMis;
			node["strategy"] = rhs.strategy;
			node["lightSamplingProb"] = rhs.lightSamplingProb;
			return node;
		}
		static bool decode(const Node& node, vtx::network::TrainingBatchGenerationSettings& rhs)
		{
			if (!node.IsMap()) {
				return false;
			}
			rhs.weightByMis = node["weightByMis"].as<float>();
			rhs.lightSamplingProb = node["lightSamplingProb"].as<float>();
			rhs.strategy = node["strategy"].as<vtx::network::SamplingStrategy>();
			rhs.isUpdated = true;
			return true;
		}
	};

	template<>
	struct convert<vtx::network::NpgSettings>
	{
		static Node encode(const vtx::network::NpgSettings& rhs)
		{
			Node node;
			node["learningRate"] = rhs.learningRate;
			node["e"] = rhs.e;
			node["constantBlendFactor"] = rhs.constantBlendFactor;
			node["samplingFractionBlend"] = rhs.samplingFractionBlend;
			node["lossType"] = rhs.lossType;
			return node;
		}
		static bool decode(const Node& node, vtx::network::NpgSettings& rhs)
		{
			if (!node.IsMap()) {
				return false;
			}
			rhs.learningRate = node["learningRate"].as<float>();
			rhs.e = node["e"].as<float>();
			rhs.constantBlendFactor = node["constantBlendFactor"].as<float>();
			rhs.samplingFractionBlend = node["samplingFractionBlend"].as<float>();
			rhs.lossType = node["lossType"].as<vtx::network::LossType>();
			rhs.isUpdated = true;
			return true;
		}
	};

	template<>
	struct convert<vtx::network::NetworkSettings>
	{
		static Node encode(const vtx::network::NetworkSettings& rhs)
		{
			Node node;
			node["active"] = rhs.active;
			node["batchSize"] = rhs.batchSize;
			node["maxTrainingStepPerFrame"] = rhs.maxTrainingStepPerFrame;
			node["doTraining"] = rhs.doTraining;
			node["maxTrainingSteps"] = rhs.maxTrainingSteps;
			node["doInference"] = rhs.doInference;
			node["inferenceIterationStart"] = rhs.inferenceIterationStart;
			node["clearOnInferenceStart"] = rhs.clearOnInferenceStart;
			node["type"] = rhs.type;
			node["trainingBatchGenerationSettings"] = rhs.trainingBatchGenerationSettings;
			node["inputSettings"] = rhs.inputSettings;
			node["pathGuidingSettings"] = rhs.pathGuidingSettings;
			node["sac"] = rhs.sac;
			node["npg"] = rhs.npg;
			return node;
		}
		static bool decode(const Node& node, vtx::network::NetworkSettings& rhs)
		{
			if (!node.IsMap()) {
				return false;
			}
			rhs.active = node["active"].as<bool>();
			rhs.batchSize = node["batchSize"].as<int>();
			rhs.maxTrainingStepPerFrame = node["maxTrainingStepPerFrame"].as<int>();
			rhs.doTraining = node["doTraining"].as<bool>();
			rhs.maxTrainingSteps = node["maxTrainingSteps"].as<int>();
			rhs.doInference = node["doInference"].as<bool>();
			rhs.inferenceIterationStart = node["inferenceIterationStart"].as<int>();
			rhs.clearOnInferenceStart = node["clearOnInferenceStart"].as<bool>();
			rhs.type = node["type"].as<vtx::network::NetworkType>();
			rhs.trainingBatchGenerationSettings = node["trainingBatchGenerationSettings"].as<vtx::network::TrainingBatchGenerationSettings>();
			rhs.inputSettings = node["inputSettings"].as<vtx::network::InputSettings>();
			rhs.pathGuidingSettings = node["pathGuidingSettings"].as<vtx::network::PathGuidingNetworkSettings>();
			rhs.sac = node["sac"].as<vtx::network::SacSettings>();
			rhs.npg = node["npg"].as<vtx::network::NpgSettings>();
			rhs.isDatasetSizeUpdated = true;
			rhs.isUpdated = true;
			return true;
		}
	};

}

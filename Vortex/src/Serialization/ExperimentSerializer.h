#pragma once
#include <yaml-cpp/yaml.h>
#include "NeuralNetworks/Experiment.h"
#include "Serialization/RenderSettingsSerializer.h"
#include "Serialization/NetworkSettingsSerializer.h"

namespace YAML
{

	template<>
	struct convert<vtx::ExperimentStages> {
		static Node encode(const vtx::ExperimentStages& rhs) {
			return Node(vtx::experimentStageNames[rhs]);
		}

		static bool decode(const Node& node, vtx::ExperimentStages& rhs) {
			const auto name = node.as<std::string>();
			for (int i = 0; i < vtx::STAGE_END; i++) {
				if (name == vtx::experimentStageNames[(vtx::ExperimentStages)i]) {
					rhs = static_cast<vtx::ExperimentStages>(i);
					return true;
				}
			}
			return false;
		}
	};
}
namespace vtx::serializer
{
	class ExperimentSerializer {

	public:
		static YAML::Node encode(const vtx::Experiment& rhs, const std::string& filePath) {
			YAML::Node node;
			node["name"] = rhs.name;
			utl::binaryDump((void*)rhs.mape.data(), rhs.mape.size() * sizeof(float), filePath);
			node["mape"] = filePath;
			node["mapeSize"] = rhs.mape.size();
			node["rendererSettings"] = rhs.rendererSettings;
			node["networkSettings"] = rhs.networkSettings;
			node["wavefrontSettings"] = rhs.wavefrontSettings;

			return node;
		}
		static bool decode(const YAML::Node& node, vtx::Experiment& rhs) {
			if (!node.IsMap()) {
				return false;
			}
			rhs.name = node["name"].as<std::string>();
			const auto mapeSize = node["mapeSize"].as<size_t>();
			const auto mapeBinOutput = node["mape"].as<std::string>();
			rhs.rendererSettings = node["rendererSettings"].as<vtx::RendererSettings>();
			rhs.networkSettings = node["networkSettings"].as<vtx::network::NetworkSettings>();
			rhs.wavefrontSettings = node["wavefrontSettings"].as<vtx::WavefrontSettings>();
			rhs.mape                 = utl::binaryLoad<float>(mapeSize, mapeBinOutput);
			rhs.storeExperiment = true;

			return true;
		}
	};

	class ExperimentsManagerSerializer {

	public:
		static YAML::Node encode(ExperimentsManager& manager, const std::string& folderPath) {
			YAML::Node node;
			YAML::Node experimentsNode;
			for (int i = 0; i < manager.experiments.size(); ++i) {
				if (manager.experiments[i].storeExperiment)
				{
					const auto& experiment = manager.experiments[i];
					std::string mapeBinPath = folderPath + "/experiment_" + std::to_string(i) + "_mape.bin";
					experimentsNode.push_back(ExperimentSerializer::encode(experiment, mapeBinPath));
				}
			}
			node["experiments"] = experimentsNode;
			node["currentExperiment"] = manager.currentExperiment;
			node["currentExperimentStep"] = manager.currentExperimentStep;
			node["width"] = manager.width;
			node["height"] = manager.height;
			node["isGroundTruthReady"] = manager.isGroundTruthReady;
			node["maxSamples"] = manager.maxSamples;

			// Save ground truth image
			if (manager.isGroundTruthReady)
			{
				const std::string groundTruthPath = folderPath + "/ground_truth.hdr";
				manager.saveGroundTruth(groundTruthPath);
				node["groundTruthImage"] = groundTruthPath;
			}
			else
			{
				node["groundTruthImage"] = "";
			}

			return node;
		}

		static bool decode(const YAML::Node& node, ExperimentsManager& manager) {
			if (!node.IsMap()) {
				return false;
			}
			YAML::Node experimentsNode = node["experiments"];
			for (const auto& experimentNode : experimentsNode) {
				Experiment experiment;
				ExperimentSerializer::decode(experimentNode, experiment);
				manager.experiments.push_back(experiment);
			}
			manager.currentExperiment = node["currentExperiment"].as<int>();
			manager.currentExperimentStep = node["currentExperimentStep"].as<int>();
			manager.width = node["width"].as<int>();
			manager.height = node["height"].as<int>();
			manager.stage = STAGE_NONE;
			manager.isGroundTruthReady = node["isGroundTruthReady"].as<bool>();
			manager.maxSamples = node["maxSamples"].as<int>();

			// Load ground truth image
			if (manager.isGroundTruthReady && node["isGroundTruthReady"])
			{
				const auto imagePath = node["groundTruthImage"].as<std::string>();
				manager.loadGroundTruth(imagePath);
			}

			return true;
		}
	};
}

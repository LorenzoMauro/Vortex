#pragma once
#include <map>
#include <memory>

#include "NetworkSettings.h"
#include "Core/Image.h"
#include "Core/Math.h"
#include "Device/UploadCode/CUDABuffer.h"
#include "Scene/Nodes/RendererSettings.h"
#include "yaml-cpp/node/node.h"

namespace vtx
{
	
}

namespace vtx
{
	namespace graph
	{
		class Renderer;
	}

	namespace network
	{
		class Network;
	}
}


namespace vtx
{
	enum ExperimentStages
	{
		STAGE_NONE,
		STAGE_REFERENCE_GENERATION,
		STAGE_MAPE_COMPUTATION,

		STAGE_END
	};

	static std::map<ExperimentStages, std::string> experimentStageNames =
	{
		{ STAGE_NONE, "Not performing Any Experiment"},
		{ STAGE_REFERENCE_GENERATION, "Generating Ground truth" },
		{ STAGE_MAPE_COMPUTATION, "Computing Mape" },
		{ STAGE_END, "End" }
	};

	struct Experiment
	{
		RendererSettings         rendererSettings;
		network::NetworkSettings networkSettings;
		WavefrontSettings        wavefrontSettings;
		std::vector<float>       mape;
		std::string              name = "Unnamed";

		void constructName(const int experimentNumber)
		{
			name = "Experiment_" + std::to_string(experimentNumber);
		}

		bool storeExperiment = false;
		bool displayExperiment = true;
	};

	class ExperimentsManager
	{
	public:

		void loadGroundTruth(const std::string& filePath)
		{
			Image image;
			image.load(filePath);
			float* hostImage = image.getData();
			width = image.getWidth();
			height = image.getHeight();

			groundTruthBuffer.resize(image.getWidth() * image.getHeight() * image.getChannels() * sizeof(float));
			groundTruthBuffer.upload(hostImage, image.getWidth() * image.getHeight() * image.getChannels());
			groundTruth = groundTruthBuffer.castedPointer<math::vec3f>();

			isGroundTruthReady = true;
		};

		void saveGroundTruth(const std::string& filePath)
		{
			std::vector<math::vec3f> hostImage(width * height);
			math::vec3f* hostImagePtr = hostImage.data();
			groundTruthBuffer.download(hostImagePtr);
			Image image;
			image.load((float*)hostImagePtr, width, height, 3);
			image.save(filePath);
		}

	public:
		std::vector<Experiment> experiments;
		int currentExperiment = 0;
		int currentExperimentStep = 0;

		int width = 800;
		int height = 800;

		ExperimentStages stage = STAGE_NONE;

		bool isGroundTruthReady = false;
		math::vec3f* groundTruth = nullptr;
		CUDABuffer groundTruthBuffer;

		int maxSamples = 1000;

	};

}

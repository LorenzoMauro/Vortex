#pragma once
#include <memory>

#include "Experiment.h"
#include "NetworkSettings.h"

namespace vtx::network
{
	class NetworkImplementation;

	struct GraphsData;

	class Network
    {
    public:
		Network();
        ~Network();

		void initNetworks();
        void train();
        void inference(const int& depth);
		void reset();

		bool doInference();
		NetworkSettings& getNeuralNetSettings();
		GraphsData& getGraphs();
		std::unique_ptr<NetworkImplementation> impl;
		NetworkSettings settings;
		ExperimentsManager experimentManager;
		bool isInitialized = false;
    };
}

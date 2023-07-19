#pragma once
#include <memory>
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
        void inference();
		void reset();

		bool doInference();
		NetworkSettings& getNeuralNetSettings();
		GraphsData& getGraphs();
		std::unique_ptr<NetworkImplementation> impl;
		NetworkType type;
    };
}

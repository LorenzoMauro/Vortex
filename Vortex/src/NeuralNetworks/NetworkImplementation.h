#pragma once
#ifndef NETWORK_IMPLEMENTATION_H
#define NETWORK_IMPLEMENTATION_H
#include "NetworkSettings.h"
#include "NeuralNetworkGraphs.h"

namespace vtx
{
	struct LaunchParams;
}

namespace vtx::network
{
	class NetworkImplementation
	{
	public:
		virtual void init() = 0;
		virtual void train() = 0;
		virtual void inference(const int& depth) = 0;
		virtual void reset() = 0;
		virtual GraphsData& getGraphs() = 0;

		void shuffleDataset(LaunchParams* params) const;

		NetworkSettings* settings;
	};

}

#endif
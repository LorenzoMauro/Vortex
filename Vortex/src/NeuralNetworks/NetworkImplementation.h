#pragma once
#ifndef NETWORK_IMPLEMENTATION_H
#define NETWORK_IMPLEMENTATION_H
#include "NetworkSettings.h"

namespace vtx::network
{
	class NetworkImplementation
	{
	public:
		virtual void init() = 0;
		virtual void train() = 0;
		virtual void inference() = 0;
		virtual void reset() = 0;
		virtual GraphsData& getGraphs() = 0;
		virtual NetworkSettings& getSettings() = 0;
	};

}

#endif
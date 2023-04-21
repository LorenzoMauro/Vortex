﻿#pragma once
#include <mi/base/types.h>

#include "Scene/Node.h"

namespace vtx::graph {
	class LightProfile : public Node
	{
	public:
		LightProfile() :
			Node(NT_MDL_LIGHTPROFILE)
		{
		}

		LightProfile(char const* _databaseName) :
			Node(NT_MDL_LIGHTPROFILE),
			databaseName(_databaseName)
		{
		}

		struct LightProfileData{
			math::vec2ui resolution;
			math::vec2f start;
			math::vec2f delta;
			const float* sourceData;
			std::vector<float> cdfData;
			size_t cdfDataSize;
			double candelaMultiplier;
			float totalPower;
		};

		void init();
		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;

		void accept(std::shared_ptr<NodeVisitor> visitor) override;

		void prepareSampling();
	public:
		std::string databaseName;
		LightProfileData lightProfileData;
		bool isInitialized = false;
		mi::Size mdlIndex;
	};
}

class LightProfile
{
public:
	
};

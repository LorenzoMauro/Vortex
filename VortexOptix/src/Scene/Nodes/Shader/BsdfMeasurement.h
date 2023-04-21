#pragma once
#include <mi/base/types.h>

#include "Scene/Node.h"

namespace vtx::graph
{
	class BsdfMeasurement : public Node
	{
	public:
		BsdfMeasurement() :
			Node(NT_MDL_BSDF)
		{
		}

		BsdfMeasurement(char const* _databaseName) :
			Node(NT_MDL_BSDF),
			databaseName(_databaseName)
		{
		}

		struct BsdfPartData
		{
			math::vec2ui angularResolution;
			unsigned int numChannels{};
			const float* srcData = nullptr;
			std::vector<float> sampleData;
			std::vector<float> albedoData;
			std::vector<float> lookupData;
			float maxAlbedo;
			bool isValid = false;
		};

		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;

		void accept(std::shared_ptr<NodeVisitor> visitor) override;

		void prepareSampling(BsdfPartData bsdfData);

		void init();
	public:
		std::string databaseName;
		BsdfPartData reflectionBsdf;
		BsdfPartData transmissionBsdf;
		bool isInitialized = false;
		mi::Size mdlIndex;
	};
}

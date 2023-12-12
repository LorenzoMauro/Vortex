#pragma once
#include <string>

namespace vtx
{
	class ExperimentsManager;
}

namespace vtx::serializer
{
	bool deserialize(const std::string& filePath, bool importScene = false, bool skipExperimentManager=false);

	void serialize(const std::string& filePath);

	void serializeBatchExperiments(const std::string& filePath);

	bool deserializeExperimentManager(const std::string& filePath);
}

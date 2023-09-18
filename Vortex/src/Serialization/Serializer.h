#pragma once
#include <memory>
#include <string>

namespace vtx
{
	class ExperimentsManager;
}

namespace vtx
{
	namespace graph
	{
		class Scene;
		class Group;
		class Renderer;
	}
}

namespace vtx::serializer
{
	bool deserialize(const std::string& filePath, const std::shared_ptr<graph::Scene>& scene);

	bool experimentDeserializer(const std::string& filePath, const std::shared_ptr<graph::Scene>& scene);

	bool serializeExperimentManger(const std::string& filePath, ExperimentsManager& em);

	ExperimentsManager deserializeExperimentManager(const std::string& filePath);

	void serialize(const std::string& filePath);

	std::string getPreviousModelPath();

	std::string getPreviousHDRIPath();

}

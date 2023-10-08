#pragma once
#include <memory>
#include <vector>

#include "Gui/PlottingWrapper.h"

namespace vtx
{
	namespace network
	{
		class Network;
		struct NetworkSettings;
		struct TrainingBatchGenerationSettings;
		struct NpgSettings;
		struct SacSettings;
		struct PathGuidingNetworkSettings;
		struct InputSettings;
		struct EncodingSettings;
	}
}

namespace vtx
{
	namespace graph
	{
		namespace shader
		{
			struct ShaderNodeSocket;
			class ShaderNode;
		}

		class Material;
		class Transform;
		class Camera;
		class Renderer;
		class Instance;
		class Group;
		class Mesh;
	}
}

namespace vtx::gui
{
	class GuiProvider
	{
	public:
		static bool drawEditGui(const std::shared_ptr<graph::Renderer>& renderer);
		static bool drawEditGui(const std::shared_ptr<graph::Camera>& camera);
		static bool drawEditGui(const std::shared_ptr<graph::Transform>& transform);
		static bool drawEditGui(const std::shared_ptr<graph::Material>& material);
		static bool drawEditGui(const std::shared_ptr<graph::Instance>& instance);
		static bool drawEditGui(const std::shared_ptr<graph::Group>& group);
		static bool drawEditGui(const std::shared_ptr<graph::Mesh>& mesh);

		static bool drawEditGui(const std::shared_ptr<graph::shader::ShaderNode>& shaderNode, const bool isNodeEditor = false);
		static bool drawEditGui(const graph::shader::ShaderNodeSocket& socket);
		static bool drawEditGui(network::EncodingSettings& settings, const std::string& encodedFeatureName);
		static bool drawEditGui(network::InputSettings& settings);
		static bool drawEditGui(network::PathGuidingNetworkSettings& settings);
		static bool drawEditGui(network::SacSettings& settings);
		static bool drawEditGui(network::NpgSettings& settings);
		static bool drawEditGui(network::TrainingBatchGenerationSettings& settings);
		static bool drawEditGui(network::NetworkSettings& settings);

		static void drawDisplayGui(network::EncodingSettings& settings, const std::string& encodedFeatureName);
		static void drawDisplayGui(network::InputSettings& settings);
		static void drawDisplayGui(network::PathGuidingNetworkSettings& settings);
		static void drawDisplayGui(network::SacSettings& settings);
		static void drawDisplayGui(network::NpgSettings& settings);
		static void drawDisplayGui(network::NetworkSettings& settings);

		static std::vector<PlotInfo> getPlots(network::Network& network);

	};
}

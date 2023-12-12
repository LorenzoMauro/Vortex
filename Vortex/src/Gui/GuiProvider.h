#pragma once
#include <memory>
#include <vector>
#include "Gui/PlottingWrapper.h"
#include "NeuralNetworks/Config/NetworkSettings.h"

namespace vtx
{
	namespace network
	{
		namespace config
		{
			struct NetworkSettings;
			struct BatchGenerationConfig;
			struct EncodingConfig;
			struct TriangleWaveEncoding;
			struct SphericalHarmonicsEncoding;
			struct OneBlobEncoding;
			struct IdentityEncoding;
			struct GridEncoding;
			struct FrequencyEncoding;
		}

		class Network;
	}

	struct Experiment;
	struct RendererSettings;

	namespace graph
	{
		namespace shader
		{
			struct ShaderNodeSocket;
			class ShaderNode;
		}

		struct Statistics;
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

		static bool                  drawEditGui(network::config::FrequencyEncoding& config);
		static bool                  drawEditGui(network::config::GridEncoding& config);
		static bool                  drawEditGui(network::config::IdentityEncoding& config);
		static bool                  drawEditGui(network::config::OneBlobEncoding& config);
		static bool                  drawEditGui(network::config::SphericalHarmonicsEncoding& config);
		static bool                  drawEditGui(network::config::TriangleWaveEncoding& config);
		static bool                  drawEditGui(const std::string& featureName, network::config::EncodingConfig& config);
		static bool                  drawEditGui(network::config::BatchGenerationConfig& settings);
		static bool                  drawEditGui(network::config::MlpSettings& config);
		static bool                  drawEditGui(network::config::MainNetEncodingConfig& config);
		static bool					 drawEditGui(network::config::AuxNetEncodingConfig& config);
		static bool                  drawEditGui(network::config::NetworkSettings& settings);
		static std::vector<PlotInfo> getPlots(network::Network& network);

		static bool drawEditGui(Experiment& experiment);

		static bool drawEditGui(RendererSettings& settings);
		static void drawDisplayGui(graph::Statistics& settings);



	};
}

#pragma once
#include "Scene/Node.h"
#include "Device/DevicePrograms/LaunchParams.h"

namespace vtx::graph
{

	class EnvironmentLight : public Node
	{
	public:
		EnvironmentLight();

		~EnvironmentLight() override;

		void init() override;

		std::vector<std::shared_ptr<Node>> getChildren() const override;

	protected:

		void accept(NodeVisitor& visitor) override;
	private:

		void computeSphericalCdf();

		void computeCdfAliasMaps();

	public:
		std::vector<float> cdfU;
		std::vector<float> cdfV;
		std::shared_ptr<graph::Texture>					envTexture;
		std::shared_ptr<graph::Transform>				transform;
		std::vector<AliasData>							aliasMap;
		std::vector<float>								importanceData;

		float invIntegral;
		float scaleLuminosity = 1.0f;
		bool								isValid = false;

	};
}
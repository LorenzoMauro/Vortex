#pragma once
#include "NodeEditorWrapper.h"
#include "Scene/Traversal.h"


namespace vtx::gui
{

	class SceneGraphVisitor : public NodeVisitor {
	public:

		SceneGraphVisitor()
		{
			collectWidthsAndDepths = true;
		}
		void visit(std::shared_ptr<graph::Instance> node) override;

		void visit(std::shared_ptr<graph::Transform> node) override;

		void visit(std::shared_ptr<graph::Group> node) override;

		void visit(std::shared_ptr<graph::Mesh> node) override;

		void visit(std::shared_ptr<graph::Material> node) override;

		void visit(std::shared_ptr<graph::Camera> node) override;

		void visit(std::shared_ptr<graph::Renderer> node) override;

		void visit(std::shared_ptr<graph::Texture> node) override;

		void visit(std::shared_ptr<graph::BsdfMeasurement> node) override;

		void visit(std::shared_ptr<graph::LightProfile> node) override;

		void visit(std::shared_ptr<graph::EnvironmentLight> node) override;

		void visit(std::shared_ptr<graph::MeshLight> node) override;

		void visit(std::shared_ptr<graph::shader::DiffuseReflection> shaderNode) override;
		void visit(std::shared_ptr<graph::shader::MaterialSurface> shaderNode)override;
		void visit(std::shared_ptr<graph::shader::Material> shaderNode) override;
		void visit(std::shared_ptr<graph::shader::ImportedNode> shaderNode) override;
		void visit(std::shared_ptr<graph::shader::PrincipledMaterial> shaderNode) override;
		void visit(std::shared_ptr<graph::shader::ColorTexture> shaderNode) override;
		void visit(std::shared_ptr<graph::shader::MonoTexture> shaderNode) override;
		void visit(std::shared_ptr<graph::shader::NormalTexture> shaderNode) override;
		void visit(std::shared_ptr<graph::shader::BumpTexture> shaderNode) override;
		void visit(std::shared_ptr<graph::shader::TextureTransform> shaderNode) override;
		void visit(std::shared_ptr<graph::shader::NormalMix> shaderNode) override;
		void visit(std::shared_ptr<graph::shader::GetChannel> shaderNode) override;

		NodeGraphUi nodeGraphUi;
	};

	class SceneGraphGui
	{
	public:
		static std::vector<int> draw(const std::shared_ptr<graph::Renderer>& renderer);
	};


}

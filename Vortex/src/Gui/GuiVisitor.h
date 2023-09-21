#pragma once
#include "Scene/Traversal.h"

namespace vtx::gui
{
	class GuiVisitor: public NodeVisitor
	{
		void callOnChildren(const std::shared_ptr<graph::Node>& node);
		void visit(const std::shared_ptr<graph::Instance>& node) override;
		void visit(const std::shared_ptr<graph::Transform>& node) override;
		void visit(const std::shared_ptr<graph::Group>& node) override;
		void visit(const std::shared_ptr<graph::Mesh>& node) override;
		void visit(const std::shared_ptr<graph::Material>& node) override;
		void visit(const std::shared_ptr<graph::Camera>& node) override;
		void visit(const std::shared_ptr<graph::Renderer>& node) override;
		void visit(const std::shared_ptr<graph::Texture>& node) override;
		void visit(const std::shared_ptr<graph::BsdfMeasurement>& node) override;
		void visit(const std::shared_ptr<graph::LightProfile>& node) override;
		void visit(const std::shared_ptr<graph::EnvironmentLight>& node) override;
		void visit(const std::shared_ptr<graph::MeshLight>& node) override;
		void visit(const std::shared_ptr<graph::shader::ShaderNode>& shaderNode) override;

	public:
		bool isNodeEditor = false;
		bool  changed = false;
	};
}

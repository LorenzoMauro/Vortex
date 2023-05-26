#pragma once
#include "Scene/Traversal.h"

namespace vtx::mdl
{
	
	class ShaderVisitor : public NodeVisitor {
	public:
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
	};
}


#pragma once
#include "ShaderOperations.h"
#include "Scene/Traversal.h"

namespace vtx::mdl
{
	
	class ShaderVisitor : public NodeVisitor {
	public:
		void visit(std::shared_ptr<graph::shader::TextureFile> shaderTexture) override;
		void visit(std::shared_ptr<graph::shader::TextureReturn> shaderTexture) override;
		void visit(std::shared_ptr<graph::shader::DiffuseReflection> shaderDiffuseReflection) override;
		void visit(std::shared_ptr<graph::shader::MaterialSurface> materialSurfaceNode)override;
		void visit(std::shared_ptr<graph::shader::Material> materialNode) override;
		void visit(std::shared_ptr<graph::shader::ImportedNode> importedNode) override;

	};
}


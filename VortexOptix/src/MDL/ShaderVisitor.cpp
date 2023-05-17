#include "ShaderVisitor.h"

#include "Scene/Nodes/Shader/mdl/ShaderNodes.h"


namespace vtx::mdl
{
	using namespace graph::shader;


		void ShaderVisitor::visit(std::shared_ptr<TextureFile> textureFileNode)
		{
			textureFileNode->sockets["texture"].parameterInfo.defaultValue = mdl::createTextureConstant(textureFileNode->path);
			textureFileNode->expression = mdl::generateFunctionExpression(textureFileNode->functionInfo.signature, textureFileNode->sockets);
		}

		void ShaderVisitor::visit(std::shared_ptr<TextureReturn> textureReturnNode)
		{
			textureReturnNode->expression = mdl::generateFunctionExpression(textureReturnNode->functionInfo.signature, textureReturnNode->sockets);
		}

		void ShaderVisitor::visit(std::shared_ptr<DiffuseReflection> shaderDiffuseReflection)
		{
			shaderDiffuseReflection->expression = mdl::generateFunctionExpression(shaderDiffuseReflection->functionInfo.signature, shaderDiffuseReflection->sockets);
		}

		void ShaderVisitor::visit(std::shared_ptr<graph::shader::MaterialSurface> materialSurfaceNode)
		{
			materialSurfaceNode->expression = mdl::generateFunctionExpression(materialSurfaceNode->functionInfo.signature, materialSurfaceNode->sockets);
		}

		void ShaderVisitor::visit(std::shared_ptr<graph::shader::Material> materialNode)
		{
			materialNode->expression = mdl::generateFunctionExpression(materialNode->functionInfo.signature, materialNode->sockets);
		}

		void ShaderVisitor::visit(std::shared_ptr<graph::shader::ImportedNode> importedNode)
		{
			importedNode->expression = mdl::generateFunctionExpression(importedNode->functionInfo.signature, importedNode->sockets);
		}

}

#include "ShaderVisitor.h"

#include "Scene/Nodes/Shader/mdl/ShaderNodes.h"


namespace vtx::mdl
{
	using namespace graph::shader;

	#define STANDARD_SHADER_NODE_VISIT(SHADER_NODE_NAME) \
	void ShaderVisitor::visit(std::shared_ptr<SHADER_NODE_NAME> shaderNode)\
	{\
		shaderNode->init();\
	}

	STANDARD_SHADER_NODE_VISIT(DiffuseReflection)
	STANDARD_SHADER_NODE_VISIT(MaterialSurface)
	STANDARD_SHADER_NODE_VISIT(Material)
	STANDARD_SHADER_NODE_VISIT(ImportedNode)
	STANDARD_SHADER_NODE_VISIT(PrincipledMaterial)
	STANDARD_SHADER_NODE_VISIT(TextureTransform)
	STANDARD_SHADER_NODE_VISIT(NormalMix)
	STANDARD_SHADER_NODE_VISIT(ColorTexture)
	STANDARD_SHADER_NODE_VISIT(MonoTexture)
	STANDARD_SHADER_NODE_VISIT(NormalTexture)
	STANDARD_SHADER_NODE_VISIT(BumpTexture)
	STANDARD_SHADER_NODE_VISIT(GetChannel)
}

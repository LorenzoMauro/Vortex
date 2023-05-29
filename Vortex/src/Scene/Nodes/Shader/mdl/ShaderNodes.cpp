#include "ShaderNodes.h"

#include "MDL/MdlTypesName.h"
#include "Scene/Traversal.h"

namespace vtx::graph::shader
{

	void ShaderNode::initializeSockets()
	{
		const std::vector<ParameterInfo> parameters = mdl::getFunctionParameters(functionInfo, name);
		vtxID socketId = 1;
		for(auto& parameter : parameters)
		{
			sockets[parameter.argumentName] = ShaderNodeSocket{nullptr, parameter, SIM::getFreeIndex(), {} };
			socketId++;
			if (socketsGroupedByGroup.count(parameter.annotation.groupName) > 0)
			{
				socketsGroupedByGroup[parameter.annotation.groupName].push_back(parameter.argumentName);
			}
			else
			{
				socketsGroupedByGroup.insert({ parameter.annotation.groupName, {parameter.argumentName} });
			}
		}
	}

	void ShaderNode::connectInput(std::string socketName, const std::shared_ptr<ShaderNode>& inputNode)
	{
		const bool doSocketExists = sockets.find(socketName) != sockets.end();
		VTX_ASSERT_CONTINUE(doSocketExists, "Trying to connect Shader Node to a not existing input socket {}!", socketName);

		if(doSocketExists)
		{
			const mi::neuraylib::IType::Kind& socketType = sockets[socketName].parameterInfo.expressionKind;
			const mi::neuraylib::IType::Kind& inputType = inputNode->outputSocket.parameterInfo.expressionKind;
			const bool                        isSameType = socketType == inputType;
			VTX_ASSERT_CONTINUE(isSameType, "Node Sockets {} Mismatch: input type {} socket type {}!", socketName, mdl::ITypeToString[inputType], mdl::ITypeToString[socketType]);
			if (isSameType)
			{
				sockets[socketName].node = inputNode;
				sockets[socketName].linkId = SIM::getFreeIndex();
			}
		}
	}

	void ShaderNode::setSocketValue(std::string socketName, const mi::base::Handle<mi::neuraylib::IExpression>& defaultExpression)
	{
		const bool doSocketExists = sockets.find(socketName) != sockets.end();
		VTX_ASSERT_CONTINUE(doSocketExists, "Trying to connect Shader Node to a not existing input socket {}!", socketName);

		if (doSocketExists)
		{
			const mi::neuraylib::IType::Kind& socketType = sockets[socketName].parameterInfo.expressionKind;
			const mi::neuraylib::IType::Kind& inputType = defaultExpression->get_type()->skip_all_type_aliases()->get_kind();
			const bool                        isSameType = socketType == inputType;
			VTX_ASSERT_CONTINUE(isSameType, "Node Sockets {} Mismatch: input type {} socket type {}!", socketName, mdl::ITypeToString[inputType], mdl::ITypeToString[socketType]);
			if (isSameType)
			{
				sockets[socketName].directExpression = defaultExpression;
			}
		}
	}

//	void TextureFile::traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors)
//	{
//		for (auto& [name, socket] : sockets)
//		{
//			if (socket.node)
//			{
//				socket.node->traverse(orderedVisitors);
//			}
//		}
//		ACCEPT(TextureFile, visitors)
//	}
//
//	void TextureReturn::traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors)
//	{
//		for (auto& [name, socket] : sockets)
//		{
//			if (socket.node)
//			{
//				socket.node->traverse(orderedVisitors);
//			}
//		}
//		ACCEPT(TextureReturn, visitors)
//	}

#define DEFINE_SHADER_NODE_TRAVERSE(NODE_NAME) \
	void NODE_NAME::traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) \
	{\
		for (auto& [name, socket] : sockets)\
		{\
			if (socket.node)\
			{\
				socket.node->traverse(orderedVisitors);\
			}\
		}\
		ACCEPT(NODE_NAME, visitors)\
	}

	DEFINE_SHADER_NODE_TRAVERSE(DiffuseReflection)
	DEFINE_SHADER_NODE_TRAVERSE(MaterialSurface)
	DEFINE_SHADER_NODE_TRAVERSE(Material)
	DEFINE_SHADER_NODE_TRAVERSE(ImportedNode)
	DEFINE_SHADER_NODE_TRAVERSE(PrincipledMaterial)
	DEFINE_SHADER_NODE_TRAVERSE(ColorTexture)
	DEFINE_SHADER_NODE_TRAVERSE(MonoTexture)
	DEFINE_SHADER_NODE_TRAVERSE(NormalTexture)
	DEFINE_SHADER_NODE_TRAVERSE(BumpTexture)
	DEFINE_SHADER_NODE_TRAVERSE(TextureTransform)
	DEFINE_SHADER_NODE_TRAVERSE(NormalMix)
	DEFINE_SHADER_NODE_TRAVERSE(GetChannel)
}

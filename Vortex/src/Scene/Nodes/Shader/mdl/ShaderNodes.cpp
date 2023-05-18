#include "ShaderNodes.h"
#include "Scene/Traversal.h"

namespace vtx::graph::shader
{

	void ShaderNode::initializeSockets()
	{
		const std::vector<mdl::ParameterInfo> parameters = mdl::getFunctionParameters(functionInfo, getID());
		for(auto& parameter : parameters)
		{
			sockets[parameter.argumentName] = ShaderNodeSocket{ nullptr, parameter };
		}
	}

	void ShaderNode::connectInput(std::string socketName, const std::shared_ptr<ShaderNode>& inputNode)
	{
		const bool doSocketExists = sockets.find(socketName) != sockets.end();
		VTX_ASSERT_CONTINUE(doSocketExists, "Trying to connect Shader Node to a not existing input socket {}!", socketName);

		if(doSocketExists)
		{
			const IType::Kind& socketType = sockets[socketName].parameterInfo.type->skip_all_type_aliases()->get_kind();
			const IType::Kind& inputType  = inputNode->functionInfo.returnType->skip_all_type_aliases()->get_kind();
			const bool         isSameType = socketType == inputType;
			VTX_ASSERT_CONTINUE(isSameType, "Node Sockets {} Mismatch: input type {} socket type {}!", socketName, shaderNodeOutputName[inputType], shaderNodeOutputName[socketType]);
			if(isSameType)
			{
				sockets[socketName].node = inputNode;
			}
		}
	}

	void ShaderNode::setSocketDefault(std::string socketName, const mi::base::Handle<IExpression>& defaultExpression)
	{
		const bool doSocketExists = sockets.find(socketName) != sockets.end();
		VTX_ASSERT_CONTINUE(doSocketExists, "Trying to connect Shader Node to a not existing input socket {}!", socketName);

		if (doSocketExists)
		{
			const IType::Kind& socketType = sockets[socketName].parameterInfo.type->skip_all_type_aliases()->get_kind();
			const IType::Kind& inputType  = defaultExpression->get_type()->skip_all_type_aliases()->get_kind();
			const bool         isSameType = socketType == inputType;
			VTX_ASSERT_CONTINUE(isSameType, "Node Sockets {} Mismatch: input type {} socket type {}!", socketName, shaderNodeOutputName[inputType], shaderNodeOutputName[socketType]);
			if (isSameType)
			{
				sockets[socketName].parameterInfo.defaultValue = defaultExpression;
			}
		}
	}

	void TextureFile::traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors)
	{
		for (auto& [name, socket] : sockets)
		{
			if (socket.node)
			{
				socket.node->traverse(orderedVisitors);
			}
		}
		ACCEPT(TextureFile, visitors)
	}

	void TextureReturn::traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors)
	{
		for (auto& [name, socket] : sockets)
		{
			if (socket.node)
			{
				socket.node->traverse(orderedVisitors);
			}
		}
		ACCEPT(TextureReturn, visitors)
	}

	void DiffuseReflection::traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors)
	{
		for (auto& [name, socket] : sockets)
		{
			if (socket.node)
			{
				socket.node->traverse(orderedVisitors);
			}
		}
		ACCEPT(DiffuseReflection, visitors)
	}

	void MaterialSurface::traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors)
	{
		for (auto& [name, socket] : sockets)
		{
			if (socket.node)
			{
				socket.node->traverse(orderedVisitors);
			}
		}
		ACCEPT(MaterialSurface, visitors)
	}

	void Material::traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors)
	{
		for (auto& [name, socket] : sockets)
		{
			if (socket.node)
			{
				socket.node->traverse(orderedVisitors);
			}
		}
		ACCEPT(Material, visitors)
	}

	void ImportedNode::traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors)
	{
		for (auto& [name, socket] : sockets)
		{
			if (socket.node)
			{
				socket.node->traverse(orderedVisitors);
			}
		}
		ACCEPT(ImportedNode, visitors)
	}
}


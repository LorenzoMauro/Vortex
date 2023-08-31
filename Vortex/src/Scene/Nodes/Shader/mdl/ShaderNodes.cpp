#include "ShaderNodes.h"

#include "MDL/MdlTypesName.h"
#include "Scene/Traversal.h"

namespace vtx::graph::shader
{
	ShaderNode::~ShaderNode()
	{
		for (auto& [name, socket] : sockets)
		{
			SIM::releaseIndex(socket.Id);
		}
	}

	ShaderNode::ShaderNode(const NodeType cNodeType, mdl::MdlFunctionInfo cFunctionInfo) :
		Node(cNodeType),
		functionInfo(std::move(cFunctionInfo))
	{
		mdl::getFunctionSignature(&functionInfo);
		generateOutputSocket();
		defineName();
		initializeSockets();
		isUpdated = true;
	}

	ShaderNode::ShaderNode(const NodeType cNodeType, std::string modulePath, std::string functionName, bool isMdlPath) :
		Node(cNodeType)
	{
		isUpdated = true;
		functionInfo = mdl::MdlFunctionInfo{};
		if (!isMdlPath)
		{
			mdl::addSearchPath(utl::getFolder(modulePath));
			modulePath = "/" + utl::getFile(modulePath);
		}
		functionInfo.module = "mdl" + mdl::pathToModuleName(modulePath);
		functionInfo.name = functionInfo.module + "::" + functionName;
		mdl::getFunctionSignature(&functionInfo);
		generateOutputSocket();
		defineName();
		initializeSockets();
	}

	void ShaderNode::generateOutputSocket()
	{
		outputSocket.Id = SIM::getFreeIndex();
		outputSocket.parameterInfo.expressionKind = functionInfo.returnType->skip_all_type_aliases()->get_kind();
		outputSocket.parameterInfo.annotation.displayName = mdl::ITypeToString[outputSocket.parameterInfo.
			expressionKind];
	}

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

	void ShaderNode::defineName()
	{
		const std::size_t lastColon = functionInfo.name.rfind("::");
		std::string       extractedName;
		if (lastColon != std::string::npos)
		{
			extractedName = functionInfo.name.substr(lastColon + 2);
		}
		name = (extractedName + "_" + std::to_string(getID()));
	}

	void ShaderNode::printSocketInfos()
	{
		auto ss = std::stringstream();
		ss << "ShaderNode: " << name << std::endl;
		for (auto& [name, socket] : sockets)
		{
			auto& [node, parameterInfo, id, expression, linkID] = socket;
			std::string socketNameType = mdl::ITypeToString[parameterInfo.expressionKind];
			ss << "\nSocket: " << name << " Type: " << socketNameType << std::endl;
		}
		VTX_INFO(ss.str());
	}

	void ShaderNode::init()
	{
		if (isUpdated)
		{
			bool Success = mdl::generateFunctionExpression(functionInfo.signature, sockets, name); 
			if (Success) {
				
					isUpdated = false; 
			}
		}
	}

//#define DEFINE_SHADER_NODE_TRAVERSE_CHILDREN(NODE_NAME) \
//	void NODE_NAME::traverseChildren(NodeVisitor& visitor) \
//	{\
//		for (auto& [name, socket] : sockets)\
//		{\
//			if (socket.node)\
//			{\
//				socket.node->traverse(visitor);\
//			}\
//		}\
//	}

#define DEFINE_SHADER_NODE_ACCEPT(NODE_NAME) \
	void NODE_NAME::accept(NodeVisitor& visitor) \
	{\
		ACCEPT(NODE_NAME, visitor)\
	}

//#define DEFINE_SHADER_NODE_GET_CHILDREN(NODE_NAME) \
//	std::vector<std::shared_ptr<Node>> NODE_NAME::getChildren() const\
//	{\
//		std::vector<std::shared_ptr<Node>> children;\
//		for (auto& [name, socket] : sockets)\
//		{\
//			if (socket.node)\
//			{\
//				children.push_back(socket.node);\
//			}\
//		}\
//		return children;\
//	}

	//DEFINE_SHADER_NODE_TRAVERSE_CHILDREN(DiffuseReflection)
	//DEFINE_SHADER_NODE_TRAVERSE_CHILDREN(MaterialSurface)
	//DEFINE_SHADER_NODE_TRAVERSE_CHILDREN(Material)
	//DEFINE_SHADER_NODE_TRAVERSE_CHILDREN(ImportedNode)
	//DEFINE_SHADER_NODE_TRAVERSE_CHILDREN(PrincipledMaterial)
	//DEFINE_SHADER_NODE_TRAVERSE_CHILDREN(ColorTexture)
	//DEFINE_SHADER_NODE_TRAVERSE_CHILDREN(MonoTexture)
	//DEFINE_SHADER_NODE_TRAVERSE_CHILDREN(NormalTexture)
	//DEFINE_SHADER_NODE_TRAVERSE_CHILDREN(BumpTexture)
	//DEFINE_SHADER_NODE_TRAVERSE_CHILDREN(TextureTransform)
	//DEFINE_SHADER_NODE_TRAVERSE_CHILDREN(NormalMix)
	//DEFINE_SHADER_NODE_TRAVERSE_CHILDREN(GetChannel)

	//DEFINE_SHADER_NODE_GET_CHILDREN(DiffuseReflection)
	//DEFINE_SHADER_NODE_GET_CHILDREN(MaterialSurface)
	//DEFINE_SHADER_NODE_GET_CHILDREN(Material)
	//DEFINE_SHADER_NODE_GET_CHILDREN(ImportedNode)
	//DEFINE_SHADER_NODE_GET_CHILDREN(PrincipledMaterial)
	//DEFINE_SHADER_NODE_GET_CHILDREN(ColorTexture)
	//DEFINE_SHADER_NODE_GET_CHILDREN(MonoTexture)
	//DEFINE_SHADER_NODE_GET_CHILDREN(NormalTexture)
	//DEFINE_SHADER_NODE_GET_CHILDREN(BumpTexture)
	//DEFINE_SHADER_NODE_GET_CHILDREN(TextureTransform)
	//DEFINE_SHADER_NODE_GET_CHILDREN(NormalMix)
	//DEFINE_SHADER_NODE_GET_CHILDREN(GetChannel)

	DEFINE_SHADER_NODE_ACCEPT(DiffuseReflection)
	DEFINE_SHADER_NODE_ACCEPT(MaterialSurface)
	DEFINE_SHADER_NODE_ACCEPT(Material)
	DEFINE_SHADER_NODE_ACCEPT(ImportedNode)
	DEFINE_SHADER_NODE_ACCEPT(PrincipledMaterial)
	DEFINE_SHADER_NODE_ACCEPT(ColorTexture)
	DEFINE_SHADER_NODE_ACCEPT(MonoTexture)
	DEFINE_SHADER_NODE_ACCEPT(NormalTexture)
	DEFINE_SHADER_NODE_ACCEPT(BumpTexture)
	DEFINE_SHADER_NODE_ACCEPT(TextureTransform)
	DEFINE_SHADER_NODE_ACCEPT(NormalMix)
	DEFINE_SHADER_NODE_ACCEPT(GetChannel)
}

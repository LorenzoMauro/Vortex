#include "ShaderNodes.h"

#include "MDL/MdlTypesName.h"
#include "Scene/Scene.h"
#include "Scene/Traversal.h"

namespace vtx::graph::shader
{
	ShaderNode::~ShaderNode()
	{
		for (auto& [name, socket] : sockets)
		{
			graph::Scene::getSim()->releaseUID(socket.Id);
		}
	}

	ShaderNode::ShaderNode(const NodeType cNodeType, mdl::MdlFunctionInfo cFunctionInfo) :
		Node(cNodeType),
		functionInfo(std::move(cFunctionInfo))
	{
		if(!cFunctionInfo.fullModulePath.empty())
		{
			mdl::addSearchPath(utl::getFolder(cFunctionInfo.fullModulePath));
		}
		mdl::getFunctionSignature(&functionInfo);
		generateOutputSocket();
		defineName();
		initializeSockets();
		state.isShaderCodeUpdated = true;
	}

	ShaderNode::ShaderNode(const NodeType cNodeType, std::string modulePath, std::string functionName, const bool isMdlPath) :
		Node(cNodeType)
	{
		state.isShaderCodeUpdated = true;
		functionInfo = mdl::MdlFunctionInfo{};
		if (!isMdlPath)
		{
			functionInfo.fullModulePath = modulePath;
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
		outputSocket.Id = graph::Scene::getSim()->getUID();
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
			sockets[parameter.argumentName] = ShaderNodeSocket{nullptr, parameter, graph::Scene::getSim()->getUID(), {} };
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
				sockets[socketName].linkId = graph::Scene::getSim()->getUID();
			}
		}
	}

	void ShaderNode::setSocketValue(std::string socketName, const mi::base::Handle<mi::neuraylib::IExpression>& defaultExpression)
	{
		if(!defaultExpression.is_valid_interface())
		{
			VTX_WARN("Trying to set a nullptr expression to socket {}!", socketName);
			return;
		}
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
		name = (extractedName + "_" + std::to_string(getUID()));
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
		if (state.isShaderCodeUpdated )
		{
			bool Success = mdl::generateFunctionExpression(functionInfo.signature, sockets, name); 
			if (Success) {
				
					state.isShaderCodeUpdated = false; 
			}
		}
	}

	void ShaderNode::resetIsShaderArgBlockUpdated()
	{
		for (const auto& child : getChildren())
		{
			if (const std::shared_ptr<ShaderNode>& shaderNode = child->as<ShaderNode>(); shaderNode)
			{
				shaderNode->resetIsShaderArgBlockUpdated();
			}
		}
		state.isShaderArgBlockUpdated = false;
	}


#define DEFINE_SHADER_NODE_ACCEPT(NODE_NAME) \
	void NODE_NAME::accept(NodeVisitor& visitor) \
	{\
		ACCEPT(NODE_NAME, visitor)\
	}

	DEFINE_SHADER_NODE_ACCEPT(ShaderNode)
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


#define DEFINE_SHADER_NODE_DESTRUCTOR(NODE_NAME)\
		NODE_NAME::~NODE_NAME()\
	{\
	}

	DEFINE_SHADER_NODE_DESTRUCTOR(DiffuseReflection)
	DEFINE_SHADER_NODE_DESTRUCTOR(MaterialSurface)
	DEFINE_SHADER_NODE_DESTRUCTOR(Material)
	DEFINE_SHADER_NODE_DESTRUCTOR(ImportedNode)
	DEFINE_SHADER_NODE_DESTRUCTOR(PrincipledMaterial)
	DEFINE_SHADER_NODE_DESTRUCTOR(ColorTexture)
	DEFINE_SHADER_NODE_DESTRUCTOR(MonoTexture)
	DEFINE_SHADER_NODE_DESTRUCTOR(NormalTexture)
	DEFINE_SHADER_NODE_DESTRUCTOR(BumpTexture)
	DEFINE_SHADER_NODE_DESTRUCTOR(TextureTransform)
	DEFINE_SHADER_NODE_DESTRUCTOR(NormalMix)
	DEFINE_SHADER_NODE_DESTRUCTOR(GetChannel)

}

#pragma once
//#include "MDL/MdlWrapper.h"
#include <mi/neuraylib/itype.h>

#include "NodesDefine.h"
#include "MDL/MdlWrapper.h"
#include "Scene/Node.h"

namespace vtx::graph::shader
{


	using SNT = mi::neuraylib::IType::Kind;

	static inline std::map<SNT, std::string> shaderNodeOutputName({
		{SNT::TK_ALIAS ,"Alias"},
		{SNT::TK_BOOL ,"Bool"},
		{SNT::TK_INT ,"Int"},
		{SNT::TK_ENUM ,"Enum"},
		{SNT::TK_FLOAT ,"Float"},
		{SNT::TK_DOUBLE ,"Double"},
		{SNT::TK_STRING ,"String"},
		{SNT::TK_VECTOR ,"Vector"},
		{SNT::TK_MATRIX ,"Matrix"},
		{SNT::TK_COLOR ,"Color"},
		{SNT::TK_ARRAY ,"Array"},
		{SNT::TK_STRUCT ,"Struct"},
		{SNT::TK_TEXTURE ,"Texture"},
		{SNT::TK_LIGHT_PROFILE ,"LightProfile"},
		{SNT::TK_BSDF_MEASUREMENT ,"BsdfMeasurement"},
		{SNT::TK_BSDF ,"Bsdf"},
		{SNT::TK_HAIR_BSDF ,"HairBsdf"},
		{SNT::TK_EDF ,"EDF"},
		{SNT::TK_VDF ,"VDF"}
	});

	struct CustomAnnotation
	{
		std::string              displayName;
		std::string              description;
		std::string              groupName;
		std::tuple<float, float> hardRange{-1.0f, -1.0f};
		std::tuple<float, float> softRange{-1.0f, -1.0f};
		bool                     use = false;

	};
	struct ShaderNodeSocket
	{
		std::shared_ptr<ShaderNode> node;
		mdl::ParameterInfo			parameterInfo;
	};

	using ShaderInputSockets = std::map<std::string, ShaderNodeSocket>;

	class ShaderNode : public Node
	{
	public:
		
		ShaderNode(const NodeType cNodeType,
				   mdl::MdlFunctionInfo cFunctionInfo) :
			Node(cNodeType),
			functionInfo(std::move(cFunctionInfo))
		{
			mdl::getFunctionSignature(&functionInfo);
			outputType = functionInfo.returnType->skip_all_type_aliases()->get_kind();
			defineName();
			initializeSockets();
			isUpdated=true;
		}

		ShaderNode(const NodeType cNodeType, std::string modulePath, std::string functionName, bool isMdlPath = false) :
			Node(cNodeType)
		{
			isUpdated = true;
			functionInfo = mdl::MdlFunctionInfo{};
			if(!isMdlPath)
			{
				mdl::addSearchPath(utl::getFolder(modulePath));
				modulePath = "/" + utl::getFile(modulePath);
			}
			functionInfo.module = "mdl" + mdl::pathToModuleName(modulePath);
			functionInfo.name = functionInfo.module + "::" + functionName;
			mdl::getFunctionSignature(&functionInfo);
			outputType = functionInfo.returnType->skip_all_type_aliases()->get_kind();
			defineName();
			initializeSockets();
		}

		//void accept(std::shared_ptr<NodeVisitor> visitor) override;

		void initializeSockets();

		void connectInput(std::string socketName, const std::shared_ptr<ShaderNode>& inputNode);

		void setSocketDefault(std::string socketName, const mi::base::Handle<mi::neuraylib::IExpression>& defaultExpression);

		void defineName()
		{
			const std::size_t lastColon = functionInfo.name.rfind("::");
			std::string extractedName;
			if (lastColon != std::string::npos) {
				extractedName = functionInfo.name.substr(lastColon + 2);
			}
			name = (extractedName + "_" + std::to_string(getID()));
		}

		void printSocketInfos()
		{
			auto ss = std::stringstream();
			ss << "ShaderNode: " << name << std::endl;
			for (auto& [name, socket] : sockets)
			{
				auto& [node, parameterInfo] = socket;
				std::string socketNameType  = shaderNodeOutputName[parameterInfo.type->skip_all_type_aliases()->get_kind()];
				ss << "\nSocket: " << name << " Type: " << socketNameType << std::endl;
			}
			VTX_INFO(ss.str());
		}

		SNT                                          outputType = mi::neuraylib::IType::TK_FORCE_32_BIT;
		ShaderInputSockets                           sockets;
		mdl::MdlFunctionInfo                         functionInfo;
		mi::base::Handle<mi::neuraylib::IExpression> expression;
		std::string                                  name;
	};


//	class TextureFile : public ShaderNode
//	{
//	public:
//		TextureFile() :ShaderNode(NT_SHADER_TEXTURE,
//								  mdl::MdlFunctionInfo{ "mdl::base",
//														"mdl::base::file_texture" })
//		{}
//
//		std::string					path;
//
//		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;
//
//	};
//
//	class TextureReturn : public ShaderNode
//	{
//		public:
//			TextureReturn(const SNT cOutputType) :ShaderNode(NT_SHADER_TEXTURE)
//			{
//				outputType = cOutputType;
//				if(outputType == IType::TK_COLOR)
//				{
//					functionInfo = mdl::MdlFunctionInfo{ "mdl::base","mdl::base::texture_return.tint" };
//				}
//				else
//				{
//					functionInfo = mdl::MdlFunctionInfo{ "mdl::base","mdl::base::texture_return.mono" };
//
//				}
//				mdl::getFunctionSignature(&functionInfo);
//				outputType = functionInfo.returnType->skip_all_type_aliases()->get_kind();
//				initializeSockets();
//				defineName();
//			}
//
//			void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;
//
//	};

	class DiffuseReflection : public ShaderNode
	{
	public:
		DiffuseReflection() :ShaderNode(NT_SHADER_DF,
										mdl::MdlFunctionInfo{ "mdl::df",
															  "mdl::df::diffuse_reflection_bsdf" })
		{}
		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;

	};

	class MaterialSurface : public ShaderNode
	{
	public:
		MaterialSurface() :ShaderNode(NT_SHADER_SURFACE,
									  mdl::MdlFunctionInfo{ "mdl",
															"mdl::material_surface",
															"mdl::material_surface(bsdf,material_emission)" })
		{}
		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;

	};

	class Material : public ShaderNode
	{
	public:
		Material() :ShaderNode(NT_SHADER_MATERIAL,
							   mdl::MdlFunctionInfo{ "mdl",
													"mdl::material",
													"mdl::material(bool,material_surface,material_surface,color,material_volume,material_geometry,hair_bsdf)" })
		{
		}
		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;

	};

	class ImportedNode : public ShaderNode
	{
	public:
		ImportedNode(std::string modulePath, std::string functionName, bool isMdlPath = false) :ShaderNode(NT_SHADER_IMPORTED, modulePath, functionName, isMdlPath)
		{}

		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;
	};

	class PrincipledMaterial : public ShaderNode
	{
	public:
		PrincipledMaterial() :ShaderNode(NT_SHADER_MATERIAL, VORTEX_PRINCIPLED_MODULE, VORTEX_PRINCIPLED_FUNCTION)
		{}

		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;
	};

	class ColorTexture : public ShaderNode
	{
	public:
		ColorTexture(const std::string& cTexturePath) : ShaderNode(NT_SHADER_COLOR, VORTEX_FUNCTIONS_MODULE, VF_COLOR_TEXTURE)
		{
			texturePath = cTexturePath;
			sockets[VF_COLOR_TEXTURE_TEXTURE_SOCKET].parameterInfo.defaultValue = mdl::createTextureConstant(texturePath);
		}

		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;

		std::string texturePath;
	};

	class MonoTexture : public ShaderNode
	{
	public:
		MonoTexture(const std::string& cTexturePath) : ShaderNode(NT_SHADER_FLOAT, VORTEX_FUNCTIONS_MODULE, VF_MONO_TEXTURE)
		{
			texturePath = cTexturePath;
			sockets[VF_MONO_TEXTURE_TEXTURE_SOCKET].parameterInfo.defaultValue = mdl::createTextureConstant(texturePath, mi::neuraylib::IType_texture::TS_2D, 1.0f);
		}

		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;

		std::string texturePath;
	};

	class NormalTexture : public ShaderNode
	{
	public:
		NormalTexture(const std::string& cTexturePath) : ShaderNode(NT_SHADER_FLOAT3, VORTEX_FUNCTIONS_MODULE, VF_NORMAL_TEXTURE)
		{
			texturePath = cTexturePath;
			sockets[VF_NORMAL_TEXTURE_TEXTURE_SOCKET].parameterInfo.defaultValue = mdl::createTextureConstant(texturePath, mi::neuraylib::IType_texture::TS_2D, 1.0f);
		}

		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;

		std::string texturePath;
	};

	class BumpTexture : public ShaderNode
	{
	public:
		BumpTexture(const std::string& cTexturePath) : ShaderNode(NT_SHADER_FLOAT3, VORTEX_FUNCTIONS_MODULE, VF_BUMP_TEXTURE)
		{
			texturePath = cTexturePath;
			sockets[VF_BUMP_TEXTURE_TEXTURE_SOCKET].parameterInfo.defaultValue = mdl::createTextureConstant(texturePath, mi::neuraylib::IType_texture::TS_2D, 1.0f);
		}

		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;

		std::string texturePath;
	};


	class TextureTransform : public ShaderNode
	{
	public:
		TextureTransform() : ShaderNode(NT_SHADER_COORDINATE, VORTEX_FUNCTIONS_MODULE, VF_TEXTURE_TRANSFORM)
		{}

		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;

		std::string texturePath;
	};


	class NormalMix : public ShaderNode
	{
	public:
		NormalMix() : ShaderNode(NT_SHADER_FLOAT3, VORTEX_FUNCTIONS_MODULE, VF_MIX_NORMAL)
		{}

		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;

		std::string texturePath;
	};


	class GetChannel : public ShaderNode
	{
	public:
		GetChannel(int cChannel) : ShaderNode(NT_SHADER_FLOAT, VORTEX_FUNCTIONS_MODULE, VF_GET_COLOR_CHANNEL)
		{
			if (cChannel > 2)
			{
				cChannel = 2;
				VTX_WARN("Setting Color Channel to Invalid Channel {}, cChannel");
			}
			else if(cChannel < 0)
			{
				cChannel = 0;
				VTX_WARN("Setting Color Channel to Invalid Channel {}, cChannel");
			}

			channel = cChannel;
			sockets[VF_GET_COLOR_CHANNEL_CHANNEL_SOCKET].parameterInfo.defaultValue = mdl::createConstantInt(channel);
		}

		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;

		int channel;
	};

}



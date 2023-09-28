#pragma once
//#include "MDL/MdlWrapper.h"
#include <mi/neuraylib/itype.h>

#include "NodesDefine.h"
#include "MDL/MdlTypesName.h"
#include "MDL/MdlWrapper.h"
#include "Scene/Node.h"
#include "Scene/SIM.h"
#include "Scene/Nodes/Shader/mdl/ShaderSocket.h"

namespace vtx::graph::shader
{
	class ShaderNode : public Node
	{
	public:
		~ShaderNode() override;

		ShaderNode(const NodeType       cNodeType,
				   mdl::MdlFunctionInfo cFunctionInfo);

		ShaderNode(const NodeType cNodeType, std::string modulePath, std::string functionName, bool isMdlPath = false);

		//void accept(std::shared_ptr<NodeVisitor> visitor) override;
		void generateOutputSocket();

		void initializeSockets();

		void connectInput(std::string socketName, const std::shared_ptr<ShaderNode>& inputNode);

		void setSocketValue(std::string                                         socketName,
							const mi::base::Handle<mi::neuraylib::IExpression>& defaultExpression);

		void defineName();

		void printSocketInfos();

		void init() override;

		void accept(NodeVisitor& visitor) override;

		void resetIsShaderArgBlockUpdated();

		std::vector<std::shared_ptr<Node>> getChildren() const override
		{
			std::vector<std::shared_ptr<Node>> children;
			for (auto& [_, socket] : sockets)
			{
				if (socket.node)
				{
					children.push_back(socket.node);
				}
			}
			return children;
		}

		ShaderInputSockets                              sockets;
		mdl::MdlFunctionInfo                            functionInfo;
		std::map<std::string, std::vector<std::string>> socketsGroupedByGroup;
		ShaderNodeSocket                                outputSocket;
	};


	class DiffuseReflection : public ShaderNode
	{
	public:
		DiffuseReflection() : ShaderNode(NT_SHADER_DF,
										 mdl::MdlFunctionInfo{
											 "mdl::df",
											 "mdl::df::diffuse_reflection_bsdf"
										 })
		{
			typeID = SIM::get()->getTypeId<DiffuseReflection>();
		}

		~DiffuseReflection() override;
	protected:
		void accept(NodeVisitor& visitor) override;
	};

	class MaterialSurface : public ShaderNode
	{
	public:
		MaterialSurface() : ShaderNode(NT_SHADER_SURFACE,
									   mdl::MdlFunctionInfo{
										   "mdl",
										   "mdl::material_surface",
										   "mdl::material_surface(bsdf,material_emission)"
									   })
		{
			typeID = SIM::get()->getTypeId<MaterialSurface>();
		}

		~MaterialSurface() override;
	protected:
		void accept(NodeVisitor& visitor) override;
	};

	class Material : public ShaderNode
	{
	public:
		Material() : ShaderNode(NT_SHADER_MATERIAL,
								mdl::MdlFunctionInfo{
									"mdl",
									"mdl::material",
									"mdl::material(bool,material_surface,material_surface,color,material_volume,material_geometry,hair_bsdf)"
								})
		{
			typeID = SIM::get()->getTypeId<Material>();
		}

		~Material() override;
	protected:
		void accept(NodeVisitor& visitor) override;
	};

	class ImportedNode : public ShaderNode
	{
	public:
		ImportedNode(std::string modulePath, std::string functionName, const bool isMdlPath = false) : ShaderNode(
			NT_SHADER_IMPORTED, modulePath, functionName, isMdlPath)
		{
			typeID = SIM::get()->getTypeId<ImportedNode>();
		}

		~ImportedNode() override;
	protected:
		void accept(NodeVisitor& visitor) override;
	};

	class PrincipledMaterial : public ShaderNode
	{
	public:
		PrincipledMaterial() : ShaderNode(NT_SHADER_MATERIAL, VORTEX_PRINCIPLED_MODULE, VORTEX_PRINCIPLED_FUNCTION)
		{
			typeID = SIM::get()->getTypeId<PrincipledMaterial>();
		}

		~PrincipledMaterial() override;
	protected:
		void accept(NodeVisitor& visitor) override;
	};

	class ColorTexture : public ShaderNode
	{
	public:
		ColorTexture(const std::string& cTexturePath) : ShaderNode(NT_SHADER_COLOR, VORTEX_FUNCTIONS_MODULE,
																   VF_COLOR_TEXTURE)
		{
			texturePath = cTexturePath;
			setSocketValue(VF_COLOR_TEXTURE_TEXTURE_SOCKET, mdl::createTextureConstant(texturePath));
			typeID = SIM::get()->getTypeId<ColorTexture>();
		}

		~ColorTexture() override;
	protected:
		void accept(NodeVisitor& visitor) override;

	public:
		std::string texturePath;
	};

	class MonoTexture : public ShaderNode
	{
	public:
		MonoTexture(const std::string& cTexturePath) : ShaderNode(NT_SHADER_FLOAT, VORTEX_FUNCTIONS_MODULE,
																  VF_MONO_TEXTURE)
		{
			texturePath = cTexturePath;
			setSocketValue(
				VF_MONO_TEXTURE_TEXTURE_SOCKET,
				mdl::createTextureConstant(texturePath, mi::neuraylib::IType_texture::TS_2D, 1.0f));
			typeID = SIM::get()->getTypeId<MonoTexture>();
		}

		~MonoTexture() override;
	protected:
		void accept(NodeVisitor& visitor) override;

	public:
		std::string texturePath;
	};

	class NormalTexture : public ShaderNode
	{
	public:
		NormalTexture(const std::string& cTexturePath) : ShaderNode(NT_SHADER_FLOAT3, VORTEX_FUNCTIONS_MODULE,
																	VF_NORMAL_TEXTURE)
		{
			texturePath = cTexturePath;
			setSocketValue(
				VF_NORMAL_TEXTURE_TEXTURE_SOCKET,
				mdl::createTextureConstant(texturePath, mi::neuraylib::IType_texture::TS_2D, 1.0f));
			typeID = SIM::get()->getTypeId<NormalTexture>();
		}

		~NormalTexture() override;
	protected:
		void accept(NodeVisitor& visitor) override;

	public:
		std::string texturePath;
	};

	class BumpTexture : public ShaderNode
	{
	public:
		BumpTexture(const std::string& cTexturePath) : ShaderNode(NT_SHADER_FLOAT3, VORTEX_FUNCTIONS_MODULE,
																  VF_BUMP_TEXTURE)
		{
			texturePath = cTexturePath;
			setSocketValue(
				VF_BUMP_TEXTURE_TEXTURE_SOCKET,
				mdl::createTextureConstant(texturePath, mi::neuraylib::IType_texture::TS_2D, 1.0f));
			typeID = SIM::get()->getTypeId<BumpTexture>();
		}

		~BumpTexture() override;
	protected:
		void accept(NodeVisitor& visitor) override;

	public:
		std::string texturePath;
	};


	class TextureTransform : public ShaderNode
	{
	public:
		TextureTransform() : ShaderNode(NT_SHADER_COORDINATE, VORTEX_FUNCTIONS_MODULE, VF_TEXTURE_TRANSFORM)
		{
			typeID = SIM::get()->getTypeId<TextureTransform>();
		}

		~TextureTransform() override;
	protected:
		void accept(NodeVisitor& visitor) override;

	public:
		std::string texturePath;
	};


	class NormalMix : public ShaderNode
	{
	public:
		NormalMix() : ShaderNode(NT_SHADER_FLOAT3, VORTEX_FUNCTIONS_MODULE, VF_MIX_NORMAL)
		{
			typeID = SIM::get()->getTypeId<NormalMix>();
		}

		~NormalMix() override;
	protected:
		void accept(NodeVisitor& visitor) override;

	public:
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
			else if (cChannel < 0)
			{
				cChannel = 0;
				VTX_WARN("Setting Color Channel to Invalid Channel {}, cChannel");
			}

			channel = cChannel;
			setSocketValue(VF_GET_COLOR_CHANNEL_CHANNEL_SOCKET, mdl::createConstantInt(channel));
			typeID = SIM::get()->getTypeId<GetChannel>();
		}

		~GetChannel() override;
	protected:
		void accept(NodeVisitor& visitor) override;

	public:
		int channel;
	};
}

#pragma once
//#include "MDL/MdlWrapper.h"
#include <mi/neuraylib/itype.h>

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
			initializeSockets();
			isUpdated=true;
			defineName();
		}

		ShaderNode(const NodeType cNodeType) :
			Node(cNodeType)
		{
			isUpdated=true;
		}


		//void accept(std::shared_ptr<NodeVisitor> visitor) override;

		void initializeSockets();

		void connectInput(std::string socketName, const std::shared_ptr<ShaderNode>& inputNode);

		void setSocketDefault(std::string socketName, const mi::base::Handle<IExpression>& defaultExpression);

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

		SNT		outputType = IType::TK_FORCE_32_BIT;
		ShaderInputSockets			sockets;
		mdl::MdlFunctionInfo		functionInfo;
		Handle<IExpression>			expression;
		std::string					name;
	};

	class TextureFile : public ShaderNode
	{
	public:
		TextureFile() :ShaderNode(NT_SHADER_TEXTURE,
								  mdl::MdlFunctionInfo{ "mdl::base",
														"mdl::base::file_texture" })
		{}

		std::string					path;

		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;

	};

	class TextureReturn : public ShaderNode
	{
		public:
			TextureReturn(const SNT cOutputType) :ShaderNode(NT_SHADER_TEXTURE)
			{
				outputType = cOutputType;
				if(outputType == IType::TK_COLOR)
				{
					functionInfo = mdl::MdlFunctionInfo{ "mdl::base","mdl::base::texture_return.tint" };
				}
				else
				{
					functionInfo = mdl::MdlFunctionInfo{ "mdl::base","mdl::base::texture_return.mono" };

				}
				mdl::getFunctionSignature(&functionInfo);
				outputType = functionInfo.returnType->skip_all_type_aliases()->get_kind();
				initializeSockets();
				defineName();
			}

			void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;

	};

	class DiffuseReflection : public ShaderNode
	{
	public:
		DiffuseReflection() :ShaderNode(NT_SHADER_DIFFUSE_REFLECTION,
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
			ImportedNode(std::string modulePath, std::string functionName) :ShaderNode(NT_SHADER_IMPORTED)
			{

				const std::string moduleFolder = utl::getFolder(modulePath);
				const std::string moduleName = "/" + utl::getFile(modulePath);
				mdl::addSearchPath(moduleFolder);
				path = modulePath;
				functionInfo = mdl::MdlFunctionInfo{};
				mdl::pathToModuleName(modulePath);
				functionInfo.module = "mdl" + mdl::pathToModuleName(moduleName);
				functionInfo.name = functionInfo.module + "::" + functionName;
				mdl::getFunctionSignature(&functionInfo);
				outputType = functionInfo.returnType->skip_all_type_aliases()->get_kind();
				initializeSockets();
				defineName();
			}

		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;

		std::string path;
	};

#define PBSDF_MODULE "E:/Dev/VortexOptix/data/mdl/OmniPBR.mdl"
#define PBSDF_FUNCTION "OmniPBR"
//  Albedo  //
#define DIFFUSE_COLOR_SOCKET			"diffuse_color_constant"	//This is the albedo base color. Default : color(0.2)
#define DIFFUSE_TEXTURE_SOCKET			"diffuse_texture"			//Default : texture_2d()
#define ALBEDO_DESATURATION_SOCKET		"albedo_desaturation"		//Desaturates the diffuse color - Default : float(0.0) - Soft Range : (float(0.0), float(1.0))
#define ALBEDO_ADD_SOCKET				"albedo_add"				//Adds a constant value to the diffuse color  - Default : float(0.0) - Soft Range : (float(-1.0), float(1.0))
#define ALBEDO_BRIGHTNESS_SOCKET		"albedo_brightness"			//Multiplier for the diffuse color  - Default : float(1.0) - Soft Range : (float(0.0), float(1.0))
#define ALBEDO_DIFFUSE_TINT_SOCKET		"diffuse_tint"				//When enabled, this color value is multiplied over the final albedo color - Default : color(1.0)

//  Reflectivity //
#define ROUGHNESS_CONSTANT_SOCKET			"reflection_roughness_constant"				//Higher roughness values lead to more blurry reflections - Default : float(0.5) - Hard Range : (float(0.0), float(1.0))
#define ROUGHNESS_TEXTURE_INFLUENCE_SOCKET	"reflection_roughness_texture_influence"	//Blends between the constant value and the lookup of the roughness texture - Default : float(0.0) - Hard Range : (float(0.0), float(1.0))
#define ROUGHNESS_TEXTURE_SOCKET			"reflectionroughness_texture"				//Default : texture_2d()
#define METALLIC_CONSTANT_SOCKET			"metallic_constant"							//Metallic Material - Default : float(0.0) - Hard Range : (float(0.0), float(1.0))
#define METALLIC_TEXTURE_INFLUENCE_SOCKET	"metallic_texture_influence"				//Blends between the constant value and the lookup of the metallic texture - Default : float(0.0) - Hard Range : (float(0.0), float(1.0))
#define METALLIC_TEXTURE_SOCKET				"metallic_texture"							//Default : texture_2d()
#define SPECULAR_LEVEL_SOCKET				"specular_level"							//The specular level (intensity) of the material - Default : float(0.5) - Soft Range : (float(0.0), float(1.0))

//  ORM  //
#define ENABLE_ORM_SOCKET	"enable_ORM_texture"//The ORM texture will be used to extract the Occlusion, Roughness and Metallic textures from R,G,B channels - Default : false
#define ORM_TEXTURE_SOCKET	"ORM_texture"		//Texture that has Occlusion, Roughness and Metallic maps stored in their respective R, G and B channels - Default : texture_2d()

//  AO Group  // 

#define AO_TO_DIFFUSE_SOCKET	"ao_to_diffuse"	//Controls the amount of ambient occlusion multiplied against the diffuse color channel - Default : 0.0 - Hard Range : (float(0.0), float(1.0))
#define AO_TEXTURE_SOCKET		"ao_texture"	//The ambient occlusion texture for the material - Default : texture_2d()

//  Emission //  

#define ENABLE_EMISSION_SOCKET			"enable_emission"			//Enables the emission of light from the material - Default : false
#define EMISSIVE_COLOR_SOCKET			"emissive_color"			//The emission color - Default : color(0.0,0.0,0.0)
#define EMISSIVE_COLOR_TEXTURE_SOCKET	"emissive_color_texture"	//The emissive color texture - Default : texture_2d() 
#define EMISSIVE_MASK_TEXTURE_SOCKET	"emissive_mask_texture"		//The texture masking the emissive color - Default : texture_2d()
#define EMISSIVE_INTENSITIY_SOCKET		"emissive_intensity"		//Intensity of the emission - Default : float(0.0)

//  Cutout Opacity  //

#define SOCKET							"enable_opacity"			//Enables the use of cutout opacity - Default : false
#define OPACITY_TEXTURE_SOCKET			"opacity_texture"			//Default : texture_2d()
#define OPACITY_CONSTANT_SOCKET			"opacity_constant"			//Opacity value between 0 and 1, when Opacity Map is not valid - Default : float(1.0) - Hard Range : (float(0.0), float(1.0))
#define ENABLE_OPACITY_TEXTURE_SOCKET	"enable_opacity_texture"	//Enables or disables the usage of the opacity texture map - Default : false
#define OPACITY_MODE_SOCKET				"opacity_mode"				//Determines how to lookup opacity from the supplied texture. mono_alpha, mono_average, mono_luminance, mono_maximum - Default : base::mono_average
#define OPACITY_THRESHOLD_SOCKET		"opacity_threshold"			//If 0, use fractional opacity values 'as is'; if > 0, remap opacity values to 1 when >= threshold and to 0 otherwise - Default : float(0.0) - Hard Range : (0.0, 1.0)

//  Normal  // 

#define NORMALMAP_FACTOR_SOCKET			"bump_factor"				//Strength of normal map - Default : float(1.0) - Soft Range : (float(0.0), float(1.0))
#define NORMALMAP_TEXTURE_SOCKET		"normalmap_texture"			//Default : texture_2d()
#define DETAIL_NORMALMAP_FACTOR_SOCKET	"detail_bump_factor"		//Strength of the detail normal - Default : float(0.3) - Soft Range : (float(0.0), float(1.0))
#define DETAIL_NORMALMAP_TEXTURE_SOCKET "detail_normalmap_texture"	//Default : texture_2d()
#define FLIP_U_SOCKET					"flip_tangent_u"			//Default : false 
#define FLIP_V_SOCKET					"flip_tangent_v"			//Default : false
#define BUMPMAP_FACTOR_SOCKET			"bumpMap_factor"			//Strength of bump map - Default : float(1.0) - Soft Range : (float(0.0), float(1.0))
#define BUMPMAP_TEXTURE_SOCKET			"bumpMap_texture"			//Default : texture_2d()

//  UVW Projection Group  //

#define PROJECT_UVW_SOCKET			"project_uvw"		//When enabled, UV coordinates will be generated by projecting them from a coordinate system - Default : false
#define UVW_WORLD_OR_OBJECT_SOCKET	"world_or_object"	//When enabled, uses world space for projection, otherwise object space is used - Default : false
#define UV_SPACE_INDEX_SOCKET		"uv_space_index"	//UV Space Index. - Default :  0 - Hard Range : (int(0), int(3))

//  UVW Adjustments Group  //
#define TEXTURE_TRANSLATE_SOCKET		"texture_translate"			//Controls position of texture. - Default :  float2(0.0) - Used only with project Uvw set to true
#define TEXTURE_ROTATE_SOCKET			"texture_rotate"			//Rotates angle of texture in degrees. - Default : float(0.0) - Used only with project Uvw set to true
#define TEXTURE_SCALE_SOCKET			"texture_scale"				//Controls the repetition of the texture. - Default :  float2(1.0) - Used only with project Uvw set to true
#define DETAIL_TEXTURE_TRANSLATE_SOCKET "detail_texture_translate"	//Controls the position of the detail texture. - Default :  float2(0.0) - Used only with project Uvw set to true
#define DETAIL_TEXTURE_ROTATE_SOCKET	"detail_texture_rotate"		//Rotates angle of the detail texture in degrees. - Default :  float(0.0) - Used only with project Uvw set to true
#define DETAIL_TEXTURE_SCALE_SOCKET		"detail_texture_scale"		//Controls the repetition of the detail texture. - Default :  float2(1.0) - Used only with project Uvw set to true

}



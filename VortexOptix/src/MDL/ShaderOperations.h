//#pragma once
//#include "MdlWrapper.h"
//#include "Scene/Nodes/Shader/mdl/ShaderNodes.h"
//
//namespace vtx::mdl
//{
//	using namespace mi;
//	using namespace base;
//	using namespace neuraylib;
//
//	#define DEFAULT_COLOR math::vec3f(0.6f, 0.6f, 0.6f)
//	#define ERROR_COLOR math::vec3f(1.0f, 0.5f, 1.0f)
//	#define DEFAULT_FLOAT 1.0f
//
//	
//	//struct FunctionInfo
//	//{
//	//	enum FunctionTypes
//	//	{
//	//		MDL_CONSTANT,
//	//		MDL_EXPRESSION
//	//	};
//	//	FunctionInfo() = default;
//	//	FunctionInfo(const std::string& _moduleName, const std::string& _functionName);
//	//	FunctionInfo(const std::string& _moduleName, const std::string& _functionName, const std::string& _signature);
//
//	//	/* Module name example: "mdl::base" */
//	//	std::string moduleName;
//	//	/* Function name example: "mdl::base::file_texture" */
//	//	std::string functionName;
//	//	std::map<std::string, ParamInfo> parameters;
//	//	FunctionTypes functionType = MDL_EXPRESSION;
//
//	//	Handle<IExpression> expression;
//
//	//private:
//	//	std::string signature;
//
//	//};
//
//#define GET_SIGNATURE(functionInfo) functionInfo.getSignature().c_str()
//
//	enum MdlFunctionId
//	{
//		MDL_FILE_TEXTURE,
//		MDL_TEXTURE_RETURN_TINT,
//		MDL_TEXTURE_RETURN_MONO,
//
//		MDL_DIFFUSE_REFLECTION,
//		MDL_MATERIAL_SURFACE,
//		MDL_MATERIAL,
//
//		NUM_MDL_FUNCTIONS
//	};
//	/*struct MdlFunctionsDataBase
//	{
//		std::map<MdlFunctionId, FunctionInfo> functions
//		{
//			{ MDL_FILE_TEXTURE, FunctionInfo{ "mdl::base", "mdl::base::file_texture" }},
//			{ MDL_TEXTURE_RETURN_TINT, FunctionInfo{ "mdl::base", "mdl::base::texture_return.tint" }},
//			{ MDL_TEXTURE_RETURN_MONO, FunctionInfo{ "mdl::base", "mdl::base::texture_return.mono" }},
//			{ MDL_DIFFUSE_REFLECTION, FunctionInfo{ "mdl::df", "mdl::df::	 " }},
//			{ MDL_MATERIAL_SURFACE, FunctionInfo{ "mdl", "mdl::material_surface", "mdl::material_surface(bsdf,material_emission)" }},
//			{ MDL_MATERIAL, FunctionInfo{ "mdl", "mdl::material", "mdl::material(bool,material_surface,material_surface,color,material_volume,material_geometry,hair_bsdf)" }}
//		};
//
//	} inline mdlFunctionDb;*/
//
//	inline std::set <std::string> forbiddenTypes{
//			"emission",
//			"backface",
//			"hair",
//			"geometry"
//	};
//	using FunctionArguments = std::map<std::string, FunctionInfo>;
//
//	FunctionInfo getTextureConstant(const std::string& texturePath, const IType_texture::Shape shape = IType_texture::TS_2D, const float gamma = 2.1f);
//
//	FunctionInfo getConstantColor(const math::vec3f& color = DEFAULT_COLOR);
//
//	FunctionInfo getConstantFloat(const float value = DEFAULT_FLOAT);
//
//	FunctionInfo createFunction(ModuleCreationParameters& moduleInfo, MdlFunctionId functionId, graph::shader::ShaderInputSockets& arguments);
//
//	void createNewFunctionInModule(const ModuleCreationParameters& moduleParameters);

//

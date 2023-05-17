#pragma once
#include <mi/mdl_sdk.h>
#include <mi/base/config.h>
#include <mi/neuraylib/ineuray.h>
#include "Core/Utils.h"
#include "Core/Options.h"
#include <string>
#include "Core/Log.h"

#include "Scene/Nodes/Shader/Shader.h"
#include "Scene/Nodes/Material.h"

namespace vtx
{
	namespace graph
	{
		namespace shader
		{
			struct ShaderNodeSocket;
		}
	}
}

namespace vtx::mdl
{
	
	bool logMessage(mi::neuraylib::IMdl_execution_context* context);

	class MdlLogger : public mi::base::Interface_implement<mi::base::ILogger> {
	public:
		void message(mi::base::Message_severity level,
		             const char* /* module_category */,
		             const mi::base::Message_details& /* details */,
		             const char* message) override
		{
			const char* severity = nullptr;

			switch (level)
			{
			case mi::base::MESSAGE_SEVERITY_FATAL:
				severity = "FATAL: ";
				VTX_ASSERT_CLOSE(false, "MDL Log {} {}", severity, message);
				break;
			case mi::base::MESSAGE_SEVERITY_ERROR:
				severity = "ERROR: ";
				VTX_ASSERT_CLOSE(false, "MDL Log {} {}", severity, message);
				break;
			case mi::base::MESSAGE_SEVERITY_WARNING:
				severity = "WARN:  ";
				VTX_WARN("MDL Log {} {}", severity, message);
				break;
			case mi::base::MESSAGE_SEVERITY_INFO:
				//return; // DEBUG No info messages.
				severity = "INFO:  ";
				VTX_INFO("MDL Log {} {}", severity, message);
				break;
			case mi::base::MESSAGE_SEVERITY_VERBOSE:
				severity = "VERBOSE:  ";
				VTX_TRACE("MDL Log {} {}", severity, message);
				break; // DEBUG No verbose messages.
			case mi::base::MESSAGE_SEVERITY_DEBUG:
				severity = "DEBUG:  ";
				VTX_TRACE("MDL Log {} {}", severity, message);
				break; // DEBUG No debug messages.
			case mi::base::MESSAGE_SEVERITY_FORCE_32_BIT:
				return;
			}
		}

		void message(const mi::base::Message_severity level,
		             const char* moduleCategory,
		             const char* message) override
		{
			this->message(level, moduleCategory, mi::base::Message_details(), message);
		}
	};

	struct TransactionInterfaces
	{

		mi::base::Handle<mi::neuraylib::ITransaction>           transaction;
		mi::base::Handle<mi::neuraylib::IValue_factory>         valueFactory;
		mi::base::Handle<mi::neuraylib::IType_factory>          typeFactory;
		mi::base::Handle<mi::neuraylib::IExpression_factory>    expressionFactory;
	};

	struct ModuleCreationParameters
	{
		void reset();

		std::string                                   moduleName;
		std::string                                   functionName;
		mi::base::Handle <mi::neuraylib::IExpression> body;

		mi::base::Handle<mi::neuraylib::IType_list>			parameters;
		int paramCount = 0;
		int potentialParameterCount = 0;
		mi::base::Handle<mi::neuraylib::IExpression_list>	defaults;
		mi::base::Handle<mi::neuraylib::IAnnotation_list>	parameterAnnotations;
		mi::base::Handle<mi::neuraylib::IAnnotation_block>	annotations;
		mi::base::Handle<mi::neuraylib::IAnnotation_block>	returnAnnotations;

		
		
	};

	struct MdlState {

		TransactionInterfaces* getTransactionInterfaces(const bool openFactories = true)
		{
			openTransaction(openFactories);
			return &tI;
		}

		void commitTransaction();

		mi::base::Handle<mi::neuraylib::INeuray>                neuray;
		mi::base::Handle<mi::neuraylib::IMdl_compiler>          compiler;
		mi::base::Handle<mi::neuraylib::IMdl_configuration>     config;
		mi::base::Handle<mi::base::ILogger>                     logger;
		mi::base::Handle<mi::neuraylib::IDatabase>              database;
		mi::base::Handle<mi::neuraylib::IScope>                 globalScope;
		mi::base::Handle<mi::neuraylib::IMdl_factory>           factory;
		mi::base::Handle<mi::neuraylib::IMdl_execution_context> context;
		mi::base::Handle<mi::neuraylib::IMdl_backend>           backend;
		mi::base::Handle<mi::neuraylib::IImage_api>             imageApi;
		mi::base::Handle<mi::neuraylib::IMdl_impexp_api>        impExpApi;
		std::vector<std::string>                                searchStartupPaths;
		std::string                                             lastError;
		mi::Sint32                                              result;
		ModuleCreationParameters								moduleCreationParameter;

	private:
		void openTransaction(const bool openFactories);
		TransactionInterfaces									tI;
	};

	
	MdlState* getState();

	/*ShutDown the MDL SDK*/
	void shutDown();

	/*Initialize the MDL SDK*/
	void init();

	/*Add a search path to the MDL SDK*/
	void addSearchPath(std::string path);

	std::string pathToModuleName(const std::string& materialPath);

	/*Compile a material from a given path*/
	void compileMaterial(const std::string& path, std::string materialName, std::string* materialDbName=nullptr);

	/*Get the shader configuration for the given material*/
	graph::Shader::Configuration determineShaderConfiguration(const std::string& materialDbName);

	/*Create a target code for the given material and configuration*/
	mi::base::Handle<mi::neuraylib::ITarget_code const> createTargetCode(const std::string& materialDbName, const graph::Shader::Configuration& config, const vtxID& shaderIndex);

	/*Analyze target Code and extract all parameters, it sets the values of the argumentBlockClone, params list and mapEnumTypes*/	
	void setMaterialParameters(const std::string& materialDbName,
							   const mi::base::Handle<mi::neuraylib::ITarget_code const>& targetCode,
							   mi::base::Handle<mi::neuraylib::ITarget_argument_block>& argumentBlockClone,
							   std::list<graph::ParamInfo>& params,
							   std::map<std::string, std::shared_ptr<graph::EnumTypeInfo>>& mapEnumTypes);

	std::shared_ptr<graph::Texture> createTextureFromFile(const std::string& filePath);

	std::string removeMdlPrefix(const std::string& name);

	std::string createTextureExpression(const std::string & filePath);
	/*Analyze mdl to prepare cuda descriptors for texture*/
	void fetchTextureData(const std::shared_ptr<graph::Texture>& textureNode);

	/*fetch data to create bsdf sampling*/
	graph::BsdfMeasurement::BsdfPartData fetchBsdfData(const std::string& bsdfDbName, const mi::neuraylib::Mbsdf_part part);

	/*fetch data to create light profile sampling*/
	graph::LightProfile::LightProfileData fetchLightProfileData(const std::string& lightDbName);

	struct MdlFunctionInfo
	{
		std::string                             module;
		std::string                             name;
		std::string                             signature;
		mi::base::Handle<const mi::neuraylib::IType> returnType;
	};

	struct ParameterInfo
	{
		std::string                                        argumentName;
		std::string                                        actualName;
		mi::base::Handle<const mi::neuraylib::IType>       type;
		mi::base::Handle<const mi::neuraylib::IExpression> defaultValue;
		mi::neuraylib::IType::Kind                         kind;
		int                                                index;
	};

	void getFunctionSignature(MdlFunctionInfo* functionInfo);

	std::vector<ParameterInfo> getFunctionParameters(const MdlFunctionInfo& functionInfo, vtxID callingNodeId);

	mi::base::Handle<mi::neuraylib::IExpression> generateFunctionExpression(const std::string& functionSignature, std::map<std::string, graph::shader::ShaderNodeSocket>& sockets);

	void setRendererModule(const std::string& rendererModule, const std::string& visibleFunction);

	std::tuple<std::string, std::string> createNewFunctionInModule(std::shared_ptr<graph::shader::ShaderNode> shaderGraph);

	mi::base::Handle<mi::neuraylib::IExpression> createConstantColor(const math::vec3f& color);

	mi::base::Handle<mi::neuraylib::IExpression> createConstantFloat(const float value);

	mi::base::Handle<mi::neuraylib::IExpression> createTextureConstant(const std::string& texturePath, const mi::neuraylib::IType_texture::Shape shape = mi::neuraylib::IType_texture::TS_2D, const float gamma = 2.2f);
	

	// Utility function to dump the arguments of a material instance or function call.
	template <class T>
	void dumpInstance(mi::neuraylib::IExpression_factory* expression_factory, const T* instance, std::string name)
	{
		using namespace mi::neuraylib;
		using namespace mi::base;
		using namespace mi;
		std::stringstream s;
		s << "Dumping material/function instance \""<<name<<" " << instance->get_mdl_function_definition() << "\":" << "\n";

		const Size                           count = instance->get_parameter_count();
		const Handle<const IExpression_list> arguments(instance->get_arguments());

		for (Size index = 0; index < count; index++) {

			Handle<const IExpression>   argument(arguments->get_expression(index));
			std::string                 name = instance->get_parameter_name(index);
			const Handle<const IString> argument_text(expression_factory->dump(argument.get(), name.c_str(), 1));
			s << "    argument " << argument_text->get_c_str() << "\n";

		}
		s << "\n";
		VTX_INFO("{}", s.str());
	}

	template <class T>
	std::string dumpDefinition(mi::neuraylib::ITransaction* transaction, mi::neuraylib::IMdl_factory* mdl_factory, const T* definition, mi::Size depth)
	{
		using namespace mi::neuraylib;
		using namespace mi::base;
		using namespace mi;
		std::stringstream ss;
		Handle<IType_factory> type_factory(mdl_factory->create_type_factory(transaction));
		Handle<IExpression_factory> expression_factory(mdl_factory->create_expression_factory(transaction));

		Size                           count = definition->get_parameter_count();
		Handle<const IType_list>       types(definition->get_parameter_types());
		Handle<const IExpression_list> defaults(definition->get_defaults());

		for (Size index = 0; index < count; index++) {

			Handle<const IType>   type(types->get_type(index));
			Handle<const IString> type_text(type_factory->dump(type.get(), depth + 1));
			std::string           name = definition->get_parameter_name(index);
			ss << "    parameter " << type_text->get_c_str() << " " << name;

			Handle<const IExpression> default_(defaults->get_expression(name.c_str()));
			if (default_.is_valid_interface()) {
				Handle<const IString> default_text(expression_factory->dump(default_.get(), 0, depth + 1));
				ss << ", default = " << default_text->get_c_str() << "\n";
			}
			else {
				ss << " (no default)" << "\n";
			}

		}

		Size temporary_count = definition->get_temporary_count();
		for (Size i = 0; i < temporary_count; ++i) {
			Handle<const IExpression> temporary(definition->get_temporary(i));
			std::stringstream name;
			name << i;
			Handle<const IString> result(expression_factory->dump(temporary.get(), name.str().c_str(), 1));
			ss << "    temporary " << result->get_c_str() << "\n";
		}

		Handle<const IExpression> body(definition->get_body());
		Handle<const IString>     result(expression_factory->dump(body.get(), 0, 1));
		if (result)
			ss << "    body " << result->get_c_str() << "\n";
		else
			ss << "    body not available for this function" << "\n";

		ss << "\n";

		return ss.str();
	}


}


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

namespace vtx::mdl
{

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

	struct State {
		mi::base::Handle<mi::neuraylib::INeuray>                neuray;
		mi::base::Handle<mi::neuraylib::IMdl_compiler>          compiler;
		mi::base::Handle<mi::neuraylib::IMdl_configuration>     config;
		mi::base::Handle<mi::base::ILogger>                     logger;
		mi::base::Handle<mi::neuraylib::IDatabase>              database;
		mi::base::Handle<mi::neuraylib::IScope>                 globalScope;
		mi::base::Handle<mi::neuraylib::IMdl_factory>           factory;
		mi::base::Handle<mi::neuraylib::IExpression_factory>    expressionFactory;
		mi::base::Handle<mi::neuraylib::IValue_factory>         valueFactory;
		mi::base::Handle<mi::neuraylib::IType_factory>          typeFactory;
		mi::base::Handle<mi::neuraylib::IMdl_execution_context> context;
		mi::base::Handle<mi::neuraylib::IMdl_backend>           backend;
		mi::base::Handle<mi::neuraylib::IImage_api>             imageApi;
		mi::base::Handle<mi::neuraylib::ITransaction>           transaction;
		mi::base::Handle<mi::neuraylib::IMdl_impexp_api>        impExpApi;
		std::vector<std::string>                                searchStartupPaths;
		std::string                                             lastError;
		mi::Sint32                                              result;
		State(): result(0)
		{
			printf("INITIALIZING STATE MDL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
		}
	};

	State* getState();

	/*ShutDown the MDL SDK*/
	void shutDown();

	/*Initialize the MDL SDK*/
	void init();

	/*Add a search path to the MDL SDK*/
	void addSearchPath(std::string path);

	/*Get the global transaction if it exists, create a new one if it doesn't*/
	mi::neuraylib::ITransaction* getGlobalTransaction();

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

	/*Analyze mdl to prepare cuda descriptors for texture*/
	void fetchTextureData(const std::shared_ptr<graph::Texture>& textureNode);

	/*fetch data to create bsdf sampling*/
	graph::BsdfMeasurement::BsdfPartData fetchBsdfData(const std::string& bsdfDbName, const mi::neuraylib::Mbsdf_part part);

	/*fetch data to create light profile sampling*/
	graph::LightProfile::LightProfileData fetchLightProfileData(const std::string& lightDbName);

}


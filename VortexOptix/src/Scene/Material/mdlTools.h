#pragma once
#include <mi/mdl_sdk.h>
#include <mi/base/config.h>
#include <mi/neuraylib/ineuray.h>
#include "Core/Utils.h"
#include "Core/Options.h"
#include <string>
#include "Core/Log.h"

#include "Scene/Nodes/Shader.h"

namespace vtx::mdl
{

	class MDLlogger : public mi::base::Interface_implement<mi::base::ILogger> {
	public:
		void message(mi::base::Message_severity level,
		             const char* /* module_category */,
		             const mi::base::Message_details& /* details */,
		             const char* message) override
		{
			const char* severity = 0;

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

		void message(mi::base::Message_severity level,
		             const char* module_category,
		             const char* message) override
		{
			this->message(level, module_category, mi::base::Message_details(), message);
		}
	};

	struct State {
		mi::base::Handle<mi::neuraylib::INeuray>                neuray;
		mi::base::Handle<mi::neuraylib::IMdl_compiler>          compiler;
		mi::base::Handle<mi::neuraylib::IMdl_configuration>     config;
		mi::base::Handle<mi::base::ILogger>                     logger;
		mi::base::Handle<mi::neuraylib::IDatabase>              database;
		mi::base::Handle<mi::neuraylib::IScope>                 global_scope;
		mi::base::Handle<mi::neuraylib::IMdl_factory>           factory;
		mi::base::Handle<mi::neuraylib::IExpression_factory>    expression_factory;
		mi::base::Handle<mi::neuraylib::IValue_factory>         value_factory;
		mi::base::Handle<mi::neuraylib::IType_factory>          type_factory;
		mi::base::Handle<mi::neuraylib::IMdl_execution_context> context;
		mi::base::Handle<mi::neuraylib::IMdl_backend>           backend;
		mi::base::Handle<mi::neuraylib::IImage_api>             imageApi;
		mi::base::Handle<mi::neuraylib::ITransaction>           transaction;
		mi::base::Handle<mi::neuraylib::IMdl_impexp_api>        impexpApi;
		std::vector<std::string>                                searchStartupPaths;
		std::string                                             lastError;
		mi::Sint32                                              result;
		State() {
			printf("INITIALIZING STATE MDL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
		}
	};

	State* getState();

	void shutDown();

	void init();

	//void loadIneuray();

	//void printLibraryVersion();

	//void configure();

	void addSearchPath(std::string path);

	//void loadPlugin(std::string path);

	//void startInterfaces();

	mi::neuraylib::ITransaction* getGlobalTransaction();

	//bool logMessage(mi::neuraylib::IMdl_execution_context* context);

	void loadShaderData(std::shared_ptr<graph::Shader> shader);

	//void storeMaterialInfo();

}


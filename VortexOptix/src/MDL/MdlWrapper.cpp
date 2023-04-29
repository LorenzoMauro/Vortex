#include "MdlWrapper.h"
#include <execution>
#include "Device/OptixWrapper.h"

namespace vtx::mdl
{
	using namespace mi;
	using namespace base;
	using namespace neuraylib;

	static State state;

	State* getState() {
		return &state;
	}

	const char* messageKindToString(IMessage::Kind message_kind)
	{
		switch (message_kind)
		{
		case IMessage::MSG_INTEGRATION:
			return "MDL SDK";
		case IMessage::MSG_IMP_EXP:
			return "Importer/Exporter";
		case IMessage::MSG_COMILER_BACKEND:
			return "Compiler Backend";
		case IMessage::MSG_COMILER_CORE:
			return "Compiler Core";
		case IMessage::MSG_COMPILER_ARCHIVE_TOOL:
			return "Compiler Archive Tool";
		case IMessage::MSG_COMPILER_DAG:
			return "Compiler DAG generator";
		default:
			break;
		}
		return "";
	}

	bool logMessage(IMdl_execution_context* context)
	{
		state.lastError = "";
		for (mi::Size i = 0; i < context->get_messages_count(); ++i)
		{
			const Handle<const IMessage> message(context->get_message(i));
			state.lastError += messageKindToString(message->get_kind());
			state.lastError += " ";
			state.lastError += ": ";
			state.lastError += message->get_string();
			switch (message->get_severity()) {
			case MESSAGE_SEVERITY_ERROR:
				VTX_ERROR(state.lastError);
				break;
			case MESSAGE_SEVERITY_WARNING:
				VTX_WARN(state.lastError);
				break;
			case MESSAGE_SEVERITY_INFO:
				VTX_INFO(state.lastError);
				break;
			case MESSAGE_SEVERITY_VERBOSE:
				VTX_TRACE(state.lastError);
				break;
			case MESSAGE_SEVERITY_DEBUG:
				VTX_TRACE(state.lastError);
				break;
			default:
				break;
			}

		}
		return context->get_error_messages_count() == 0;
	}

	void loadIneuray()
	{
		std::string filename = utl::absolutePath(getOptions()->dllPath + "libmdl_sdk.dll");
		const HMODULE handle = LoadLibraryA(filename.c_str());
		VTX_ASSERT_CLOSE(handle, "Error Loading {} with error code {}", filename, GetLastError());

		void* symbol = GetProcAddress(handle, "mi_factory");
		VTX_ASSERT_CLOSE(symbol, "ERROR: GetProcAddress(handle, \"mi_factory\") failed with error {}", GetLastError());

		state.neuray = mi_factory<INeuray>(symbol);
		if (!state.neuray.get())
		{
			const Handle<IVersion> version(mi_factory<IVersion>(symbol));
			VTX_ASSERT_CLOSE(version, "ERROR: Incompatible library. Could not determine version.");
			VTX_ASSERT_CLOSE(!version, "ERROR: Library version {} does not match header version {}", version->get_product_version(), MI_NEURAYLIB_PRODUCT_VERSION_STRING);
		}

		VTX_INFO("MDL SDK Loaded");
	}

	void loadPlugin(std::string path)
	{
		const Handle<IPlugin_configuration> plugin_conf(state.neuray->get_api_component<IPlugin_configuration>());

		// Try loading the requested plugin before adding any special handling
		state.result = plugin_conf->load_plugin_library(path.c_str());
		VTX_ASSERT_CLOSE(state.result == 0, "load_plugin( {} ) failed with {}", path, state.result);
	}

	void configure()
	{
		// Create the MDL compiler.
		state.compiler = state.neuray->get_api_component<IMdl_compiler>();
		VTX_ASSERT_CLOSE(state.compiler, "ERROR: Initialization of MDL compiler failed.");

		// Configure Neuray.
		state.config = state.neuray->get_api_component<IMdl_configuration>();
		VTX_ASSERT_CLOSE(state.config, "ERROR: Retrieving MDL configuration failed.");

		state.logger = make_handle(new MdlLogger());
		state.config->set_logger(state.logger.get());

		// Path for search
		state.config->add_mdl_system_paths(); //MDL_SYSTEM_PATH is set to "C:\ProgramData\NVIDIA Corporation\mdl\" by default
		state.config->add_mdl_user_paths(); //MDL_USER_PATH is set to "C:\Users\<username>\Documents\mdl\" by default
		// Add all additional MDL and resource search paths defined inside the system description file as well.
		for (auto const& path : state.searchStartupPaths)
		{
			addSearchPath(path);
		}
		// Load plugins.

		std::string freeimagePath = utl::absolutePath(getOptions()->dllPath + "freeimage.dll");
		const HMODULE handle = LoadLibraryA(freeimagePath.c_str());
		VTX_ASSERT_CLOSE(handle, "Error Loading {} with error code {}", freeimagePath, GetLastError());

		const std::string ddsPath = getOptions()->dllPath + "dds.dll";
		const std::string nv_freeimagePath = utl::absolutePath(getOptions()->dllPath + "nv_freeimage.dll");


		loadPlugin(nv_freeimagePath);
		loadPlugin(ddsPath);

	}

	void startInterfaces()
	{

		VTX_ASSERT_CLOSE(state.neuray->start() == 0, "FATAL: Starting MDL SDK failed.");

		state.database = state.neuray->get_api_component<IDatabase>();

		state.globalScope = state.database->get_global_scope();

		state.factory = state.neuray->get_api_component<IMdl_factory>();

		state.impExpApi = state.neuray->get_api_component<IMdl_impexp_api>();

		ITransaction* transaction = getGlobalTransaction();

		state.expressionFactory = state.factory->create_expression_factory(transaction);
		state.valueFactory = state.factory->create_value_factory(transaction);
		state.typeFactory = state.factory->create_type_factory(transaction);

		// Configure the execution context.
		// Used for various configurable operations and for querying warnings and error messages.
		// It is possible to have more than one, in order to use different settings.
		state.context = state.factory->create_execution_context();

		state.context->set_option("internal_space", "coordinate_world");  // equals default
		state.context->set_option("bundle_resources", false);             // equals default
		state.context->set_option("meters_per_scene_unit", 1.0f);         // equals default
		state.context->set_option("mdl_wavelength_min", 380.0f);          // equals default
		state.context->set_option("mdl_wavelength_max", 780.0f);          // equals default
		// If true, the "geometry.normal" field will be applied to the MDL state prior to evaluation of the given DF.
		state.context->set_option("include_geometry_normal", true);       // equals default 

		const Handle<IMdl_backend_api> mdl_backend_api(state.neuray->get_api_component<IMdl_backend_api>());

		state.backend = mdl_backend_api->get_backend(IMdl_backend_api::MB_CUDA_PTX);

		state.result = state.backend->set_option("num_texture_spaces", std::to_string(getOptions()->numTextureSpaces).c_str());
		VTX_ASSERT_CLOSE(state.result == 0, "Error with number of texture spaces");

		state.result = state.backend->set_option("num_texture_results", std::to_string(getOptions()->numTextureResults).c_str());
		VTX_ASSERT_CLOSE(state.result == 0, "Error with number of texture results");

		// Use SM 5.0 for Maxwell and above.
		state.result = state.backend->set_option("sm_version", "50");
		VTX_ASSERT_CLOSE(state.result == 0, "Error with sm version");

		state.result = state.backend->set_option("tex_lookup_call_mode", "direct_call");
		VTX_ASSERT_CLOSE(state.result == 0, "Error with tex look up mode");

		state.result = state.backend->set_option("enable_auxiliary", "on");
		VTX_ASSERT_CLOSE(state.result == 0, "Error with enable auxiliary");

		state.result = state.backend->set_option("fast_math", "on");
		VTX_ASSERT_CLOSE(state.result == 0, "Error with tex fast_math mode");

		if (getOptions()->enable_derivatives) //Not supported in this renderer
		{
			// Option "texture_runtime_with_derivs": Default is disabled.
			// We enable it to get coordinates with derivatives for texture lookup functions.
			state.result = state.backend->set_option("texture_runtime_with_derivs", "on");
			VTX_ASSERT_CLOSE(state.result == 0, "Error with texture runtime with derivatives");
		}

		if (getOptions()->isDebug) {
			Sint32 result;
			result = state.backend->set_option("opt_level", getOptions()->mdlOptLevel);
			VTX_ASSERT_CLOSE((result == 0), "Error with opt level");
			//result = state.backend->set_option("enable_exceptions", "on");
		}
		else {
			VTX_ASSERT_CLOSE((state.backend->set_option("inline_aggressively", "on") == 0), "Error with inline aggressive");
		}

		// FIXME Determine what scene data the renderer needs to provide here.
		// FIXME scene_data_names is not a supported option anymore!
		//if (state.mdl_backend->set_option("scene_data_names", "*") != 0)
		//{
		//  return false;
		//}

		state.imageApi = state.neuray->get_api_component<IImage_api>();
	}

	void printLibraryVersion() {
		// Print library version information.
		Handle<const IVersion> version(
			state.neuray->get_api_component<const IVersion>());

		VTX_INFO("MDL SDK header version          = {}", MI_NEURAYLIB_PRODUCT_VERSION_STRING);
		VTX_INFO("MDL SDK library product name    = {}", version->get_product_name());
		VTX_INFO("MDL SDK library product version = {}", version->get_product_version());
		VTX_INFO("MDL SDK library build number    = {}", version->get_build_number());
		VTX_INFO("MDL SDK library build date      = {}", version->get_build_date());
		VTX_INFO("MDL SDK library build platform  = {}", version->get_build_platform());
		VTX_INFO("MDL SDK library version string  = {}", version->get_string());

		Uuid neuray_id_libraray = version->get_neuray_iid();
		Uuid neuray_id_interface = INeuray::IID();

		VTX_INFO("MDL SDK header interface ID     = <{}, {}, {}, {}>",
		         neuray_id_interface.m_id1,
		         neuray_id_interface.m_id2,
		         neuray_id_interface.m_id3,
		         neuray_id_interface.m_id4);
		VTX_INFO("MDL SDK library interface ID    = <{}, {}, {}, {}>",
		         neuray_id_libraray.m_id1,
		         neuray_id_libraray.m_id2,
		         neuray_id_libraray.m_id3,
		         neuray_id_libraray.m_id4);

		version = 0;
	}

	void init() {
		loadIneuray();
		printLibraryVersion();
		for (std::string path : getOptions()->mdlSearchPath) {
			state.searchStartupPaths.push_back(utl::absolutePath(path));
		}
		configure();
		startInterfaces();
	}

	void shutDown() {

		VTX_INFO("Shutdown: MDL");
		state.imageApi.reset();
		state.backend.reset();
		state.context.reset();
		state.factory.reset();
		state.globalScope.reset();
		state.database.reset();
		state.config.reset();
		state.compiler.reset();
		state.neuray->shutdown();
	}

	void addSearchPath(std::string path)
	{
		state.result = state.config->add_mdl_path(path.c_str());
		VTX_ASSERT_CONTINUE(state.result == 0, "add_mdl_path( {} ) failed with {}", path, state.result);

		state.result = state.config->add_resource_path(path.c_str());
		VTX_ASSERT_CONTINUE(state.result == 0, "add_resource_path( {} ) failed with {}", path, state.result);
	}
        
	ITransaction* getGlobalTransaction() {
		if (!state.transaction || !(state.transaction.get()->is_open())) {
			state.transaction = make_handle<ITransaction>(state.globalScope->create_transaction());
			return state.transaction.get();
		}
		else {
			return state.transaction.get();
		}

	}

	/*path expressed as relative to the search path added to mdl sdk*/
	std::string pathToModuleName(const std::string& materialPath) {
		// Replace backslashes with colons
		std::string output;

		for (size_t i = 0; i < materialPath.size(); ++i) {
			if (materialPath[i] == '\\') {
				output.push_back(':');
				output.push_back(':');
			}
			else {
				output.push_back(materialPath[i]);
			}
		}

		// Remove the file extension
		const std::size_t last_dot = output.find_last_of('.');
		if (last_dot != std::string::npos) {
			output.erase(last_dot);
		}

		return output;
	};

	std::string getMaterialDatabaseName(const IModule* module, const IString* moduleDatabaseName, const std::string& materialName) {
		std::string materialDatabaseName = std::string(moduleDatabaseName->get_c_str()) + "::" + materialName;

		// Return input if it already contains a signature.
		if (materialDatabaseName.back() == ')')
		{
			return materialDatabaseName;
		}

		const Handle<const mi::IArray> result(module->get_function_overloads(materialDatabaseName.c_str()));

		// Not supporting multiple function overloads with the same name but different signatures.
		if (!result || result->get_length() != 1)
		{
			return std::string();
		}

		const Handle<const mi::IString> overloads(result->get_element<mi::IString>(static_cast<mi::Size>(0)));

		return overloads->get_c_str();

	}
	
	void compileMaterial(const std::string& path, std::string materialName, std::string* materialDbName)
	{
		Sint32 result;
		ITransaction* transaction = getGlobalTransaction();

		std::string moduleName = pathToModuleName(path);

		result = state.impExpApi->load_module(transaction, moduleName.c_str(), state.context.get());

		do {
			VTX_ASSERT_RETURN((result >= 0 && logMessage(state.context.get())), state.lastError);
			const Handle<const IString> moduleDatabaseName(state.factory->get_db_module_name(moduleName.c_str()));
			// access module
			const Handle<const IModule> module(transaction->access<IModule>(moduleDatabaseName.get()->get_c_str()));
			VTX_ASSERT_BREAK((module.is_valid_interface()), "Error with module access");

			// define material module Name and verify no overload
			const std::string materialDatabaseName = getMaterialDatabaseName(module.get(), moduleDatabaseName.get(), materialName);
			if (materialDbName != nullptr) {
				*materialDbName = materialDatabaseName;
			}
			VTX_ASSERT_BREAK(!materialDatabaseName.empty(), "Error with retrieving material {} in the module {}, material might have overload", materialName, moduleName);

			// Create material Definition
			const Handle<const IFunction_definition> material_definition(transaction->access<IFunction_definition>(materialDatabaseName.c_str()));
			VTX_ASSERT_BREAK((material_definition.is_valid_interface()), "Error with material definition creation for material {} in module {}", materialName, moduleName);

			//Create material Call
			const Handle<IFunction_call> materialCall(material_definition->create_function_call(0, &result));
			VTX_ASSERT_BREAK((materialCall.is_valid_interface() && !(result != 0)), "Error with material instance creation for material {} in module {}", materialName, moduleName);

			//Create material Instance
			const Handle<IMaterial_instance> materialInstance(materialCall->get_interface<IMaterial_instance>());
			VTX_ASSERT_BREAK((materialInstance.is_valid_interface()), "Error with material instance creation for material {} in module {}", materialName, moduleName);

			//Create compiled material
			const Uint32 flags = IMaterial_instance::CLASS_COMPILATION;
			const Handle<ICompiled_material> compiledMaterial(materialInstance->create_compiled_material(flags, state.context.get()));
			VTX_ASSERT_BREAK((logMessage(state.context.get())), state.lastError);

			transaction->store(compiledMaterial.get(), ("compiledMaterial_" + materialDatabaseName).c_str());
		} while (0);

	}

	bool isValidDistribution(IExpression const* expr)
	{
		if (expr == nullptr)
		{
			return false;
		}

		if (expr->get_kind() == IExpression::EK_CONSTANT)
		{
			const Handle<IExpression_constant const> expr_constant(expr->get_interface<IExpression_constant>());
			const Handle<IValue const> value(expr_constant->get_value());

			if (value->get_kind() == IValue::VK_INVALID_DF)
			{
				return false;
			}
		}

		return true;
	}

	graph::Shader::Configuration determineShaderConfiguration(const std::string& materialDbName)
	{
		ITransaction* transaction = getGlobalTransaction();
		graph::Shader::Configuration config;
		do {
			const Handle<const ICompiled_material> compiledMaterial((transaction->access(("compiledMaterial_" + materialDbName).c_str())->get_interface<ICompiled_material>()));
			//const ICompiled_material* compiledMaterial = shader->compilation.compiledMaterial.get();
			config.isThinWalledConstant = false;
			config.thinWalled = false;

			Handle<IExpression const> thin_walled_expr(compiledMaterial->lookup_sub_expression("thin_walled"));
			if (thin_walled_expr->get_kind() == IExpression::EK_CONSTANT)
			{
				config.isThinWalledConstant = true;

				Handle<IExpression_constant const> expr_const(thin_walled_expr->get_interface<IExpression_constant const>());
				Handle<IValue_bool const> value_bool(expr_const->get_value<IValue_bool>());

				config.thinWalled = value_bool->get_value();
			}

			Handle<IExpression const> surface_scattering_expr(compiledMaterial->lookup_sub_expression("surface.scattering"));

			config.isSurfaceBsdfValid = isValidDistribution(surface_scattering_expr.get()); // True if surface.scattering != bsdf().

			config.isBackfaceBsdfValid = false;

			// The backface scattering is only used for thin-walled materials.
			if (!config.isThinWalledConstant || config.thinWalled)
			{
				// When backface == bsdf() MDL uses the surface scattering on both sides, irrespective of the thin_walled state.
				Handle<IExpression const> backface_scattering_expr(compiledMaterial->lookup_sub_expression("backface.scattering"));

				config.isBackfaceBsdfValid = isValidDistribution(backface_scattering_expr.get()); // True if backface.scattering != bsdf().

				if (config.isBackfaceBsdfValid)
				{
					// Only use the backface scattering when it's valid and different from the surface scattering expression.
					config.isBackfaceBsdfValid = (compiledMaterial->get_slot_hash(SLOT_SURFACE_SCATTERING) !=
												  compiledMaterial->get_slot_hash(SLOT_BACKFACE_SCATTERING));
				}
			}

			// Surface EDF.
			Handle<IExpression const> surface_edf_expr(compiledMaterial->lookup_sub_expression("surface.emission.emission"));

			config.isSurfaceEdfValid = isValidDistribution(surface_edf_expr.get());

			config.isSurfaceIntensityConstant = true;
			config.surfaceIntensity = mi::math::Color(0.0f, 0.0f, 0.0f);
			config.isSurfaceIntensityModeConstant = true;
			config.surfaceIntensityMode = 0; // == intensity_radiant_exitance;

			if (config.isSurfaceEdfValid)
			{
				// Surface emission intensity.
				Handle<IExpression const> surface_intensity_expr(compiledMaterial->lookup_sub_expression("surface.emission.intensity"));

				config.isSurfaceIntensityConstant = false;

				if (surface_intensity_expr->get_kind() == IExpression::EK_CONSTANT)
				{
					Handle<IExpression_constant const> intensity_const(surface_intensity_expr->get_interface<IExpression_constant const>());
					Handle<IValue_color const> intensity_color(intensity_const->get_value<IValue_color>());

					if (get_value(intensity_color.get(), config.surfaceIntensity) == 0)
					{
						config.isSurfaceIntensityConstant = true;
					}
				}

				// Surface emission mode. This is a uniform and normally the default intensity_radiant_exitance
				Handle<IExpression const> surface_intensity_mode_expr(compiledMaterial->lookup_sub_expression("surface.emission.mode"));

				config.isSurfaceIntensityModeConstant = false;

				if (surface_intensity_mode_expr->get_kind() == IExpression::EK_CONSTANT)
				{
					Handle<IExpression_constant const> expr_const(surface_intensity_mode_expr->get_interface<IExpression_constant const>());
					Handle<IValue_enum const> value_enum(expr_const->get_value<IValue_enum>());

					config.surfaceIntensityMode = value_enum->get_value();

					config.isSurfaceIntensityModeConstant = true;
				}
			}

			// Backface EDF.
			config.isBackfaceEdfValid = false;
			// DEBUG Is any of this needed at all or is the BSDF init() function handling all this?
			config.isBackfaceIntensityConstant = true;
			config.backfaceIntensity = mi::math::Color(0.0f, 0.0f, 0.0f);
			config.isBackfaceIntensityModeConstant = true;
			config.backfaceIntensityMode = 0; // == intensity_radiant_exitance;
			config.useBackfaceEdf = false;
			config.useBackfaceIntensity = false;
			config.useBackfaceIntensityMode = false;

			// A backface EDF is only used on thin-walled materials with a backface.emission.emission != edf()
			if (!config.isThinWalledConstant || config.thinWalled)
			{
				Handle<IExpression const> backface_edf_expr(compiledMaterial->lookup_sub_expression("backface.emission.emission"));

				config.isBackfaceEdfValid = isValidDistribution(backface_edf_expr.get());

				if (config.isBackfaceEdfValid)
				{
					// Backface emission intensity.
					Handle<IExpression const> backface_intensity_expr(compiledMaterial->lookup_sub_expression("backface.emission.intensity"));

					config.isBackfaceIntensityConstant = false;

					if (backface_intensity_expr->get_kind() == IExpression::EK_CONSTANT)
					{
						Handle<IExpression_constant const> intensity_const(backface_intensity_expr->get_interface<IExpression_constant const>());
						Handle<IValue_color const> intensity_color(intensity_const->get_value<IValue_color>());

						if (get_value(intensity_color.get(), config.backfaceIntensity) == 0)
						{
							config.isBackfaceIntensityConstant = true;
						}
					}

					// Backface emission mode. This is a uniform and normally the default intensity_radiant_exitance.
					Handle<IExpression const> backface_intensity_mode_expr(compiledMaterial->lookup_sub_expression("backface.emission.mode"));

					config.isBackfaceIntensityModeConstant = false;

					if (backface_intensity_mode_expr->get_kind() == IExpression::EK_CONSTANT)
					{
						Handle<IExpression_constant const> expr_const(backface_intensity_mode_expr->get_interface<IExpression_constant const>());
						Handle<IValue_enum const> value_enum(expr_const->get_value<IValue_enum>());

						config.backfaceIntensityMode = value_enum->get_value();

						config.isBackfaceIntensityModeConstant = true;
					}

					// When surface and backface expressions are identical, reuse the surface expression to generate less code.
					config.useBackfaceEdf = (compiledMaterial->get_slot_hash(SLOT_SURFACE_EMISSION_EDF_EMISSION) !=
											 compiledMaterial->get_slot_hash(SLOT_BACKFACE_EMISSION_EDF_EMISSION));

					// If the surface and backface emission use different intensities then use the backface emission intensity.
					config.useBackfaceIntensity = (compiledMaterial->get_slot_hash(SLOT_SURFACE_EMISSION_INTENSITY) !=
												   compiledMaterial->get_slot_hash(SLOT_BACKFACE_EMISSION_INTENSITY));

					// If the surface and backface emission use different modes (radiant exitance vs. power) then use the backface emission intensity mode.
					config.useBackfaceIntensityMode = (compiledMaterial->get_slot_hash(SLOT_SURFACE_EMISSION_MODE) !=
													   compiledMaterial->get_slot_hash(SLOT_BACKFACE_EMISSION_MODE));
				}
			}

			config.isIorConstant = true;
			config.ior = mi::math::Color(1.0f, 1.0f, 1.0f);

			Handle<IExpression const> ior_expr(compiledMaterial->lookup_sub_expression("ior"));
			if (ior_expr->get_kind() == IExpression::EK_CONSTANT)
			{
				Handle<IExpression_constant const> expr_const(ior_expr->get_interface<IExpression_constant const>());
				Handle<IValue_color const> value_color(expr_const->get_value<IValue_color>());

				if (get_value(value_color.get(), config.ior) == 0)
				{
					config.isIorConstant = true;
				}
			}
			else
			{
				config.isIorConstant = false;
			}

			// If the VDF is valid, it is the df::anisotropic_vdf(). ::vdf() is not a valid VDF.
			// Though there aren't any init, sample, eval or pdf functions genereted for a VDF.
			Handle<IExpression const> volume_vdf_expr(compiledMaterial->lookup_sub_expression("volume.scattering"));

			config.isVdfValid = isValidDistribution(volume_vdf_expr.get());

			// Absorption coefficient. Can be used without valid VDF.
			config.isAbsorptionCoefficientConstant = true;  // Default to constant and no absorption.
			config.useVolumeAbsorption = false; // If there is no abosorption, the absorption coefficient is constant zero.
			config.absorptionCoefficient = mi::math::Color(0.0f, 0.0f, 0.0f); // No absorption.

			Handle<IExpression const> volume_absorption_coefficient_expr(compiledMaterial->lookup_sub_expression("volume.absorption_coefficient"));

			if (volume_absorption_coefficient_expr->get_kind() == IExpression::EK_CONSTANT)
			{
				Handle<IExpression_constant const> expr_const(volume_absorption_coefficient_expr->get_interface<IExpression_constant const>());
				Handle<IValue_color const> value_color(expr_const->get_value<IValue_color>());

				if (get_value(value_color.get(), config.absorptionCoefficient) == 0)
				{
					config.isAbsorptionCoefficientConstant = true;

					if (config.absorptionCoefficient[0] != 0.0f || config.absorptionCoefficient[1] != 0.0f || config.absorptionCoefficient[2] != 0.0f)
					{
						config.useVolumeAbsorption = true;
					}
				}
			}
			else
			{
				config.isAbsorptionCoefficientConstant = false;
				config.useVolumeAbsorption = true;
			}

			// Scattering coefficient. Only used when there is a valid VDF. 
			config.isScatteringCoefficientConstant = true; // Default to constant and no scattering. Assumes invalid VDF.
			config.useVolumeScattering = false;
			config.scatteringCoefficient = mi::math::Color(0.0f, 0.0f, 0.0f); // No scattering

			// Directional bias (Henyey_Greenstein g factor.) Only used when there is a valid VDF and volume scattering coefficient not zero.
			config.isDirectionalBiasConstant = true;
			config.directionalBias = 0.0f;

			// The anisotropic_vdf() is the only valid VDF. 
			// The scattering_coefficient, directional_bias (and emission_intensity) are only needed when there is a valid VDF.
			if (config.isVdfValid)
			{
				Handle<IExpression const> volume_scattering_coefficient_expr(compiledMaterial->lookup_sub_expression("volume.scattering_coefficient"));

				if (volume_scattering_coefficient_expr->get_kind() == IExpression::EK_CONSTANT)
				{
					Handle<IExpression_constant const> expr_const(volume_scattering_coefficient_expr->get_interface<IExpression_constant const>());
					Handle<IValue_color const> value_color(expr_const->get_value<IValue_color>());

					if (get_value(value_color.get(), config.scatteringCoefficient) == 0)
					{
						config.isScatteringCoefficientConstant = true;

						if (config.scatteringCoefficient[0] != 0.0f || config.scatteringCoefficient[1] != 0.0f || config.scatteringCoefficient[2] != 0.0f)
						{
							config.useVolumeScattering = true;
						}
					}
				}
				else
				{
					config.isScatteringCoefficientConstant = false;
					config.useVolumeScattering = true;
				}

				Handle<IExpression const> volume_directional_bias_expr(compiledMaterial->lookup_sub_expression("volume.scattering.directional_bias"));

				if (volume_directional_bias_expr->get_kind() == IExpression::EK_CONSTANT)
				{
					config.isDirectionalBiasConstant = true;

					Handle<IExpression_constant const> expr_const(volume_directional_bias_expr->get_interface<IExpression_constant const>());
					Handle<IValue_float const> value_float(expr_const->get_value<IValue_float>());

					// 0.0f is isotropic. No need to distinguish. The sampleHenyeyGreenstein() function takes this as parameter anyway.
					config.directionalBias = value_float->get_value();
				}
				else
				{
					config.isDirectionalBiasConstant = false;
				}

				// volume.scattering.emission_intensity is not supported by this renderer.
				// Also the volume absorption and scattering coefficients are assumed to be homogeneous in this renderer.
			}

			// geometry.displacement is not supported by this renderer.

			// geometry.normal is automatically handled because of set_option("include_geometry_normal", true);

			config.cutoutOpacity = 1.0f; // Default is fully opaque.
			config.isCutoutOpacityConstant = compiledMaterial->get_cutout_opacity(&config.cutoutOpacity); // This sets cutout opacity to -1.0 when it's not constant!
			config.useCutoutOpacity = !config.isCutoutOpacityConstant || config.cutoutOpacity < 1.0f;

			Handle<IExpression const> hair_bsdf_expr(compiledMaterial->lookup_sub_expression("hair"));

			config.isHairBsdfValid = isValidDistribution(hair_bsdf_expr.get()); // True if hair != hair_bsdf().

			// Check if front face is emissive

			bool isSurfaceEmissive = false;
			bool isBackfaceEmissive = false;
			bool isThinWalled = false;

			if (config.isSurfaceEdfValid) {
				if (!config.isSurfaceIntensityConstant) {
					isSurfaceEmissive = true;
				}
				else if (config.surfaceIntensity[0] != 0.0f || config.surfaceIntensity[1] != 0.0f || config.surfaceIntensity[2] != 0.0f) {
					isSurfaceEmissive = true;
				}
			}

			if (!config.isThinWalledConstant || (config.isThinWalledConstant && config.thinWalled)) { // To be emissive on the backface it needs to be thinWalled
				isThinWalled = true;
			}

			if (config.isBackfaceEdfValid) {
				if (!config.isBackfaceIntensityConstant) {
					isBackfaceEmissive = true;
				}
				else if (config.backfaceIntensity[0] != 0.0f || config.backfaceIntensity[1] != 0.0f || config.backfaceIntensity[2] != 0.0f) {
					isBackfaceEmissive = true;
				}
			}
			config.isEmissive = isSurfaceEmissive || (isThinWalled && isBackfaceEmissive);

			compiledMaterial->release();
			
		} while (0);
		transaction->commit();
		return config;
	}

	std::vector<Target_function_description> createShaderDescription(const graph::Shader::Configuration& config, const vtxID& shaderIndex, graph::Shader::FunctionNames& fNames) {

		// These are all expressions required for a materials which does everything supported in this renderer. 
		// The Target_function_description only stores the C-pointers to the base names!
		// Make sure these are not destroyed as long as the descs vector is used.

		fNames = graph::Shader::FunctionNames(std::to_string(shaderIndex));

		// Centralize the init functions in a single material init().
		// This will only save time when there would have been multiple init functions inside the shader.
		// Also for very complicated materials with cutout opacity this is most likely a loss,
		// because the geometry.cutout is only needed inside the anyhit program and 
		// that doesn't need additional evalations for the BSDFs, EDFs, or VDFs at that point.
		std::vector<Target_function_description> descriptions;

		descriptions.emplace_back("init", fNames.init.c_str());

		if (!config.isThinWalledConstant)
		{
			descriptions.emplace_back("thin_walled", fNames.thinWalled.c_str());
		}
		if (config.isSurfaceBsdfValid)
		{
			descriptions.emplace_back("surface.scattering", fNames.surfaceScattering.c_str());
		}
		if (config.isSurfaceEdfValid)
		{
			descriptions.emplace_back("surface.emission.emission", fNames.surfaceEmissionEmission.c_str());
			if (!config.isSurfaceIntensityConstant)
			{
				descriptions.emplace_back("surface.emission.intensity", fNames.surfaceEmissionIntensity.c_str());
			}
			if (!config.isSurfaceIntensityModeConstant)
			{
				descriptions.emplace_back("surface.emission.mode", fNames.surfaceEmissionMode.c_str());
			}
		}
		if (config.isBackfaceBsdfValid)
		{
			descriptions.emplace_back("backface.scattering", fNames.backfaceScattering.c_str());
		}
		if (config.isBackfaceEdfValid)
		{
			if (config.useBackfaceEdf)
			{
				descriptions.emplace_back("backface.emission.emission", fNames.backfaceEmissionEmission.c_str());
			}
			if (config.useBackfaceIntensity && !config.isBackfaceIntensityConstant)
			{
				descriptions.emplace_back("backface.emission.intensity", fNames.backfaceEmissionIntensity.c_str());
			}
			if (config.useBackfaceIntensityMode && !config.isBackfaceIntensityModeConstant)
			{
				descriptions.emplace_back("backface.emission.mode", fNames.backfaceEmissionMode.c_str());
			}
		}
		if (!config.isIorConstant)
		{
			descriptions.emplace_back("ior", fNames.ior.c_str());
		}
		if (!config.isAbsorptionCoefficientConstant)
		{
			descriptions.emplace_back("volume.absorption_coefficient", fNames.volumeAbsorptionCoefficient.c_str());
		}
		if (config.isVdfValid)
		{
			// DAR This fails in ILink_unit::add_material(). The MDL SDK is not generating functions for VDFs!
			//descriptions.push_back(Target_function_description("volume.fNames...c_str()));

			// The scattering coefficient and directional bias are not used when there is no valid VDF.
			if (!config.isScatteringCoefficientConstant)
			{
				descriptions.emplace_back("volume.scattering_coefficient", fNames.volumeScatteringCoefficient.c_str());
			}

			if (!config.isDirectionalBiasConstant)
			{
				descriptions.emplace_back("volume.scattering.directional_bias", fNames.volumeDirectionalBias.c_str());
			}

			// volume.scattering.emission_intensity is not implemented.
		}

		// geometry.displacement is not implemented.

		// geometry.normal is automatically handled because of set_option("include_geometry_normal", true);

		if (config.useCutoutOpacity)
		{
			descriptions.emplace_back("geometry.cutout_opacity", fNames.geometryCutoutOpacity.c_str());
		}
		if (config.isHairBsdfValid)
		{
			descriptions.emplace_back("hair", fNames.hairBsdf.c_str());
		}

		return descriptions;
	}

	Handle<ITarget_code const> createTargetCode(const std::string& materialDbName, const graph::Shader::Configuration& config,const vtxID& shaderIndex)
	{
		ITransaction* transaction = getGlobalTransaction();
		Handle<ITarget_code const> targetCode;
		do {

			Handle<const ICompiled_material> compiledMaterial(transaction->access<ICompiled_material>(("compiledMaterial_" + materialDbName).c_str()));

			//const Handle<const ICompiled_material> compiledMaterial((transaction->access(("compiledMaterial_" + materialDbName).c_str())->get_interface<ICompiled_material>()));
			const Handle<const IFunction_definition> materialDefinition(transaction->access<IFunction_definition>(materialDbName.c_str()));

			graph::Shader::FunctionNames fNames;
			std::vector<Target_function_description> descriptions = createShaderDescription(config, shaderIndex, fNames);

			VTX_INFO("Creating target code for shader {} index {} with {} functions.", materialDbName, shaderIndex, descriptions.size());
			const Handle<ILink_unit> link_unit(state.backend->create_link_unit(transaction, state.context.get()));
			VTX_ASSERT_BREAK((logMessage(state.context.get())), state.lastError);
			const Sint32 result = link_unit->add_material(compiledMaterial.get(), descriptions.data(), descriptions.size(),state.context.get());
			VTX_ASSERT_BREAK((result == 0 && logMessage(state.context.get())), state.lastError);

			targetCode = make_handle(state.backend->translate_link_unit(link_unit.get(), state.context.get()));
			VTX_ASSERT_BREAK((logMessage(state.context.get())), state.lastError);
			//mi::neuraylib::ITarget_code::Prototype_language

		} while (false);

		transaction->commit();

		return targetCode;
	}

	graph::ParamInfo generateParamInfo(
		const size_t index,
		const Handle<const ICompiled_material>& compiledMat,
		char* argBlockData,
		Handle<ITarget_value_layout const>& argBlockLayout,
		const Handle<IAnnotation_list const>& annoList,
		std::map<std::string, std::shared_ptr<graph::EnumTypeInfo>>&	mapEnumTypes
	)
	{
		using graph::EnumValue;
		using graph::EnumTypeInfo;
		using graph::ParamInfo;

		const char* name = compiledMat->get_parameter_name(index);

		if (name == nullptr) return {};

		Handle argument = make_handle<IValue const>(compiledMat->get_argument(index));
		const IValue::Kind kind = argument->get_kind();
		auto paramKind = ParamInfo::PK_UNKNOWN;
		auto paramArrayElemKind = ParamInfo::PK_UNKNOWN;
		Size paramArraySize = 0;
		Size paramArrayPitch = 0;
		const EnumTypeInfo* enumType = nullptr;

		switch (kind)
		{
		case IValue::VK_FLOAT:
			paramKind = ParamInfo::PK_FLOAT;
			break;
		case IValue::VK_COLOR:
			paramKind = ParamInfo::PK_COLOR;
			break;
		case IValue::VK_BOOL:
			paramKind = ParamInfo::PK_BOOL;
			break;
		case IValue::VK_INT:
			paramKind = ParamInfo::PK_INT;
			break;
		case IValue::VK_VECTOR:
			{
				const Handle val		= make_handle<const IValue_vector>(argument->get_interface<const IValue_vector>());
				const Handle valType	= make_handle<const IType_vector>(val->get_type());

				if (const Handle elemType = make_handle<const IType_atomic>(valType->get_element_type()); elemType->get_kind() == IType::TK_FLOAT)
				{
					switch (valType->get_size())
					{
					case 2:
						paramKind = ParamInfo::PK_FLOAT2;
						break;
					case 3:
						paramKind = ParamInfo::PK_FLOAT3;
						break;
					}
				}
			}
			break;
		case IValue::VK_ARRAY:
			{
				const Handle val		= make_handle<const IValue_array>(argument->get_interface<const IValue_array>());
				const Handle valType	= make_handle<const IType_array>(val->get_type());

				// we currently only support arrays of some values
				switch (const Handle elemType = make_handle<const IType>(valType->get_element_type()); elemType->get_kind())
				{
				case IType::TK_FLOAT:
					paramArrayElemKind = ParamInfo::PK_FLOAT;
					break;
				case IType::TK_COLOR:
					paramArrayElemKind = ParamInfo::PK_COLOR;
					break;
				case IType::TK_BOOL:
					paramArrayElemKind = ParamInfo::PK_BOOL;
					break;
				case IType::TK_INT:
					paramArrayElemKind = ParamInfo::PK_INT;
					break;
				case IType::TK_VECTOR:
					{
						const Handle valType = make_handle<const IType_vector>(elemType->get_interface<const IType_vector>());

						if (const Handle vElemType = make_handle<const IType_atomic>(valType->get_element_type()); vElemType->get_kind() == IType::TK_FLOAT)
						{
							switch (valType->get_size())
							{
							case 2:
								paramArrayElemKind = ParamInfo::PK_FLOAT2;
								break;
							case 3:
								paramArrayElemKind = ParamInfo::PK_FLOAT3;
								break;
							}
						}
					}
					break;
				default:
					break;
				}
				if (paramArrayElemKind != ParamInfo::PK_UNKNOWN)
				{
					paramKind		= ParamInfo::PK_ARRAY;
					paramArraySize	= valType->get_size();

					// determine pitch of array if there are at least two elements
					if (paramArraySize > 1)
					{
						const Target_value_layout_state arrayState(argBlockLayout->get_nested_state(index));
						const Target_value_layout_state nextElemState(argBlockLayout->get_nested_state(1, arrayState));

						IValue::Kind kind;
						Size paramSize;

						const Size startOffset = argBlockLayout->get_layout(kind, paramSize, arrayState);
						const Size nextOffset = argBlockLayout->get_layout(kind, paramSize, nextElemState);

						paramArrayPitch = nextOffset - startOffset;
					}
				}
			}
			break;
		case IValue::VK_ENUM:
			{
				const Handle val		= make_handle<const IValue_enum>(argument->get_interface<const IValue_enum>());
				const Handle valType	= make_handle<const IType_enum>(val->get_type());

				EnumTypeInfo* info = nullptr;
				std::string enumTypeName = valType->get_symbol();

				if (auto it = mapEnumTypes.find(enumTypeName); it != mapEnumTypes.end()) {
					info = it->second.get();
				}
				// prepare info for this enum type if not seen so far
				else {
					std::shared_ptr<EnumTypeInfo> enumTypeInfo(new EnumTypeInfo());
					for (Size i = 0, n = valType->get_size(); i < n; ++i)
					{
						std::string enumName = valType->get_value_name(i);
						int			enumValue = valType->get_value_code(i);
						enumTypeInfo->add(enumName, enumValue);
					}
					mapEnumTypes[enumTypeName] = enumTypeInfo;
					info = enumTypeInfo.get();
				}
				enumType = info;
				paramKind = ParamInfo::PK_ENUM;
			}
			break;
		case IValue::VK_STRING:
			paramKind = ParamInfo::PK_STRING;
			break;
		case IValue::VK_TEXTURE:
			paramKind = ParamInfo::PK_TEXTURE;
			break;
		case IValue::VK_LIGHT_PROFILE:
			paramKind = ParamInfo::PK_LIGHT_PROFILE;
			break;
		case IValue::VK_BSDF_MEASUREMENT:
			paramKind = ParamInfo::PK_BSDF_MEASUREMENT;
			break;
		default:
			// Unsupported? -> skip
			return {};
		}

		// Get the offset of the argument within the target argument block
		const Target_value_layout_state targetValueLayoutState(argBlockLayout->get_nested_state(index));
		IValue::Kind kind2;
		Size paramSize;
		const Size offset = argBlockLayout->get_layout(kind2, paramSize, targetValueLayoutState);
		if (kind != kind2)
		{
			return {};  // layout is invalid -> skip
		}

		char* dataPtr = argBlockData + offset;

		ParamInfo paramInfo(index,
							name,
							name,
							/*groupName=*/ "",
							paramKind,
							paramArrayElemKind,
							paramArraySize,
							paramArrayPitch,
							dataPtr,
							enumType);

		// Check for annotation info
		Handle annoBlock = make_handle<IAnnotation_block const>(annoList->get_annotation_block(name));
		if (annoBlock)
		{
			Annotation_wrapper annos(annoBlock.get());
			Size annoIndex;

			annoIndex = annos.get_annotation_index("::anno::hard_range(float,float)");
			if (annoIndex == static_cast<Size>(-1))
			{
				annoIndex = annos.get_annotation_index("::anno::soft_range(float,float)");
			}
			if (annoIndex != static_cast<Size>(-1))
			{
				annos.get_annotation_param_value(annoIndex, 0, paramInfo.rangeMin());
				annos.get_annotation_param_value(annoIndex, 1, paramInfo.rangeMax());
			}
			
			annoIndex = annos.get_annotation_index("::anno::display_name(string)");
			if (annoIndex != static_cast<Size>(-1))
			{
				char const* displayName = nullptr;
				annos.get_annotation_param_value(annoIndex, 0, displayName);
				paramInfo.setDisplayName(displayName);
			}
			annoIndex = annos.get_annotation_index("::anno::in_group(string)");
			if (annoIndex != static_cast<Size>(-1))
			{
				char const* groupName = nullptr;
				annos.get_annotation_param_value(annoIndex, 0, groupName);
				paramInfo.setGroupName(groupName);
			}
		}

		return paramInfo;
	}

	void setMaterialParameters(
		const std::string& materialDbName, 
		const Handle<ITarget_code const>& targetCode, 
		Handle<ITarget_argument_block>& argumentBlockClone, 
		std::list<graph::ParamInfo>& params,
		std::map<std::string, std::shared_ptr<graph::EnumTypeInfo>>& mapEnumTypes)
	{
		using graph::EnumValue;
		using graph::EnumTypeInfo;
		using graph::ParamInfo;

		char* argBlockData = nullptr;
		Handle<ITarget_value_layout const> argBlockLayout;
		if (targetCode->get_argument_block_count() > 0)
		{
			const auto argumentBlock = make_handle<ITarget_argument_block const>(targetCode->get_argument_block(0));

			argumentBlockClone = Handle(argumentBlock->clone());
			argBlockData = argumentBlockClone->get_data();
			argBlockLayout = make_handle<ITarget_value_layout const>(targetCode->get_argument_block_layout(0));
		}
		else
		{
			VTX_WARN("MDL WRAPPER: It was impossible to create argument block clone for target code");
		}

		ITransaction* transaction = getGlobalTransaction();
		{
			const Handle	compiledMaterial		= make_handle(transaction->access<ICompiled_material>(("compiledMaterial_" + materialDbName).c_str()));
			const Handle	materialDefinition		= make_handle(transaction->access<IFunction_definition>(materialDbName.c_str()));
			const Handle	annoList				= make_handle<const IAnnotation_list>(materialDefinition->get_parameter_annotations());
			const Size		numParams				= compiledMaterial->get_parameter_count();

			for (Size j = 0; j < numParams; ++j)
			{
				ParamInfo paramInfo = generateParamInfo(j, compiledMaterial, argBlockData, argBlockLayout, annoList, mapEnumTypes);

				// Add the parameter information as last entry of the corresponding group, or to the
				// end of the list, if no group name is available.
				if (paramInfo.groupName() != nullptr)
				{
					bool groupFound = false;
					for (auto it = params.begin(); it != params.end(); ++it)
					{
						const bool sameGroup = (it->groupName() != nullptr && strcmp(it->groupName(), paramInfo.groupName()) == 0);
						if (groupFound && !sameGroup)
						{
							params.insert(it, paramInfo);
							return;
						}
						if (sameGroup)
						{
							groupFound = true;
						}
					}
				}
				params.push_back(paramInfo);
			}
		}
		transaction->commit();
	}

	void fetchTextureData(const std::shared_ptr<graph::Texture>& textureNode)
	{
		ITransaction* transaction = getGlobalTransaction();
		{

			const auto* mdlState = getState();
			const Handle<IImage_api>& imageApi = mdlState->imageApi;

			const ITarget_code::Texture_shape	shape					= textureNode->shape;
			std::string							textureDbName			= textureNode->databaseName;

			//The following objects will be created by analyzing the mdl texture information
			size_t&								pixelBytesSize			= textureNode->pixelBytesSize;
			CUarray_format_enum&				format					= textureNode->format;
			math::vec4ui&						dimension				= textureNode->dimension;
			std::vector<const void*>&			imageLayersPointers		= textureNode->imageLayersPointers;

			// Access texture image and canvas
			const Handle						texture					= make_handle(transaction->access<ITexture>(textureNode->databaseName.c_str()));
			const Handle						image					= make_handle<const IImage>(transaction->access<IImage>(texture->get_image()));
			Handle								canvas					= make_handle<const ICanvas>(image->get_canvas(0, 0, 0));
			//const Float32						effectiveGamma			= texture->get_effective_gamma(0, 0);

			if (image->is_uvtile() || image->is_animated())
			{
				VTX_ERROR("MDL TEXTURE: uvtile and/or animated textures not supported! Texture name {}", textureDbName);
				return;
			}

			// MDL pixel types.
			//"Sint8"      // Signed 8-bit integer
			//"Sint32"     // Signed 32-bit integer
			//"Float32"    // 32-bit IEEE-754 single-precision floating-point number
			//"Float32<2>" // 2 x Float32
			//"Float32<3>" // 3 x Float32
			//"Float32<4>" // 4 x Float32
			//"Rgb"        // 3 x Uint8 representing RGB color
			//"Rgba"       // 4 x Uint8 representing RGBA color
			//"Rgbe"       // 4 x Uint8 representing RGBE color
			//"Rgbea"      // 5 x Uint8 representing RGBEA color
			//"Rgb_16"     // 3 x Uint16 representing RGB color
			//"Rgba_16"    // 4 x Uint16 representing RGBA color
			//"Rgb_fp"     // 3 x Float32 representing RGB color
			//"Color"      // 4 x Float32 representing RGBA color
			const std::string imageType(image->get_type(0, 0));

			if (imageType == "Rgb")
			{
				canvas = imageApi->convert(canvas.get(), "Rgba"); // Append an alpha channel with 0xFF.
			}
			if (imageType == "Rgba")
			{
				format = CU_AD_FORMAT_UNSIGNED_INT8;
				pixelBytesSize = sizeof(Uint8);
			}
			else {
				format = CU_AD_FORMAT_FLOAT;
				pixelBytesSize = sizeof(Float32);

				// Convert image to linear color space if necessary.
				if (texture->get_effective_gamma(0, 0) != 1.0f)
				{
					// Copy/convert to float4 canvas and adjust gamma from "effective gamma" to 1.0.
					const Handle gammaCanvas = make_handle<ICanvas>(imageApi->convert(canvas.get(), "Color"));
					gammaCanvas->set_gamma(texture->get_effective_gamma(0, 0));
					imageApi->adjust_gamma(gammaCanvas.get(), 1.0f);
					canvas = gammaCanvas;
				}
				else if (imageType != "Color" && imageType != "Float32<4>")
				{
					canvas = imageApi->convert(canvas.get(), "Color");
				}
			}
			dimension.x = canvas->get_resolution_x();
			dimension.y = canvas->get_resolution_y();
			// Copy image data to GPU array depending on texture shape
			if (shape == mi::neuraylib::ITarget_code::Texture_shape_cube ||
				shape == mi::neuraylib::ITarget_code::Texture_shape_3d ||
				shape == mi::neuraylib::ITarget_code::Texture_shape_bsdf_data)
			{
				dimension.z = canvas->get_layers_size();
				// Cubemap and 3D texture objects require 3D CUDA arrays.
				VTX_ASSERT_BREAK((shape != ITarget_code::Texture_shape_cube || dimension.z == 6),
								 "ERROR: prepareTextureMDL() Invalid number of layers ({}), cube Maps must have 6 layers!", dimension.z);
			}
			else
			{
				dimension.z = 0;
			}
			dimension.w = 4;
			textureNode->effectiveGamma = texture->get_effective_gamma(0, 0);


			//Fetch pointers to the texture layers data (for non 3d images the number of layers is 0!)
			for (unsigned int z = 0; z < canvas->get_layers_size(); ++z)
			{
				const auto tile = make_handle<const ITile>(canvas->get_tile(z));
				void* dst = malloc(dimension.x*dimension.y*pixelBytesSize* dimension.w);
				memcpy(dst, tile->get_data(), dimension.x * dimension.y * pixelBytesSize * dimension.w);
				imageLayersPointers.push_back(dst);
			}
		}
		transaction->commit();
	}

	graph::BsdfMeasurement::BsdfPartData fetchBsdfData(const std::string& bsdfDbName, const Mbsdf_part part)
	{
		ITransaction* transaction = getGlobalTransaction();
		graph::BsdfMeasurement::BsdfPartData data{};
		{
			bool success = true;
			// Get access to the MBSDF data by the texture database name from the target code.
			const Handle bsdfMeasurement = make_handle<const IBsdf_measurement>(transaction->access<IBsdf_measurement>(bsdfDbName.c_str()));

			switch (part)
			{
				case(MBSDF_DATA_REFLECTION):
				{
					if (const Handle reflectionDataset = make_handle(bsdfMeasurement->get_reflection<Bsdf_isotropic_data>()))
					{
						data.angularResolution.x = reflectionDataset->get_resolution_theta();
						data.angularResolution.y = reflectionDataset->get_resolution_phi();
						data.numChannels = (reflectionDataset->get_type() == BSDF_SCALAR) ? 1 : 3;
						// {1, 3} * (index_theta_in * (res_phi * res_theta) + index_theta_out * res_phi + index_phi)
						data.srcData = make_handle<const IBsdf_buffer>(reflectionDataset->get_bsdf_buffer())->get_data();
					}
				}
				case(MBSDF_DATA_TRANSMISSION):
				{
					if (const Handle transmissionDataset = make_handle(bsdfMeasurement->get_transmission<Bsdf_isotropic_data>()))
					{
						data.angularResolution.x = transmissionDataset->get_resolution_theta();
						data.angularResolution.y = transmissionDataset->get_resolution_phi();
						data.numChannels = (transmissionDataset->get_type() == BSDF_SCALAR) ? 1 : 3;
						// {1, 3} * (index_theta_in * (res_phi * res_theta) + index_theta_out * res_phi + index_phi)
						data.srcData = make_handle<const IBsdf_buffer>(transmissionDataset->get_bsdf_buffer())->get_data();
					}
				}
			}
			
		}
		transaction->commit();
		return data;
	}


	graph::LightProfile::LightProfileData fetchLightProfileData(const std::string& lightDbName)
	{
		ITransaction* transaction = getGlobalTransaction();
		graph::LightProfile::LightProfileData data{};
		{
			// Get access to the light_profile data.
			const Handle lightProfile = make_handle<const ILightprofile>(transaction->access<ILightprofile>(lightDbName.c_str()));

			data.resolution.x = lightProfile->get_resolution_theta();
			data.resolution.y = lightProfile->get_resolution_phi();
			data.start.x = lightProfile->get_theta(0);
			data.start.y = lightProfile->get_phi(0);
			data.delta.x = lightProfile->get_theta(1) - data.start.x;
			data.delta.y = lightProfile->get_phi(1) - data.start.y;
			data.sourceData = lightProfile->get_data();
			data.candelaMultiplier = lightProfile->get_candela_multiplier();

		}
		transaction->commit();
		return data;
	}

	// Utility function to dump the arguments of a material instance or function call.
	template <class T>
	void dumpInstance(IExpression_factory* expression_factory, const T* instance)
	{
		std::stringstream s;
		s << "Dumping material/function instance \"" << instance->get_mdl_function_definition() << "\":" << "\n";

		const mi::Size count = instance->get_parameter_count();
		const Handle<const IExpression_list> arguments(instance->get_arguments());

		for (mi::Size index = 0; index < count; index++) {

			Handle<const IExpression> argument(arguments->get_expression(index));
			std::string name = instance->get_parameter_name(index);
			const Handle<const mi::IString> argument_text(expression_factory->dump(argument.get(), name.c_str(), 1));
			s << "    argument " << argument_text->get_c_str() << "\n";

		}
		s << "\n";
		VTX_INFO("{}", s.str());
	}

	template <class T>
	std::string dumpDefinition(ITransaction* transaction, IMdl_factory* mdl_factory, const T* definition, Size depth)
	{
		std::stringstream ss;
		Handle<IType_factory> type_factory(mdl_factory->create_type_factory(transaction));
		Handle<IExpression_factory> expression_factory(mdl_factory->create_expression_factory(transaction));

		mi::Size count = definition->get_parameter_count();
		Handle<const IType_list> types(definition->get_parameter_types());
		Handle<const IExpression_list> defaults(definition->get_defaults());

		for (mi::Size index = 0; index < count; index++) {

			Handle<const IType> type(types->get_type(index));
			Handle<const mi::IString> type_text(type_factory->dump(type.get(), depth + 1));
			std::string name = definition->get_parameter_name(index);
			ss << "    parameter " << type_text->get_c_str() << " " << name;

			Handle<const IExpression> default_(defaults->get_expression(name.c_str()));
			if (default_.is_valid_interface()) {
				Handle<const mi::IString> default_text(expression_factory->dump(default_.get(), 0, depth + 1));
				ss << ", default = " << default_text->get_c_str() << "\n";
			}
			else {
				ss << " (no default)" << "\n";
			}

		}

		mi::Size temporary_count = definition->get_temporary_count();
		for (mi::Size i = 0; i < temporary_count; ++i) {
			Handle<const IExpression> temporary(definition->get_temporary(i));
			std::stringstream name;
			name << i;
			Handle<const mi::IString> result(expression_factory->dump(temporary.get(), name.str().c_str(), 1));
			ss << "    temporary " << result->get_c_str() << "\n";
		}

		Handle<const IExpression> body(definition->get_body());
		Handle<const mi::IString> result(expression_factory->dump(body.get(), 0, 1));
		if (result)
			ss << "    body " << result->get_c_str() << "\n";
		else
			ss << "    body not available for this function" << "\n";

		ss << "\n";

		return ss.str();
	}

	void dumpModuleInfo(const IModule* module, IMdl_factory* factory, ITransaction* transaction, bool dumpDefinitions = false) {

		std::stringstream ss;
		// Print the module name and the file name it was loaded from.
		ss << "Loaded file " << module->get_filename() << "\n";
		ss << "Found module " << module->get_mdl_name() << "\n";
		ss << "\n";

		// Dump imported modules.
		const Size module_count = module->get_import_count();
		if (module_count > 0) {
			ss << "The module imports the following modules:" << "\n";
			for (Size i = 0; i < module_count; i++)
				ss << "    " << module->get_import(i) << "\n";
		}
		else {
			ss << "The module doesn't import other modules." << "\n";
		}

		// Dump exported types.
		const Handle<IType_factory> type_factory(factory->create_type_factory(transaction));
		const Handle<const IType_list> types(module->get_types());
		if (types->get_size() > 0) {
			ss << "\n";
			ss << "The module contains the following types: " << "\n";
			for (Size i = 0; i < types->get_size(); ++i) {
				Handle<const IType> type(types->get_type(i));
				const Handle<const IString> result(type_factory->dump(type.get(), 1));
				ss << "    " << result->get_c_str() << "\n";
			}
		}
		else {
			ss << "The module doesn't contain any types " << "\n";
		}

		// Dump exported constants.
		const Handle<IValue_factory> value_factory(factory->create_value_factory(transaction));
		const Handle<const IValue_list> constants(module->get_constants());
		if (constants->get_size() > 0) {
			ss << "\n";
			ss << "The module contains the following constants: " << "\n";
			for (Size i = 0; i < constants->get_size(); ++i) {
				const char* name = constants->get_name(i);
				Handle<const IValue> constant(constants->get_value(i));
				const Handle<const IString> result(value_factory->dump(constant.get(), 0, 1));
				ss << "    " << name << " = " << result->get_c_str() << "\n";
			}
		}
		else {
			ss << "The module doesn't contain any constants " << "\n";
		}

		// Dump function definitions of the module.
		const Size function_count = module->get_function_count();
		if (function_count > 0) {
			ss << "\n";
			ss << "The module contains the following function definitions:" << "\n";
			for (Size i = 0; i < function_count; i++) {
				ss << "    " << module->get_function(i) << "\n";
				// Dump a function definition from the module.
				if (dumpDefinitions) {
					ss << "Dumping function definition \"" << module->get_function(i) << "\":"
						<< "\n";
					Handle<const IFunction_definition> function_definition(transaction->access<IFunction_definition>(module->get_function(i)));
					ss << dumpDefinition(transaction, factory, function_definition.get(), 1);
				}
			}
		}
		else {
			ss << "The module doesn't contains any function definitions" << "\n";
		}


		// Dump material definitions of the module.
		const Size material_count = module->get_material_count();
		if (material_count > 0) {
			ss << "\n";
			ss << "The module contains the following material definitions:" << "\n";
			for (Size i = 0; i < material_count; i++) {
				ss << "    " << module->get_material(i) << "\n";
				// Dump a material definition from the module.
				if (dumpDefinitions) {
					ss << "Dumping material definition \"" << module->get_material(i) << "\":"
						<< "\n";
					Handle<const IFunction_definition> material_definition(transaction->access<IFunction_definition>(module->get_material(i)));
					ss << dumpDefinition(transaction, factory, material_definition.get(), 1);
				}
			}
		}
		else {
			ss << "The module doesn't contains any material definitions" << "\n";
		}


		// Dump the resources referenced by this module
		if (module->get_resources_count() == 0) {
			ss << "This Module has no resources \n";
		}
		else {
			ss << "Dumping resources of this module: \n";
			for (Size r = 0, rn = module->get_resources_count(); r < rn; ++r)
			{
				const Handle<const IValue_resource> resource(
					module->get_resource(r));
				const char* db_name = resource->get_value();
				const char* mdl_file_path = resource->get_file_path();

				if (db_name == nullptr)
				{
					// resource is either not used and therefore has not been loaded or
					// could not be found.
					ss << "    db_name:               none" << "\n";
					ss << "    mdl_file_path:         " << mdl_file_path << "\n"
						<< "\n";
					continue;
				}
				ss << "    db_name:               " << db_name << "\n";
				ss << "    mdl_file_path:         " << mdl_file_path << "\n";

				const Handle<const IType_resource> type(
					resource->get_type());
				switch (type->get_kind())
				{
				case IType::TK_TEXTURE:
					{
						const Handle<const ITexture> texture(
							transaction->access<ITexture>(db_name));
						if (texture)
						{
							const Handle<const IImage> image(
								transaction->access<IImage>(texture->get_image()));

							for (Size f = 0, fn = image->get_length(); f < fn; ++f)
								for (Size t = 0, tn = image->get_frame_length(f); t < tn; ++t)
								{
									const char* resolved_file_path = image->get_filename(f, t);
									ss << "    resolved_file_path[" << f << "," << t << "]: "
										<< resolved_file_path << "\n";
								}
						}
						break;
					}

				case IType::TK_LIGHT_PROFILE:
					{
						const Handle<const ILightprofile> light_profile(
							transaction->access<ILightprofile>(db_name));
						if (light_profile)
						{
							const char* resolved_file_path = light_profile->get_filename();
							ss << "    resolved_file_path:    " << resolved_file_path << "\n";
						}
						break;
					}

				case IType::TK_BSDF_MEASUREMENT:
					{
						const Handle<const IBsdf_measurement> mbsdf(
							transaction->access<IBsdf_measurement>(db_name));
						if (mbsdf)
						{
							const char* resolved_file_path = mbsdf->get_filename();
							ss << "    resolved_file_path:    " << resolved_file_path << "\n";
						}
						break;
					}

				default:
					break;
				}
			}
			ss << "\n";
		}
		VTX_INFO("Module Info:\n{}", ss.str());
	}

}

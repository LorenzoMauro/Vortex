#include "MDLtools.h"
#include <map>
#include <execution>

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
			Handle<const IMessage> message(context->get_message(i));
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
		std::string filename = utl::absolutePath(getOptions()->dll_path + "libmdl_sdk.dll");
		HMODULE handle = LoadLibraryA(filename.c_str());
		VTX_ASSERT_CLOSE(handle, "Error Loading {} with error code {}", filename, GetLastError());

		void* symbol = GetProcAddress(handle, "mi_factory");
		VTX_ASSERT_CLOSE(symbol, "ERROR: GetProcAddress(handle, \"mi_factory\") failed with error {}", GetLastError());

		state.neuray = mi_factory<INeuray>(symbol);
		if (!state.neuray.get())
		{
			Handle<IVersion> version(mi_factory<IVersion>(symbol));
			VTX_ASSERT_CLOSE(version, "ERROR: Incompatible library. Could not determine version.");
			VTX_ASSERT_CLOSE(!version, "ERROR: Library version {} does not match header version {}", version->get_product_version(), MI_NEURAYLIB_PRODUCT_VERSION_STRING);
		}

		VTX_INFO("MDL SDK Loaded");
	}

	void loadPlugin(std::string path)
	{
		Handle<IPlugin_configuration> plugin_conf(state.neuray->get_api_component<IPlugin_configuration>());

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

		state.logger = make_handle(new MDLlogger());
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

		std::string freeimagePath = utl::absolutePath(getOptions()->dll_path + "freeimage.dll");
		HMODULE handle = LoadLibraryA(freeimagePath.c_str());
		VTX_ASSERT_CLOSE(handle, "Error Loading {} with error code {}", freeimagePath, GetLastError());

		std::string ddsPath = getOptions()->dll_path + "dds.dll";
		std::string nv_freeimagePath = utl::absolutePath(getOptions()->dll_path + "nv_freeimage.dll");


		loadPlugin(nv_freeimagePath);
		loadPlugin(ddsPath);

	}


	void startInterfaces()
	{

		VTX_ASSERT_CLOSE(state.neuray->start() == 0, "FATAL: Starting MDL SDK failed.");

		state.database = state.neuray->get_api_component<IDatabase>();

		state.global_scope = state.database->get_global_scope();

		state.factory = state.neuray->get_api_component<IMdl_factory>();

		state.impexpApi = state.neuray->get_api_component<IMdl_impexp_api>();

		ITransaction* transaction = getGlobalTransaction();

		state.expression_factory = state.factory->create_expression_factory(transaction);
		state.value_factory = state.factory->create_value_factory(transaction);
		state.type_factory = state.factory->create_type_factory(transaction);

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

		Handle<IMdl_backend_api> mdl_backend_api(state.neuray->get_api_component<IMdl_backend_api>());

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

		if (getOptions()->enable_derivatives) //Not supported in this renderer
		{
			// Option "texture_runtime_with_derivs": Default is disabled.
			// We enable it to get coordinates with derivatives for texture lookup functions.
			state.result = state.backend->set_option("texture_runtime_with_derivs", "on");
			VTX_ASSERT_CLOSE(state.result == 0, "Error with texture runtime with derivatives");
		}

		if (getOptions()->isDebug) {
			Sint32 result;
			result = state.backend->set_option("opt_level", "0");
			VTX_ASSERT_CLOSE((result == 0), "Error with opt level");
			result = state.backend->set_option("enable_exceptions", "on");

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
		state.imageApi.reset();
		state.backend.reset();
		state.context.reset();
		state.factory.reset();
		state.global_scope.reset();
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
			state.transaction = make_handle<ITransaction>(state.global_scope->create_transaction());
			return state.transaction.get();
		}
		else {
			return state.transaction.get();
		}

	}

	/*path expressed as relative to the search path added to mdl sdk*/
	std::string pathToModuleName(std::string materialPath) {
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
		std::size_t last_dot = output.find_last_of('.');
		if (last_dot != std::string::npos) {
			output.erase(last_dot);
		}

		return output;
	};

	std::string getMaterialDatabaseName(const IModule* module, const IString* moduleDatabaseName, std::string materialName) {
		std::string materialDatabaseName = std::string(moduleDatabaseName->get_c_str()) + "::" + materialName;

		// Return input if it already contains a signature.
		if (materialDatabaseName.back() == ')')
		{
			return materialDatabaseName;
		}

		mi::base::Handle<const mi::IArray> result(module->get_function_overloads(materialDatabaseName.c_str()));

		// Not supporting multiple function overloads with the same name but different signatures.
		if (!result || result->get_length() != 1)
		{
			return std::string();
		}

		mi::base::Handle<const mi::IString> overloads(result->get_element<mi::IString>(static_cast<mi::Size>(0)));

		return overloads->get_c_str();

	}
        
	bool isValidDistribution(IExpression const* expr)
	{
		if (expr == nullptr)
		{
			return false;
		}

		if (expr->get_kind() == IExpression::EK_CONSTANT)
		{
			Handle<IExpression_constant const> expr_constant(expr->get_interface<IExpression_constant>());
			Handle<IValue const> value(expr_constant->get_value());

			if (value->get_kind() == IValue::VK_INVALID_DF)
			{
				return false;
			}
		}

		return true;
	}

	void determineShaderConfiguration(std::shared_ptr<graph::Shader> shader)
	{
		graph::ShaderConfiguration& config = shader->config;
		const ICompiled_material* compiledMaterial = shader->compilation.compiledMaterial.get();

		config.isThinWalledConstant = false;
		config.thin_walled = false;

		Handle<IExpression const> thin_walled_expr(compiledMaterial->lookup_sub_expression("thin_walled"));
		if (thin_walled_expr->get_kind() == IExpression::EK_CONSTANT)
		{
			config.isThinWalledConstant = true;

			Handle<IExpression_constant const> expr_const(thin_walled_expr->get_interface<IExpression_constant const>());
			Handle<IValue_bool const> value_bool(expr_const->get_value<IValue_bool>());

			config.thin_walled = value_bool->get_value();
		}

		Handle<IExpression const> surface_scattering_expr(compiledMaterial->lookup_sub_expression("surface.scattering"));

		config.isSurfaceBsdfValid = isValidDistribution(surface_scattering_expr.get()); // True if surface.scattering != bsdf().

		config.isBackfaceBsdfValid = false;

		// The backface scattering is only used for thin-walled materials.
		if (!config.isThinWalledConstant || config.thin_walled)
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

		config.is_surface_edf_valid = isValidDistribution(surface_edf_expr.get());

		config.is_surface_intensity_constant = true;
		config.surface_intensity = mi::math::Color(0.0f, 0.0f, 0.0f);
		config.is_surface_intensity_mode_constant = true;
		config.surface_intensity_mode = 0; // == intensity_radiant_exitance;

		if (config.is_surface_edf_valid)
		{
			// Surface emission intensity.
			Handle<IExpression const> surface_intensity_expr(compiledMaterial->lookup_sub_expression("surface.emission.intensity"));

			config.is_surface_intensity_constant = false;

			if (surface_intensity_expr->get_kind() == IExpression::EK_CONSTANT)
			{
				Handle<IExpression_constant const> intensity_const(surface_intensity_expr->get_interface<IExpression_constant const>());
				Handle<IValue_color const> intensity_color(intensity_const->get_value<IValue_color>());

				if (get_value(intensity_color.get(), config.surface_intensity) == 0)
				{
					config.is_surface_intensity_constant = true;
				}
			}

			// Surface emission mode. This is a uniform and normally the default intensity_radiant_exitance
			Handle<IExpression const> surface_intensity_mode_expr(compiledMaterial->lookup_sub_expression("surface.emission.mode"));

			config.is_surface_intensity_mode_constant = false;

			if (surface_intensity_mode_expr->get_kind() == IExpression::EK_CONSTANT)
			{
				Handle<IExpression_constant const> expr_const(surface_intensity_mode_expr->get_interface<IExpression_constant const>());
				Handle<IValue_enum const> value_enum(expr_const->get_value<IValue_enum>());

				config.surface_intensity_mode = value_enum->get_value();

				config.is_surface_intensity_mode_constant = true;
			}
		}

		// Backface EDF.
		config.is_backface_edf_valid = false;
		// DEBUG Is any of this needed at all or is the BSDF init() function handling all this?
		config.is_backface_intensity_constant = true;
		config.backface_intensity = mi::math::Color(0.0f, 0.0f, 0.0f);
		config.is_backface_intensity_mode_constant = true;
		config.backface_intensity_mode = 0; // == intensity_radiant_exitance;
		config.use_backface_edf = false;
		config.use_backface_intensity = false;
		config.use_backface_intensity_mode = false;

		// A backface EDF is only used on thin-walled materials with a backface.emission.emission != edf()
		if (!config.isThinWalledConstant || config.thin_walled)
		{
			Handle<IExpression const> backface_edf_expr(compiledMaterial->lookup_sub_expression("backface.emission.emission"));

			config.is_backface_edf_valid = isValidDistribution(backface_edf_expr.get());

			if (config.is_backface_edf_valid)
			{
				// Backface emission intensity.
				Handle<IExpression const> backface_intensity_expr(compiledMaterial->lookup_sub_expression("backface.emission.intensity"));

				config.is_backface_intensity_constant = false;

				if (backface_intensity_expr->get_kind() == IExpression::EK_CONSTANT)
				{
					Handle<IExpression_constant const> intensity_const(backface_intensity_expr->get_interface<IExpression_constant const>());
					Handle<IValue_color const> intensity_color(intensity_const->get_value<IValue_color>());

					if (get_value(intensity_color.get(), config.backface_intensity) == 0)
					{
						config.is_backface_intensity_constant = true;
					}
				}

				// Backface emission mode. This is a uniform and normally the default intensity_radiant_exitance.
				Handle<IExpression const> backface_intensity_mode_expr(compiledMaterial->lookup_sub_expression("backface.emission.mode"));

				config.is_backface_intensity_mode_constant = false;

				if (backface_intensity_mode_expr->get_kind() == IExpression::EK_CONSTANT)
				{
					Handle<IExpression_constant const> expr_const(backface_intensity_mode_expr->get_interface<IExpression_constant const>());
					Handle<IValue_enum const> value_enum(expr_const->get_value<IValue_enum>());

					config.backface_intensity_mode = value_enum->get_value();

					config.is_backface_intensity_mode_constant = true;
				}

				// When surface and backface expressions are identical, reuse the surface expression to generate less code.
				config.use_backface_edf = (compiledMaterial->get_slot_hash(SLOT_SURFACE_EMISSION_EDF_EMISSION) !=
					compiledMaterial->get_slot_hash(SLOT_BACKFACE_EMISSION_EDF_EMISSION));

				// If the surface and backface emission use different intensities then use the backface emission intensity.
				config.use_backface_intensity = (compiledMaterial->get_slot_hash(SLOT_SURFACE_EMISSION_INTENSITY) !=
					compiledMaterial->get_slot_hash(SLOT_BACKFACE_EMISSION_INTENSITY));

				// If the surface and backface emission use different modes (radiant exitance vs. power) then use the backface emission intensity mode.
				config.use_backface_intensity_mode = (compiledMaterial->get_slot_hash(SLOT_SURFACE_EMISSION_MODE) !=
					compiledMaterial->get_slot_hash(SLOT_BACKFACE_EMISSION_MODE));
			}
		}

		config.is_ior_constant = true;
		config.ior = mi::math::Color(1.0f, 1.0f, 1.0f);

		Handle<IExpression const> ior_expr(compiledMaterial->lookup_sub_expression("ior"));
		if (ior_expr->get_kind() == IExpression::EK_CONSTANT)
		{
			Handle<IExpression_constant const> expr_const(ior_expr->get_interface<IExpression_constant const>());
			Handle<IValue_color const> value_color(expr_const->get_value<IValue_color>());

			if (get_value(value_color.get(), config.ior) == 0)
			{
				config.is_ior_constant = true;
			}
		}
		else
		{
			config.is_ior_constant = false;
		}

		// If the VDF is valid, it is the df::anisotropic_vdf(). ::vdf() is not a valid VDF.
		// Though there aren't any init, sample, eval or pdf functions genereted for a VDF.
		Handle<IExpression const> volume_vdf_expr(compiledMaterial->lookup_sub_expression("volume.scattering"));

		config.is_vdf_valid = isValidDistribution(volume_vdf_expr.get());

		// Absorption coefficient. Can be used without valid VDF.
		config.is_absorption_coefficient_constant = true;  // Default to constant and no absorption.
		config.use_volume_absorption = false; // If there is no abosorption, the absorption coefficient is constant zero.
		config.absorption_coefficient = mi::math::Color(0.0f, 0.0f, 0.0f); // No absorption.

		Handle<IExpression const> volume_absorption_coefficient_expr(compiledMaterial->lookup_sub_expression("volume.absorption_coefficient"));

		if (volume_absorption_coefficient_expr->get_kind() == IExpression::EK_CONSTANT)
		{
			Handle<IExpression_constant const> expr_const(volume_absorption_coefficient_expr->get_interface<IExpression_constant const>());
			Handle<IValue_color const> value_color(expr_const->get_value<IValue_color>());

			if (get_value(value_color.get(), config.absorption_coefficient) == 0)
			{
				config.is_absorption_coefficient_constant = true;

				if (config.absorption_coefficient[0] != 0.0f || config.absorption_coefficient[1] != 0.0f || config.absorption_coefficient[2] != 0.0f)
				{
					config.use_volume_absorption = true;
				}
			}
		}
		else
		{
			config.is_absorption_coefficient_constant = false;
			config.use_volume_absorption = true;
		}

		// Scattering coefficient. Only used when there is a valid VDF. 
		config.is_scattering_coefficient_constant = true; // Default to constant and no scattering. Assumes invalid VDF.
		config.use_volume_scattering = false;
		config.scattering_coefficient = mi::math::Color(0.0f, 0.0f, 0.0f); // No scattering

		// Directional bias (Henyey_Greenstein g factor.) Only used when there is a valid VDF and volume scattering coefficient not zero.
		config.is_directional_bias_constant = true;
		config.directional_bias = 0.0f;

		// The anisotropic_vdf() is the only valid VDF. 
		// The scattering_coefficient, directional_bias (and emission_intensity) are only needed when there is a valid VDF.
		if (config.is_vdf_valid)
		{
			Handle<IExpression const> volume_scattering_coefficient_expr(compiledMaterial->lookup_sub_expression("volume.scattering_coefficient"));

			if (volume_scattering_coefficient_expr->get_kind() == IExpression::EK_CONSTANT)
			{
				Handle<IExpression_constant const> expr_const(volume_scattering_coefficient_expr->get_interface<IExpression_constant const>());
				Handle<IValue_color const> value_color(expr_const->get_value<IValue_color>());

				if (get_value(value_color.get(), config.scattering_coefficient) == 0)
				{
					config.is_scattering_coefficient_constant = true;

					if (config.scattering_coefficient[0] != 0.0f || config.scattering_coefficient[1] != 0.0f || config.scattering_coefficient[2] != 0.0f)
					{
						config.use_volume_scattering = true;
					}
				}
			}
			else
			{
				config.is_scattering_coefficient_constant = false;
				config.use_volume_scattering = true;
			}

			Handle<IExpression const> volume_directional_bias_expr(compiledMaterial->lookup_sub_expression("volume.scattering.directional_bias"));

			if (volume_directional_bias_expr->get_kind() == IExpression::EK_CONSTANT)
			{
				config.is_directional_bias_constant = true;

				Handle<IExpression_constant const> expr_const(volume_directional_bias_expr->get_interface<IExpression_constant const>());
				Handle<IValue_float const> value_float(expr_const->get_value<IValue_float>());

				// 0.0f is isotropic. No need to distinguish. The sampleHenyeyGreenstein() function takes this as parameter anyway.
				config.directional_bias = value_float->get_value();
			}
			else
			{
				config.is_directional_bias_constant = false;
			}

			// volume.scattering.emission_intensity is not supported by this renderer.
			// Also the volume absorption and scattering coefficients are assumed to be homogeneous in this renderer.
		}

		// geometry.displacement is not supported by this renderer.

		// geometry.normal is automatically handled because of set_option("include_geometry_normal", true);

		config.cutout_opacity = 1.0f; // Default is fully opaque.
		config.is_cutout_opacity_constant = compiledMaterial->get_cutout_opacity(&config.cutout_opacity); // This sets cutout opacity to -1.0 when it's not constant!
		config.use_cutout_opacity = !config.is_cutout_opacity_constant || config.cutout_opacity < 1.0f;

		Handle<IExpression const> hair_bsdf_expr(compiledMaterial->lookup_sub_expression("hair"));

		config.is_hair_bsdf_valid = isValidDistribution(hair_bsdf_expr.get()); // True if hair != hair_bsdf().
	}

	std::vector<Target_function_description> createShaderDescription(std::shared_ptr<graph::Shader> shader) {

		// These are all expressions required for a materials which does everything supported in this renderer. 
		// The Target_function_description only stores the C-pointers to the base names!
		// Make sure these are not destroyed as long as the descs vector is used.

		shader->fNames = graph::FunctionNames(std::to_string(shader->getID()));

		// Centralize the init functions in a single material init().
		// This will only save time when there would have been multiple init functions inside the shader.
		// Also for very complicated materials with cutout opacity this is most likely a loss,
		// because the geometry.cutout is only needed inside the anyhit program and 
		// that doesn't need additional evalations for the BSDFs, EDFs, or VDFs at that point.
		std::vector<Target_function_description> descriptions;

		descriptions.push_back(Target_function_description("init", shader->fNames.init.c_str()));

		if (!shader->config.isThinWalledConstant)
		{
			descriptions.push_back(Target_function_description("thin_walled", shader->fNames.thin_walled.c_str()));
		}
		if (shader->config.isSurfaceBsdfValid)
		{
			descriptions.push_back(Target_function_description("surface.scattering", shader->fNames.surface_scattering.c_str()));
		}
		if (shader->config.is_surface_edf_valid)
		{
			descriptions.push_back(Target_function_description("surface.emission.emission", shader->fNames.surface_emission_emission.c_str()));
			if (!shader->config.is_surface_intensity_constant)
			{
				descriptions.push_back(Target_function_description("surface.emission.intensity", shader->fNames.surface_emission_intensity.c_str()));
			}
			if (!shader->config.is_surface_intensity_mode_constant)
			{
				descriptions.push_back(Target_function_description("surface.emission.mode", shader->fNames.surface_emission_mode.c_str()));
			}
		}
		if (shader->config.isBackfaceBsdfValid)
		{
			descriptions.push_back(Target_function_description("backface.scattering", shader->fNames.backface_scattering.c_str()));
		}
		if (shader->config.is_backface_edf_valid)
		{
			if (shader->config.use_backface_edf)
			{
				descriptions.push_back(Target_function_description("backface.emission.emission", shader->fNames.backface_emission_emission.c_str()));
			}
			if (shader->config.use_backface_intensity && !shader->config.is_backface_intensity_constant)
			{
				descriptions.push_back(Target_function_description("backface.emission.intensity", shader->fNames.backface_emission_intensity.c_str()));
			}
			if (shader->config.use_backface_intensity_mode && !shader->config.is_backface_intensity_mode_constant)
			{
				descriptions.push_back(Target_function_description("backface.emission.mode", shader->fNames.backface_emission_mode.c_str()));
			}
		}
		if (!shader->config.is_ior_constant)
		{
			descriptions.push_back(Target_function_description("ior", shader->fNames.ior.c_str()));
		}
		if (!shader->config.is_absorption_coefficient_constant)
		{
			descriptions.push_back(Target_function_description("volume.absorption_coefficient", shader->fNames.volume_absorption_coefficient.c_str()));
		}
		if (shader->config.is_vdf_valid)
		{
			// DAR This fails in ILink_unit::add_material(). The MDL SDK is not generating functions for VDFs!
			//descriptions.push_back(Target_function_description("volume.shader->fNames...c_str()));

			// The scattering coefficient and directional bias are not used when there is no valid VDF.
			if (!shader->config.is_scattering_coefficient_constant)
			{
				descriptions.push_back(Target_function_description("volume.scattering_coefficient", shader->fNames.volume_scattering_coefficient.c_str()));
			}

			if (!shader->config.is_directional_bias_constant)
			{
				descriptions.push_back(Target_function_description("volume.scattering.directional_bias", shader->fNames.volume_directional_bias.c_str()));
			}

			// volume.scattering.emission_intensity is not implemented.
		}

		// geometry.displacement is not implemented.

		// geometry.normal is automatically handled because of set_option("include_geometry_normal", true);

		if (shader->config.use_cutout_opacity)
		{
			descriptions.push_back(Target_function_description("geometry.cutout_opacity", shader->fNames.geometry_cutout_opacity.c_str()));
		}
		if (shader->config.is_hair_bsdf_valid)
		{
			descriptions.push_back(Target_function_description("hair", shader->fNames.hair_bsdf.c_str()));
		}

		return descriptions;
	}

	void loadShaderData(std::shared_ptr<graph::Shader> shader)
	{
		Sint32 result;
		ITransaction* transaction = getGlobalTransaction();

		std::string& path = shader->path;
		std::string& materialName = shader->name;
		Handle<const ICompiled_material> compiledMaterial = shader->compilation.compiledMaterial;
		Handle<const ITarget_code> taregetCode = taregetCode;

		//Early Exit if the Shader has already been loaded.
		if (taregetCode.is_valid_interface()) {
			VTX_INFO("Shader with module {} and material {} already elaborated", path, materialName);
			return;
		}

		std::string moduleName = pathToModuleName(shader->path);

		result = state.impexpApi->load_module(transaction, moduleName.c_str(), state.context.get());

		do {
			VTX_ASSERT_RETURN((result >= 0 && logMessage(state.context.get())), state.lastError);
			Handle<const IString> moduleDatabaseName(state.factory->get_db_module_name(moduleName.c_str()));
			// access module
			Handle<const IModule> module(transaction->access<IModule>(moduleDatabaseName.get()->get_c_str()));
			VTX_ASSERT_BREAK((module.is_valid_interface()), "Error with module access");

			// define material module Name and verify no overload
			std::string materialDatabaseName = getMaterialDatabaseName(module.get(), moduleDatabaseName.get(), shader->name);
			VTX_ASSERT_BREAK(!materialDatabaseName.empty(), "Error with retrieving material {} in the module {}, material might have overload", shader->name, moduleName);

			// Create material Definition
			Handle<const IFunction_definition> material_definition(transaction->access<IFunction_definition>(materialDatabaseName.c_str()));
			VTX_ASSERT_BREAK((material_definition.is_valid_interface()), "Error with material definition creation for material {} in module {}", shader->name, moduleName);

			//Create material Call
			Handle<IFunction_call> materialCall(material_definition->create_function_call(0, &result));
			VTX_ASSERT_BREAK((materialCall.is_valid_interface() && !(result != 0)), "Error with material instance creation for material {} in module {}", shader->name, moduleName);

			//Create material Instance
			Handle<IMaterial_instance> materialInstance(materialCall->get_interface<IMaterial_instance>());
			VTX_ASSERT_BREAK((materialInstance.is_valid_interface()), "Error with material instance creation for material {} in module {}", shader->name, moduleName);

			//Create compiled material
			Uint32 flags = IMaterial_instance::CLASS_COMPILATION;
			compiledMaterial = materialInstance->create_compiled_material(flags, state.context.get());
			VTX_ASSERT_BREAK((logMessage(state.context.get())), state.lastError);

			//Get Compiled material Hash
			shader->compilation.compilationHash = compiledMaterial->get_hash();

			// Analize the compiled Material to get its property and generate function descriptions
			determineShaderConfiguration(shader);
			//std::string suffix = std::to_string(materialHash.m_id1) + std::to_string(materialHash.m_id2) + std::to_string(materialHash.m_id3) + std::to_string(materialHash.m_id4);
			std::string suffix = "0";
			std::vector<Target_function_description> descriptions = createShaderDescription(shader);

			Handle<ILink_unit> link_unit(state.backend->create_link_unit(transaction, state.context.get()));
			VTX_ASSERT_BREAK((logMessage(state.context.get())), state.lastError);
			result = link_unit->add_material(compiledMaterial.get(), descriptions.data(), descriptions.size(), state.context.get());
			VTX_ASSERT_BREAK((result == 0 && logMessage(state.context.get())), state.lastError);


			taregetCode = state.backend->translate_link_unit(link_unit.get(), state.context.get());
			VTX_ASSERT_BREAK((logMessage(state.context.get())), state.lastError);

			for (Size i = 1, n = taregetCode->get_texture_count(); i < n; ++i) {
				shader->textures.emplace_back(taregetCode->get_texture(i), taregetCode->get_texture_shape(i));
			}

			if (taregetCode->get_light_profile_count() > 0)
			{
				for (mi::Size i = 1, n = taregetCode->get_light_profile_count(); i < n; ++i)
				{
					shader->lightProfiles.emplace_back(taregetCode->get_light_profile(i));
				}
			}

			if (taregetCode->get_bsdf_measurement_count() > 0)
			{
				for (mi::Size i = 1, n = taregetCode->get_bsdf_measurement_count(); i < n; ++i)
				{
					shader->bsdfMeasurements.emplace_back(taregetCode->get_bsdf_measurement(i));
				}
			}

			if (taregetCode->get_argument_block_count() > 0)
			{
				shader->argumentBlock = taregetCode->get_argument_block(0);
			}

			if (taregetCode->get_argument_block_count() > 0)
			{
				shader->argLayout = taregetCode->get_argument_block_layout(0);
			}

			if (getOptions()->isDebug) {
				std::string code = taregetCode->get_code();

				// Print generated PTX source code to the console.
				//std::cout << code << std::endl;

				// Dump generated PTX source code to a local folder for offline comparisons.
				const std::string filename = std::string("./mdl_ptx/") + shader->name + std::string("_") + utl::getDateTime() + std::string(".ptx");

				utl::saveString(filename, code);
			}
		} while (0);

		transaction->commit();

	}

	//void storeMaterialInfo() {
	//    int indexShader;
	//    Handle<IFunction_definition>        mat_def;
	//    Handle<ICompiled_material>          comp_mat;
	//    Handle<ITarget_value_layout>        arg_block_layout;
	//    Handle<ITarget_argument_block>      arg_block;

	//    int m_indexShader;
	//    std::string m_name;
	//    Handle<ITarget_argument_block> m_arg_block;


	//    m_indexShader = indexShader;
	//    m_name = mat_def->get_mdl_name();

	//    char* arg_block_data = nullptr;

	//    if (arg_block != nullptr)
	//    {
	//        m_arg_block = Handle<ITarget_argument_block>(arg_block->clone());
	//        arg_block_data = m_arg_block->get_data();
	//    }

	//    Handle<IAnnotation_list const> anno_list(mat_def->get_parameter_annotations());

	//    for (Size j = 0, num_params = comp_mat->get_parameter_count(); j < num_params; ++j)
	//    {
	//        const char* name = comp_mat->get_parameter_name(j);
	//        if (name == nullptr)
	//        {
	//            continue;
	//        }

	//        // Determine the type of the argument
	//        Handle<IValue const> arg(comp_mat->get_argument(j));
	//        IValue::Kind kind = arg->get_kind();

	//        Param_info::Param_kind param_kind = Param_info::PK_UNKNOWN;
	//        Param_info::Param_kind param_array_elem_kind = Param_info::PK_UNKNOWN;
	//        Size               param_array_size = 0;
	//        Size               param_array_pitch = 0;

	//        const Enum_type_info* enum_type = nullptr;

	//        switch (kind)
	//        {
	//            case IValue::VK_FLOAT:
	//                param_kind = Param_info::PK_FLOAT;
	//                break;
	//            case IValue::VK_COLOR:
	//                param_kind = Param_info::PK_COLOR;
	//                break;
	//            case IValue::VK_BOOL:
	//                param_kind = Param_info::PK_BOOL;
	//                break;
	//            case IValue::VK_INT:
	//                param_kind = Param_info::PK_INT;
	//                break;
	//            case IValue::VK_VECTOR:
	//            {
	//                Handle<IValue_vector const> val(arg.get_interface<IValue_vector const>());
	//                Handle<IType_vector const> val_type(val->get_type());
	//                Handle<IType_atomic const> elem_type(val_type->get_element_type());

	//                if (elem_type->get_kind() == IType::TK_FLOAT)
	//                {
	//                    switch (val_type->get_size())
	//                    {
	//                        case 2:
	//                            param_kind = Param_info::PK_FLOAT2;
	//                            break;
	//                        case 3:
	//                            param_kind = Param_info::PK_FLOAT3;
	//                            break;
	//                    }
	//                }
	//            }
	//            break;
	//            case IValue::VK_ARRAY:
	//            {
	//                Handle<IValue_array const> val(arg.get_interface<IValue_array const>());
	//                Handle<IType_array const> val_type(val->get_type());
	//                Handle<IType const> elem_type(val_type->get_element_type());

	//                // we currently only support arrays of some values
	//                switch (elem_type->get_kind())
	//                {
	//                    case IType::TK_FLOAT:
	//                        param_array_elem_kind = Param_info::PK_FLOAT;
	//                        break;
	//                    case IType::TK_COLOR:
	//                        param_array_elem_kind = Param_info::PK_COLOR;
	//                        break;
	//                    case IType::TK_BOOL:
	//                        param_array_elem_kind = Param_info::PK_BOOL;
	//                        break;
	//                    case IType::TK_INT:
	//                        param_array_elem_kind = Param_info::PK_INT;
	//                        break;
	//                    case IType::TK_VECTOR:
	//                    {
	//                        Handle<IType_vector const> val_type(elem_type.get_interface<IType_vector const>());
	//                        Handle<IType_atomic const> velem_type(val_type->get_element_type());

	//                        if (velem_type->get_kind() == IType::TK_FLOAT)
	//                        {
	//                            switch (val_type->get_size())
	//                            {
	//                                case 2:
	//                                    param_array_elem_kind = Param_info::PK_FLOAT2;
	//                                    break;
	//                                case 3:
	//                                    param_array_elem_kind = Param_info::PK_FLOAT3;
	//                                    break;
	//                            }
	//                        }
	//                    }
	//                    break;
	//                    default:
	//                        break;
	//                }
	//                if (param_array_elem_kind != Param_info::PK_UNKNOWN)
	//                {
	//                    param_kind = Param_info::PK_ARRAY;
	//                    param_array_size = val_type->get_size();

	//                    // determine pitch of array if there are at least two elements
	//                    if (param_array_size > 1)
	//                    {
	//                        Target_value_layout_state array_state(arg_block_layout->get_nested_state(j));
	//                        Target_value_layout_state next_elem_state(arg_block_layout->get_nested_state(1, array_state));

	//                        IValue::Kind kind;
	//                        Size param_size;

	//                        Size start_offset = arg_block_layout->get_layout(kind, param_size, array_state);
	//                        Size next_offset = arg_block_layout->get_layout(kind, param_size, next_elem_state);

	//                        param_array_pitch = next_offset - start_offset;
	//                    }
	//                }
	//            }
	//            break;
	//            case IValue::VK_ENUM:
	//            {
	//                Handle<IValue_enum const> val(arg.get_interface<IValue_enum const>());
	//                Handle<IType_enum const> val_type(val->get_type());

	//                // prepare info for this enum type if not seen so far
	//                const Enum_type_info* info = get_enum_type(val_type->get_symbol());
	//                if (info == nullptr)
	//                {
	//                    std::shared_ptr<Enum_type_info> p(new Enum_type_info());

	//                    for (Size i = 0, n = val_type->get_size(); i < n; ++i)
	//                    {
	//                        p->add(val_type->get_value_name(i), val_type->get_value_code(i));
	//                    }
	//                    add_enum_type(val_type->get_symbol(), p);
	//                    info = p.get();
	//                }
	//                enum_type = info;

	//                param_kind = Param_info::PK_ENUM;
	//            }
	//            break;
	//            case IValue::VK_STRING:
	//                param_kind = Param_info::PK_STRING;
	//                break;
	//            case IValue::VK_TEXTURE:
	//                param_kind = Param_info::PK_TEXTURE;
	//                break;
	//            case IValue::VK_LIGHT_PROFILE:
	//                param_kind = Param_info::PK_LIGHT_PROFILE;
	//                break;
	//            case IValue::VK_BSDF_MEASUREMENT:
	//                param_kind = Param_info::PK_BSDF_MEASUREMENT;
	//                break;
	//            default:
	//                // Unsupported? -> skip
	//                continue;
	//        }

	//        // Get the offset of the argument within the target argument block
	//        Target_value_layout_state state(arg_block_layout->get_nested_state(j));
	//        IValue::Kind kind2;
	//        Size param_size;
	//        Size offset = arg_block_layout->get_layout(kind2, param_size, state);
	//        if (kind != kind2)
	//        {
	//            continue;  // layout is invalid -> skip
	//        }

	//        Param_info param_info(j,
	//                              name,
	//                              name,
	//                              /*group_name=*/ "",
	//                              param_kind,
	//                              param_array_elem_kind,
	//                              param_array_size,
	//                              param_array_pitch,
	//                              arg_block_data + offset,
	//                              enum_type);

	//        // Check for annotation info
	//        Handle<IAnnotation_block const> anno_block(anno_list->get_annotation_block(name));
	//        if (anno_block)
	//        {
	//            Annotation_wrapper annos(anno_block.get());
	//            Size anno_index = annos.get_annotation_index("::anno::hard_range(float,float)");
	//            if (anno_index != Size(-1))
	//            {
	//                annos.get_annotation_param_value(anno_index, 0, param_info.range_min());
	//                annos.get_annotation_param_value(anno_index, 1, param_info.range_max());
	//            }
	//            else
	//            {
	//                anno_index = annos.get_annotation_index("::anno::soft_range(float,float)");
	//                if (anno_index != Size(-1))
	//                {
	//                    annos.get_annotation_param_value(anno_index, 0, param_info.range_min());
	//                    annos.get_annotation_param_value(anno_index, 1, param_info.range_max());
	//                }
	//            }
	//            anno_index = annos.get_annotation_index("::anno::display_name(string)");
	//            if (anno_index != Size(-1))
	//            {
	//                char const* display_name = nullptr;
	//                annos.get_annotation_param_value(anno_index, 0, display_name);
	//                param_info.set_display_name(display_name);
	//            }
	//            anno_index = annos.get_annotation_index("::anno::in_group(string)");
	//            if (anno_index != Size(-1))
	//            {
	//                char const* group_name = nullptr;
	//                annos.get_annotation_param_value(anno_index, 0, group_name);
	//                param_info.set_group_name(group_name);
	//            }
	//        }

	//        add_sorted_by_group(param_info);
	//    }

	//}

	// Utility function to dump the arguments of a material instance or function call.
	template <class T>
	void dumpInstance(IExpression_factory* expression_factory, const T* instance)
	{
		std::stringstream s;
		s << "Dumping material/function instance \"" << instance->get_mdl_function_definition() << "\":" << "\n";

		mi::Size count = instance->get_parameter_count();
		Handle<const IExpression_list> arguments(instance->get_arguments());

		for (mi::Size index = 0; index < count; index++) {

			Handle<const IExpression> argument(arguments->get_expression(index));
			std::string name = instance->get_parameter_name(index);
			Handle<const mi::IString> argument_text(expression_factory->dump(argument.get(), name.c_str(), 1));
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
		Size module_count = module->get_import_count();
		if (module_count > 0) {
			ss << "The module imports the following modules:" << "\n";
			for (Size i = 0; i < module_count; i++)
				ss << "    " << module->get_import(i) << "\n";
		}
		else {
			ss << "The module doesn't import other modules." << "\n";
		}

		// Dump exported types.
		Handle<IType_factory> type_factory(factory->create_type_factory(transaction));
		Handle<const IType_list> types(module->get_types());
		if (types->get_size() > 0) {
			ss << "\n";
			ss << "The module contains the following types: " << "\n";
			for (Size i = 0; i < types->get_size(); ++i) {
				Handle<const IType> type(types->get_type(i));
				Handle<const IString> result(type_factory->dump(type.get(), 1));
				ss << "    " << result->get_c_str() << "\n";
			}
		}
		else {
			ss << "The module doesn't contain any types " << "\n";
		}

		// Dump exported constants.
		Handle<IValue_factory> value_factory(factory->create_value_factory(transaction));
		Handle<const IValue_list> constants(module->get_constants());
		if (constants->get_size() > 0) {
			ss << "\n";
			ss << "The module contains the following constants: " << "\n";
			for (Size i = 0; i < constants->get_size(); ++i) {
				const char* name = constants->get_name(i);
				Handle<const IValue> constant(constants->get_value(i));
				Handle<const IString> result(value_factory->dump(constant.get(), 0, 1));
				ss << "    " << name << " = " << result->get_c_str() << "\n";
			}
		}
		else {
			ss << "The module doesn't contain any constants " << "\n";
		}

		// Dump function definitions of the module.
		Size function_count = module->get_function_count();
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
		Size material_count = module->get_material_count();
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
				Handle<const IValue_resource> resource(
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

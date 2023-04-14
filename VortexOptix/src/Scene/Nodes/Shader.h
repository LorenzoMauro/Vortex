#pragma once
#include <string>
#include <algorithm>
#include <mi/mdl_sdk.h>
#include "Scene/Node.h"

namespace vtx::graph
{

	using namespace mi;
	using namespace base;
	using namespace neuraylib;
            
	struct ShaderConfiguration {
		// The state of the expressions :
		bool isThinWalledConstant = false;
		bool isSurfaceBsdfValid = false;
		bool isBackfaceBsdfValid = false;
		bool is_surface_edf_valid = false;
		bool is_surface_intensity_constant = false;
		bool is_surface_intensity_mode_constant = false;
		bool is_backface_edf_valid = false;
		bool is_backface_intensity_constant = false;
		bool is_backface_intensity_mode_constant = false;
		bool use_backface_edf = false;
		bool use_backface_intensity = false;
		bool use_backface_intensity_mode = false;
		bool is_ior_constant = false;
		bool is_vdf_valid = false;
		bool is_absorption_coefficient_constant = false;
		bool use_volume_absorption = false;
		bool is_scattering_coefficient_constant = false;
		bool is_directional_bias_constant = false;
		bool use_volume_scattering = false;
		bool is_cutout_opacity_constant = false;
		bool use_cutout_opacity = false;
		bool is_hair_bsdf_valid = false;

		// The constant expression values:
		bool            thin_walled = false;
		mi::math::Color surface_intensity;
		Sint32 surface_intensity_mode{};
		mi::math::Color backface_intensity;
		Sint32 backface_intensity_mode{};
		mi::math::Color ior;
		mi::math::Color absorption_coefficient;
		mi::math::Color scattering_coefficient;
		Float32 directional_bias{};
		Float32 cutout_opacity{};

		bool isEmissive(ShaderConfiguration config) const
		{
			// Check if front face is emissive
			if (config.is_surface_edf_valid) {
				if (!config.is_surface_intensity_constant) {
					return true;
				}
				else {
					if (config.surface_intensity[0] != 0.0f || config.surface_intensity[1] != 0.0f || config.surface_intensity[2] != 0.0f) {
						return true;
					}
				}
			}

			// To be emissive on the backface it needs to be thinWalled
			if (config.isThinWalledConstant) {
				if (!config.thin_walled) {
					return false;
				}
			}

			// Check if back face is emissive
			if (config.is_backface_edf_valid) {
				if (!config.is_backface_intensity_constant) {
					return true;
				}
				else {
					if (config.backface_intensity[0] != 0.0f || config.backface_intensity[1] != 0.0f || config.backface_intensity[2] != 0.0f) {
						return true;
					}
				}
			}

			return false;
		}
	};
            
	struct FunctionNames {

		FunctionNames() = default;

		FunctionNames(const std::string& suffix) {
			init = "__direct_callable__init" + suffix;
			thin_walled = "__direct_callable__thin_walled" + suffix;
			surface_scattering = "__direct_callable__surface_scattering" + suffix;
			surface_emission_emission = "__direct_callable__surface_emission_emission" + suffix;
			surface_emission_intensity = "__direct_callable__surface_emission_intensity" + suffix;
			surface_emission_mode = "__direct_callable__surface_emission_mode" + suffix;
			backface_scattering = "__direct_callable__backface_scattering" + suffix;
			backface_emission_emission = "__direct_callable__backface_emission_emission" + suffix;
			backface_emission_intensity = "__direct_callable__backface_emission_intensity" + suffix;
			backface_emission_mode = "__direct_callable__backface_emission_mode" + suffix;
			ior = "__direct_callable__ior" + suffix;
			volume_absorption_coefficient = "__direct_callable__volume_absorption_coefficient" + suffix;
			volume_scattering_coefficient = "__direct_callable__volume_scattering_coefficient" + suffix;
			volume_directional_bias = "__direct_callable__volume_directional_bias" + suffix;
			geometry_cutout_opacity = "__direct_callable__geometry_cutout_opacity" + suffix;
			hair_bsdf = "__direct_callable__hair" + suffix;
		}

		std::string init;
		std::string thin_walled;
		std::string surface_scattering;
		std::string surface_emission_emission;
		std::string surface_emission_intensity;
		std::string surface_emission_mode;
		std::string backface_scattering;
		std::string backface_emission_emission;
		std::string backface_emission_intensity;
		std::string backface_emission_mode;
		std::string ior;
		std::string volume_absorption_coefficient;
		std::string volume_scattering_coefficient;
		std::string volume_directional_bias;
		std::string geometry_cutout_opacity;
		std::string hair_bsdf;
	};
		
	struct CompilationResult {
		Handle<ICompiled_material const>    compiledMaterial;
		Handle<ITarget_code const>          targetCode;
		Uuid 							    compilationHash{};      
	};

	struct TextureInfo {
		std::string                                                 databaseName;
		ITarget_code::Texture_shape                  shape;

		TextureInfo()
			: shape(ITarget_code::Texture_shape_invalid) {}

		TextureInfo(char const* _databaseName, ITarget_code::Texture_shape _shape)
			: databaseName(_databaseName), shape(_shape) {}
	};

	/// Information required to load a light profile.
	struct LightProfileInfo
	{
		std::string databaseName;

		LightProfileInfo() = default;

		LightProfileInfo(char const* _databaseName)
			: databaseName(_databaseName) {}
	};

	/// Information required to load a BSDF measurement.
	struct BsdfMeasuremnetInfo
	{
		std::string databaseName;

		BsdfMeasuremnetInfo() = default;

		BsdfMeasuremnetInfo(char const* _databaseName)
			: databaseName(_databaseName) {}
	};
       
	class Shader : public Node {
	public:
		Shader() : Node(NT_MDLSHADER) {};

		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;

		void accept(std::shared_ptr<NodeVisitor> visitor) override;

		ShaderConfiguration                                                 config;
		FunctionNames                                                       fNames;
		CompilationResult                                                   compilation;

		//Probably the next 3 vectors could be transformed into nodes that get created on shader loading.
		std::vector<TextureInfo>                                            textures;
		std::vector<LightProfileInfo>                                       lightProfiles;
		std::vector<BsdfMeasuremnetInfo>                                    bsdfMeasurements;
		Handle<ITarget_value_layout const> argLayout;
		Handle<ITarget_argument_block const> argumentBlock;
		std::string                                                         nameReference;	// example : "default"
		std::string                                                         name;			// example : "bsdf_diffuse_reflection"
		std::string                                                         path;			// example : "mdl\\bsdf_diffuse_reflection.mdl"
		bool                                                                isProcessed = false;
	};
}


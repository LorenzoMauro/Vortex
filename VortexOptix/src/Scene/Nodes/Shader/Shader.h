#pragma once
#include <string>
#include <algorithm>
#include <mi/mdl_sdk.h>
#include "Scene/Node.h"
#include "Device/OptixWrapper.h"
#include "Texture.h"
#include "LightProfile.h"
#include "BsdfMeasurement.h"

namespace vtx::graph
{
	using namespace mi;
	using namespace base;
	using namespace neuraylib;
        
	class Shader : public Node {
	public:
		Shader() : Node(NT_MDL_SHADER) {};

		struct Configuration {
			// The state of the expressions :
			bool isThinWalledConstant = false;
			bool isSurfaceBsdfValid = false;
			bool isBackfaceBsdfValid = false;
			bool isSurfaceEdfValid = false;
			bool isSurfaceIntensityConstant = false;
			bool isSurfaceIntensityModeConstant = false;
			bool isBackfaceEdfValid = false;
			bool isBackfaceIntensityConstant = false;
			bool isBackfaceIntensityModeConstant = false;
			bool useBackfaceEdf = false;
			bool useBackfaceIntensity = false;
			bool useBackfaceIntensityMode = false;
			bool isIorConstant = false;
			bool isVdfValid = false;
			bool isAbsorptionCoefficientConstant = false;
			bool useVolumeAbsorption = false;
			bool isScatteringCoefficientConstant = false;
			bool isDirectionalBiasConstant = false;
			bool useVolumeScattering = false;
			bool isCutoutOpacityConstant = false;
			bool useCutoutOpacity = false;
			bool isHairBsdfValid = false;

			// The constant expression values:
			bool            thinWalled = false;
			mi::math::Color surfaceIntensity;
			Sint32			surfaceIntensityMode{};
			mi::math::Color backfaceIntensity;
			Sint32			backfaceIntensityMode{};
			mi::math::Color ior;
			mi::math::Color absorptionCoefficient;
			mi::math::Color scatteringCoefficient;
			Float32			directionalBias{};
			Float32			cutoutOpacity{};
			bool            isEmissive = false;

		};

		struct FunctionNames {

			FunctionNames() = default;

			FunctionNames(const std::string& suffix) {
				init = "__direct_callable__init_" + suffix;
				thinWalled = "__direct_callable__thin_walled_" + suffix;

				surfaceScattering = "__direct_callable__surface_scattering_" + suffix;

				surfaceEmissionEmission = "__direct_callable__surface_emission_emission_" + suffix;
				surfaceEmissionIntensity = "__direct_callable__surface_emission_intensity_" + suffix;
				surfaceEmissionMode = "__direct_callable__surface_emission_mode_" + suffix;

				backfaceScattering = "__direct_callable__backface_scattering_" + suffix;

				backfaceEmissionEmission = "__direct_callable__backface_emission_emission_" + suffix;
				backfaceEmissionIntensity = "__direct_callable__backface_emission_intensity_" + suffix;
				backfaceEmissionMode = "__direct_callable__backface_emission_mode_" + suffix;

				ior = "__direct_callable__ior_" + suffix;

				volumeAbsorptionCoefficient = "__direct_callable__volume_absorption_coefficient_" + suffix;
				volumeScatteringCoefficient = "__direct_callable__volume_scattering_coefficient_" + suffix;
				volumeDirectionalBias = "__direct_callable__volume_directional_bias_" + suffix;

				geometryCutoutOpacity = "__direct_callable__geometry_cutout_opacity_" + suffix;

				hairBsdf = "__direct_callable__hair_" + suffix;
			}

			std::string init;
			std::string thinWalled;
			std::string surfaceScattering;
			std::string surfaceEmissionEmission;
			std::string surfaceEmissionIntensity;
			std::string surfaceEmissionMode;
			std::string backfaceScattering;
			std::string backfaceEmissionEmission;
			std::string backfaceEmissionIntensity;
			std::string backfaceEmissionMode;
			std::string ior;
			std::string volumeAbsorptionCoefficient;
			std::string volumeScatteringCoefficient;
			std::string volumeDirectionalBias;
			std::string geometryCutoutOpacity;
			std::string hairBsdf;
		};

		struct DevicePrograms
		{
			std::shared_ptr<optix::ProgramOptix> pgInit;
			std::shared_ptr<optix::ProgramOptix> pgThinWalled;

			std::shared_ptr<optix::ProgramOptix> pgSurfaceScatteringSample;
			std::shared_ptr<optix::ProgramOptix> pgSurfaceScatteringEval;
			std::shared_ptr<optix::ProgramOptix> pgSurfaceScatteringAuxiliary;

			std::shared_ptr<optix::ProgramOptix> pgBackfaceScatteringSample;
			std::shared_ptr<optix::ProgramOptix> pgBackfaceScatteringEval;
			std::shared_ptr<optix::ProgramOptix> pgBackfaceScatteringAuxiliary;

			std::shared_ptr<optix::ProgramOptix> pgSurfaceEmissionEval;
			std::shared_ptr<optix::ProgramOptix> pgSurfaceEmissionIntensity;
			std::shared_ptr<optix::ProgramOptix> pgSurfaceEmissionIntensityMode;

			std::shared_ptr<optix::ProgramOptix> pgBackfaceEmissionEval;
			std::shared_ptr<optix::ProgramOptix> pgBackfaceEmissionIntensity;
			std::shared_ptr<optix::ProgramOptix> pgBackfaceEmissionIntensityMode;

			std::shared_ptr<optix::ProgramOptix> pgIor;

			std::shared_ptr<optix::ProgramOptix> pgVolumeAbsorptionCoefficient;
			std::shared_ptr<optix::ProgramOptix> pgVolumeScatteringCoefficient;
			std::shared_ptr<optix::ProgramOptix> pgVolumeDirectionalBias;

			std::shared_ptr<optix::ProgramOptix> pgGeometryCutoutOpacity;

			std::shared_ptr<optix::ProgramOptix> pgHairSample;
			std::shared_ptr<optix::ProgramOptix> pgHairEval;

			bool isEmissive = false;
			bool isThinWalled = true;
			bool hasOpacity = false;
			//unsigned int  flags;
		};

		void init();

		void createChildResources();

		const Configuration& getConfiguration();

		const char* getTargetCode();

		Handle<ITarget_code const> getTargetCodeHandle();

		void createPrograms();

		const DevicePrograms& getPrograms();

		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;

		void accept(std::shared_ptr<NodeVisitor> visitor) override;
		std::string getMaterialDbName();

		bool isEmissive();

		std::vector<std::shared_ptr<graph::Texture>>  getTextures();
		std::vector<std::shared_ptr<graph::BsdfMeasurement>>  getBsdfs();
		std::vector<std::shared_ptr<graph::LightProfile>>  getLightProfiles();

		std::string                                                         name;			// example : "bsdf_diffuse_reflection"
		std::string                                                         materialDbName;	// example : "bsdf_diffuse_reflection"
		std::string                                                         path;			// example : "mdl\\bsdf_diffuse_reflection.mdl"

		Configuration														config;
		Handle<ITarget_code const>											targetCode;
		DevicePrograms														devicePrograms;

		//Child Resources
		std::vector<std::shared_ptr<graph::Texture>>                        textures;
		std::vector<std::shared_ptr<graph::LightProfile>>					lightProfiles;
		std::vector<std::shared_ptr<graph::BsdfMeasurement>>                bsdfMeasurements;

		Handle<ITarget_value_layout const>									argLayout;
		Handle<ITarget_argument_block const>								argumentBlock;
		bool                                                                isInitialized = false;
	};
}


#include "Material.h"
#include "Scene/Traversal.h"
#include "MDL/MdlWrapper.h"
#include "Scene/Nodes/Shader/mdl/ShaderNodes.h"
#include "Shader/BsdfMeasurement.h"
#include "Shader/LightProfile.h"
#include "Shader/Texture.h"

namespace vtx::graph
{
	void Material::init()
	{

		mdl::compileMaterial(materialGraph->name, materialGraph->functionInfo.signature);

		config = mdl::determineShaderConfiguration(materialGraph->functionInfo.signature);
		targetCode = mdl::createTargetCode(materialGraph->functionInfo.signature, config, getID());

		createPrograms();
		createChildResources();

		mdl::setMaterialParameters(materialGraph->functionInfo.signature, targetCode, argBlock, params, mapEnumTypes);

		std::string enable_emission = "enable_emission";
		std::string enable_opacity = "enable_opacity";
		bool* enableEmission = nullptr;
		bool* enableOpacity = nullptr;
		for(std::pair<const std::string, std::vector<ParamInfo>> param : params)
		{
			for(auto paramInfo : param.second)
			{
				std::string name = paramInfo.displayName();

				if (name.find(enable_emission) != std::string::npos)
				{
					enableEmission = &paramInfo.data<bool>();
				}
				else if (name.find(enable_opacity) != std::string::npos)
				{
					enableOpacity = &paramInfo.data<bool>();
				}
			}

		}
		if(enableEmission != nullptr)
		{
			config.emissivityToggle = enableEmission;

		}
		if(enableOpacity != nullptr)
		{
			config.opacityToggle = enableOpacity;
		}
		isInitialized = true;
	}

	size_t Material::getArgumentBlockSize()
	{
		if (!argBlock)
		{
			return 0;
		}
		return argBlock->get_size();
	}

	char* Material::getArgumentBlockData()
	{
		if (!argBlock)
		{
			return nullptr;
		}
		return argBlock->get_data();
	}

	void Material::traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors)
	{
		if(materialGraph)
		{
			if(materialGraph->isUpdated)
			{
				auto materialCasted = std::dynamic_pointer_cast<graph::shader::Material>(materialGraph);
				mdl::createShaderGraphFunctionCalls(materialGraph);
				//shader->materialDbName = materialGraph->functionInfo.signature;
				//shader->materialCallName = materialGraph->name;
				//shader->name = materialGraph->name;
				//auto [moduleName, functionName] =mdl::createNewFunctionInModule(materialGraph);
				//shader->name = functionName;
				//shader->path = moduleName;
			}
		}

		//shader->traverse(orderedVisitors);
		if(!isInitialized)
		{
			init();
		}
		for (auto& texture : textures)
		{
			texture->traverse(orderedVisitors);
		}
		for (auto& lightProfile : lightProfiles)
		{
			lightProfile->traverse(orderedVisitors);
		}
		for (auto& bsdfMeasurement : bsdfMeasurements)
		{
			bsdfMeasurement->traverse(orderedVisitors);
		}
		ACCEPT(Material,visitors);
	}

	/*void Material::accept(std::shared_ptr<NodeVisitor> visitor)
	{
		visitor->visit(sharedFromBase<Material>());
	}*/


	void Material::createPrograms()
	{
		auto module = std::make_shared<optix::ModuleOptix>();
		module->name = name;
		module->code = targetCode->get_code();
		if (getOptions()->directCallable)
		{
			const auto fNames = graph::FunctionNames(std::to_string(getID()));

			devicePrograms.pgInit = optix::createDcProgram(module, fNames.init);

			if (!config.isThinWalledConstant)
			{
				devicePrograms.pgThinWalled = optix::createDcProgram(module, (fNames.thinWalled));
			}

			if (config.isSurfaceBsdfValid)
			{
				devicePrograms.pgSurfaceScatteringSample = optix::createDcProgram(module, fNames.surfaceScattering + "_sample");
				devicePrograms.pgSurfaceScatteringEval = optix::createDcProgram(module, fNames.surfaceScattering + "_evaluate");
				devicePrograms.pgSurfaceScatteringAuxiliary = optix::createDcProgram(module, fNames.surfaceScattering + "_auxiliary");
			}

			if (config.isBackfaceBsdfValid)
			{
				devicePrograms.pgBackfaceScatteringSample = optix::createDcProgram(module, fNames.backfaceScattering + "_sample");
				devicePrograms.pgBackfaceScatteringEval = optix::createDcProgram(module, fNames.backfaceScattering + "_evaluate");
				devicePrograms.pgBackfaceScatteringAuxiliary = optix::createDcProgram(module, fNames.backfaceScattering + "_auxiliary");
			}

			if (config.isSurfaceEdfValid)
			{
				devicePrograms.pgSurfaceEmissionEval = optix::createDcProgram(module, fNames.surfaceEmissionEmission + "_evaluate");

				if (!config.isSurfaceIntensityConstant)
				{
					devicePrograms.pgSurfaceEmissionIntensity = optix::createDcProgram(module, fNames.surfaceEmissionIntensity);
				}

				if (!config.isSurfaceIntensityModeConstant)
				{
					devicePrograms.pgSurfaceEmissionIntensityMode = optix::createDcProgram(module, fNames.surfaceEmissionMode);
				}
			}

			if (config.isBackfaceEdfValid)
			{
				if (config.useBackfaceEdf)
				{
					devicePrograms.pgBackfaceEmissionEval = optix::createDcProgram(module, fNames.backfaceEmissionEmission + "_evaluate");
				}
				else // Surface and backface expressions were identical. Reuse the code of the surface expression.
				{
					devicePrograms.pgBackfaceEmissionEval = devicePrograms.pgSurfaceEmissionEval;
				}

				if (config.useBackfaceIntensity)
				{
					if (!config.isBackfaceIntensityConstant)
					{
						devicePrograms.pgBackfaceEmissionIntensity = optix::createDcProgram(module, fNames.backfaceEmissionIntensity);
					}
				}
				else // Surface and backface expressions were identical. Reuse the code of the surface expression.
				{
					devicePrograms.pgBackfaceEmissionIntensity = devicePrograms.pgSurfaceEmissionIntensity;
				}

				if (config.useBackfaceIntensityMode)
				{
					if (!config.isBackfaceIntensityModeConstant)
					{
						devicePrograms.pgBackfaceEmissionIntensityMode = optix::createDcProgram(module, fNames.backfaceEmissionMode);
					}
				}
				else // Surface and backface expressions were identical. Reuse the code of the surface expression.
				{
					devicePrograms.pgBackfaceEmissionIntensityMode = devicePrograms.pgSurfaceEmissionIntensityMode;
				}
			}

			if (!config.isIorConstant)
			{
				devicePrograms.pgIor = optix::createDcProgram(module, fNames.ior);
			}

			if (!config.isAbsorptionCoefficientConstant)
			{
				devicePrograms.pgVolumeAbsorptionCoefficient = optix::createDcProgram(module, fNames.volumeAbsorptionCoefficient);
			}

			if (config.isVdfValid)
			{
				// The MDL SDK doesn't generate code for the volume.scattering expression.
				// Means volume scattering must be implemented by the renderer and only the parameter expresssions can be generated.

				// The volume scattering coefficient and direction bias are only used when there is a valid VDF. 
				if (!config.isScatteringCoefficientConstant)
				{
					devicePrograms.pgVolumeScatteringCoefficient = optix::createDcProgram(module, fNames.volumeScatteringCoefficient);
				}

				if (!config.isDirectionalBiasConstant)
				{
					devicePrograms.pgVolumeDirectionalBias = optix::createDcProgram(module, fNames.volumeDirectionalBias);
				}

				// volume.scattering.emission_intensity not implemented.
			}

			if (config.useCutoutOpacity)
			{
				//devicePrograms.hasOpacity = true;

				if (!config.isCutoutOpacityConstant)
				{
					devicePrograms.pgGeometryCutoutOpacity = optix::createDcProgram(module, fNames.geometryCutoutOpacity);
				}
			}

			if (config.isHairBsdfValid)
			{
				devicePrograms.pgHairSample = optix::createDcProgram(module, (fNames.hairBsdf + "_sample"));
				devicePrograms.pgHairEval = optix::createDcProgram(module, (fNames.hairBsdf + "_evaluate"));
			}
		}
		else
		{
			devicePrograms.pgEvaluateMaterial = optix::createDcProgram(module, "__direct_callable__EvaluateMaterial", getID());
		}
	}


	void Material::createChildResources()
	{
		// TODO We have to store the textures, light profiles and bsdf indices since these will be refencered by the mdl lookup functions
		for (Size i = 1, n = targetCode->get_texture_count(); i < n; ++i) {
			auto texture = std::make_shared<graph::Texture>(targetCode->get_texture(i), targetCode->get_texture_shape(i));
			texture->mdlIndex = i;
			textures.emplace_back(texture);
		}

		if (targetCode->get_light_profile_count() > 0)
		{
			for (mi::Size i = 1, n = targetCode->get_light_profile_count(); i < n; ++i)
			{
				auto lightProfile = std::make_shared<graph::LightProfile>(targetCode->get_light_profile(i));
				lightProfile->mdlIndex = i;
				lightProfiles.emplace_back(lightProfile);
			}
		}

		if (targetCode->get_bsdf_measurement_count() > 0)
		{
			for (mi::Size i = 1, n = targetCode->get_bsdf_measurement_count(); i < n; ++i)
			{
				auto bsdfMeasurement = std::make_shared<graph::BsdfMeasurement>(targetCode->get_bsdf_measurement(i));
				bsdfMeasurement->mdlIndex = i;
				bsdfMeasurements.emplace_back(bsdfMeasurement);
			}
		}
	}


	const Configuration& Material::getConfiguration()
	{
		if (!isInitialized)
		{
			init();
		}
		return config;
	}


	const DevicePrograms& Material::getPrograms()
	{
		if (!isInitialized)
		{
			init();
		}
		return devicePrograms;
	}


	std::vector<std::shared_ptr<graph::Texture>>  Material::getTextures()
	{
		if (!isInitialized)
		{
			init();
		}
		return textures;
	}

	std::vector<std::shared_ptr<graph::BsdfMeasurement>> Material::getBsdfs()
	{
		if (!isInitialized)
		{
			init();
		}
		return bsdfMeasurements;
	}

	std::vector<std::shared_ptr<graph::LightProfile>> Material::getLightProfiles()
	{
		if (!isInitialized)
		{
			init();
		}
		return lightProfiles;
	}

	bool Material::isThinWalled()
	{
		if (!config.isThinWalledConstant || (config.isThinWalledConstant && config.isThinWalled)) {
			return true;
		}
		return false;
	}

	bool Material::useEmission()
	{
		if (!isInitialized)
		{
			init();
		}
		bool thinWalled = isThinWalled();

		bool isSurfaceEmissive = false;
		if (config.isSurfaceEdfValid) {
			if (!config.isSurfaceIntensityConstant) {
				isSurfaceEmissive = true;
			}
			else if (config.surfaceIntensity != math::vec3f(0.0f)) {
				isSurfaceEmissive = true;
			}
		}


		bool isBackfaceEmissive = false;
		if (config.isBackfaceEdfValid) {
			if (!config.isBackfaceIntensityConstant) {
				isBackfaceEmissive = true;
			}
			else if (config.backfaceIntensity != math::vec3f(0.0f)) {
				isBackfaceEmissive = true;
			}
		}
		isBackfaceEmissive = (thinWalled && isBackfaceEmissive);
		bool isEmissive = isSurfaceEmissive || isBackfaceEmissive;// To be emissive on the backface it needs to be isThinWalled

		if (config.emissivityToggle) //The material has a enable emission option
		{
			if (*(config.emissivityToggle))
			{
				return isEmissive;
			}
			return false;
		}
		return isEmissive;
	}

	bool Material::useOpacity()
	{
		if (!isInitialized)
		{
			init();
		}
		if (config.opacityToggle) //The material has a enable emission option
		{
			if (*(config.opacityToggle))
			{
				return config.useCutoutOpacity;
			}
			return false;
		}
		return config.useCutoutOpacity;
	};
}

